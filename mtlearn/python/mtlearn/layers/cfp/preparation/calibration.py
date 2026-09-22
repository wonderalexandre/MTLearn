from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch

from ..storage import DiskStore
from ._calibration_measurement import CalibrationStopped, distribution, measure, recommend
from ._prepared_reader import contract, describe, open_dataset
from ._process_reader import process_description
from ._read_diagnostics import available_memory, process_memory


def _integer(name, value, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer >= {minimum}" + (f" and <= {maximum}." if maximum else "."))


def _positions(count, limit):
    if limit == 1:
        return [count // 2]
    return [i * (count - 1) // (limit - 1) for i in range(limit)]


def _select(store, manifest, expected, max_samples, sample_indices, metadata_samples, workspace, check):
    dataset = open_dataset(store, manifest, None, expected)
    count = len(dataset)
    if not count:
        raise ValueError("Calibration requires a nonempty complete manifest.")
    if sample_indices is None:
        inspected = _positions(count, min(metadata_samples, count))
    else:
        inspected = list(sample_indices)
        if not inspected or len(inspected) > max_samples or any(type(i) is not int or not 0 <= i < count for i in inspected):
            raise ValueError("sample_indices must contain 1..max_samples valid manifest positions.")
    descriptions = []
    for position in inspected:
        check()
        descriptions.append(describe(store, dataset, (position,), None, workspace, None, True))
    if sample_indices is None:
        descriptions.sort(key=lambda d: (d["payload_bytes"], d["indices"]))
        descriptions = [descriptions[i] for i in _positions(len(descriptions), min(max_samples, len(descriptions))) ]
    return descriptions, len(inspected)


def _candidates(descriptions, modes, ram_cache_bytes, worker_memory_bytes, workspace, validation, immutable):
    largest = max(d["payload_bytes"] for d in descriptions)
    common = dict(batch_size=1, shuffle=False, read_workspace_bytes=workspace,
                  validation=validation, immutable=immutable, close_timeout=2.)
    result = []
    for mode in ("synchronous", *modes):
        workers = {"process1": 1, "process2": 2}.get(mode, 0)
        prefetch = mode == "thread" or bool(workers)
        mmap = mode != "buffered"
        batch_bytes = largest * (1 if mmap else 2) + workspace
        if workers:
            batch_bytes = max(process_description(d, 2**63)["reserved_bytes"] for d in descriptions)
        tasks = min(2, len(descriptions)) if prefetch else 1
        ram = ram_cache_bytes if mode == "ram" else 0
        queue = tasks * batch_bytes if prefetch else 0
        options = common | dict(mmap=mmap, max_ram_bytes=ram, prefetch=prefetch,
            max_prefetch_bytes=queue, max_prefetch_tasks=tasks, max_batch_bytes=batch_bytes,
            num_workers=workers, worker_threads=1, worker_memory_bytes=worker_memory_bytes if workers else None)
        active = largest + (workspace if workers else 0)
        planned = active + (queue if prefetch else batch_bytes) + ram + min(workers, tasks) * (worker_memory_bytes or 0)
        result.append({"mode": mode, "options": options, "planned_memory_bytes": planned,
                       "status": "pending", "memory_available": True, "trials": [], "admissions": []})
    return result


def calibrate_disk_cache(path, manifest, *, max_samples, max_seconds, max_memory_bytes,
                         max_cache_read_bytes, sample_indices=None, metadata_samples=128,
                         modes=("buffered", "thread"), repeats=3, warmup_rounds=1,
                         ram_cache_bytes=0, worker_memory_bytes=None, read_workspace_bytes=8 * 1024**2,
                         validation="always", immutable=False, min_available_memory_bytes=0,
                         max_rss_bytes=None, progress=None, cancel=None):
    for name, value in (("max_memory_bytes", max_memory_bytes), ("max_cache_read_bytes", max_cache_read_bytes)):
        _integer(name, value)
    _integer("max_samples", max_samples, maximum=64)
    _integer("metadata_samples", metadata_samples, maximum=1024)
    _integer("repeats", repeats, maximum=10)
    _integer("warmup_rounds", warmup_rounds, minimum=0, maximum=2)
    for name, value in (("ram_cache_bytes", ram_cache_bytes), ("read_workspace_bytes", read_workspace_bytes),
                        ("min_available_memory_bytes", min_available_memory_bytes)):
        _integer(name, value, minimum=0)
    if worker_memory_bytes is not None:
        _integer("worker_memory_bytes", worker_memory_bytes)
    if max_rss_bytes is not None:
        _integer("max_rss_bytes", max_rss_bytes)
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive.")
    if sample_indices is not None and (not isinstance(sample_indices, Sequence) or isinstance(sample_indices, (str, bytes))):
        raise ValueError("sample_indices must be a finite sequence.")
    if sample_indices is not None and not 1 <= len(sample_indices) <= max_samples:
        raise ValueError("sample_indices must contain 1..max_samples positions.")
    if not isinstance(modes, (tuple, list)) or len(modes) > 5 or any(type(mode) is not str for mode in modes):
        raise ValueError("modes must be a sequence of at most five mode names.")
    if len(set(modes)) != len(modes) or set(modes) - {"buffered", "thread", "ram", "process1", "process2"}:
        raise ValueError("Supported comparison modes: buffered, thread, ram, process1, process2, without duplicates.")
    if "ram" in modes and not ram_cache_bytes:
        raise ValueError("The ram comparison requires an explicit positive ram_cache_bytes.")
    if any(mode.startswith("process") for mode in modes) and worker_memory_bytes is None:
        raise ValueError("Process comparisons require worker_memory_bytes per process.")
    if sample_indices is None and metadata_samples < max_samples:
        raise ValueError("metadata_samples must be >= max_samples for automatic selection.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None.")
    if cancel is not None and not callable(cancel) and not callable(getattr(cancel, "is_set", None)):
        raise TypeError("cancel must be callable, an event, or None.")
    started = time.monotonic()
    report = {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "path": str(Path(path).resolve()), "manifest": manifest, "status": "running",
        "scope": "prepared_payload_read_and_touch", "page_cache_controlled": False,
        "source_and_targets_read": False, "model_executed": False,
        "physical_disk_read_bytes": None, "device_available_memory_bytes": None,
        "device_memory_reason": "CPU-only calibration; accelerator context is not queried or initialized.",
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(),
                    "torch_threads": torch.get_num_threads(), "caller": process_memory()},
        "budgets": {"max_samples": max_samples, "max_seconds": max_seconds, "max_memory_bytes": max_memory_bytes,
                    "max_cache_read_bytes": max_cache_read_bytes, "max_rss_bytes": max_rss_bytes,
                    "min_available_memory_bytes": min_available_memory_bytes},
        "reserved_cache_read_bytes": 0, "candidates": [], "samples": [], "repeats": repeats,
        "warmup_rounds": warmup_rounds,
        "limitations": ["Memory reservations are estimates for reader buffers and declared worker overhead, not a hard RSS cap.",
            "RSS observations are sampled and may miss peaks; summing processes can double-count shared pages.",
            "Worker CPU is cumulative through its latest observation, excluding later shutdown work.",
            "Logical bytes count file sizes submitted for load/map, checksums, and tensor touch separately; they are not physical SSD traffic.",
            "Time and RSS guards are cooperative between reads and hash chunks; an in-flight I/O and closing can exceed the limit.",
            "Each trial opens new readers; LRU/session validation are reset. There is no artificial RAM priming.",
            "Payload hashing is the consumer workload. This recommendation does not predict end-to-end training speed.",
            "The sampled largest entry is not a guaranteed maximum of the full dataset."]}
    def check():
        if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
            raise CalibrationStopped("cancelled")
        if time.monotonic() - started >= max_seconds:
            raise CalibrationStopped("time_limit")
        available = available_memory()
        if available is not None and available < min_available_memory_bytes:
            raise CalibrationStopped("available_memory_limit")
    expected_signatures = None
    try:
        check()
        with DiskStore(path, readonly=True, validation=validation, immutable=immutable) as store:
            expected = contract(store, manifest)
            generation = store._read_generation()
            descriptions, inspected = _select(store, manifest, expected, max_samples, sample_indices,
                                              metadata_samples, read_workspace_bytes, check)
            report["contract"] = dict(zip(("config", "source_version", "preprocessing_version", "split", "sample_count"), expected))
            report["contract"]["config"] = json.loads(expected[0])
            report["contract_sha256"] = hashlib.sha256(json.dumps(expected).encode()).hexdigest()
            report["metadata_samples_inspected"] = inspected
            report["selection"] = "explicit_order" if sample_indices is not None else "size_quantiles_of_uniform_metadata_sample"
            report["samples"] = [{"position": d["indices"][0], "sample_id": d["sample_ids"][0],
                                  "registered_payload_bytes": d["payload_bytes"],
                                  "shape": json.loads(store._db.execute(
                                      "SELECT shape FROM samples WHERE manifest=? AND position=?",
                                      (manifest, d["indices"][0])).fetchone()[0])} for d in descriptions]
            indices = [d["indices"][0] for d in descriptions]
            candidates = _candidates(descriptions, modes, ram_cache_bytes, worker_memory_bytes,
                                     read_workspace_bytes, validation, immutable)
            report["candidates"] = candidates
            trial_budget = 3 * sum(d["payload_bytes"] for d in descriptions)
            for candidate in candidates:
                if candidate["planned_memory_bytes"] > max_memory_bytes:
                    candidate.update(status="skipped", reason="memory_budget")
            if candidates[0]["status"] == "skipped":
                raise CalibrationStopped("baseline_memory_budget")
            for round_index in range(warmup_rounds + repeats):
                warmup = round_index < warmup_rounds
                order = candidates if round_index == 0 or round_index % 2 == 0 else list(reversed(candidates))
                for candidate in order:
                    check()
                    if candidate["status"] in ("skipped", "failed"):
                        continue
                    if store._read_generation() != generation or contract(store, manifest) != expected:
                        raise CalibrationStopped("manifest_changed")
                    available = available_memory()
                    candidate["memory_available"] &= available is not None
                    candidate["admissions"].append({"round": round_index, "available_memory_bytes": available,
                                                    "required_memory_bytes": candidate["planned_memory_bytes"] + min_available_memory_bytes})
                    if available is not None and candidate["planned_memory_bytes"] + min_available_memory_bytes > available:
                        candidate.update(status="skipped", reason="available_memory_budget")
                        if candidate is candidates[0]:
                            raise CalibrationStopped("baseline_available_memory_budget")
                        continue
                    if report["reserved_cache_read_bytes"] + trial_budget > max_cache_read_bytes:
                        raise CalibrationStopped("cache_read_budget")
                    report["reserved_cache_read_bytes"] += trial_budget
                    if progress:
                        progress({"phase": "trial_start", "mode": candidate["mode"], "round": round_index,
                                  "warmup": warmup, "reserved_cache_read_bytes": report["reserved_cache_read_bytes"]})
                    trial = measure(path, manifest, indices, candidate["options"], check, max_rss_bytes)
                    trial.update(round=round_index, warmup=warmup, available_memory_before_bytes=available)
                    candidate["trials"].append(trial)
                    if trial["status"] != "complete":
                        raise CalibrationStopped(trial.get("reason", "trial_failed"))
                    if expected_signatures is None:
                        expected_signatures = trial["signatures"]
                    trial["equivalent"] = trial["signatures"] == expected_signatures
                    if not trial["equivalent"]:
                        raise CalibrationStopped("payload_equivalence_failed")
                    counters = trial["counters"]
                    if counters["sync_fallback_batches"] or counters.get("prefetch_reason"):
                        candidate.update(status="skipped", reason="effective_mode_fallback")
                    else:
                        candidate["status"] = "complete"
                    if progress:
                        progress({"phase": "trial_complete", "mode": candidate["mode"], "round": round_index,
                                  "warmup": warmup, "wall_seconds": trial["wall_seconds"], "equivalent": trial["equivalent"]})
            if store._read_generation() != generation:
                raise CalibrationStopped("manifest_changed")
            report["status"] = "complete"
    except CalibrationStopped as exc:
        report.update(status="stopped", reason=str(exc))
    for candidate in report["candidates"]:
        measured = [t for t in candidate["trials"] if not t["warmup"] and t["status"] == "complete" and t.get("equivalent")]
        for key in ("wall_seconds", "wait_seconds", "touch_seconds", "caller_cpu_seconds"):
            candidate[key] = distribution([t[key] for t in measured])
    if report["candidates"]:
        report["recommendation"] = recommend(report["candidates"], repeats, report["status"] == "complete")
    else:
        report["recommendation"] = {"mode": "synchronous", "validated": False, "reason": "no_completed_baseline"}
    report["elapsed_seconds"] = time.monotonic() - started
    return report
