import hashlib
import json
import statistics
import time

import torch

from ._read_diagnostics import process_memory
from .prepared_dataloader import PreparedDataLoader


class CalibrationStopped(Exception):
    pass


def batch_digest(batch, check):
    digest = hashlib.sha256()
    touched = 0
    def metadata(value):
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    metadata((batch.sample_ids, batch.shape))
    for sample in batch.samples:
        for channel in sample:
            for key, prepared in sorted(channel.items()):
                metadata((key, prepared.format_version, [a.name for a in prepared.feature_spec.attributes]))
                for group, values in (("info", prepared.info), ("attributes", prepared.raw_attributes)):
                    for name, value in sorted(values.items(), key=lambda pair: str(pair[0])):
                        metadata((group, str(name)))
                        if torch.is_tensor(value):
                            metadata((str(value.dtype), tuple(value.shape)))
                            data = memoryview(value.numpy()).cast("B")
                            for offset in range(0, len(data), 1024**2):
                                check()
                                digest.update(data[offset:offset + 1024**2])
                            touched += len(data)
                        else:
                            metadata(value)
    return digest.hexdigest(), touched


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {"count": len(values), "median": median, "min": ordered[0], "max": ordered[-1],
            "median_absolute_deviation": statistics.median(abs(n - median) for n in ordered)}


def measure(path, manifest, indices, options, check, max_rss_bytes):
    started, cpu_started = time.perf_counter(), time.process_time()
    loader = iterator = batch = None
    result = {"status": "running", "signatures": [], "sample_ids": [], "wait_seconds": 0.,
              "touch_seconds": 0., "touched_tensor_bytes": 0, "memory_observations": []}
    def observe():
        observation = {"caller": process_memory(), "workers": list(loader.counters().get("worker_memory_observations", ())) if loader else []}
        result["memory_observations"].append(observation)
        rss = [item["rss_bytes"] for item in [observation["caller"], *observation["workers"]]]
        if max_rss_bytes is not None:
            if any(value is None for value in rss):
                raise CalibrationStopped("rss_unavailable")
            if sum(rss) > max_rss_bytes:
                raise CalibrationStopped("rss_limit")
    try:
        check()
        observe()
        loader = PreparedDataLoader(path, manifest, sampler=indices,
            generator=torch.Generator().manual_seed(0), collect_read_metrics=True, **options)
        iterator = iter(loader)
        for _ in indices:
            check()
            before = time.perf_counter()
            batch = next(iterator)
            result["wait_seconds"] += time.perf_counter() - before
            observe()
            check()
            before = time.perf_counter()
            signature, touched = batch_digest(batch, check)
            result["touch_seconds"] += time.perf_counter() - before
            result["signatures"].append(signature)
            result["sample_ids"].extend(batch.sample_ids)
            result["touched_tensor_bytes"] += touched
            observe()
            batch = None
        check()
        result["status"] = "complete"
    except CalibrationStopped as exc:
        result.update(status="stopped", reason=str(exc))
    except Exception as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}: {str(exc)[:4096]}")
    finally:
        batch = None
        closed = True
        if loader is not None:
            closed = loader.close()
            result["counters"] = loader.counters()
        result["reader_closed"] = closed
        if not closed:
            result.update(status="stopped", reason="reader_still_closing")
        result["wall_seconds"] = time.perf_counter() - started
        result["caller_cpu_seconds"] = time.process_time() - cpu_started
    return result


def recommend(candidates, repeats, complete):
    baseline = candidates[0]
    result = {"mode": "synchronous", "options": None, "scope": "prepared_payload_read_and_touch",
              "reason": "incomplete_calibration", "validated": False}
    if not complete or repeats < 3:
        return result
    reference = baseline.get("wall_seconds")
    if reference is None or reference["count"] != repeats:
        return result
    result.update(reason="no_consistent_gain", validated=True, options=baseline["options"])
    eligible = [c for c in candidates[1:] if c["status"] == "complete" and c["memory_available"]
                and c["wall_seconds"]["count"] == repeats and c["wall_seconds"]["max"] < reference["min"]]
    if eligible:
        chosen = min(eligible, key=lambda c: c["wall_seconds"]["median"])
        result.update(mode=chosen["mode"], options=chosen["options"], reason="separated_repeated_wall_times",
                      median_ratio=chosen["wall_seconds"]["median"] / reference["median"])
    return result
