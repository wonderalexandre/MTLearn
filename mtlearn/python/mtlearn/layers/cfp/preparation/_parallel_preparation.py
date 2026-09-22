"""Bounded CPU workers; only the calling process owns the manifest.

Inputs are canonical uint8 arrays loaded by the caller. Workers never receive a
Dataset, model, store handle or normalizer, and return only file descriptors.
"""
from collections import deque
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from ..._helpers import to_numpy_u8
from ._identity import (canonical_json, digest_file, image_identity, identity_key,
                        preprocessor_config, preprocessor_from_config)
from ._persistent_preparation import _bind_sample, sample_image
from ..storage._disk_format import pack
from ..storage._writer_lock import _QuotaWriter, _WriterLock
from ._preparation_progress import emit


def validate_parallel_options(store, num_workers, max_in_flight, worker_threads, max_sample_bytes, max_retries):
    for name, value, minimum in (("num_workers", num_workers, 0), ("worker_threads", worker_threads, 1),
                                 ("max_sample_bytes", max_sample_bytes, 1), ("max_retries", max_retries, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if max_in_flight is not None and (type(max_in_flight) is not int or max_in_flight < 1):
        raise ValueError("max_in_flight must be a positive integer or None.")
    if num_workers:
        from ..storage import DiskStore
        if not isinstance(store, DiskStore):
            raise ValueError("num_workers > 0 currently requires a DiskStore; use num_workers=0 for RAM/no retention.")
        if mp.current_process().daemon or torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("Start preparation in the main application, outside DataLoader workers or another daemon pool.")
    elif max_retries or max_in_flight is not None:
        raise ValueError("max_retries/max_in_flight require num_workers > 0.")
    return max_in_flight if max_in_flight is not None else max(1, num_workers)


def _initialize_worker(config, threads):
    # Runtime setters also cover libraries imported while spawn imports __main__.
    # This backend has no independent OpenMP region; Torch owns its CPU runtime.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    import cv2
    # Zero disables OpenCV parallel regions even on Apple GCD builds, which
    # ignore positive thread counts. Keep image I/O below the worker budget.
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    return preprocessor_from_config(config)


def _write_entry(preprocessor, image, entry, budget, limit):
    """Serialize under a shared byte counter, without a SQLite connection.

    One writer at a time bounds serialization buffers and prevents two workers
    from spending the same free quota. Tree construction remains parallel.
    Published replacements can make the counter conservative until the next run.
    """
    prepared = preprocessor.prepare_u8(image, entry["tree_key"])
    data = pack(entry["identity"], prepared)
    path = Path(entry["temporary"])
    with budget.get_lock():
        writer = None
        try:
            with path.open("xb") as handle:
                writer = _QuotaWriter(handle, limit - budget.value,
                    min_free_disk_bytes=entry.get("min_free_disk_bytes", 0), disk_path=path.parent)
                torch.save(data, writer)
                handle.flush()
                os.fsync(handle.fileno())
            size = path.stat().st_size
            budget.value += size
        except BaseException as exc:
            path.unlink(missing_ok=True)
            if writer is not None and writer.exceeded:
                raise OSError("DiskStore quota exceeded; increase max_disk_bytes and resume. Completed entries were preserved.") from exc
            if writer is not None and writer.reserve_exceeded:
                raise OSError("DiskStore free-space reserve reached; free disk space and resume. Completed entries were preserved.") from exc
            raise
    return {"kind": "entry", "key": entry["key"], "temporary": str(path),
            "sha256": digest_file(path), "summary": data["summary"], "size_bytes": size}


def _worker_loop(connection, config, threads, budget, limit, lease_path):
    lease, task = None, None
    try:
        lease = _WriterLock(lease_path)
        preprocessor = _initialize_worker(config, threads)
        # A live coordinator must acknowledge readiness after this lease exists.
        # A new writer cannot recover files while an orphan worker is active.
        connection.send({"kind": "ready"})
        while True:
            task = connection.recv()
            if task is None:
                return
            try:
                for entry in task["entries"]:
                    message = _write_entry(preprocessor, task["images"][entry["channel"]], entry, budget, limit)
                    connection.send(message)
                    # Backpressure: do not retain/build a second payload until
                    # the coordinator validates and publishes this one.
                    if connection.recv() != "published":
                        return
                connection.send({"kind": "done"})
            except Exception as exc:
                connection.send({"kind": "error", "error": f"{type(exc).__name__}: {exc}",
                                 "quota": isinstance(exc, OSError) and any(text in str(exc) for text in
                                     ("quota exceeded", "free-space reserve"))})
            finally:
                for entry in task["entries"]:
                    Path(entry["temporary"]).unlink(missing_ok=True)
                task = None
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        connection.send({"kind": "fatal", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()
        if lease is not None:
            lease.close()
            Path(lease_path).unlink(missing_ok=True)


class _Workers:
    def __init__(self, preprocessor, store, count, threads):
        self.context = mp.get_context("spawn")
        self.budget = self.context.Value("q", store._tensor_file_bytes())
        self.slots = []
        try:
            for _ in range(count):
                parent, child = self.context.Pipe()
                lease_path = str(store.path / f".worker.{uuid.uuid4().hex}.lock")
                process = self.context.Process(target=_worker_loop,
                    args=(child, preprocessor_config(preprocessor), threads, self.budget, store.max_disk_bytes, lease_path),
                    daemon=True, name="mtlearn-cfp-prepare")
                slot = {"process": process, "connection": parent, "task": None, "ready": False, "lease_path": lease_path}
                self.slots.append(slot)
                try:
                    process.start()
                finally:
                    child.close()
        except BaseException:
            self.close()
            raise

    def check(self):
        for slot in self.slots:
            code = slot["process"].exitcode
            if code is not None:
                raise RuntimeError(f"CFP CPU worker exited unexpectedly (exit code {code}); resume the same manifest. "
                                   "Abrupt process failures are not retried automatically.")

    def close(self):
        # Public Process APIs, including on Python versions without executor
        # terminate_workers(). Discard locks/IPC after any forced termination.
        for slot in self.slots:
            if slot["process"].is_alive():
                slot["process"].terminate()
        for slot in self.slots:
            process = slot["process"]
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            slot["connection"].close()
            Path(slot["lease_path"]).unlink(missing_ok=True)


def _new_task(preprocessor, source, index, sample_ids, store, manifest, max_sample_bytes):
    sample_id = str(index) if sample_ids is None else (sample_ids(index) if callable(sample_ids) else sample_ids[index])
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_ids must be nonempty strings.")
    item = source[index]
    image, target = sample_image(item)
    # The original source item is transient. Only the bounded canonical CPU copy
    # is held for dispatch/retries; masks and arbitrary source objects stay here.
    if image.numel() > max_sample_bytes:
        raise ValueError("Canonical image exceeds max_sample_bytes; increase this per-image input budget or use num_workers=0.")
    images = np.stack([np.ascontiguousarray(to_numpy_u8(channel)) for channel in image])
    entries, bindings = {}, []
    for channel, pixels in enumerate(images):
        trees = {}
        for tree_key in preprocessor.tree_specs:
            identity = image_identity(preprocessor, pixels, tree_key)
            key = identity_key(identity)
            trees[tree_key] = key
            entries.setdefault(key, {"key": key, "identity": identity, "channel": channel, "tree_key": tree_key,
                                    "min_free_disk_bytes": store.min_free_disk_bytes,
                                    "temporary": str(store._files / f".{key}.{uuid.uuid4().hex}.tmp")})
        bindings.append(trees)
    old = store._db.execute("SELECT * FROM samples WHERE manifest=? AND position=?", (manifest, index)).fetchone()
    if old is not None and (old["sample_id"] != sample_id or old["bindings"] != canonical_json(bindings)):
        raise ValueError("Source content/order changed under the same manifest version; use a new manifest name.")
    return {"index": index, "sample_id": sample_id, "images": images, "bindings": bindings,
            "shape": list(image.shape), "entries": entries, "remaining": None, "attempt": 0}


def _missing_entries(store, task):
    missing = []
    for entry in task["entries"].values():
        try:
            prepared = store.get(entry["key"])
        except ValueError:
            prepared = None  # Same repair policy as sequential get_or_prepare.
        if prepared is None:
            missing.append(entry)
        del prepared
    return missing


def _dispatch(store, slot, task):
    with store._db:
        for entry in task["remaining"]:
            store._db.execute("""INSERT INTO entries(key,identity,state) VALUES (?,?,'preparing')
                ON CONFLICT(key) DO UPDATE SET state='preparing',error=NULL""",
                (entry["key"], canonical_json(entry["identity"])))
    slot["task"] = task
    # Never send source, targets, models, statistical contracts or stores.
    slot["connection"].send({"images": task["images"], "entries": task["remaining"]})


def _publish(store, task, message):
    entry = task["entries"].get(message["key"])
    if entry is None or message["temporary"] != entry["temporary"]:
        raise ValueError("Worker returned an unexpected preparation descriptor.")
    path = Path(entry["temporary"])
    if path.stat().st_size != message["size_bytes"] or digest_file(path) != message["sha256"]:
        raise ValueError("Worker preparation checksum/size mismatch.")
    identity, prepared, summary = store._load_file(path, full=True)
    if identity != entry["identity"] or summary != message["summary"]:
        raise ValueError("Worker preparation identity/summary mismatch.")
    del prepared
    from ..storage._writer_lock import fsync_directory
    destination = store._entry_path(entry["key"])
    os.replace(path, destination)
    fsync_directory(store._files)
    store._register_valid(entry["key"], identity, destination, summary, message["sha256"])
    store._writes += 1


def prepare_parallel(preprocessor, source, store, manifest, sample_ids, cancel, *,
                     num_workers, max_in_flight, worker_threads, max_sample_bytes, max_retries, progress=None):
    """Return (completed count, cancelled); references and input queues are bounded."""
    workers, pending, next_index, completed = None, deque(), 0, 0
    try:
        while pending or next_index < len(source) or (workers and any(s["task"] is not None for s in workers.slots)):
            if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
                return completed, True
            if workers:
                workers.check()
            active = [s for s in workers.slots if s["task"] is not None] if workers else []
            if next_index < len(source) and len(pending) + len(active) < max_in_flight:
                pending.append(_new_task(preprocessor, source, next_index, sample_ids, store, manifest, max_sample_bytes))
                next_index += 1
            busy_keys = {e["key"] for s in active for e in s["task"]["remaining"]}
            # Only the bounded queue is scanned. Identical inputs wait for their
            # existing task, so shared content is never prepared concurrently.
            for _ in range(len(pending)):
                task = pending.popleft()
                if busy_keys.intersection(task["entries"]):
                    task["remaining"] = None
                    pending.append(task)
                    continue
                if task["remaining"] is None:
                    task["remaining"] = _missing_entries(store, task)
                if not task["remaining"]:
                    _bind_sample(store, manifest, task["index"], task["sample_id"], task["bindings"], task["shape"])
                    completed += 1
                    emit(progress, manifest, completed, len(source), position=task["index"], sample_id=task["sample_id"])
                    del task
                    continue
                if workers is None:
                    workers = _Workers(preprocessor, store, min(num_workers, max_in_flight, len(source)), worker_threads)
                idle = next((s for s in workers.slots if s["task"] is None and s["ready"]), None)
                if idle is None:
                    pending.append(task)
                    continue
                _dispatch(store, idle, task)
                busy_keys.update(e["key"] for e in task["remaining"])
                del task
            if workers is None:
                continue
            connections = {s["connection"]: s for s in workers.slots if s["task"] is not None or not s["ready"]}
            # Poll cancellation/death even if a native task is slow or stuck.
            for connection in wait(list(connections), timeout=0.05) if connections else ():
                slot = connections[connection]
                task = slot["task"]
                try:
                    message = connection.recv()
                except EOFError as exc:
                    raise RuntimeError("CFP CPU worker disconnected; resume the same manifest.") from exc
                if message["kind"] == "ready":
                    slot["ready"] = True
                elif message["kind"] == "entry":
                    _publish(store, task, message)
                    connection.send("published")
                elif message["kind"] == "done":
                    _bind_sample(store, manifest, task["index"], task["sample_id"], task["bindings"], task["shape"])
                    completed += 1
                    slot["task"] = None
                    emit(progress, manifest, completed, len(source), position=task["index"], sample_id=task["sample_id"])
                elif message["kind"] == "error":
                    if message["quota"]:
                        raise OSError(message["error"])
                    if task["attempt"] >= max_retries:
                        raise RuntimeError(f"CFP preparation failed for sample {task['sample_id']!r}: {message['error']}. "
                                           "Completed entries were preserved; resume the same manifest.")
                    task["attempt"] += 1
                    task["remaining"] = None
                    pending.append(task)
                    slot["task"] = None
                elif message["kind"] == "fatal":
                    raise RuntimeError(f"CFP CPU worker initialization failed: {message['error']}")
                else:
                    raise ValueError("Unknown worker result.")
                del task
        return completed, False
    finally:
        if workers:
            workers.close()
            pending.extend(s["task"] for s in workers.slots if s["task"] is not None)
        # Join before cleanup: no worker can write a temporary after it is removed.
        for task in pending:
            for entry in task["entries"].values():
                Path(entry["temporary"]).unlink(missing_ok=True)
                with store._db:
                    store._db.execute("UPDATE entries SET state='failed',error='Preparation stopped before publication' "
                                      "WHERE key=? AND state='preparing'", (entry["key"],))
