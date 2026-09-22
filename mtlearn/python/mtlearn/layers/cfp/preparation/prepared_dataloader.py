from collections import deque
from collections.abc import Sequence
import math
import os
import threading
import weakref

from torch.utils.data import DataLoader

from ..storage import DiskStore
from ._prepared_reader import contract, describe, open_dataset, load_batch
from ._threaded_reader import ReaderState, run_reader, failure_message
from ._reader_process import factory_descriptor
from ._process_reader import ProcessReaderState, ProcessReaders, process_description


class PreparedDataLoadingError(RuntimeError):
    pass


class PreparedDataLoader:
    def __init__(self, path, manifest, *, source=None, batch_size=1, shuffle=False, sampler=None,
                 generator=None, drop_last=False, prefetch=False, max_prefetch_bytes=0,
                 max_prefetch_tasks=1, max_batch_bytes=None, read_workspace_bytes=8 * 1024**2,
                 source_memory_bytes=None, source_thread_safe=False, max_ram_bytes=0, mmap=True,
                 validation="always", immutable=False, max_validation_entries=1024, close_timeout=5.0,
                 num_workers=0, worker_threads=1, worker_memory_bytes=None,
                 source_factory=None, source_factory_kwargs=None, collect_read_metrics=False):
        if type(collect_read_metrics) is not bool:
            raise ValueError("collect_read_metrics must be a boolean.")
        self.collect_read_metrics = collect_read_metrics
        if type(num_workers) is not int or num_workers not in (0, 1, 2):
            raise ValueError("num_workers must be 0, 1, or 2; larger reader pools require calibration.")
        if type(worker_threads) is not int or worker_threads < 1:
            raise ValueError("worker_threads must be a positive integer.")
        if worker_memory_bytes is not None and (type(worker_memory_bytes) is not int or worker_memory_bytes <= 0):
            raise ValueError("worker_memory_bytes must be a positive integer or None.")
        if source_factory is not None and not callable(source_factory):
            raise ValueError("source_factory must be callable or None.")
        if source_factory_kwargs is not None and (type(source_factory_kwargs) is not dict or source_factory is None):
            raise ValueError("source_factory_kwargs requires a factory and must be a dictionary.")
        self.source_factory = source_factory
        self.source_factory_kwargs = {} if source_factory_kwargs is None else source_factory_kwargs
        self._source_descriptor = None
        process_reason = None
        if num_workers and source is not None and source_factory is None:
            process_reason = "source_factory_required"
        elif num_workers and source_factory is not None:
            try:
                self._source_descriptor = factory_descriptor(source_factory, self.source_factory_kwargs)
            except (ValueError, TypeError, AttributeError, ImportError):
                process_reason = "source_factory_not_serializable"
        requested = prefetch or bool(num_workers)
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        for name, value, minimum in (("max_prefetch_bytes", max_prefetch_bytes, 0),
                ("max_prefetch_tasks", max_prefetch_tasks, 1), ("read_workspace_bytes", read_workspace_bytes, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name, value in (("prefetch", prefetch), ("source_thread_safe", source_thread_safe)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean.")
        if max_batch_bytes is not None and (type(max_batch_bytes) is not int or max_batch_bytes <= 0):
            raise ValueError("max_batch_bytes must be a positive integer or None.")
        if source_memory_bytes is not None and not callable(source_memory_bytes) and (
                type(source_memory_bytes) is not int or source_memory_bytes < 0):
            raise ValueError("source_memory_bytes must be a nonnegative integer, callable, or None.")
        if (source is not None or source_factory is not None) and max_batch_bytes is not None and source_memory_bytes is None:
            raise ValueError("A bounded batch with a source requires source_memory_bytes.")
        if type(close_timeout) not in (int, float) or not math.isfinite(close_timeout) or not 0 <= close_timeout <= 60:
            raise ValueError("close_timeout must be finite and between 0 and 60 seconds.")
        self._config = dict(path=path, readonly=True, max_ram_bytes=max_ram_bytes, mmap=mmap,
                           validation=validation, immutable=immutable, max_validation_entries=max_validation_entries)
        with DiskStore(**self._config) as store:
            self._contract = contract(store, manifest)
            self._length = self._contract[-1]
        if source is not None and len(source) != self._length:
            raise ValueError("Source length does not match the prepared manifest.")
        self._index_loader = DataLoader(range(self._length), batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, generator=generator, drop_last=drop_last, num_workers=0, collate_fn=tuple)
        self.num_workers_requested = num_workers
        self.prefetch_requested = requested
        self.prefetch_reason = process_reason
        if process_reason is not None:
            pass
        elif requested and not num_workers and source_factory is not None:
            self.prefetch_reason = "source_factory_requires_synchronous_iteration"
        elif requested and not num_workers and source is not None and not source_thread_safe:
            self.prefetch_reason = "source_not_declared_thread_safe"
        elif requested and (source is not None or source_factory is not None) and source_memory_bytes is None:
            self.prefetch_reason = "source_memory_bound_unavailable"
        elif requested and sampler is not None and not isinstance(sampler, Sequence):
            self.prefetch_reason = "lazy_sampler_requires_synchronous_iteration"
        elif requested and shuffle and generator is None:
            self.prefetch_reason = "shuffle_requires_a_dedicated_generator"
        self.prefetch = requested and self.prefetch_reason is None
        self.num_workers = num_workers if self.prefetch else 0
        if self.num_workers and worker_memory_bytes is None:
            raise ValueError("Process readers require worker_memory_bytes for runtime/source overhead per process.")
        self.worker_threads, self.worker_memory_bytes = worker_threads, worker_memory_bytes
        if self.prefetch and (max_prefetch_bytes == 0 or max_batch_bytes is None):
            raise ValueError("Prefetch requires explicit positive max_prefetch_bytes and max_batch_bytes.")
        self.source, self.manifest = source, manifest
        self.max_prefetch_bytes, self.max_prefetch_tasks = max_prefetch_bytes, max_prefetch_tasks
        self.max_batch_bytes, self.read_workspace_bytes = max_batch_bytes, read_workspace_bytes
        self.source_memory_bytes, self.close_timeout = source_memory_bytes, close_timeout
        self._pid, self._thread_id = os.getpid(), threading.get_ident()
        self._active = None
        self._state = ReaderState()
        self._closed = False

    def _check(self):
        if os.getpid() != self._pid or threading.get_ident() != self._thread_id:
            raise RuntimeError("Create and consume the prepared loader in one caller process/thread.")
        if self._closed:
            raise RuntimeError("PreparedDataLoader is closed.")

    def __len__(self):
        return len(self._index_loader)

    def __iter__(self):
        self._check()
        active = None if self._active is None else self._active()
        if active is not None and not active.closed:
            raise RuntimeError("Close or exhaust the current prepared iterator before starting another epoch.")
        if self._state.thread is not None and self._state.thread.is_alive():
            raise RuntimeError("The previous reader is still closing; wait for its I/O before starting another epoch.")
        if self._state.counters().get("processes_alive", 0):
            raise RuntimeError("The previous reader processes are still closing.")
        iterator = _PreparedIterator(self)
        self._active = weakref.ref(iterator)
        self._state = iterator.state
        return iterator

    def counters(self):
        return {"prefetch_requested": self.prefetch_requested, "prefetch_enabled": self.prefetch,
                "prefetch_reason": self.prefetch_reason, "max_prefetch_bytes": self.max_prefetch_bytes,
                "max_prefetch_tasks": self.max_prefetch_tasks, "max_batch_bytes": self.max_batch_bytes,
                "num_workers_requested": self.num_workers_requested, "num_workers": self.num_workers,
                **self._state.counters()}

    def close(self):
        active = None if self._active is None else self._active()
        closed = active.close() if active is not None else not (self.counters()["thread_alive"] or self.counters().get("processes_alive", 0))
        self._closed = True
        return closed

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, *args):
        if not self.close() and exc_type is None:
            raise TimeoutError("Prepared reader is still finishing I/O; counters()['thread_alive'] reports its state.")


class _PreparedIterator:
    def __init__(self, owner):
        self.owner, self.state = owner, ReaderState()
        self.state.active = owner._state.active
        if owner.num_workers:
            self.state = ProcessReaderState(owner._state.active,
                getattr(owner._state, "shared_active", weakref.WeakValueDictionary()), owner.worker_memory_bytes)
        elif hasattr(owner._state, "shared_active"):
            self.state.shared_active = owner._state.shared_active
        self.pending, self.waiting = deque(), None
        self.exhausted = self.closed = False
        self.sequence = 0
        self.store = None
        self.processes = None
        try:
            config = dict(owner._config)
            if owner.prefetch:
                config["max_ram_bytes"] = 0
            self.store = DiskStore(**config)
            source = owner.source
            if not owner.num_workers and source is None and owner.source_factory is not None:
                source = owner.source_factory(**owner.source_factory_kwargs)
            self.dataset = open_dataset(self.store, owner.manifest, source, owner._contract)
            self.indices = iter(owner._index_loader)
            if owner.num_workers:
                self.processes = ProcessReaders(owner, self.state)
            elif owner.prefetch:
                self.state.thread = threading.Thread(target=run_reader,
                    args=(self.state, owner._config, owner.manifest, owner.source, owner._contract),
                    daemon=True, name="mtlearn-cfp-reader")
                self.state.thread.start()
            else:
                self.state.reader_closed = False
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        return self

    def _next_description(self):
        if self.exhausted:
            return None
        indices = None
        try:
            indices = next(self.indices)
        except StopIteration:
            self.exhausted = True
            return None
        except Exception as exc:
            self.exhausted = True
            return {"error": failure_message(exc, {"indices": indices})}
        try:
            description = describe(self.store, self.dataset, indices, self.owner.source_memory_bytes,
                self.owner.read_workspace_bytes, self.owner.max_batch_bytes, self.owner._config["mmap"])
            return process_description(description, self.owner.max_batch_bytes) if self.owner.num_workers else description
        except Exception as exc:
            self.exhausted = True
            return {"error": failure_message(exc, {"indices": indices})}

    def _fill(self):
        if self.pending and self.pending[-1][0] != "thread":
            return
        while len(self.pending) < self.owner.max_prefetch_tasks:
            description = self.waiting or self._next_description()
            self.waiting = None
            if description is None:
                return
            if "error" in description:
                self.pending.append(("error", description["error"]))
                return
            size = description["reserved_bytes"]
            if size > self.owner.max_prefetch_bytes or description.get("process_oversized"):
                self.pending.append(("sync", description))
                return
            with self.state.condition:
                if sum(self.state.reservations.values()) + size > self.owner.max_prefetch_bytes:
                    self.waiting = description
                    return
                key = self.sequence
                self.sequence += 1
                self.state.reserve(key, description)
            self.pending.append(("thread", (key, description)))

    def _synchronous(self, description):
        failure = None
        value = None
        try:
            if self.owner.num_workers and self.dataset.source is None and self.owner.source_factory is not None:
                self.dataset = open_dataset(self.store, self.owner.manifest,
                    self.owner.source_factory(**self.owner.source_factory_kwargs), self.owner._contract)
            value = load_batch(self.dataset, description)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failure = failure_message(exc, description)
        if failure is not None:
            raise PreparedDataLoadingError(failure)
        if not self.owner.prefetch:
            self.state.update_reader(self.store)
        else:
            with self.state.condition:
                self.state.caller_counters = self.store.counters()
        return value

    def __next__(self):
        self.owner._check()
        if self.closed:
            raise StopIteration
        try:
            if not self.owner.prefetch:
                description = self._next_description()
                if description is None:
                    self.close()
                    raise StopIteration
                if "error" in description:
                    raise PreparedDataLoadingError(description["error"])
                value = self._synchronous(description)
                self.state.scheduled += 1
            else:
                self._fill()
                if self.processes is not None:
                    self.processes.poll()
                if not self.pending:
                    self.close()
                    raise StopIteration
                kind, data = self.pending.popleft()
                if kind == "error":
                    raise PreparedDataLoadingError(data)
                if kind == "sync":
                    value = self._synchronous(data)
                    self.state.fallback += 1
                    self.state.scheduled += 1
                else:
                    key, description = data
                    if self.processes is not None:
                        value, failure = self.processes.take(key, description, self.dataset.preprocessor)
                    else:
                        with self.state.condition:
                            self.state.condition.wait_for(lambda: key in self.state.results or self.state.startup_error is not None)
                            if self.state.startup_error is not None:
                                raise PreparedDataLoadingError(self.state.startup_error)
                            value, failure = self.state.results.pop(key)
                            self.state.reservations.pop(key)
                            self.state.condition.notify_all()
                    if failure is not None:
                        raise PreparedDataLoadingError(failure)
            self.state.active.track(value)
            self.state.delivered += 1
            if self.owner.prefetch:
                self._fill()
                if self.processes is not None:
                    self.processes.poll()
            return value
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, "closed", True):
            if self.processes is not None:
                return self.processes.close(self.owner.close_timeout)
            return self.state.thread is None or not self.state.thread.is_alive()
        self.closed = True
        if self.processes is not None:
            self.pending.clear()
            self.waiting = None
            try:
                return self.processes.close(self.owner.close_timeout)
            finally:
                if self.store is not None:
                    self.store.close()
                self.dataset = None
                self.indices = None
        with self.state.condition:
            self.state.stop = True
            self.state.jobs.clear()
            self.state.results.clear()
            for key in list(self.state.reservations):
                if key != self.state.reading:
                    del self.state.reservations[key]
            self.state.condition.notify_all()
        self.pending.clear()
        self.waiting = None
        if self.store is not None:
            self.store.close()
        self.dataset = None
        self.indices = None
        if self.state.thread is not None:
            self.state.thread.join(timeout=self.owner.close_timeout)
        else:
            self.state.retained.clear()
            self.state.reader_closed = True
        return self.state.thread is None or not self.state.thread.is_alive()

    def __del__(self):
        if hasattr(self, "owner") and os.getpid() == self.owner._pid and threading.get_ident() == self.owner._thread_id:
            self.close()


def build_prepared_dataloader(path, manifest, **kwargs):
    return PreparedDataLoader(path, manifest, **kwargs)
