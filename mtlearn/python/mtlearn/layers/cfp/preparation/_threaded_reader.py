from collections import deque
import threading

from ..storage import DiskStore
from ._loader_memory import ActiveBuffers, tensor_buffers
from ._prepared_reader import open_dataset, load_batch


def failure_message(exc, description):
    return f"Prepared loading failed for samples {description.get('sample_ids', description.get('indices'))!r}: {type(exc).__name__}: {str(exc)[:4096]}"


class ReaderState:
    def __init__(self):
        self.condition = threading.Condition()
        self.jobs, self.results = deque(), {}
        self.reservations = {}
        self.reading = None
        self.stop = False
        self.thread = None
        self.startup_error = None
        self.retained = {}
        self.active = ActiveBuffers()
        self.reader_counters = {}
        self.caller_counters = {}
        self.scheduled = self.delivered = self.fallback = 0
        self.peak_bytes = self.peak_tasks = 0
        self.reader_closed = True

    def reserve(self, key, description):
        self.reservations[key] = description["reserved_bytes"]
        self.peak_bytes = max(self.peak_bytes, sum(self.reservations.values()))
        self.peak_tasks = max(self.peak_tasks, len(self.reservations))
        self.scheduled += 1
        self.jobs.append((key, description))
        self.condition.notify_all()

    def update_reader(self, store):
        values = store.counters()
        retained = {("torch", pointer): size for pointer, (size, count) in store._memory._storage_counts.items()}
        with self.condition:
            self.reader_counters = values
            self.retained = retained

    def counters(self):
        with self.condition:
            queued = {}
            for value, failure in self.results.values():
                if value is not None:
                    queued.update(tensor_buffers(value))
            active = self.active.buffers()
            return {"scheduled_batches": self.scheduled, "delivered_batches": self.delivered,
                    "queued_batches": len(self.results), "building_batches": int(self.reading is not None),
                    "prefetch_reserved_bytes": sum(self.reservations.values()),
                    "building_reserved_bytes": self.reservations.get(self.reading, 0),
                    "peak_prefetch_reserved_bytes": self.peak_bytes, "peak_prefetch_tasks": self.peak_tasks,
                    "sync_fallback_batches": self.fallback, "queued_bytes": sum(queued.values()),
                    "active_bytes": sum(active.values()), "retained_bytes": sum(self.retained.values()),
                    "known_tensor_bytes": sum((queued | active | self.retained).values()),
                    "reader_closed": self.reader_closed,
                    "thread_alive": self.thread is not None and self.thread.is_alive(),
                    **{key: self.reader_counters.get(key, 0) + self.caller_counters.get(key, 0)
                       for key in ("checksum_reads", "checksum_bytes", "logical_load_bytes", "ram_hits", "disk_hits")}}


def run_reader(state, config, manifest, source, expected_contract):
    store = None
    try:
        store = DiskStore(**config)
        dataset = open_dataset(store, manifest, source, expected_contract)
        with state.condition:
            state.reader_closed = False
        while True:
            with state.condition:
                state.condition.wait_for(lambda: state.stop or state.jobs)
                if state.stop:
                    break
                key, description = state.jobs.popleft()
                state.reading = key
            value = failure = None
            try:
                value = load_batch(dataset, description)
            except BaseException as exc:
                failure = failure_message(exc, description)
            state.update_reader(store)
            with state.condition:
                state.reading = None
                if state.stop:
                    state.reservations.pop(key, None)
                else:
                    state.results[key] = (value, failure)
                state.condition.notify_all()
            value = None
    except BaseException as exc:
        with state.condition:
            state.startup_error = f"Prepared reader failed: {type(exc).__name__}: {str(exc)[:4096]}"
            state.condition.notify_all()
    finally:
        if store is not None:
            store.close()
        with state.condition:
            state.reader_closed = True
            state.retained.clear()
            state.reading = None
            if state.stop or state.startup_error:
                state.jobs.clear()
                state.results.clear()
                state.reservations.clear()
            state.condition.notify_all()
