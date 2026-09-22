import multiprocessing as mp
from multiprocessing.connection import wait
import time

from ._reader_process import run_process, send, receive, MESSAGE_BYTES
from ._shared_batch import SharedBatchLease, read_shared_batch
from ._threaded_reader import ReaderState, failure_message


class ProcessReaderState(ReaderState):
    def __init__(self, active, shared_active, worker_memory_bytes):
        super().__init__()
        self.active = active
        self.shared_active = shared_active
        self.worker_memory_bytes = worker_memory_bytes
        self.slots = []
        self.leases = {}
        self.forced_stops = 0

    def counters(self):
        values = super().counters()
        queued = sum(packet["used_bytes"] for packet, error in self.results.values() if packet is not None)
        busy = [slot["task"] for slot in self.slots if slot["task"] is not None]
        values.update(queued_bytes=queued, known_tensor_bytes=queued + values["active_bytes"],
            retained_bytes=sum(slot["retained_bytes"] for slot in self.slots),
            building_batches=len(busy), building_reserved_bytes=sum(self.reservations.get(key, 0) for key in busy),
            processes_alive=sum(slot["process"] is not None and slot["process"].is_alive() for slot in self.slots),
            worker_pids=tuple(slot["pid"] for slot in self.slots),
            worker_ram_limits=tuple(slot["ram_limit"] for slot in self.slots),
            worker_base_reserved_bytes=len(self.slots) * self.worker_memory_bytes,
            shared_pending_bytes=sum(lease.size for lease in self.leases.values()),
            shared_active_bytes=sum(lease.size for lease in list(self.shared_active.values())),
            worker_memory_observations=tuple(slot["memory_observation"] for slot in self.slots if slot["memory_observation"] is not None),
            **{key: sum(slot.get(key, 0) for slot in self.slots) + self.caller_counters.get(key, 0)
               for key in ("checksum_reads", "checksum_bytes", "logical_load_bytes", "ram_hits", "disk_hits")},
            forced_worker_stops=self.forced_stops)
        return values


class ProcessReaders:
    def __init__(self, owner, state):
        self.state = state
        context = mp.get_context("spawn")
        count = min(owner.num_workers, owner.max_prefetch_tasks, len(owner))
        try:
            for index in range(count):
                parent, child = context.Pipe()
                config = dict(owner._config)
                quotient, remainder = divmod(config["max_ram_bytes"], count)
                config["max_ram_bytes"] = quotient + (index < remainder)
                process = context.Process(target=run_process,
                    args=(child, config, owner.manifest, owner._contract, owner._source_descriptor, owner.worker_threads, owner.collect_read_metrics),
                    daemon=True, name="mtlearn-cfp-reader")
                slot = {"process": process, "pid": None, "connection": parent, "task": None, "ready": False,
                        "retained_bytes": 0, "checksum_reads": 0, "ram_limit": config["max_ram_bytes"], "memory_observation": None}
                state.slots.append(slot)
                try:
                    process.start()
                    slot["pid"] = process.pid
                finally:
                    child.close()
            state.reader_closed = not count
        except BaseException:
            self.close(owner.close_timeout)
            raise

    def _failed(self, slot, error):
        if slot["task"] is None:
            self.state.startup_error = error
        else:
            self.state.results[slot["task"]] = (None, error)
        slot["task"] = None
        slot["ready"] = False

    def poll(self, timeout=0):
        state = self.state
        connections = {slot["connection"]: slot for slot in state.slots if slot["process"] is not None}
        for connection in wait(list(connections), timeout=timeout) if connections else ():
            slot = connections[connection]
            try:
                message = receive(connection)
                if message["kind"] == "ready":
                    if slot["ready"] or slot["task"] is not None:
                        raise ValueError("Unexpected prepared reader startup message.")
                    slot["ready"] = True
                elif message["kind"] == "result":
                    key = message["key"]
                    if key != slot["task"] or key is None:
                        raise ValueError("Prepared reader returned an unexpected task ID.")
                    state.results[key] = message["packet"], message["error"]
                    slot["retained_bytes"] = message["retained_bytes"]
                    for metric in ("checksum_reads", "checksum_bytes", "logical_load_bytes", "ram_hits", "disk_hits", "memory_observation"):
                        slot[metric] = message[metric]
                    slot["task"] = None
                else:
                    raise RuntimeError(message.get("error", "Invalid prepared reader message."))
            except Exception as exc:
                key = slot["task"]
                self._failed(slot, f"Prepared worker {slot['pid']} failed for task {key}: {type(exc).__name__}: {str(exc)[:4096]}")
                connection.close()
                process = slot["process"]
                if process.is_alive():
                    process.terminate()
                process.join(timeout=0.2)
                if not process.is_alive():
                    process.close()
                    slot["process"] = None
        for slot in state.slots:
            process = slot["process"]
            if process is not None and process.exitcode is not None:
                self._failed(slot, f"Prepared worker {slot['pid']} exited unexpectedly with code {process.exitcode}.")
                slot["connection"].close()
                process.join()
                process.close()
                slot["process"] = None
                slot["retained_bytes"] = 0
        for slot in state.slots:
            if not state.jobs or state.startup_error:
                break
            if slot["process"] is None or not slot["ready"] or slot["task"] is not None:
                continue
            key, description = state.jobs.popleft()
            state.leases[key] = SharedBatchLease(description["transport_bytes"])
            slot["task"] = key
            send(slot["connection"], (key, description, state.leases[key].memory.name))

    def take(self, key, description, preprocessor):
        state = self.state
        while key not in state.results and state.startup_error is None:
            self.poll(0.05)
        if state.startup_error is not None:
            return None, f"Samples {description['sample_ids']!r}: {state.startup_error}"
        packet, error = state.results.pop(key)
        lease = state.leases.pop(key, None)
        value = None
        if error is None:
            try:
                value = read_shared_batch(packet, lease, preprocessor, description["sample_ids"])
            except Exception as exc:
                error = failure_message(exc, description)
        if lease is not None:
            lease.unlink()
            if value is None:
                lease.close()
            else:
                state.shared_active[lease.memory.name] = lease
        state.reservations.pop(key, None)
        return value, None if error is None else f"Samples {description['sample_ids']!r}: {error}"

    def close(self, timeout):
        state = self.state
        state.stop = True
        state.jobs.clear()
        for slot in state.slots:
            process = slot["process"]
            if process is not None and process.is_alive() and slot["task"] is None:
                try:
                    send(slot["connection"], None)
                except (OSError, EOFError):
                    pass
            elif process is not None and process.is_alive():
                process.terminate()
                state.forced_stops += 1
        deadline = time.monotonic() + timeout
        for slot in state.slots:
            process = slot["process"]
            if process is None:
                continue
            if process.pid is not None:
                process.join(timeout=max(0, deadline - time.monotonic()))
                if process.is_alive():
                    process.kill()
                    state.forced_stops += 1
                    process.join(timeout=1)
                if process.is_alive():
                    continue
                process.close()
            slot["process"] = None
            slot["connection"].close()
            slot["task"] = None
            slot["retained_bytes"] = 0
        alive = any(slot["process"] is not None and slot["process"].is_alive() for slot in state.slots)
        if not alive:
            state.results.clear()
            state.reservations.clear()
            for lease in state.leases.values():
                lease.close()
            state.leases.clear()
        state.reader_closed = not alive
        return not alive


def process_description(description, max_batch_bytes):
    transport = description["payload_bytes"] + 2 * sum(n or 0 for n in description["source_limits"]) + description["workspace_bytes"]
    reserved = description["reserved_bytes"] + transport + 2 * MESSAGE_BYTES
    result = dict(description, transport_bytes=transport, reserved_bytes=reserved)
    if reserved > max_batch_bytes:
        result["process_oversized"] = True
    return result
