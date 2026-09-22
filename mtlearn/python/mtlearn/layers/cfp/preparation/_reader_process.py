import importlib
import json
import pickle
import sys
from multiprocessing.shared_memory import SharedMemory

import torch

from ..storage import DiskStore
from ._prepared_reader import open_dataset, load_batch
from ._shared_batch import write_shared_batch
from ._threaded_reader import failure_message

MESSAGE_BYTES = 256 * 1024


def factory_descriptor(factory, kwargs):
    if factory is None:
        return None
    module, name = getattr(factory, "__module__", None), getattr(factory, "__qualname__", "")
    if not module or "<" in name or (module == "__main__" and not getattr(sys.modules[module], "__file__", None)):
        raise ValueError("Source factory must be importable by a spawned process.")
    resolved = importlib.import_module(module)
    for part in name.split("."):
        resolved = getattr(resolved, part)
    if resolved is not factory:
        raise ValueError("Source factory must resolve to its module-level definition.")
    remaining = 64 * 1024
    def check(value, depth=0):
        nonlocal remaining
        remaining -= 16
        if remaining < 0 or depth > 32:
            raise ValueError("Source factory configuration exceeds its metadata budget.")
        if type(value) is str:
            remaining -= 6 * len(value)
        elif type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError("Source factory configuration requires string mapping keys.")
                check(key, depth + 1)
                check(item, depth + 1)
        elif type(value) in (list, tuple):
            for item in value:
                check(item, depth + 1)
        elif type(value) is int:
            if value.bit_length() > 1024:
                raise ValueError("Source factory integer exceeds the metadata budget.")
            remaining -= 320
        elif value is not None and type(value) not in (bool, float):
            raise ValueError("Source factory configuration accepts only JSON values, without tensors or handles.")
        if remaining < 0:
            raise ValueError("Source factory configuration exceeds 64 KiB.")
    try:
        check(kwargs)
    finally:
        check = None
    encoded = json.dumps(kwargs, allow_nan=False)
    if len(encoded.encode()) > 64 * 1024:
        raise ValueError("Source factory configuration exceeds 64 KiB; use paths to source metadata.")
    return module, name, encoded


def create_source(descriptor):
    if descriptor is None:
        return None
    module, name, encoded = descriptor
    factory = importlib.import_module(module)
    for part in name.split("."):
        factory = getattr(factory, part)
    return factory(**json.loads(encoded))


def send(connection, value):
    data = pickle.dumps(value, protocol=5)
    if len(data) > MESSAGE_BYTES:
        raise ValueError("Prepared reader metadata exceeds the bounded IPC message size.")
    connection.send_bytes(data)


def receive(connection):
    return pickle.loads(connection.recv_bytes(MESSAGE_BYTES))


def run_process(connection, config, manifest, expected_contract, source_descriptor, worker_threads, collect_read_metrics=False):
    store = None
    try:
        torch.set_num_threads(worker_threads)
        torch.set_num_interop_threads(1)
        source = create_source(source_descriptor)
        store = DiskStore(**config)
        dataset = open_dataset(store, manifest, source, expected_contract)
        send(connection, {"kind": "ready"})
        while True:
            command = receive(connection)
            if command is None:
                break
            key, description, name = command
            memory = value = packet = None
            error = None
            memory_observation = None
            try:
                memory = SharedMemory(name=name)
                value = load_batch(dataset, description)
                packet = write_shared_batch(value, memory)
                if collect_read_metrics:
                    from ._read_diagnostics import process_memory
                    memory_observation = process_memory()
            except BaseException as exc:
                error = failure_message(exc, description)
            finally:
                value = None
                if memory is not None:
                    memory.close()
            counters = store.counters()
            send(connection, {"kind": "result", "key": key, "packet": packet, "error": error,
                "retained_bytes": counters["retained_bytes"], "memory_observation": memory_observation,
                **{key: counters[key] for key in ("checksum_reads", "checksum_bytes", "logical_load_bytes", "ram_hits", "disk_hits")}})
            packet = command = description = None
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            send(connection, {"kind": "fatal", "error": f"{type(exc).__name__}: {str(exc)[:4096]}"})
        except (EOFError, OSError):
            pass
    finally:
        if store is not None:
            store.close()
        connection.close()
