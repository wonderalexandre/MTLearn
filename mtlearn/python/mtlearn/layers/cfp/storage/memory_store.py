"""Explicit CPU preparation cache with a byte-bounded LRU."""
from collections import OrderedDict

from .null_store import NullStore
from ._storages import storages
from ..preparation.prepared_morphology import PreparedMorphology


class MemoryStore(NullStore):
    """Retain unique CPU storages up to ``max_bytes`` (not a process RSS limit).

    Entries larger than the budget are returned without admission. Eviction drops
    only the store's reference; active batches and their backward remain valid.
    Keys supplied directly to this low-level store must identify immutable content.
    CFPPreprocessor calculates keys from canonical pixels and preparation options.
    """

    retains_entries = True

    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer.")
        super().__init__()
        self._max_bytes = max_bytes
        self._entries = OrderedDict()
        self._storage_counts = {}
        self._retained_bytes = 0

    @property
    def max_bytes(self):
        return self._max_bytes

    def get(self, key):
        if key not in self._entries:
            self._misses += 1
            return None
        self._hits += 1
        self._entries.move_to_end(key)
        return self._borrow(self._entries[key])

    def _remove(self, key):
        prepared = self._entries.pop(key)
        for pointer in storages(prepared):
            size, count = self._storage_counts[pointer]
            if count == 1:
                del self._storage_counts[pointer]
                self._retained_bytes -= size
            else:
                self._storage_counts[pointer] = size, count - 1

    def put(self, key, prepared):
        if not isinstance(prepared, PreparedMorphology):
            raise TypeError("Stores accept PreparedMorphology objects only.")
        prepared.validate()
        hash(key)  # Fail before changing retention if the key is not hashable.
        buffers = storages(prepared)
        if key in self._entries:
            self._remove(key)
        if sum(buffers.values()) > self.max_bytes:
            self._oversized += 1
            return self._borrow(prepared)
        while self._retained_bytes + sum(size for pointer, size in buffers.items()
                                         if pointer not in self._storage_counts) > self.max_bytes:
            self._remove(next(iter(self._entries)))
            self._evictions += 1
        # Never store the consumer wrapper itself: its lifetime measures use.
        from dataclasses import replace
        self._entries[key] = replace(prepared)
        for pointer, size in buffers.items():
            _, count = self._storage_counts.get(pointer, (size, 0))
            self._storage_counts[pointer] = size, count + 1
            if count == 0:
                self._retained_bytes += size
        return self._borrow(prepared)

    def clear(self):
        self._entries.clear()
        self._storage_counts.clear()
        self._retained_bytes = 0

    def info(self):
        resident = {key: value[0] for key, value in self._storage_counts.items()}
        return {
            "max_bytes": self.max_bytes, "retained_bytes": self._retained_bytes,
            "entries": len(self._entries), "hits": self._hits, "misses": self._misses,
            "evictions": self._evictions, "oversized_entries": self._oversized,
            **self._consumer_info(resident),
        }
