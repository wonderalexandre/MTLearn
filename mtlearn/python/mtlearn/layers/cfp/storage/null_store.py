"""CPU preparation without persistent retention."""
from ._storages import _ConsumerTracker
from ..preparation.prepared_morphology import PreparedMorphology


class NullStore(_ConsumerTracker):
    """Return prepared data to callers without retaining it.

    Stores are local to one process and intended for sequential use. Active-byte
    metrics cover returned preparation handles, including handles kept alive by
    CFP autograd; they exclude normalized/device tensors and allocator overhead.
    """

    retains_entries = False

    def __init__(self):
        super().__init__()
        self._hits = self._misses = self._evictions = self._oversized = 0

    def get(self, key):
        self._misses += 1
        return None

    def put(self, key, prepared):
        """Return a consumer handle; callers must treat its tensors as read-only."""
        if not isinstance(prepared, PreparedMorphology):
            raise TypeError("Stores accept PreparedMorphology objects only.")
        prepared.validate()
        return self._borrow(prepared)

    def get_or_prepare(self, key, factory):
        """Resolve an immutable content key or invoke the zero-argument factory."""
        prepared = self.get(key)
        return self.put(key, factory()) if prepared is None else prepared

    def clear(self):
        """Drop retained entries without invalidating consumer handles or counters."""

    def info(self):
        return {
            "max_bytes": 0, "retained_bytes": 0, "entries": 0,
            "hits": self._hits, "misses": self._misses,
            "evictions": self._evictions, "oversized_entries": self._oversized,
            **self._consumer_info({}),
        }
