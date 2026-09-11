"""Storage accounting shared by CPU preparation stores."""
import weakref


def storages(prepared):
    result = {}
    for value in (*prepared.info.values(), *prepared.raw_attributes.values()):
        if hasattr(value, "untyped_storage"):
            storage = value.untyped_storage()
            result[storage.data_ptr()] = storage.nbytes()
    return result


class _ConsumerTracker:
    def __init__(self):
        self._consumers = weakref.WeakValueDictionary()

    def _borrow(self, prepared):
        # A separate wrapper distinguishes consumer ownership from store ownership.
        # Tensor storage is shared, never copied or reused as a writable buffer.
        from dataclasses import replace
        borrowed = replace(prepared)
        self._consumers[id(borrowed)] = borrowed
        return borrowed

    def _consumer_info(self, resident):
        active = {}
        consumers = list(self._consumers.values())
        for prepared in consumers:
            active.update(storages(prepared))
        return {
            "active_entries": len(consumers),
            "active_bytes": sum(active.values()),
            "active_not_retained_bytes": sum(size for key, size in active.items() if key not in resident),
            "live_bytes": sum((resident | active).values()),
        }
