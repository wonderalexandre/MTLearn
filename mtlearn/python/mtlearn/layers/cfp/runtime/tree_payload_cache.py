"""Small in-memory cache for CFP tree payloads."""

from __future__ import annotations


class TreePayloadCache:
    """Cache tree payloads by sample key and tree key."""

    def __init__(self):
        self._payloads: dict[str, dict[str, object]] = {}
        self.norm_epoch_by_key: dict[str, int] = {}

    def get(self, sample_key: str, tree_key: str):
        """Return a cached payload or ``None``."""
        return self._payloads.get(sample_key, {}).get(tree_key)

    def has(self, sample_key: str, tree_key: str) -> bool:
        """Return whether a payload exists for ``sample_key`` and ``tree_key``."""
        return tree_key in self._payloads.get(sample_key, {})

    def set(self, sample_key: str, tree_key: str, payload) -> None:
        """Store a payload."""
        self._payloads.setdefault(sample_key, {})[tree_key] = payload

    def set_epoch(self, sample_key: str, epoch: int) -> None:
        """Record the normalization epoch used by one sample cache."""
        self.norm_epoch_by_key[sample_key] = int(epoch)

    def invalidate_sample_normalization(self, sample_key: str) -> None:
        """Force one sample's normalized attributes to refresh."""
        self.norm_epoch_by_key[sample_key] = -1

    def replace_norm_attrs(self, sample_key: str, norm_attrs_by_tree_key) -> None:
        """Replace normalized attributes for all cached trees of one sample."""
        for tree_key, norm_attrs in norm_attrs_by_tree_key.items():
            self._payloads[sample_key][tree_key]["norm_attrs"] = norm_attrs

    def sample_payloads(self, sample_key: str):
        """Return cached payloads for one sample."""
        return self._payloads.get(sample_key, {})

    def sample_keys(self):
        """Return cached sample keys."""
        return self._payloads.keys()

    def sample_count(self) -> int:
        """Return the number of cached sample/channel keys."""
        return len(self._payloads)

    def payload_count(self) -> int:
        """Return the number of cached sample/tree payloads."""
        return sum(len(per_sample) for per_sample in self._payloads.values())

    def clear(self) -> None:
        """Drop all cached payloads."""
        self._payloads.clear()
        self.norm_epoch_by_key.clear()

    def __len__(self) -> int:
        return self.payload_count()
