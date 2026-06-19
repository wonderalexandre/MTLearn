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

    def _view(self, field: str) -> dict[str, dict[str, object]]:
        return {
            sample_key: {
                tree_key: payload[field]
                for tree_key, payload in per_sample.items()
            }
            for sample_key, per_sample in self._payloads.items()
        }

    def tree_info(self) -> dict[str, dict[str, object]]:
        """Return a compatibility view of cached tree metadata."""
        return self._view("info")

    def base_attrs(self) -> dict[str, dict[str, object]]:
        """Return a compatibility view of cached raw attributes."""
        return self._view("base_attrs")

    def norm_attrs(self) -> dict[str, dict[str, object]]:
        """Return a compatibility view of cached normalized attributes."""
        return self._view("norm_attrs")

    def valuation_increments(self) -> dict[str, dict[str, object]]:
        """Return a compatibility view of cached valuation increments."""
        return self._view("valuation_increments")

    def clear(self) -> None:
        """Drop all cached payloads."""
        self._payloads.clear()
        self.norm_epoch_by_key.clear()

    def __len__(self) -> int:
        return sum(len(per_sample) for per_sample in self._payloads.values())
