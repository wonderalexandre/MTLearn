"""Cached DataLoader construction for CFP layers.

The production CFP layer can avoid rebuilding morphology trees during every
forward pass by wrapping a user-provided ``DataLoader`` with stable sample
indices and precomputing tree payloads. This module owns that workflow:

- preserve the original loader's batching options where practical;
- wrap dataset samples as ``((x, idx), y)``;
- validate that ``x`` satisfies the CFP image-intensity contract;
- populate tree/attribute caches and, for training data, dataset statistics.

The cached loader intentionally disables shuffling because the stable index is
the cache key. Callers can still shuffle at a higher level after cache creation
if they preserve the emitted sample indices.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..._helpers import IndexedDatasetWrapper
from .cache_input_contract import validate_cfp_cache_batch_x


class CachedDataLoaderBuilder:
    """Build DataLoaders that precompute CFP tree payload caches."""

    def build_cached(self, layer, dataloader):
        """Wrap a training DataLoader and precompute cache/statistics.

        The training path updates dataset-level normalization statistics while
        building the per-sample tree cache, then freezes those statistics before
        returning the wrapped loader.
        """

        new_loader = self.wrap_dataloader(dataloader)

        layer._stats_frozen = False
        with torch.no_grad():
            self.precompute(layer, new_loader, update_stats=True)

        layer.freeze_ds_stats()
        layer.refresh_cached_normalization()
        return new_loader

    def build_fixed_stats(self, layer, dataloader, *, index_offset: int = 0):
        """Wrap a DataLoader and precompute caches without updating stats.

        This path is used for validation/test splits after training statistics
        have already been built or loaded. ``index_offset`` lets callers keep
        cache keys disjoint across splits whose local dataset indices overlap.
        """

        layer._require_fixed_dataset_stats()
        new_loader = self.wrap_dataloader(dataloader, index_offset=index_offset)

        with torch.no_grad():
            self.precompute(layer, new_loader, update_stats=False)

        return new_loader

    @staticmethod
    def wrap_dataloader(dataloader, *, index_offset: int = 0):
        """Return a DataLoader whose dataset yields stable sample indexes.

        The original dataset is wrapped by ``IndexedDatasetWrapper`` so each
        sample carries the cache index used by CFP forward calls. Batch size,
        workers, pinned-memory behavior, and custom collate functions are
        preserved from the source loader.
        """

        dataset_wrapped = IndexedDatasetWrapper(dataloader.dataset, index_offset=index_offset)
        return DataLoader(
            dataset_wrapped,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=False,
            collate_fn=dataloader.collate_fn,
            persistent_workers=getattr(dataloader, "persistent_workers", False),
        )

    @staticmethod
    def precompute(layer, indexed_loader, *, update_stats: bool) -> None:
        """Populate layer tree payload caches from an indexed DataLoader.

        Each sample/channel pair becomes a base cache key of the form
        ``"{dataset_index}_{channel_index}"``. All configured tree specs are then
        built for that key, optionally updating dataset statistics.
        """

        for (x, idx), _ in indexed_loader:
            # Validate before building morphology trees; bad intensity scales can
            # preserve tensor shape while changing the tree's gray-level order.
            validate_cfp_cache_batch_x(
                x,
                expected_channels=layer.in_channels,
                sample_indices=idx,
            )
            batch_size, channels, _, _ = x.shape
            for batch_index in range(batch_size):
                for channel_index in range(channels):
                    base_key = f"{int(idx[batch_index])}_{channel_index}"
                    for tree_key in layer._tree_spec_by_key:
                        layer._ensure_tree_payload_cached(
                            base_key,
                            x[batch_index, channel_index],
                            tree_key,
                            update_stats=update_stats,
                        )
