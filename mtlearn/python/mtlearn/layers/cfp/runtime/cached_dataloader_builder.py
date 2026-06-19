"""Cached DataLoader construction for CFP layers."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..._helpers import IndexedDatasetWrapper


class CachedDataLoaderBuilder:
    """Build DataLoaders that precompute CFP tree payload caches."""

    def build_cached(self, layer, dataloader):
        """Wrap a training DataLoader and precompute cache/statistics."""
        new_loader = self.wrap_dataloader(dataloader)

        layer._stats_frozen = False
        with torch.no_grad():
            self.precompute(layer, new_loader, update_stats=True)

        layer.freeze_ds_stats()
        layer.refresh_cached_normalization()
        return new_loader

    def build_fixed_stats(self, layer, dataloader, *, index_offset: int = 0):
        """Wrap a DataLoader and precompute caches without updating stats."""
        layer._require_fixed_dataset_stats()
        new_loader = self.wrap_dataloader(dataloader, index_offset=index_offset)

        with torch.no_grad():
            self.precompute(layer, new_loader, update_stats=False)

        return new_loader

    @staticmethod
    def wrap_dataloader(dataloader, *, index_offset: int = 0):
        """Return a DataLoader whose dataset yields stable sample indexes."""
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
        """Populate layer tree payload caches from an indexed DataLoader."""
        for (x, idx), _ in indexed_loader:
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
