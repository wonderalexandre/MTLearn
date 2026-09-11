"""Streaming fit of CFP statistics with one temporary morphology at a time."""
import torch

from .cfp_preprocessor import CFPPreprocessor
from ..normalization.attribute_normalizer import AttributeNormalizer
from ..normalization.statistics_snapshot import StatisticsSnapshot
from ..._helpers import to_numpy_u8


def fit_statistics(layer, loader):
    """Fit the finite stream actually yielded by loader, then install atomically.

    Sampler order, repeated samples, batching and drop_last are owned by loader.
    The new fit replaces the previous fit; a failed or empty pass leaves it intact.
    Memory includes the caller's active batch plus one prepared image/tree.
    """
    from ..runtime.cache_input_contract import validate_cfp_cache_batch_x
    preprocessor = CFPPreprocessor.from_layer(layer)
    accumulator = AttributeNormalizer(layer.scale_mode, layer.eps,
                                      clipped_zscore_radius=layer.clipped_zscore_radius,
                                      clipped_zscore_floor=layer.clipped_zscore_floor)
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch if torch.is_tensor(batch) else batch[0]
            # Also accept the public indexed-loader form ((x, index), targets).
            if isinstance(x, (list, tuple)) and len(x) == 2 and torch.is_tensor(x[0]):
                x = x[0]
            validate_cfp_cache_batch_x(x, expected_channels=layer.in_channels)
            for image in x:
                for channel in image:
                    canonical = to_numpy_u8(channel)
                    for tree_key in preprocessor.tree_specs:
                        prepared = preprocessor.prepare_u8(canonical, tree_key)
                        for attribute, raw in prepared.raw_attributes.items():
                            accumulator.merge(accumulator.summarize(layer._stat_key(tree_key, attribute), raw.view(-1)))
                        # Release the previous tree and its last attribute before preparing another.
                        del prepared, raw
                    del canonical, channel
                sample_count += 1
                del image
            del x, batch
        if sample_count == 0:
            raise ValueError("fit_stats requires a non-empty finite training loader.")
        snapshot = StatisticsSnapshot(layer._statistics_contract(), accumulator.ds_stats, sample_count=sample_count)
    layer.set_stats(snapshot)
    return snapshot
