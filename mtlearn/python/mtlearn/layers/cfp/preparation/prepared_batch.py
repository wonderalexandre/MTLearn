"""Dense batches of immutable, model-independent CPU morphology."""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from .prepared_morphology import PreparedMorphology


@dataclass(frozen=True, eq=False)
class PreparedBatch:
    """Morphology grouped as ``samples[batch][channel][tree_key]``.

    All channels share a spatial shape. Logical sample IDs affect CFPContext,
    never cache identity. Tensors are shared and read-only by contract. A batch
    owns its active data even after the corresponding store entries are evicted.
    """

    samples: tuple
    sample_ids: tuple | None = None

    def __post_init__(self):
        samples = tuple(tuple(MappingProxyType(dict(channel)) for channel in sample)
                        for sample in self.samples)
        object.__setattr__(self, "samples", samples)
        ids = tuple(str(i) for i in range(len(samples))) if self.sample_ids is None else tuple(self.sample_ids)
        if len(ids) != len(samples) or any(not isinstance(i, str) for i in ids):
            raise ValueError("sample_ids must contain one string per sample.")
        object.__setattr__(self, "sample_ids", ids)
        if not samples or not samples[0]:
            raise ValueError("PreparedBatch requires at least one sample and channel.")
        shape = None
        for sample in samples:
            if len(sample) != len(samples[0]):
                raise ValueError("All prepared samples must have the same channel count.")
            for channel in sample:
                if not channel:
                    raise ValueError("Each prepared channel requires at least one tree.")
                for key, prepared in channel.items():
                    if not isinstance(prepared, PreparedMorphology) or key != prepared.tree_key:
                        raise ValueError("Prepared channel keys must match their morphology trees.")
                    prepared.validate()
                    if shape is not None and prepared.image_shape != shape:
                        raise ValueError("All prepared channels must share a spatial shape.")
                    shape = prepared.image_shape

    @property
    def shape(self):
        first = next(iter(self.samples[0][0].values()))
        return len(self.samples), len(self.samples[0]), *first.image_shape

    def sample_key(self, batch_index, channel_index):
        return f"{self.sample_ids[batch_index]}_{channel_index}"

    def validate_for(self, layer):
        """Check compatibility before transferring tensors or scoring any sample.

        Extra trees/attributes are allowed, enabling preparation shared by models
        with different feature subsets. Required raw dtypes must match the layer.
        """
        if self.shape[1] != layer.in_channels:
            raise ValueError(f"Prepared channels do not match in_channels={layer.in_channels}.")
        dtype = torch.float64 if np.dtype(layer.attribute_dtype) == np.dtype(np.float64) else torch.float32
        for sample in self.samples:
            for channel in sample:
                for key, attrs in layer._scoring_attrs_by_tree_key.items():
                    if key not in channel:
                        raise ValueError(f"Prepared batch is missing tree configuration {key}.")
                    prepared = channel[key]
                    prepared.validate()
                    if not set(attrs).issubset(prepared.raw_attributes):
                        raise ValueError(f"Prepared batch is missing attributes for {key}.")
                    if any(prepared.raw_attributes[attr].dtype != dtype for attr in attrs):
                        raise ValueError("Prepared attribute_dtype does not match the layer.")
