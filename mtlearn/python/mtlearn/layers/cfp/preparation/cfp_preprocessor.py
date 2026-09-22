"""Morphological preparation with no model, normalization or retained cache."""
from __future__ import annotations

from types import MappingProxyType
import hashlib
import numpy as np
import torch
import mtlearn

from .... import morphology
from ..._helpers import build_tree, to_numpy_u8
from ..specs import FeatureSpec, TreeSpec
from .prepared_morphology import PreparedMorphology


class CFPPreprocessor:
    """Build CPU topology and raw attributes, one image/channel/tree at a time."""

    def __init__(self, *, tree_specs, features, attribute_dtype=np.float32, morphology_module=morphology):
        self.tree_specs = MappingProxyType(dict(tree_specs))
        self.features = MappingProxyType(dict(features))
        if not self.tree_specs or self.tree_specs.keys() != self.features.keys():
            raise ValueError("Preparation requires matching tree and feature specifications.")
        if any(not isinstance(spec, TreeSpec) or spec.cache_key() != key for key, spec in self.tree_specs.items()):
            raise ValueError("Tree keys must match their TreeSpec.")
        if any(not isinstance(spec, FeatureSpec) or spec.normalization is not None for spec in self.features.values()):
            raise ValueError("Preparation requires raw FeatureSpec objects.")
        self.attribute_dtype = np.dtype(attribute_dtype)
        if self.attribute_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("attribute_dtype must be float32 or float64.")
        self.morphology = morphology_module

    @classmethod
    def from_layer(cls, layer):
        """Copy only tree/feature contracts, never parameters, stats or device."""
        return cls(
            tree_specs={key: TreeSpec(spec.tree_type, spec.tos_interpolation,
                                     spec.tos_infinity_seed_row, spec.tos_infinity_seed_col)
                        for key, spec in layer._tree_spec_by_key.items()},
            features={key: FeatureSpec(tuple(sorted(attrs, key=lambda a: a.name)))
                      for key, attrs in layer._scoring_attrs_by_tree_key.items()},
            attribute_dtype=layer.attribute_dtype,
        )

    def get_config(self):
        from ._identity import preprocessor_config
        return preprocessor_config(self)

    @classmethod
    def from_config(cls, config):
        from ._identity import preprocessor_from_config
        return preprocessor_from_config(config)

    def prepare_image(self, image, tree_key=None, *, input_id=None):
        """Validate and prepare one 2D image using the existing uint8 conversion.

        Inputs may be tensors or NumPy arrays. Multiple trees require an explicit
        tree_key. No images or returned payloads are retained by this object.
        """
        from ..runtime.cache_input_contract import validate_cfp_cache_batch_x
        image = image if torch.is_tensor(image) else torch.as_tensor(np.asarray(image))
        if image.ndim != 2:
            raise ValueError("prepare_image expects one two-dimensional channel.")
        validate_cfp_cache_batch_x(image[None, None], expected_channels=1)
        if tree_key is None:
            if len(self.tree_specs) != 1:
                raise ValueError("tree_key is required when preparing multiple tree configurations.")
            tree_key = next(iter(self.tree_specs))
        return self.prepare_u8(to_numpy_u8(image), tree_key, input_id=input_id)

    def prepare_batch(self, images, *, store=None, sample_ids=None):
        """Prepare one dense BCHW batch; default to no persistent retention.

        Explicit stores key the effective uint8 pixels, shape, tree, attributes,
        dtype and process-local implementation identity. Logical IDs only label
        the returned batch. Keep batch size small to bound active memory.
        """
        from ..runtime.cache_input_contract import validate_cfp_cache_batch_x
        from ..storage import NullStore
        from .prepared_batch import PreparedBatch
        images = images if torch.is_tensor(images) else torch.as_tensor(np.asarray(images))
        validate_cfp_cache_batch_x(images)
        if images.shape[0] == 0 or images.shape[1] == 0:
            raise ValueError("prepare_batch requires at least one sample and channel.")
        ids = tuple(str(i) for i in range(len(images))) if sample_ids is None else tuple(sample_ids)
        if len(ids) != len(images) or any(not isinstance(i, str) for i in ids):
            raise ValueError("sample_ids must contain one string per sample.")
        store = NullStore() if store is None else store
        samples = []
        for sample in images:
            channels = []
            for image in sample:
                canonical = np.ascontiguousarray(to_numpy_u8(image.detach()))
                # No hashing on the default no-retention path. Hash once per
                # channel, even when several tree configurations are requested.
                digest = hashlib.sha256(memoryview(canonical)).hexdigest() if store.retains_entries else None
                payloads = {}
                for key in self.tree_specs:
                    if getattr(store, "persistent", False):
                        from ._identity import image_identity
                        identity = image_identity(self, canonical, key, pixel_digest=digest)
                    else:
                        identity = ("cfp-ram-v1", type(self), id(self.morphology), digest,
                                    canonical.shape, key, self.attribute_dtype.str,
                                    tuple(attr.name for attr in self.features[key].attributes))
                    payloads[key] = store.get_or_prepare(
                        identity, lambda key=key: self.prepare_u8(canonical, key))
                channels.append(payloads)
            samples.append(channels)
        return PreparedBatch(samples, ids)

    def prepare(self, source, *, store=None, manifest=None, source_version=None,
                preprocessing_version=None, split="train", sample_ids=None, collect_stats=None, cancel=None,
                num_workers=0, max_in_flight=None, worker_threads=1, max_sample_bytes=64 * 1024**2, max_retries=0,
                progress=None):
        """Prepare a finite map-style source, returning only progress/statistics.

        A persistent store requires named/versioned source and preprocessing
        contracts. Pass collect_stats=layer.get_statistics_contract() for a
        train-only snapshot. With a DiskStore, num_workers > 0 prepares in spawned
        CPU processes. Source/targets stay in the caller; only canonical uint8
        images enter the bounded queue (max_in_flight, default num_workers).
        max_sample_bytes limits each canonical input, not native workspace/RSS.
        max_retries retries ordinary task errors on the same pixels; abrupt
        worker death and quota exhaustion require an explicit resume.
        A cancel callable/event is checked between samples and while waiting.
        Resume by repeating the same call; completed content is verified/reused.
        """
        from ._persistent_preparation import prepare_source
        return prepare_source(self, source, store=store, manifest=manifest,
            source_version=source_version, preprocessing_version=preprocessing_version,
            split=split, sample_ids=sample_ids, collect_stats=collect_stats, cancel=cancel,
            num_workers=num_workers, max_in_flight=max_in_flight, worker_threads=worker_threads,
            max_sample_bytes=max_sample_bytes, max_retries=max_retries, progress=progress)

    def prepare_or_reuse(self, source=None, *, path, manifest, source_version, preprocessing_version,
                         split="train", sample_ids=None, collect_stats=None, mode="reuse",
                         max_disk_bytes=None, min_free_disk_bytes=0, pilot_samples=0,
                         disk_safety_factor=1.2, cancel=None, progress=None, num_workers=0,
                         max_in_flight=None, worker_threads=1, max_sample_bytes=64 * 1024**2, max_retries=0):
        from ._disk_preparation import prepare_or_reuse
        return prepare_or_reuse(self, source, path=path, manifest=manifest, source_version=source_version,
            preprocessing_version=preprocessing_version, split=split, sample_ids=sample_ids,
            collect_stats=collect_stats, mode=mode, max_disk_bytes=max_disk_bytes,
            min_free_disk_bytes=min_free_disk_bytes, pilot_samples=pilot_samples,
            disk_safety_factor=disk_safety_factor, cancel=cancel, progress=progress,
            num_workers=num_workers, max_in_flight=max_in_flight, worker_threads=worker_threads,
            max_sample_bytes=max_sample_bytes, max_retries=max_retries)

    def build_tree(self, image_np, spec):
        return build_tree(image_np, spec.tree_type, tos_interpolation=spec.tos_interpolation,
                          tos_infinity_seed_row=spec.tos_infinity_seed_row,
                          tos_infinity_seed_col=spec.tos_infinity_seed_col)

    def compute_tree_info(self, tree, spec):
        residues, tpre, tpost, parent, node_of_pixel = (
            mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
        )
        num_nodes = tpre.numel()
        if int(tpost.max().item()) != num_nodes:
            raise ValueError("CFP preparation requires compact preorder metadata; rebuild the native extension.")
        return {
            "residues": residues, "tpre": tpre, "tpost": tpost,
            "parent": parent, "node_of_pixel": node_of_pixel,
            "num_rows": tree.num_rows, "num_cols": tree.num_columns,
            "tree_type": spec.tree_type,
        }

    @torch.no_grad()
    def prepare_u8(self, image_np, tree_key, *, input_id=None):
        """Prepare an already canonical CPU uint8 image without normalization."""
        if not isinstance(image_np, np.ndarray) or image_np.dtype != np.uint8 or image_np.ndim != 2 or not image_np.size:
            raise ValueError("prepare_u8 requires a non-empty 2D uint8 NumPy image.")
        spec = self.tree_specs[tree_key]
        tree = self.build_tree(np.ascontiguousarray(image_np), spec)
        info = self.compute_tree_info(tree, spec)
        attrs = {}
        for attr_type in self.features[tree_key].attributes:
            values = self.morphology.compute_attributes(tree, [attr_type], dtype=self.attribute_dtype)[1]
            attrs[attr_type] = torch.as_tensor(values, device="cpu").reshape(-1, 1)
        return PreparedMorphology(spec, self.features[tree_key], info, attrs, input_id=input_id)
