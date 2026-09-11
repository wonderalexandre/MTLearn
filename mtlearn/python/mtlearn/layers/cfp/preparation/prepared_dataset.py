"""Lazy, ordered views over complete prepared manifests."""
import json
import operator

import torch
from torch.utils.data import Dataset, default_collate, get_worker_info

from ._identity import preprocessor_from_config
from ._persistent_preparation import manifest_record, sample_image, image_bindings
from .prepared_batch import PreparedBatch


class PreparedDataset(Dataset):
    """Resolve one manifest sample at a time, optionally paired with a live source.

    With ``source``, input content is verified on every access and its target is
    returned. Without a source this is an immutable prepared snapshot, independent
    of current image files and labels. The source must match manifest order;
    apply samplers or Subset to this PreparedDataset for training order changes.
    P3 consumption uses DataLoader num_workers=0; each DiskStore is process-local.
    """

    def __init__(self, store, manifest, source=None):
        row = manifest_record(store, manifest)
        self.store, self.manifest, self.source = store, manifest, source
        self.preprocessor = preprocessor_from_config(json.loads(row["config"]))
        self._length = row["sample_count"]
        if source is not None and len(source) != self._length:
            raise ValueError("Source length does not match the prepared manifest.")

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if get_worker_info() is not None:
            raise RuntimeError("PreparedDataset currently requires DataLoader num_workers=0.")
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        manifest_record(self.store, self.manifest)
        row = self.store._db.execute("SELECT * FROM samples WHERE manifest=? AND position=?",
                                     (self.manifest, index)).fetchone()
        if row is None:
            raise RuntimeError("Manifest sample is missing; resume preparation.")
        bindings = json.loads(row["bindings"])
        image = target = None
        if self.source is not None:
            image, target = sample_image(self.source[index])
            if (image_bindings(self.preprocessor, image) != bindings or list(image.shape) != json.loads(row["shape"])):
                raise ValueError("Source pixels/shape changed; prepare a new manifest before pairing with targets.")
        try:
            channels = []
            for trees in bindings:
                payloads = {}
                for tree_key, key in trees.items():
                    payload = self.store.get(key)
                    if payload is None:
                        raise FileNotFoundError(f"Missing or incomplete prepared entry {key}; resume with the source.")
                    payloads[tree_key] = payload
                channels.append(payloads)
            batch = PreparedBatch((channels,), (row["sample_id"],))
        except (FileNotFoundError, ValueError):
            if image is None or self.store.readonly:
                raise
            # Release any partial read before constructing a missing tree.
            channels.clear()
            payloads.clear()
            payload = None
            batch = self.preprocessor.prepare_batch(image.unsqueeze(0), store=self.store, sample_ids=(row["sample_id"],))
        return batch if target is None else (batch, target)


def collate_prepared(samples):
    """Collate prepared samples (and optional targets) without copying raw tensors."""
    if not samples:
        raise ValueError("Cannot collate an empty prepared batch.")
    paired = isinstance(samples[0], (tuple, list))
    batches = [sample[0] for sample in samples] if paired else samples
    if any(not isinstance(batch, PreparedBatch) or batch.shape[0] != 1 for batch in batches):
        raise ValueError("collate_prepared expects single-sample PreparedBatch entries.")
    result = PreparedBatch(tuple(batch.samples[0] for batch in batches),
                           tuple(batch.sample_ids[0] for batch in batches))
    return (result, default_collate([sample[1] for sample in samples])) if paired else result
