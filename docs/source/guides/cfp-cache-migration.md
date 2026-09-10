# CFP preparation and cache migration

Use this page to choose a storage policy and migrate an existing CFP experiment.
The {doc}`connected-filter-preprocessing` guide describes the layer, its scoring
and detailed preparation options; {doc}`pytorch-integration` covers model
composition, devices and checkpoints. Public signatures are in
{doc}`../api/python/cfp_preparation`.

## Choose a policy

| Policy | Retained morphology | Cost on a later pass |
| --- | --- | --- |
| Ordinary images or `NullStore` (new API default) | None after consumers release it | Build trees and raw attributes again. |
| `MemoryStore(max_bytes=...)` | CPU tensors under a byte-bounded LRU | Reuse hits; rebuild misses. A dataset larger than the cache may have few hits. |
| `DiskStore(path, max_disk_bytes=..., max_ram_bytes=0)` | Local tensor files; optional bounded RAM | Read and validate files, then normalize the active batch. |
| `build_dataloader_cached` (compatibility API) | Whole selected dataset, including normalized payloads | Reuse the legacy indexed cache. Its original memory behavior remains. |

The new stores contain topology, pixel ownership, altitude residues and **raw**
attributes. They contain no scorer weights, scores, target labels, normalized
features or autograd graph. Linear and MLP scorers can share compatible raw data;
each layer keeps its own parameters and normalization state.

## Start from the existing layer

These examples assume a configured `layer` and a finite `train_dataset` yielding
`(CHW image, target)` pairs. Keep the original quantization, filter specification,
loss and optimizer. Use `batch_size=1` to preserve original resolution when images
have different shapes or when the active batch is large.

### Fit once without retention

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
snapshot = layer.fit_stats(train_loader)  # training split only; before epochs

for images, targets in train_loader:
    response = layer(images)
    # Compute the existing loss and backward here.
```

`fit_stats` consumes the loader's actual stream, including sampler multiplicity
and `drop_last`. It freezes the same node-weighted statistics used by the legacy
path. Forward does not refit them. This option avoids retaining the whole dataset
but reconstructs morphology during later passes. Use `load_stats`, a checkpoint
or `set_stats(snapshot)` when a compatible fit already exists. Statistical fitting
is unnecessary for `scale_mode="none"`; that mode changes attribute scaling and
should not be selected merely to avoid caching.

### Add a bounded RAM store

After fitting or loading statistics:

```python
from mtlearn.layers.cfp import CFPPreprocessor, MemoryStore

preprocessor = CFPPreprocessor.from_layer(layer)
store = MemoryStore(max_bytes=256 * 1024**2)
for images, targets in train_loader:
    batch = preprocessor.prepare_batch(images, store=store)
    response = layer(batch)
    # Compute loss and backward before releasing the active batch.
print(store.info())
store.clear()
```

Omitting `store` from `prepare_batch` uses no retained cache. An entry larger than
the RAM budget is returned for immediate use without evicting other entries to
make an impossible admission. It still requires enough working memory to build
and consume that image. An LRU quota does not limit the process's total memory.

### Prepare once on SSD

For repeated epochs on deterministic inputs, combine preparation, per-image
summaries and persistence in one pass. This avoids a separate tree-building pass
for `fit_stats`. In a script, call parallel preparation behind the usual
`if __name__ == "__main__":` guard; notebook-defined sources remain in the parent
process. This function also works sequentially with `num_workers=0`.

```python
from mtlearn.layers.cfp import (
    CFPPreprocessor, DiskStore, PreparedDataset, collate_prepared,
)
from torch.utils.data import DataLoader


def prepare_training(layer, train_dataset, cache_path, *, num_workers=0):
    preprocessor = CFPPreprocessor.from_layer(layer)
    with DiskStore(cache_path, max_disk_bytes=4 * 1024**3,
                   max_ram_bytes=0) as writer:
        result = preprocessor.prepare(
            train_dataset,
            store=writer,
            manifest="train-v1",
            source_version="dataset-and-split-v1",
            preprocessing_version="original-resolution-u8-v1",
            split="train",
            collect_stats=layer.get_statistics_contract(),
            num_workers=num_workers,
            max_in_flight=max(1, num_workers),
            worker_threads=1,
        )
        if result.status != "complete":
            raise RuntimeError("Resume preparation before training")
        return result.statistics


if __name__ == "__main__":
    snapshot = prepare_training(layer, train_dataset, "cfp-cache", num_workers=2)
    layer.set_stats(snapshot)
    # A compatible second layer can call second_layer.set_stats(snapshot).
    with DiskStore("cfp-cache", readonly=True, max_ram_bytes=0) as reader:
        prepared_train = PreparedDataset(reader, "train-v1", source=train_dataset)
        loader = DataLoader(prepared_train, batch_size=1, shuffle=True,
                            num_workers=0, collate_fn=collate_prepared)
        for batch, targets in loader:
            response = layer(batch)
            # Compute the existing loss and backward here.
```

The 4 GiB quota is illustrative, not a default or an estimate for a full dataset.
Prepare a representative subset in a separate pilot cache and measure file bytes
and peak memory before selecting a quota. Prepare test/validation data with a
separate manifest, `split="evaluation"`, and no `collect_stats`; keep the training
snapshot installed. A completed train manifest can also supply
`reader.statistics("train-v1", layer.get_statistics_contract())` on restart, using
small persisted summaries instead of scanning tensor files. Install it once
before epochs. Changing the scorer does not require building the trees again.

## Ownership and lifetime

`PreparedMorphology` and `PreparedBatch` share CPU buffers. Their mappings are
protected; tensors are read-only **by contract**, not by a hardware write barrier.
Clone a tensor before changing it. The layer normalizes and transfers the active
data needed by scoring/reconstruction to its device. Frozen statistical moments
remain on CPU; derived normalization constants are reused per version/device/dtype.

A live batch and an autograd graph can keep buffers alive after LRU eviction or
`store.clear()`. Autograd retains the necessary prepared owner through backward;
`retain_graph=True` extends that lifetime. Release unused responses, batches and
graphs instead of collecting an epoch of outputs. Do not close a store while a
loader still needs to retrieve samples. Closing it releases its resources but
does not delete valid SSD entries.

`info()` distinguishes retained tensors from live consumer handles:

| Field | Meaning |
| --- | --- |
| `retained_bytes` | Unique CPU storages retained by the store. |
| `active_bytes` | Unique storages referenced by tracked live consumer handles. |
| `active_not_retained_bytes` | Active storages outside retained entries. |
| `live_bytes` | Union of retained and tracked active storages, without double counting. |
| `disk_bytes` / `manifest_bytes` | Tensor-file bytes / database and manifest overhead, separately. |

These counters exclude Python/native temporaries, decoded images, optimizer and
activation memory, accelerator allocations and OS page cache. Extracting a tensor
and keeping it after its prepared wrapper dies is outside handle accounting.
Measure total process memory separately; memory-mapped files still consume physical
pages when accessed. CPU and GPU share the same physical budget on unified-memory
machines. SSD storage cannot make an individually oversized image fit in memory.

## Identity, interruption and process limits

Prepared morphology format 2 stores `node_of_pixel` as `uint32` (four bytes per
pixel), with a checked range of 0 through 4,294,967,295. Other traversal indices
remain `int64`, including `parent`, which permits -1. Reconstruction temporarily
converts the pixel map to `int64` for PyTorch indexing in forward and backward;
the retained map stays compact. This halves the map's retained CPU/device and
disk storage, but does not halve total training memory or eliminate indexing
workspace. Rebuild the native extension together with the Python package.

Format 1 stores are rejected without converting their entries. Use a separate
cache directory for format 2; existing caches are not migrated automatically.
Model checkpoints remain independent of the cache format.

A physical key includes effective pixels after CFP's canonical uint8 conversion,
shape, tree settings, ordered raw attributes, attribute dtype and format/backend
identity. Merely occupying the first network layer does not make its input fixed:
a crop, intensity transform or resize before CFP changes the prepared content.
For fixed inputs, weight updates and shuffled access do not invalidate raw entries.

A manifest separately records source/preprocessing versions, ordered logical IDs
and the split role. Keep the paired source in manifest order; apply `Subset` or a
sampler to `PreparedDataset` to change training order. Two logical samples with the
same pixels can share a file but each contributes to the train statistics. A retry
of the same logical sample contributes once. Changed input content/order under an
existing manifest raises an error: choose a new manifest/version for a new source.
Changing the compiled backend identity requires a separate cache directory; cache
files are not portable model checkpoints.

Reopen a writable store and repeat the same preparation request to resume. Complete
entries and their summaries are reused; owned incomplete files are recovered or
removed. Quota exhaustion stops new writes and preserves completed entries. Raise
the explicit quota or select a smaller source with its own manifest. The quota
covers temporary and published tensor files, not SQLite/WAL, filesystem allocation
or OS page cache. Valid disk entries are never silently deleted to make space.
`clear()` empties only the RAM tier.

Each writable store belongs to one process and holds a writer lock. Preparation
workers use CPU `spawn`; the parent bounds canonical inputs with `max_in_flight`
and `max_sample_bytes`, and workers publish files instead of returning large
payloads. `num_workers>0` currently requires SSD. Start with one worker, one pending
task and `worker_threads=1`, measure aggregate peak memory, then compare two workers.
Use `num_workers=0` on the consuming DataLoader: parallel preparation and loader
workers are separate controls. Do not overlap preparation and training until their
combined memory has been measured. See the layer guide for cancellation, retries,
worker leases and detailed failure behavior.

## Existing checkpoints and notebooks

No checkpoint migration is required. Model checkpoints and `save_stats` contain
parameters/statistics, not cache paths, quotas or prepared tensors. A model loaded
from an old checkpoint can consume ordinary images without opening an SSD cache.
Changing a storage policy does not change parameter names or the inference contract.

Keep using the existing indexed loader when its resource cost is appropriate.
`cached_sample_count()` describes that legacy cache; `store.info()` describes the
new stores. The original screws notebook remains a legacy example. The separate
`205_SA_L3D14M3` experiment uses shared SSD preparation and accumulates evaluation
counts and losses by batch; it retains only a selected image for visualization.
