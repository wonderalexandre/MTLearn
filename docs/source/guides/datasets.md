---
orphan: true
---

# Image Datasets, Splits and Manifests

This guide uses `mtlearn.datasets` to pair images, define reproducible splits,
audit original files and reopen a saved selection. It also shows ordinary
PyTorch loading and integration with the prepared CFP cache. Run the Python
blocks in order with a build of MTLearn that includes `SplitManifest` and
`PairedImageDataset.from_suffixes`. The example uses CPU and creates six
synthetic image pairs under `artifacts/datasets-example`.

## Choose a Dataset

| Entry point | Input organization | Sample identity |
| --- | --- | --- |
| `PairedImageDataset.from_suffixes` | One directory with literal prefixes and suffixes | Text between the declared filename parts |
| `PairedImageDataset.from_folders` | Separate input and target directories | Matching filename stems |
| `PairedImageDataset.from_pairs` | Explicit `(sample_id, input_path, target_path)` records | Supplied strings |
| `PairedImageDataset.from_manifest` | Saved `SplitManifest` and current root directories | Saved strings and order |
| `GeneratedTargetImageDataset` | Input images and a target-generating callable | Returned filename; supply IDs explicitly for split helpers |

The historical `PairedImageDataset(...)` constructor remains available. The
factories above provide explicit pair records, IDs and reconstructible reader
configuration. Import public APIs from `mtlearn.datasets`; internal modules are
implementation details.

Sample IDs identify dataset records. They are unrelated to tree node IDs or
pixel IDs. Groups identify acquisitions, subjects or other units that your
protocol must keep together. The library validates the supplied mapping and
does not infer the correct scientific grouping from filenames or pixels.

## Create and Read Paired Images

```python
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from mtlearn.datasets import (
    GeneratedTargetImageDataset,
    PairedImageDataset,
    SplitManifest,
    audit_image_pairs,
    split_ids_by_group,
    subsets_from_ids,
    validate_split_ids,
)

run_path = Path("artifacts/datasets-example").resolve()
image_root = run_path / "pairs"
image_root.mkdir(parents=True, exist_ok=True)
groups = {
    "01": "scene-a",
    "1": "scene-a",
    "scene-b-0": "scene-b",
    "scene-b-1": "scene-b",
    "scene-c-0": "scene-c",
    "scene-c-1": "scene-c",
}
for index, sample_id in enumerate(groups):
    grid = np.arange(16 * 20, dtype=np.uint16).reshape(16, 20)
    image = ((grid + 19 * index) % 256).astype(np.uint8)
    target = np.where(image >= 128, 255, 0).astype(np.uint8)
    for suffix, array in (("_in", image), ("_target", target)):
        if not cv2.imwrite(str(image_root / f"{sample_id}{suffix}.png"), array):
            raise RuntimeError(f"Could not write {sample_id}{suffix}.png")

reader_options = {
    "strict_grayscale_uint8": True,
    "binary_target": True,
    "dtype": torch.float32,
    "scale_in": True,
    "scale_out": True,
}
dataset = PairedImageDataset.from_suffixes(
    image_root,
    suffix_in="_in",
    suffix_target="_target",
    extensions=(".png",),
    ordering="textual",
    unmatched="error",
    **reader_options,
)
image, target, sample_id = dataset[0]
assert image.shape == target.shape == (1, 16, 20)
assert image.dtype == target.dtype == torch.float32
assert set(torch.unique(target).tolist()) <= {0.0, 1.0}
print(dataset.sample_ids)
```

IDs are nonempty, case-sensitive strings: `"01"` and `"1"` remain distinct.
The suffix factory scans direct files in textual order without reading pixels.
Prefixes and suffixes are literal filename fragments. Duplicate IDs within a
role and files matching both roles always fail. `unmatched="warn"` or
`"ignore"` can exclude incomplete pairs; inspect `missing_input_ids` and
`missing_target_ids`. Neither option resolves ambiguous pairs.

Items contain `(input_tensor, target_tensor, sample_id)`. Tensors have shapes
`(C, H, W)` and `(C_target, H, W)` on CPU. Indexing performs decoding and
transformation; it does not repeat the file audit. `records` exposes immutable
records with `sample_id`, `input_path` and `target_path`; `pairs` contains only
the two paths, without an ID field.

### Select the Reading Policy

| Option | Effective behavior |
| --- | --- |
| `grayscale_in=True`, `grayscale_target=True` | Ordinary OpenCV grayscale decoding; False selects RGB decoding |
| `strict_grayscale_uint8=True` | Require native 2D uint8 arrays for both roles; grayscale flags must be True |
| `binary_target=True` | Require a native 2D uint8 target with values in either `{0, 1}` or `{0, 255}`; return zero/one values and bypass `scale_out` |
| `invert_in`, `invert_target` | Apply `255 - image` before resizing; default False |
| `num_rows`, `num_cols` | Supply both to resize; area interpolation for inputs and nearest interpolation for targets |
| `dtype` | Convert both tensors to this Torch dtype before scaling; default `torch.float32` |
| `scale_in`, `scale_out` | Divide converted values by 255; default True; integer tensors can promote to the default floating dtype |

Strict reading and binary targets are opt-in; the example enables both because
it generates uint8 binary masks. `binary_target=True` rejects
`invert_target=True`. For inverted 0/255 masks, use `binary_target=False`,
`invert_target=True` and `scale_out=True`, and validate binary values explicitly
in an audit callback. A 0/1 mask with ordinary reading and `scale_out=True`
would become zero and 1/255, so choose its scaling policy deliberately.

For RGB inputs, use `grayscale_in=False` and `strict_grayscale_uint8=False`.
Ordinary color decoding returns RGB and drops alpha channels. Ordinary
OpenCV reading converts images to uint8; setting the tensor dtype to float64
or uint16 does not restore precision discarded during decoding. For example,
a native uint16 PNG can be recorded as uint16 by an audit while the ordinary
reader decodes it to uint8. Input and target spatial sizes must match before
resizing in the explicit factories.

`dataset.get_preprocessing_contract()` describes actual reader operations for
both roles, including decoding, channel order, inversion, interpolation,
scaling, conversion `dtype` and effective `output_dtype`. Keep reader settings
fixed during an execution. The current factories describe deterministic image
reading; they do not serialize arbitrary augmentation callables.

### Use Separate Folders or Explicit Records

The following creates a second organization of the same synthetic files and
reconstructs the same tensors using the other factories.

```python
input_root = run_path / "inputs"
target_root = run_path / "targets"
input_root.mkdir(exist_ok=True)
target_root.mkdir(exist_ok=True)
for record in dataset.records:
    shutil.copy2(record.input_path, input_root / f"{record.sample_id}.png")
    shutil.copy2(record.target_path, target_root / f"{record.sample_id}.png")

folder_dataset = PairedImageDataset.from_folders(
    input_root, target_root, extensions=(".png",),
    ordering="textual", unmatched="error", **reader_options,
)
explicit_dataset = PairedImageDataset.from_pairs(
    [(record.sample_id, record.input_path, record.target_path)
     for record in dataset.records],
    ordering="provided", **reader_options,
)
assert folder_dataset.sample_ids == explicit_dataset.sample_ids == dataset.sample_ids
```

`from_pairs` preserves supplied order by default. Both it and `from_folders`
retain their historical `ordering="numeric"` option, which normalizes decimal
IDs: `"01"` and `"1"` then collide. Use textual or provided order when the
strings themselves define identity. The new suffix factory accepts only
textual order.

## Split by IDs and Groups

Define splits before extracting patches. Images from one acquisition must not
cross splits when that acquisition is the unit of independence. Here all six
images belong to three groups, and each split receives one complete group.

```python
split_ids = split_ids_by_group(
    dataset.sample_ids,
    groups,
    group_counts={"train": 1, "validation": 1, "test": 1},
    seed=42,
)
report = validate_split_ids(
    dataset.sample_ids, split_ids, groups=groups, require_complete=True,
)
subsets = subsets_from_ids(
    dataset, split_ids, groups=groups, require_complete=True,
)
assert report.sample_counts == {"train": 2, "validation": 2, "test": 2}
assert report.group_counts == {"train": 1, "validation": 1, "test": 1}
print(report.to_dict())

train_loader = DataLoader(subsets["train"], batch_size=2, shuffle=False, num_workers=0)
images, targets, batch_ids = next(iter(train_loader))
assert images.shape == targets.shape == (2, 1, 16, 20)
assert list(batch_ids) == split_ids["train"]
```

You can supply your own `split_ids` mapping instead of generating it. Split
names are arbitrary nonempty strings. The validators preserve the supplied
order, reject duplicate or unknown IDs and reject groups that cross splits.
They use metadata without reading dataset items. Subsets retain the original
dataset and item type; nested subsets are supported.

`group_counts` counts groups, not images or percentages, and must assign all
available groups using positive integers. Group names are sorted textually
before a local `numpy.random.RandomState(seed)` permutation; mapping key order
assigns the groups to splits, and source order is retained within each group.
Global Python, NumPy and Torch RNG states are unchanged. Persist the resulting
IDs: the seed alone is insufficient to identify a selection.

| Validation policy | Default for split helpers | Meaning |
| --- | --- | --- |
| `require_complete` | False | When True, assign every source ID |
| `require_complete_groups` | True | Include every member of each selected group, relative to the supplied source universe |
| `allow_empty` | False | When True, individual splits can be empty; at least one must contain samples |
| `groups=None` | Grouping disabled | A supplied mapping must cover exactly the source IDs; `{}` does not disable grouping |

For a reduced execution that intentionally cuts a group, create a separate view
with an explicit partial-group policy. Keep the full selection for provenance.
Repeated source positions are invalid for partitioning by unique IDs; repeated
positions in a training sampler remain valid.

```python
small_train = subsets_from_ids(
    dataset,
    {"quick_check": split_ids["train"][:1]},
    groups=groups,
    require_complete=False,
    require_complete_groups=False,
)["quick_check"]
assert len(small_train) == 1
```

## Audit Original Files

An audit records streamed SHA-256 file hashes, native shapes and dtypes, and
optional consumer metadata. It decodes one pair at a time with
`cv2.IMREAD_UNCHANGED`. Callback arrays are read-only and precede inversion,
resize, tensor conversion and scaling. Native color arrays use BGR/BGRA;
ordinary reader tensors use RGB. The default audit verifies readability and
spatial agreement without imposing binary masks, uint8 or equal channel counts.

```python
def validate_sample(record, input_array, target_array):
    if target_array.ndim != 2 or target_array.dtype != np.uint8:
        raise ValueError(f"Expected a 2D uint8 mask for {record.sample_id}")
    values = np.unique(target_array)
    if not set(values.tolist()) <= {0, 255}:
        raise ValueError(f"Nonbinary mask for {record.sample_id}")
    return {
        "mask_values": [int(value) for value in values],
        "original_zero_fraction": float(np.mean(target_array == 0)),
    }

roots = {"input": image_root, "target": image_root}
audit = audit_image_pairs(dataset, roots=roots, validate_sample=validate_sample)
assert audit.sources["01"]["input"]["dtype"] == "uint8"
assert audit.sources["01"]["metadata"]["mask_values"] == [0, 255]
assert audit.preprocessing == dataset.get_preprocessing_contract()
```

Return None or a JSON dictionary with string keys and finite numeric values.
Convert NumPy scalars to Python values, as above. Raise an exception to reject
a sample. Do not mutate files or reader settings, retain arrays, or use the
callback to transform samples. Metadata is stored under `sources[id]["metadata"]`
and does not overwrite required fields.

For separate input and target directories, pass their respective roots. Every
source must resolve within its named root, including through symbolic links.
The audit detects ordinary file changes during inspection and rechecks file
signatures when used to create a manifest. It does not lock the source directory;
repeat validation when files may have changed.

## Save, Verify and Relocate a Manifest

```python
manifest = SplitManifest.from_audit(
    audit, split_ids, groups=groups,
    provenance={"split_seed": 42, "split_algorithm": "numpy.RandomState"},
)
manifest_path = run_path / "selection.json"
manifest.save(manifest_path)

restored = SplitManifest.load(manifest_path)
assert restored.to_dict() == manifest.to_dict()
assert restored.validate_files(roots=roots)["valid"]
assert restored.validate_preprocessing(dataset)["valid"]

relocated_root = run_path / "relocated-pairs"
shutil.copytree(image_root, relocated_root, dirs_exist_ok=True)
relocated_roots = {"input": relocated_root, "target": relocated_root}
assert restored.validate_files(roots=relocated_roots)["valid"]
train_dataset = PairedImageDataset.from_manifest(
    manifest_path,
    roots=relocated_roots,
    split="train",
    expected_fingerprint=restored.fingerprint(),
)
assert list(train_dataset.sample_ids) == split_ids["train"]
assert restored.validate_preprocessing(train_dataset)["valid"]
```

`SplitManifest` uses `format="mtlearn.dataset-manifest"` and `schema_version=1`.
Construction and loading validate IDs, splits, groups, policies and reader
contracts. Properties and `to_dict()` return defensive copies. `from_audit`
requires complete coverage by default and derives preprocessing from the audited
reader; an optional `preprocessing=` mapping must match that contract exactly.
The parent directory must exist for `save`, which uses a unique temporary file
and atomic replacement.

File references are relative POSIX paths under named roots. Absolute paths,
Windows drive paths, `.` or `..` components and backslashes are rejected.
Resolving a reference rejects symbolic links that escape its root. Root
locations are supplied at execution time and can change without changing the
saved document or content fingerprints.

Loading a manifest does not read images. `validate_files` explicitly checks
existence and hashes; `raise_on_error=False` returns issues with sample IDs and
roles. `validate_preprocessing` compares an explicit reader, its subset, or a
contract mapping. File hashes alone cannot detect a changed inversion or resize
policy. Readers that scale integer tensors also depend on Torch's default
floating dtype, which must agree with the saved effective dtype.

`from_manifest` resolves only the saved records, uses the saved reader settings
and checks the expected manifest fingerprint when supplied. It checks file
existence and containment, without rehashing images. New unrelated files in the
directory do not enter the selection. A selected split must be nonempty; use
`subsets_from_ids(..., allow_empty=True)` for empty views. The manifest and source
files must remain available and unchanged while workers use them.

## Distinguish Content, Selection and Preprocessing

```python
identities = restored.fingerprints()
train_identities = restored.fingerprints(split="train")
print(train_identities)
```

| Fingerprint kind | Recorded meaning |
| --- | --- |
| `input` | Sorted `(sample_id, original input file SHA-256)` pairs |
| `target` | Sorted `(sample_id, original target file SHA-256)` pairs |
| `splits` | Ordered split membership, groups and policies |
| `preprocessing_input` | Effective input reader operations |
| `preprocessing_target` | Effective target reader operations |
| `manifest` | Full validated document excluding provenance, including relative paths and audit metadata |

A split-scoped fingerprint excludes sources and groups from other splits.
Preprocessing is shared by all splits. Reordering a split changes its selection
identity without changing its input content fingerprint. JSON map order and
formatting do not affect fingerprints; list order does. Provenance is excluded.

These file hashes differ from `DatasetSource.target_fingerprint()`, which reads
and hashes the effective target tensors, and from the cache's identity of input
pixels and morphological preparation. Recoding a file can change its byte hash
without changing its decoded pixels. A label-only change affects target
identity; it need not invalidate input morphology. Keep these components
separate when identifying experiments and caches.

## Prepare and Read a CFP Cache

Use `DatasetSource.from_paired_dataset` to adapt paired items to `(input, target)`
while retaining IDs separately. Creating the source directly from a named
manifest split gives workers a compact manifest reference; arbitrary large
Subset index lists or explicit pair lists can exceed the descriptor budget.
Audit and verify files before opening the cache.

```python
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer
from mtlearn.layers.cfp import (
    CFPPreprocessor,
    DatasetCacheConfig,
    DatasetSource,
    open_dataset_cache,
    prepare_dataset_cache,
)

source = DatasetSource.from_paired_dataset(train_dataset)
cfp = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=[{
        "tree_type": "max-tree",
        "attributes": [morphology.AttributeType.AREA],
    }],
    scale_mode="dataset_clipped_zscore01",
    device="cpu",
)
cache_name = "train-" + "-".join(
    train_identities[kind][:12]
    for kind in ("splits", "input", "preprocessing_input")
)
cache_config = DatasetCacheConfig.from_preprocessor(
    CFPPreprocessor.from_layer(cfp),
    path=run_path / "cache",
    manifest=cache_name,
    source_version=train_identities["input"],
    preprocessing_version=train_identities["preprocessing_input"],
    split="train",
)
preparation = prepare_dataset_cache(
    cache_config, source,
    max_disk_bytes=32 * 1024**2,
    collect_stats=cfp.get_statistics_contract(),
)
if preparation.status != "complete" or preparation.statistics is None:
    raise RuntimeError(f"Preparation incomplete: {preparation.status}")

with open_dataset_cache(
    cache_config, source, layer=cfp, statistics=preparation.statistics,
    batch_size=1, num_workers=0,
) as loader:
    for prepared, target in loader:
        response = cfp.forward_prepared(prepared)
        assert response.shape == target.shape
```

The CFP cache stores CPU morphological trees and raw node attributes. A max-tree
represents upper-level connected components; `AREA` measures each node's full
support in pixels. CFP computes learned scores and reconstructs responses during
forward execution. Targets continue to come from the source dataset.

Fit normalization statistics on the training selection and reuse them for
validation and test. Prepare evaluation caches with `split="evaluation"` and
without `collect_stats`; open them using the training statistics. Changing only
evaluation files does not change the training fingerprints used above.

The example's cache manifest name includes selection order, input content and
input preprocessing. It excludes target hashes and target preprocessing.
Keep the library's existing pixel/preparation identity for individual cache
entries: do not substitute the full dataset-manifest fingerprint as a tree key.
If tree or attribute requirements change incompatibly, choose a new cache
manifest or preparation path as well.

To enable prepared-reader processes, pass `num_workers=1` or `2` together with
explicit `worker_memory_bytes`, `max_prefetch_bytes`, `max_batch_bytes`,
`source_memory_bytes` and `read_workspace_bytes` budgets appropriate to the
images and trees. Check `loader.num_workers` and `loader.prefetch` to confirm the
requested mode is active. The manifest path and root locations must be accessible
from each process. In a script, place process-starting code under
`if __name__ == "__main__":`. Configuration serialization itself neither audits
files nor decodes images.

## Generate Targets with a Callable

Use `GeneratedTargetImageDataset` when targets are deterministic computations
from source images. Its callback receives a copy of the decoded image after
input inversion and resize, before tensor conversion and scaling. It must
return a NumPy image of the same spatial size. Color input is RGB here, unlike
the native audit callback's BGR/BGRA arrays.

```python
def threshold_target(image_array):
    return np.where(image_array >= 128, 255, 0).astype(np.uint8)

generated = GeneratedTargetImageDataset(
    input_root, target_fn=threshold_target,
    extensions=(".png",), grayscale=True,
    dtype=torch.float32, scale_in=True, scale_out=True,
)
generated_ids = [path.stem for path in sorted(input_root.glob("*.png"))]
generated_subsets = subsets_from_ids(
    generated, split_ids, sample_ids=generated_ids,
    groups=groups, require_complete=True,
)
image, target, filename = generated_subsets["train"][0]
assert image.shape == target.shape == (1, 16, 20)
assert filename.endswith(".png")
```

Generated-target items return filenames and do not expose the explicit paired
reader's `sample_ids` contract; supply ordered IDs without iterating items to
infer them. Audit and manifest reconstruction in this guide require actual
input/target files and an explicit paired reader. To persist generated targets
in that format, materialize them as files and build explicit pairs. The caller
owns the target-generation configuration and its reproducibility.
