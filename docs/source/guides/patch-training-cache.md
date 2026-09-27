---
orphan: true
---

# Patch Training with a Prepared CFP Cache

This guide trains a local segmentation model on patches of a global response
from `ConnectedFilterPreprocessingLayer` (CFP). It uses the public
`mtlearn.patches`, `mtlearn.training` and `mtlearn.layers.cfp` APIs. Run the Python
blocks in order with a build of MTLearn that includes these modules. The example
uses CPU and synthetic data so that it can run without an external dataset.

## What Is Cached and What Is Learned

CFP constructs morphological trees for complete images. The preparation cache
stores CPU tree metadata and raw node attributes. Scoring parameters, normalized
attributes used by the current forward, reconstructed responses and autograd
graphs are computed during training. A cached response would become stale after
an optimizer step.

For each image batch, compute the global CFP response once. The consumer then
runs on one patch at a time, keeping the consumer activations required for one
patch backward. Gradients from all patches accumulate before one global
backward and one optimizer step. The complete response, its accumulated gradient,
the producer graph and the active prepared batch still need memory.

| Name | Meaning in this guide |
| --- | --- |
| Morphological tree | A max-tree or min-tree of a complete image channel |
| Node attributes | Raw measurements such as `AREA` and `GRAY_LEVEL_HEIGHT` |
| Prepared batch | A `PreparedBatch` holding fixed CPU morphology for a batch |
| Response | The differentiable image tensor reconstructed by CFP |
| Core | The square patch region that contributes to the main loss |
| Halo | Context around the core; context outside the image is filled with zeros |
| Logits | The consumer output before sigmoid or softmax |

Core coordinates are zero-based image rows and columns. Patch geometry does not
refer to tree nodes, node supports or proper parts.

## Define Data and Models

A source yields an image shaped `(C, H, W)` and a target shaped `(C_target, H, W)`.
The prepared reader adds the batch dimension. Sample IDs and source order must
be stable. Keep image preprocessing deterministic while reusing a manifest.

```python
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer
from mtlearn.layers.cfp import (
    CFPPreprocessor,
    DatasetCacheConfig,
    DatasetSource,
    open_dataset_cache,
    prepare_dataset_cache,
)
from mtlearn.patches import patch_grid, reconstruct_logits
from mtlearn.training import backward_from_global_response


torch.manual_seed(12)
device = torch.device("cpu")
run_path = Path("artifacts/patch-training")
run_path.mkdir(parents=True, exist_ok=True)


def make_source(split, count, seed):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randint(
        0, 256, (count, 1, 32, 40), dtype=torch.uint8, generator=generator,
    )
    targets = (images >= 128).to(torch.float32)
    return DatasetSource(
        TensorDataset(images, targets),
        extract_sample=lambda sample: (sample[0], sample[1]),
        sample_ids=[f"{split}-{index:04d}" for index in range(count)],
    )


train_source = make_source("train", count=4, seed=21)
cfp = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=[
        {
            "name": "max_area_height",
            "tree_type": "max-tree",
            "attributes": [
                morphology.AttributeType.AREA,
                morphology.AttributeType.GRAY_LEVEL_HEIGHT,
            ],
            "regularizers": [
                {"kind": "edge_score_monotonicity", "weight": 0.01},
            ],
        },
        {
            "name": "min_area",
            "tree_type": "min-tree",
            "attributes": [morphology.AttributeType.AREA],
        },
    ],
    scale_mode="dataset_clipped_zscore01",
    clamp=None,
    device=device,
)
consumer = torch.nn.Sequential(
    torch.nn.Conv2d(cfp.out_channels, 8, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(8, 1, kernel_size=3, padding=1),
).to(device)
preprocessor = CFPPreprocessor.from_layer(cfp)
```

Replace the synthetic source with your dataset while retaining the explicit
extraction and IDs. For a `PairedImageDataset`, use
`DatasetSource.from_paired_dataset(dataset)`. Images must be finite and
nonnegative. CFP preserves uint8 pixels; floating inputs with maximum at most
1.5 are scaled by 255 before conversion to uint8. Use uint8 for images already
expressed as integer gray levels, including dark images.

The consumer receives `cfp.out_channels`, equal to input channels times filter
specs, and returns one binary logit channel here. The core and halo must be
compatible with the consumer architecture. This two-convolution example accepts
any positive spatial size; an encoder-decoder may impose divisibility constraints.

## Prepare a Disk Cache and Training Statistics

Use the full training split to collect normalization statistics during
preparation. There is no separate `fit_stats` pass in this example.

```python
train_config = DatasetCacheConfig.from_preprocessor(
    preprocessor,
    path=run_path / "cache",
    manifest="train",
    source_version="synthetic-train-v1",
    preprocessing_version="uint8-identity-v1",
    split="train",
)
preparation = prepare_dataset_cache(
    train_config,
    train_source,
    max_disk_bytes=64 * 1024**2,
    collect_stats=cfp.get_statistics_contract(),
)
if preparation.status != "complete":
    raise RuntimeError(f"Preparation stopped: {preparation.status}: {preparation.reason}")
if preparation.statistics is None:
    raise RuntimeError("Training statistics were not collected")
cfp.set_stats(preparation.statistics)
cfp.save_stats(str(run_path / "training-stats.pt"))
```

`prepare_dataset_cache` explicitly prepares missing entries and can repair
invalid entries using the source. Repeating it with the same source and contracts
reuses completed content. Inspect `status` before training; cancellation or a
resource limit can leave useful partial preparation that must be resumed.

`open_dataset_cache` opens completed preparation for reading. It verifies sample
IDs and positions, checks the layer's tree and attribute requirements, and
validates source image content when loading samples. It does not construct
missing trees or fit statistics. Targets come from the live source; the morphology
cache does not record label revisions for you.

Version the source and preprocessing whenever their meaning changes. Use another
manifest for an incompatible source or preparation contract. A scorer or learning
rate change does not require new raw morphology when tree and attribute
requirements remain compatible. Patch size, stride and halo also do not change
the full-image preparation contract.

## Train One Global Response per Batch

The main callback receives core logits, core targets and optional core weights.
It returns one differentiable scalar. The coordinator divides by the number of
patches. In this example, weighting uses a mean over all core elements; a
weight-normalized reduction would be a different objective.

```python
def main_loss(logits, target, weights):
    values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weights is not None:
        values = values * weights
    return values.mean()


parameters = list(cfp.parameters()) + list(consumer.parameters())
optimizer = torch.optim.AdamW(parameters, lr=1e-3)
patch_options = {"size": 16, "stride": 12, "halo": 2}
shuffle_generator = torch.Generator().manual_seed(31)

with open_dataset_cache(
    train_config,
    train_source,
    layer=cfp,
    batch_size=1,
    shuffle=True,
    generator=shuffle_generator,
    num_workers=0,
    prefetch=False,
    max_ram_bytes=0,
) as train_loader:
    for epoch in range(2):
        cfp.train()
        consumer.train()
        epoch_loss = 0.0
        batch_count = 0
        for prepared, target_cpu in train_loader:
            target = target_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            response = cfp.forward_prepared(prepared) / 255.0
            patches = patch_grid(*response.shape[-2:], **patch_options)
            parameter_loss = cfp.regularization_penalty(prepared)
            result, response_gradient = backward_from_global_response(
                response,
                consumer,
                target,
                patches,
                main_loss=main_loss,
                parameter_loss=parameter_loss,
                strategy="local",
            )
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_loss += result.total_loss
            batch_count += 1
            del prepared, target_cpu, target, response, parameter_loss, response_gradient
        print(f"epoch={epoch + 1} loss={epoch_loss / batch_count:.6f}")
```

The division by 255 is an explicit scale choice for CFP's reconstructed gray
levels. The patch APIs do not rescale tensors. Keep the same choice during
validation and inference. The regularization penalty already includes the
weights configured in each filter spec. Pass it once; do not multiply by the
patch count or call `.backward()` on it again.

Do not detach the global response before the coordinator call when learning CFP
parameters. The coordinator detaches local inputs internally, accumulates their
gradients, and then differentiates the original producer graph. The returned
`response_gradient` is detached and is for inspection, not a second backward.
The `PreparedBatch` remains valid throughout this computation; avoid retaining
old batches, responses or gradients in logging lists.

`batch_size=1` also supports images with different spatial sizes. Larger batches
require matching image shapes. Each image must be at least as large as the core.
The reader does not resize or pad different images into one dense batch.

## Loss and Gradient Contracts

For `K` patches, the objective is the arithmetic mean of the `K` main losses,
plus an optional weighted global auxiliary loss and an optional parameter
penalty. Overlap therefore repeats supervision where cores overlap. It is not a
uniform full-image pixel mean or the loss of averaged reconstructed logits.

- `pixel_weights`, when supplied, must be finite, nonnegative and have exactly
  the target shape and device. Targets and weights must not require gradients.
  A fully masked core can return a graph-connected zero such as
  `(logits * 0).sum()`; returning a detached zero is rejected.
- `auxiliary_loss(response, target)` may depend on that response and constants.
  It runs once globally, even when `auxiliary_weight=0` for reporting. A positive
  weight requires a differentiable response and a loss connected to it.
- `parameter_loss` is already weighted and must be independent of the response
  tensor. CFP's `regularization_penalty(prepared)` satisfies this contract by
  using the same prepared morphology and current scoring parameters.
- The main callback must not capture the global response or another autograd
  graph reused across patches. Producer and consumer must not share parameters.
- `strategy="local"` explicitly adds local gradients at their image coordinates;
  `"global_leaf"` obtains the same extraction adjoint through a detached global
  leaf. Neither divides gradients by pixel coverage. The valid halo may receive
  gradients even outside supervised cores.
- If a differentiable response is disconnected from all patch losses and there
  is no active auxiliary loss, its returned gradient is zero, but the coordinator
  does not manufacture zero `.grad` values on the producer. Parameter penalties
  can still contribute. This distinction matters for momentum and weight decay.

For multiclass segmentation, use integer targets shaped `(B, 1, H, W)` and let
the main callback call `F.cross_entropy(logits, target[:, 0])`. The consumer
must produce one logit channel per class. Apply sigmoid or softmax only where
the selected loss or evaluation protocol requires it.

## Validate with Frozen Training Statistics

Prepare validation images under a separate manifest with `split="evaluation"`
and without `collect_stats`. Continue using the statistics fitted on training.

```python
validation_source = make_source("validation", count=2, seed=22)
validation_config = DatasetCacheConfig.from_preprocessor(
    preprocessor,
    path=run_path / "cache",
    manifest="validation",
    source_version="synthetic-validation-v1",
    preprocessing_version="uint8-identity-v1",
    split="evaluation",
)
validation_preparation = prepare_dataset_cache(
    validation_config,
    validation_source,
    max_disk_bytes=64 * 1024**2,
)
if validation_preparation.status != "complete":
    raise RuntimeError(f"Validation preparation stopped: {validation_preparation.reason}")

cfp.eval()
consumer.eval()
with open_dataset_cache(
    validation_config, validation_source, layer=cfp, batch_size=1, prefetch=False,
) as validation_loader:
    with torch.no_grad():
        for prepared, target_cpu in validation_loader:
            response = cfp.forward_prepared(prepared) / 255.0
            patches = patch_grid(*response.shape[-2:], **patch_options)
            logits = reconstruct_logits(consumer, response, patches)
            loss = F.binary_cross_entropy_with_logits(logits, target_cpu.to(device))
            probability = logits.sigmoid()
            print(f"validation_loss={float(loss):.6f}")
            del prepared, target_cpu, response, logits, probability, loss
```

`reconstruct_logits` averages overlapping core logits and requires full image
coverage. It does not change model modes or apply sigmoid, softmax or a
threshold. Average logits before the chosen probability transform. Training may
use an explicit partial list of patches, but reconstruction rejects uncovered
pixels.

This validation loss uses reconstructed full-image logits and has a different
reduction from the training mean over patch losses. Patch inference also need
not equal a single full-image model call: padding, receptive fields,
normalization and global operations can change the result. In training,
BatchNorm buffers and dropout follow the sequential patch calls and their order.

## Reopen the Cache in Another Run

Keep the original source IDs, ordering and preprocessing when reopening.
Recover the preparation contract from the manifest and load saved training
statistics before forwarding through a newly constructed CFP layer.

```python
reopened_config = DatasetCacheConfig.from_manifest(run_path / "cache", "train")
cfp.load_stats(str(run_path / "training-stats.pt"))
with open_dataset_cache(
    reopened_config, train_source, layer=cfp, batch_size=1, prefetch=False,
) as reopened_loader:
    print(reopened_loader.opening_report)
```

This reopens morphology and statistics; it does not restore trained weights or
optimizer state. For model weights and CFP configurations, place `cfp` and
`consumer` in a parent `torch.nn.Module` and use `mtlearn.layers.save_checkpoint`
and `load_checkpoint`. Exact training continuation also needs optimizer state,
epoch and step, random-number generator states (including the reader's shuffle
generator), and the data, loss and patch contracts. Record label revisions as
well as the morphology manifest. The patch coordinator does not own checkpoints.

## Use an Explicit RAM Cache Instead

For a small dataset, a `MemoryStore` can retain preparation within one process.
Fit or restore training statistics first, as above. The same store can be reused
across batches and epochs. This is an alternative to disk preparation.

```python
from mtlearn.layers.cfp import MemoryStore

ram_store = MemoryStore(max_bytes=16 * 1024**2)
image, target = train_source[0]
prepared = preprocessor.prepare_batch(
    image.unsqueeze(0), store=ram_store, sample_ids=(train_source.sample_ids[0],),
)
with torch.no_grad():
    response = cfp.forward_prepared(prepared) / 255.0
print(ram_store.info())
del prepared, response
```

Feed that prepared batch into the same training step when using this alternative.
`MemoryStore` keys the effective uint8 pixels and preparation options. Eviction
drops the store's reference; it does not invalidate a batch in active use. A
budget smaller than one entry allows preparation but cannot retain that entry.
`max_bytes` and the disk reader's `max_ram_bytes` bound retained CPU storage;
they are not limits on process memory, active tensors or accelerator memory.

## Instrumentation and Scope

Pass `diagnostics=True` and `diagnostic_parameters=cfp.parameters()` to measure
the response and producer gradient norms for main, weighted auxiliary and
parameter contributions. These measurements perform extra vector-Jacobian
products and assume backward operations and hooks have no random or other
stateful effects. Select parameters for measurement without changing which
parameters receive gradients.

Pass `timing=True` independently to collect `model_forward_loss_seconds`,
`model_backward_seconds`, `auxiliary_seconds` when applicable, and
`producer_backward_seconds` when a global backward runs. These stage timings
synchronize the device and exclude preparation, the initial CFP forward,
extraction, diagnostics and optimizer work. Validation and Python scalar
reporting may synchronize even with timing disabled.

The supported contract is first-order differentiation with float32 or float64
responses and logits on CPU, and float32 on supported accelerators. Prepared
CFP execution on MPS requires a compatible PyTorch runtime; the known UInt32
transfer limitation in PyTorch 2.8 occurs before patch training. CPU is the
portable choice for this example. AMP, GradScaler, DDP, higher-order gradients
and `torch.compile` are outside the patch-training contract.

If a runtime error occurs after some patches, gradients and model buffers may
already have changed. Do not step the optimizer. Clear gradients, address the
cause, and recompute the forward before retrying. Restore model buffers too if
retrying must reproduce the original step exactly.
