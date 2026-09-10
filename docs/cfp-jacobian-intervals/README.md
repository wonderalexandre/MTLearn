# CFP Jacobian interval comparison

The refactor was measured in two separate stages on 2026-09-10. The final CFP uses compact preorder intervals. The backend keeps its existing DFS event convention.

## Stages

1. **Events without sorting.** Native metadata still used entry/exit events. Preparation recovered traversal order with a linear scatter/gather. Backward accumulated gradients directly at pixel-owner entry positions, scanned the event vector, and subtracted interval endpoints. This removed rank conversion as well as the fallback sort.
2. **Compact preorder.** Native CFP extraction counted entries only, so a subtree occupies `[pre, post)` and `post - pre` is its node count. Preparation inverted the preorder permutation directly. The same interval-product code as stage 1 now scans the smaller time domain.

The [stage 1 patch](stage1-events-no-sort.patch) preserves the production changes of the first stage relative to the original checkout. It excludes the later compact-index, format-version, test, and documentation changes. Its native reference build remains in `build/cfp-node-u32-release`; stage 2 was built independently in `build/cfp-compact-jacobian`.

## Protocol and limits

- Platform: `macOS-26.6.2-arm64-arm-64bit`; PyTorch `2.10.0`; FP32; four CPU threads.
- Noise and nested-square images at 64, 256, and 512 pixels per side; max/min trees; CPU and MPS; one input seed (71).
- Three warm-ups and ten timed repetitions per operator; medians and quartiles retained. Variant order rotates across repetitions, and MPS synchronizes around each measurement.
- Forward and backward are isolated operator microbenchmarks. Tree construction, native metadata extraction, attributes, normalization, transfers, autograd, and optimization are excluded. Order-construction timing is recorded separately on CPU.
- The original cached backward receives its order and time extent, as the original public layer did. Its timing does **not** include an argsort. Stage 1 runtime improvements come from removing rank conversions and redundant scans; removal of preparation sorting is measured separately.
- Both stages were executed before/after the native change. Stage 2 additionally compares all three algorithms in one process using equivalent event/compact metadata, with the legacy reference retained in the benchmark script.
- These local microbenchmarks do not establish end-to-end training speedups, universal GPU performance, or statistical significance across inputs. Small latency differences may be timing variability.

## Backward comparison

The table reports medians of the per-case speedup ratios over 12 cases per device in the stage 2 run. A ratio above 1 means the variant is faster than the original cached backward.

| Device | Original / events without sorting | Original / compact | Events / compact |
| --- | ---: | ---: | ---: |
| CPU | 0.99x | 1.06x | 1.09x |
| MPS | 2.30x | 2.24x | 0.98x |

Selected 512 x 512 max-tree cases, with medians in milliseconds:

| Image | Device | Original backward | Events backward | Compact backward |
| --- | --- | ---: | ---: | ---: |
| noise | CPU | 1.079 | 0.748 | 0.666 |
| noise | MPS | 1.617 | 0.729 | 0.778 |
| nested | CPU | 0.578 | 0.692 | 0.701 |
| nested | MPS | 1.260 | 0.808 | 0.800 |

The event and compact scans improve the MPS backward substantially relative to the former rank conversion. Compact intervals add a modest CPU improvement for node-rich images. With few nodes and many pixels, the direct pixel-to-preorder gather can cost more than the original proper-part reduction on CPU; compact intervals do not remove that cost.

## Order construction and memory

For the 512 x 512 noise max-tree (94,182 nodes), CPU order construction takes:

| Method | Median (ms) |
| --- | ---: |
| original_argsort | 6.089 |
| events_linear | 0.423 |
| compact_linear | 0.102 |

The forward event/prefix length drops from 188,364 to 94,183 elements in this example. For FP32, a single such vector drops from 753,456 to 376,732 bytes. These are exact buffer sizes, **not** measured process or accelerator peak memory. Per-node pre/post tensors still each contain T indices. At the time of these measurements, the retained order fields shared one CPU storage. The final cleanup removes that storage entirely (8 bytes per node on CPU).

## Numerical checks

FP32 operator outputs were compared with FP64 interval products on the same signals. Largest absolute errors across the benchmark cases:

| Variant | Forward | Backward |
| --- | ---: | ---: |
| original_cached | 3.913e-04 | 6.604e-09 |
| events_scan | 3.913e-04 | 8.112e-09 |
| compact_scan | 4.376e-04 | 5.007e-09 |

The automated tests separately check hand-computable pixel supports, the adjoint identity, full-output finite differences, native parent-chain subtree membership, inactive slots after pruning, CPU/MPS behavior, and frozen checkpoint references. No frozen numerical fixtures were regenerated.

## Validation

- Stage 1: 81 existing tests and 3 new interval/no-sorting tests passed.
- Stage 2: the expanded CFP, gradient, binding, and morphology suite passed all
  **629 tests**, including persistent/parallel preparation and frozen checkpoints.
- Both C++ checks passed (`mtl_public_morphology_test`, `mtl_interpreter_test`).
- Final compact-only cleanup: **672 tests passed** (658 CFP/gradient/binding/morphology/paper tests plus 14 tests for the updated pilot and operator benchmark consumers). The three-way benchmark also completed a 16 x 16 smoke run on CPU and MPS; no new performance conclusions are drawn from this smoke run.
- Python syntax parsing, notebook Python-cell syntax, and `git diff --check` passed. The complete notebook was not executed; its updated autograd calls are covered by the gradient suite.

The expanded Python test selection was:

```bash
python -m pytest -q -p no:cacheprovider \
  mtlearn/tests/python/test_cfp_*.py \
  mtlearn/tests/python/test_gradchecks.py \
  mtlearn/tests/python/test_bindings.py \
  mtlearn/tests/python/test_morphology_api.py \
  mtlearn/tests/python/test_tip2026_stage10.py

python -m pytest -q -p no:cacheprovider \
  mtlearn/tests/python/test_tip2026_stage4_loss_pilot.py \
  mtlearn/tests/python/test_tip2026_stage8_b50_benchmarks.py
```

## Compact-only API

After the staged comparisons, the compact implementation was consolidated without
legacy compatibility. The historical JSON results above predate this cleanup.

- Reconstruction helpers take only the signal or pixel gradients, `tpre`, `tpost`, and `node_of_pixel`. The autograd boundary additionally takes image dimensions. The affine CFP autograd function uses the same compact metadata, without `parent` or traversal-order arguments.
- `order_forward`, `order_backward`, and `num_times` are removed from prepared metadata. Preparation no longer constructs an inverse preorder permutation. Forward sizes its buffer from the node tensor length plus one; backward uses the node tensor length, without reading device scalars on the CPU.
- Production accepts only compact intervals. Historical event algorithms remain isolated in this benchmark script so the three-way comparison is reproducible. The script's order-construction timings describe the historical stages; current compact preparation does not perform that step.
- Prepared morphology, persistent identities, serialized entries, and DiskStore metadata use format 4. Format 1/2/3 stores are rejected without rewriting them; prepare a new store after rebuilding the native extension. No migration path is provided.
- Model parameters, checkpoint contracts, and normalization statistics keep their existing formats.

## Reproduction

From a checkout with the compact native extension built:

```bash
PYTHONPATH=mtlearn/python:build/cfp-compact-jacobian/mtlearn/bindings \
python scripts/benchmarks/cfp_jacobian_intervals.py \
  --stage compact --output /tmp/cfp-stage2.json
```

For the event stage, apply the stage 1 patch to the original sources, use the corresponding native build, and pass `--stage events`. The compact-stage benchmark reproduces all three operator variants without reverting the checkout.

The local Python environment had an editable installation pointing to a different checkout. Validation used a temporary `sitecustomize.py` that excluded that installation finder, and checked the source/native paths recorded in the JSON files. No installed package or environment configuration was changed.

Raw measurements: [stage 1](stage1-events.json), [stage 2](stage2-compact.json).

## Image-folder comparison

The same three variants were also measured on all 39 images in `dat/misc256`,
preserving RGB channels. See the [per-image comparison](misc256/README.md),
including raw repetitions, input hashes, and results grouped equally by image.
