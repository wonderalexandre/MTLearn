# ICPR 2026 Experiment Notebooks

This directory contains RRPR notebooks for the experiments presented in the
paper:

> Wonder A. L. Alves, Lucas de P. O. Santos, Ronaldo F. Hashimoto, Nicolas
> Passat, Anderson H. R. Souza, Dennis J. Silva, Yukiko Kenmochi.
> **A trainable connected filter preprocessing layer based on component
> trees.** International Conference on Pattern Recognition (ICPR), 2026, Lyon,
> France. [hal-05575141](https://hal.science/hal-05575141/)

Each notebook is a rerunnable `run_000` artifact for one dataset/backbone
configuration. The paper tables report statistics over 10 independent runs; the
notebooks expose the corresponding single-run locations used for inspection and
aggregation.

## Python Environment Requirements

Use a clean Python environment, preferably from `venv` or `conda`. The package
supports Python `>=3.9`; the RRPR reference environment used Python `3.12`.

The native `_mtlearn` extension is built against the active PyTorch
installation. If the review machine needs a specific CUDA, MPS, or CPU PyTorch
build, install that PyTorch build before installing MTLearn.

The ICPR 2026 notebooks are repository artifacts and are not installed into
`site-packages` by `pip`. Clone this repository and run the notebooks from the
checkout. The `notebooks` extra installs the notebook execution dependencies:
`ipykernel`, `matplotlib`, `nbformat`, `papermill`, `pandas`, `scikit-learn`,
`scipy`, and `segmentation-models-pytorch`. JupyterLab is optional and only
needed for interactive notebook editing.

## Reviewer Quick Start

Clone the repository and install the checkout:

```bash
git clone --recurse-submodules https://github.com/wonderalexandre/MTLearn.git
cd MTLearn
git checkout rrpr-icpr2026  # or the commit/branch listed in the RRPR form
python -m pip install -U pip
python scripts/install_release_dependencies.py --build-tools --torch none
python -m pip install -e ".[notebooks]" --no-build-isolation
```

See `docs/installation.md` and `docs/development.md` for PyPI installs,
editable source installs, validation commands, and environment details. The
repository also includes `requirements.txt` and `environment.yml` snapshots.

For interactive use, install JupyterLab in the same environment:

```bash
python -m pip install jupyterlab
```

After downloading the datasets described below, run one notebook with Jupyter or
Papermill. For example:

```bash
papermill \
  notebooks/ICPR2026/ICPR2026_screw_segmentation_ConvNet_run_000.ipynb \
  /tmp/ICPR2026_screw_segmentation_ConvNet_run_000.ipynb
```

The notebooks choose the training device automatically in this order: `cuda`,
`mps`, then `cpu`.

The file `icpr2026_utils.py` contains local RRPR helpers for dataset
download/location, seeding, training, threshold selection, metric tables,
qualitative panels, and attribute ablation. It is local to this reproducibility
package and is not part of the public MTLearn API.

If running multiple notebooks in JupyterLab on the same GPU, restart the kernel
between runs when memory is not released by the frontend. Papermill runs each
notebook in a separate process and is the recommended path for repeated
executions.

## Data

The screw segmentation dataset is public and handled by the MTLearn dataset
downloader. The plant segmentation dataset is downloaded with the dataset key
`plants_segmentation`, but it is not distributed publicly by this repository.
For RRPR review, download the required datasets with:

```bash
python -m mtlearn.data screws_segmentation
python -m mtlearn.data plants_segmentation --url "<review-url>"
```

The private plant review URL is provided to reviewers in the submission form.
The source-checkout helper provides equivalent commands:

```bash
python notebooks/ICPR2026/icpr2026_utils.py --download-data screws_segmentation
python notebooks/ICPR2026/icpr2026_utils.py --download-data plants_segmentation --url "<review-url>"
```

The local RRPR helper can check dataset status with:

```bash
python notebooks/ICPR2026/icpr2026_utils.py --list-data
```

## Configurations

| Dataset | Backbone | Notebook |
| --- | --- | --- |
| Plants | ConvNet | [ICPR2026_plant_segmentation_ConvNet_run_000.ipynb](ICPR2026_plant_segmentation_ConvNet_run_000.ipynb) |
| Plants | ED3-NN | [ICPR2026_plant_segmentation_ED3-NN_run_000.ipynb](ICPR2026_plant_segmentation_ED3-NN_run_000.ipynb) |
| Plants | U-Net MobileNetV2 | [ICPR2026_plant_segmentation_U-Net-MobileNetV2_run_000.ipynb](ICPR2026_plant_segmentation_U-Net-MobileNetV2_run_000.ipynb) |
| Screws | ConvNet | [ICPR2026_screw_segmentation_ConvNet_run_000.ipynb](ICPR2026_screw_segmentation_ConvNet_run_000.ipynb) |
| Screws | ED3-NN | [ICPR2026_screw_segmentation_ED3-NN_run_000.ipynb](ICPR2026_screw_segmentation_ED3-NN_run_000.ipynb) |
| Screws | U-Net MobileNetV2 | [ICPR2026_screw_segmentation_U-Net-MobileNetV2_run_000.ipynb](ICPR2026_screw_segmentation_U-Net-MobileNetV2_run_000.ipynb) |

## Hardware and Runtime

The executed notebooks were prepared with Python 3.12 on a CUDA runtime using
an NVIDIA T4 GPU. Runtime depends on the assigned machine and dataset cache
state. The recorded Papermill durations for the provided `run_000` notebooks
were approximately:

| Notebook | Approximate runtime |
| --- | ---: |
| `ICPR2026_screw_segmentation_ConvNet_run_000.ipynb` | 5 min |
| `ICPR2026_screw_segmentation_ED3-NN_run_000.ipynb` | 3 h 18 min |
| `ICPR2026_screw_segmentation_U-Net-MobileNetV2_run_000.ipynb` | 25 min |
| `ICPR2026_plant_segmentation_ConvNet_run_000.ipynb` | 7 min |
| `ICPR2026_plant_segmentation_ED3-NN_run_000.ipynb` | 34 min |
| `ICPR2026_plant_segmentation_U-Net-MobileNetV2_run_000.ipynb` | 18 min |

The screw notebooks are the recommended first checks because their data path
is public and automated.

## Reproducibility Scope

The notebooks are kept as rerunnable protocol artifacts. Regenerate outputs in
the review environment so logs, plots, threshold analysis, ablation tables, and
final metrics correspond to the local hardware and package build. The provided
notebooks report single-run metrics, so a fresh run can differ from the paper
means and standard deviations even when the training protocol is unchanged.

The paper aggregates were computed from 10 independent seeds per configuration.
Use `run_icpr2026_seeds.py` to repeat the same notebook protocol with the paper
seeds. For each run, the script writes CSV artifacts under
`notebooks/ICPR2026_runs/<config>/runs/run_XXX/`, including `base.csv` and
`cfp.csv`. To reproduce the aggregate tables, aggregate the rows with
`Evaluation Split == "Test"` and `Threshold Protocol == "train_pr"`.

The `Paper Output Map` identifies the notebook sections and variables used for
Tables 1-3 and Figure 3.

## Method Overview

Each configuration compares two models:

- **Baseline**: the segmentation backbone operates directly on the input image.
- **CFP variant**: a trainable morphological preprocessing layer (Connected
  Filter Preprocessing) is applied before the backbone, using attributes
  derived from component trees (MTLearn).

## Paper Protocol

The notebooks follow the experimental protocol used to reproduce the ICPR 2026
results. The paper text states the Adam settings for the training protocol; the
notebooks make the separate CFP parameter-group learning rate explicit.

| Item | Value |
| --- | --- |
| Image preprocessing | grayscale, resized to `588 x 660` pixels |
| Split protocol | Paper: 10 random 70/30 train/test splits; notebooks: one fixed `run_000` split |
| Representative seed | `2684470948` |
| Epochs | `100` |
| Optimizer | Adam, betas `(0.9, 0.999)`, epsilon `1e-8` |
| Backbone learning rate | `1e-3` |
| CFP parameter-group learning rate | `5e-2` |
| Batch size | `8` |
| Weight decay | `1e-7` |
| Early stopping | none |
| Loss | class-balanced `BCEWithLogitsLoss` |
| CFP auxiliary loss | summed BCE on the CFP output |
| CFP schedule | `gamma` linearly decays from `1` at epoch `0` to `0` at epoch `50` |
| CFP initialization | near-identity, with pass-through probability `0.995` |
| Threshold selection | `train_pr`: selected on the training split by maximizing F1 on the precision-recall curve, matching the protocol used by the paper notebooks |

CFP attribute sets are dataset-specific:

| Dataset | Attributes |
| --- | --- |
| Plants / PP | area, inertia, gray-level height, gray-level variance, perimeter |
| Screws / SWP | circularity, inertia, area, length of minor axis, gray-level height, length of major axis, rectangularity |

## What is included

Each notebook presents:

- training curves;
- qualitative predictions;
- decision threshold analysis;
- final evaluation metrics.

The ED3-NN notebooks also include a rerunnable CFP attribute-ablation section
for the `run_000` model. The ConvNet and U-Net MobileNetV2 notebooks omit this
section so that Table 3 is reproduced only where the paper reports it.

## Paper Output Map

The paper reports aggregate values over 10 runs. The notebooks expose the
corresponding `run_000` locations from which values, qualitative panels, and
attribute-ablation diagnostics are extracted:

| Paper item | Notebook location |
| --- | --- |
| Table 1, PP metrics | Plant notebooks, `Baseline Evaluation` and `CFP Evaluation`; values are extracted from `results_df_base` and `results_df`. |
| Table 2, SWP metrics | Screw notebooks, `Baseline Evaluation` and `CFP Evaluation`; values are extracted from `results_df_base` and `results_df`. |
| Table 3(a), PP attribute ablation | `ICPR2026_plant_segmentation_ED3-NN_run_000.ipynb`, `Attribute Ablation`. |
| Table 3(b), SWP attribute ablation | `ICPR2026_screw_segmentation_ED3-NN_run_000.ipynb`, `Attribute Ablation`. |
| Figure 3, qualitative examples | ConvNet and ED3-NN notebooks, `Figure 3 Panels`. U-Net MobileNetV2 is not used for Figure 3. |

The ConvNet and U-Net notebooks are still needed for Tables 1 and 2. They simply
do not reproduce Table 3.

## Ten-Run Execution

The original paper campaign used the same 10 seeds for every
dataset/backbone configuration. They are generated from the master seed `42`
with `np.random.SeedSequence(42).spawn(10)`:

| RUN_ID | SEED |
| ---: | ---: |
| 0 | 2684470948 |
| 1 | 4091952314 |
| 2 | 233227757 |
| 3 | 3276785861 |
| 4 | 3644269654 |
| 5 | 1206282609 |
| 6 | 3543069911 |
| 7 | 3479688010 |
| 8 | 877132087 |
| 9 | 1244265337 |

List the available configurations and seeds:

```bash
python notebooks/ICPR2026/run_icpr2026_seeds.py --list
```

Run a single smoke execution:

```bash
python notebooks/ICPR2026/run_icpr2026_seeds.py \
  --configs screw_convnet \
  --run-ids 0
```

Run all screw configurations:

```bash
python notebooks/ICPR2026/run_icpr2026_seeds.py --configs screws
```

Run all plant configurations:

```bash
python notebooks/ICPR2026/run_icpr2026_seeds.py --configs plants
```

Run the full 10-run campaign for all six configurations:

```bash
python notebooks/ICPR2026/run_icpr2026_seeds.py --configs all
```

By default, executed notebooks are written under
`notebooks/ICPR2026_runs/<configuration>/executions/`. Use `--output-dir` to
choose a different location and `--kernel` to select a Jupyter kernel. The
runner appends per-execution status and timing to
`notebooks/ICPR2026_runs/execution_summary.csv`.

## License and Artifacts

The MTLearn source code is distributed under the GPL-3.0-only license; see
`LICENSE` at the repository root. No pretrained weights or checkpoints are
required for these experiments. Each notebook trains the baseline model and the
CFP-enhanced variant from scratch.
