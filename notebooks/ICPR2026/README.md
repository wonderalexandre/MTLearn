# ICPR 2026 Reproducibility Guide (RRPR)

This document supports the **Reproducible Research in Pattern Recognition (RRPR)** review for:

- **A trainable connected filter preprocessing layer based on component trees**,
  Wonder A. L. Alves, Lucas de P. O. Santos, Ronaldo F. Hashimoto, Nicolas
  Passat, Anderson H. R. Souza, Dennis J. Silva, Yukiko Kenmochi,
  *ICPR 2026, Lyon, France* (`hal-05575141`).

<p>
  <a href="https://iapr-tc22-rrl.github.io/icpr2026/results/"><img src="RRPR_badge.svg" alt="ICPR 2026 RRPR reproducibility badge" width="110" align="left" hspace="14" vspace="4" /></a>
  <b>Reproducibility badge.</b> This work was awarded the ICPR 2026 reproducibility badge by the
  IAPR TC22 Reproducible Research Label initiative. Full evaluation results are published at
  <a href="https://iapr-tc22-rrl.github.io/icpr2026/results/">iapr-tc22-rrl.github.io/icpr2026/results</a>.
</p>

Its scope is reviewer-centric: provide the minimum, complete steps to reproduce the reported paper results from the same artifacts used in the evaluation campaign.

## Purpose and Scope

Use this guide to:

- prepare a compatible environment and dataset access,
- run the six notebook configurations over the fixed seeds used by the paper,
- inspect and validate generated outputs,
- rebuild the aggregate paper table used for comparison.

Only the materials and commands in this folder are required, plus access credentials/URL for the restricted plant-segmentation dataset.

## Repository and Output Layout

`notebooks/ICPR2026/` is the reproducibility bundle containing the six notebooks and helpers:

- `run_icpr2026_seeds.py` (batch execution across fixed seeds),
- `icpr2026_utils.py` (shared utilities, including dataset bootstrap),
- `ICPR2026_*_run_000.ipynb` (six entrypoint notebooks).

By default, outputs are written under `<root> = notebooks/ICPR2026_runs/`.

Run identifiers are `run_000` through `run_009` and map to deterministic seeds:

- `run_000` → seed `2684470948`
- `run_001` → seed `4091952314`
- `run_002` → seed `233227757`
- `run_003` → seed `3276785861`
- `run_004` → seed `3644269654`
- `run_005` → seed `1206282609`
- `run_006` → seed `3543069911`
- `run_007` → seed `3479688010`
- `run_008` → seed `877132087`
- `run_009` → seed `1244265337`

Seeds are generated from `np.random.SeedSequence(42).spawn(10)`.

## Output directory layout

All executions write to `<root>/` with these structures:

- `<root>/execution_summary.csv`
- `<root>/paper_table_metrics.csv`
- `<root>/<config>/executions/`
- `<root>/<config>/runs/run_###/`

`<config>` is one of:
`plant_convnet`, `plant_ed3`, `plant_unet`, `screw_convnet`, `screw_ed3`, `screw_unet`.

Example:

```text
<root>/
  execution_summary.csv
  paper_table_metrics.csv
  plant_convnet/
    executions/
      ICPR2026_plant_segmentation_ConvNet_run_000.ipynb
      ICPR2026_plant_segmentation_ConvNet_run_001.ipynb
      ...
    runs/
      run_000/
        base.csv
        cfp.csv
      run_001/
        base.csv
        cfp.csv
      ...
  screw_ed3/
    executions/
      ICPR2026_screw_segmentation_ED3-NN_run_000.ipynb
      ...
    runs/
      run_000/
        base.csv
        cfp.csv
      ...
```

Meaning of generated files:

- `<root>/execution_summary.csv`: one row per executed notebook run with start/end timestamps, status, elapsed time, and file locations.
- `<root>/paper_table_metrics.csv`: aggregated table (mean/std) built from available `base.csv`/`cfp.csv` rows.
  This table is updated as runs complete and may be partial until all planned runs are executed.
- `<root>/<config>/executions/`: executed notebook copies for each configuration and run.
- `<root>/<config>/runs/run_###/base.csv`: Baseline metrics for one run.
- `<root>/<config>/runs/run_###/cfp.csv`: CFP metrics for one run.
- `base.csv` and `cfp.csv` include both train and test rows and threshold metadata.

## Environment Setup

Use a clean environment (`venv` or `conda`) with Python `>=3.9`.
The RRPR reference environment used Python `3.12`.

```bash
git clone --recurse-submodules https://github.com/wonderalexandre/MTLearn.git
cd MTLearn
python -m pip install -U pip
python scripts/install_release_dependencies.py --build-tools
python -m pip install -e ".[notebooks]" --no-build-isolation
```

If the environment already has the exact Torch stack you want to keep, preserve
it with `--torch none`:

```bash
python scripts/install_release_dependencies.py --build-tools --torch none
python -m pip install -e ".[notebooks]" --no-build-isolation
python -m pip install jupyterlab
```

If specific PyTorch builds are needed for CUDA/MPS/CPU, install those first.

For additional install and validation instructions, see: [docs/installation.md](../../docs/installation.md) and [docs/development.md](../../docs/development.md).

## Data Preparation

Prepare datasets (screws public, plants via private RRPR URL):

```bash
python notebooks/ICPR2026/icpr2026_utils.py --download-data screws_segmentation
python notebooks/ICPR2026/icpr2026_utils.py --download-data plants_segmentation --url "<review-url>"
```

Useful dataset-helper options:

- `--all`: download both ICPR datasets.
- `--data-dir`: override default root (for both download and listing).
- `--force`: re-download even when files already exist.
- `--keep-archive`: keep the downloaded archive locally.

Check resolved dataset roots:

```bash
python notebooks/ICPR2026/icpr2026_utils.py --list-data
```

When using a non-default dataset path, pass the same root to list commands:

```bash
python notebooks/ICPR2026/icpr2026_utils.py --all --data-dir /path/to/mtlearn_data
python notebooks/ICPR2026/icpr2026_utils.py --list-data --data-dir /path/to/mtlearn_data
```

## Reproduction Entry Point

All RRPR reruns are launched by:

```text
python notebooks/ICPR2026/run_icpr2026_seeds.py
```

The launcher controls the whole campaign by:

- selecting one or more experiment configurations (`--configs`);
- selecting one or more deterministic seeds (`--run-ids`);
- executing the corresponding notebooks through Papermill;
- writing outputs under `<root>` and regenerating `paper_table_metrics.csv`.
- tracking campaign progress with a global progress bar.

Start here to understand how the campaign is executed.

```bash
python notebooks/ICPR2026/run_icpr2026_seeds.py --help
```

## Execution Protocol

To reproduce the paper results, run the six notebooks for each fixed seed used in the campaign:

1. choose experiment configurations with `--configs`;
2. choose deterministic seeds with `--run-ids`;
3. execute all planned notebook runs;
4. validate outputs and aggregated tables.

The full paper run is:

```bash
python notebooks/ICPR2026/run_icpr2026_seeds.py --configs all --run-ids 0 1 2 3 4 5 6 7 8 9
```

This produces:

- two experiment rows per configuration/run in `runs/run_###/base.csv` and `runs/run_###/cfp.csv`;
- campaign-level execution log in `<root>/execution_summary.csv`;
- aggregate table in `<root>/paper_table_metrics.csv` (mean/std over the executed runs).

Reviewer workflow generally runs this in two phases: one smoke test (e.g. `--configs screw_convnet --run-ids 0`) and then the full required subset for the paper tables.

## Experiment Configurations

Each row is one executable RRPR configuration. The `Config key` is the identifier accepted by `run_icpr2026_seeds.py --configs`.

| Dataset | Backbone | Config key | Notebook |
| --- | --- | --- | --- |
| Plants | ConvNet | `plant_convnet` | [ICPR2026_plant_segmentation_ConvNet_run_000.ipynb](ICPR2026_plant_segmentation_ConvNet_run_000.ipynb) |
| Plants | ED3-NN | `plant_ed3` | [ICPR2026_plant_segmentation_ED3-NN_run_000.ipynb](ICPR2026_plant_segmentation_ED3-NN_run_000.ipynb) |
| Plants | U-Net MobileNetV2 | `plant_unet` | [ICPR2026_plant_segmentation_U-Net-MobileNetV2_run_000.ipynb](ICPR2026_plant_segmentation_U-Net-MobileNetV2_run_000.ipynb) |
| Screws | ConvNet | `screw_convnet` | [ICPR2026_screw_segmentation_ConvNet_run_000.ipynb](ICPR2026_screw_segmentation_ConvNet_run_000.ipynb) |
| Screws | ED3-NN | `screw_ed3` | [ICPR2026_screw_segmentation_ED3-NN_run_000.ipynb](ICPR2026_screw_segmentation_ED3-NN_run_000.ipynb) |
| Screws | U-Net MobileNetV2 | `screw_unet` | [ICPR2026_screw_segmentation_U-Net-MobileNetV2_run_000.ipynb](ICPR2026_screw_segmentation_U-Net-MobileNetV2_run_000.ipynb) |

## Reviewer Workflow

1. Set up environment and datasets.
2. Confirm plan:
   `python notebooks/ICPR2026/run_icpr2026_seeds.py --list`.
3. Validate planned execution paths:
   `python notebooks/ICPR2026/run_icpr2026_seeds.py --configs all --run-ids 0 --dry-run`.
4. Run a smoke test:
   `python notebooks/ICPR2026/run_icpr2026_seeds.py --configs screw_convnet --run-ids 0`.
5. Run the required seed/config campaign.
6. The campaign automatically validates generated outputs and prints a validation summary.

## Command Reference

| Goal | Command | Expected result |
| --- | --- | --- |
| List fixed seeds and configs | `python notebooks/ICPR2026/run_icpr2026_seeds.py --list` | Prints full config and seed mapping. |
| Print execution matrix only | `python notebooks/ICPR2026/run_icpr2026_seeds.py --configs all --run-ids 0 --dry-run` | Prints input notebook/output notebook/results path for each planned run. |
| Smoke test (single configuration, seed 0) | `python notebooks/ICPR2026/run_icpr2026_seeds.py --configs screw_convnet --run-ids 0` | Writes `executions/` and `runs/run_000/` for the selected config and updates summary CSVs. |
| Full campaign | `python notebooks/ICPR2026/run_icpr2026_seeds.py --configs all --run-ids 0 1 2 3 4 5 6 7 8 9` | Writes all runs for all 6 configs, plus `execution_summary.csv` and `paper_table_metrics.csv`. |
| Resume partial subset | `python notebooks/ICPR2026/run_icpr2026_seeds.py --configs plants --run-ids 0 1 2 3 4 --continue-on-error` | Continues remaining selected runs even when one execution fails. |
| Run with custom output directory | `python notebooks/ICPR2026/run_icpr2026_seeds.py --configs all --run-ids 0 1 2 3 4 --output-dir /path/to/ICPR2026_runs` | Runs are written under the custom path and `paper_table_metrics.csv` is regenerated there. |
| Validate finished runs | `python notebooks/ICPR2026/run_icpr2026_seeds.py --configs all --run-ids 0 1 2 3 4 5 6 7 8 9 --validate-only` | Checks execution summary, required result CSVs, and prints aggregate table metadata. |

Optional arguments:

- `--output-dir`: custom output root.
- `--kernel`: Jupyter kernel passed to Papermill.
- `--progress-bar`: show per-cell notebook progress (printed above the global campaign bar when enabled).
- `--no-global-progress`: disable campaign-level progress bar (enabled by default).
- `--skip-existing`: skip execution if output notebook already exists.
- `--continue-on-error`: continue remaining runs after one failure.
- `--dry-run`: print planned runs without executing.
- `--configs`: accepts `all`, `plants`, `screws`, or each config key above.
- `--run-ids`: accepts integers `0..9` (default).
- `--validate-only`: validate existing campaign outputs and exit.
- `--no-auto-validate`: skip the automatic validation summary after normal execution.

## Validate the produced tables

Use the script validation mode (optional, for manual re-checks):

```bash
python notebooks/ICPR2026/run_icpr2026_seeds.py --configs all --run-ids 0 1 2 3 4 5 6 7 8 9 --validate-only
```

Use `--output-dir` if your campaign wrote to a non-default folder.

## Method and Threshold Protocol

Protocol values match the paper campaign:

- Image preprocessing: grayscale, resized to `588 x 660`.
- Split protocol: 10 random 70/30 train/test splits.
- Epochs: `100`.
- Optimizer: Adam (`betas=(0.9, 0.999)`, `eps=1e-8`).
- Backbone learning rate: `1e-3`.
- CFP learning rate: `5e-2`.
- Batch size: `8`.
- Weight decay: `1e-7`.
- Early stopping: none.
- Loss: class-balanced `BCEWithLogitsLoss`.
- CFP auxiliary loss: summed BCE on CFP output.
- CFP schedule: `gamma` linearly from `1` at epoch `0` to `0` at epoch `50`.
- CFP init: near-identity, pass-through probability `0.995`.
- Threshold protocol: `train_pr` (criterion: F1, source: Train).

CFP attributes:

- Plants / PP (Plant Phenotyping): area, inertia, gray-level height, gray-level variance, perimeter.
- Screws / SWP (Screw Wire Processing): circularity, inertia, area, length of minor axis, gray-level height, length of major axis, rectangularity.

Each notebook reports:

- **Baseline**: model trained directly on inputs.
- **CFP**: model with trainable Connected Filter Preprocessing layer.

Only ED3-NN notebooks include the rerunnable attribute-ablation block used for paper Table 3.

## Paper-to-Artifact Mapping

| Paper item | Notebook source |
| --- | --- |
| Table 1 (PP) | Plant notebooks, `Baseline Evaluation` and `CFP Evaluation` using `results_df_base` and `results_df`. |
| Table 2 (SWP) | Screw notebooks, `Baseline Evaluation` and `CFP Evaluation` using `results_df_base` and `results_df`. |
| Table 3(a) | `ICPR2026_plant_segmentation_ED3-NN_run_000.ipynb`, `Attribute Ablation`. |
| Table 3(b) | `ICPR2026_screw_segmentation_ED3-NN_run_000.ipynb`, `Attribute Ablation`. |
| Figure 3 | ConvNet and ED3-NN notebooks, `Figure 3 Panels`. |

## Hardware and Runtime

Planning estimates from the reference environment (hardware/driver/stack dependent):

| Notebook | Approx. runtime (1 run) | Estimated (10 runs) |
| --- | ---: | ---: |
| `ICPR2026_screw_segmentation_ConvNet_run_000.ipynb` | 3m 39s | 36m 30s |
| `ICPR2026_screw_segmentation_ED3-NN_run_000.ipynb` | 22m 59s | 3h 49m |
| `ICPR2026_screw_segmentation_U-Net-MobileNetV2_run_000.ipynb` | 19m 28s | 3h 15m |
| `ICPR2026_plant_segmentation_ConvNet_run_000.ipynb` | 6m 39s | 1h 7m |
| `ICPR2026_plant_segmentation_ED3-NN_run_000.ipynb` | 29m 59s | 4h 59m |
| `ICPR2026_plant_segmentation_U-Net-MobileNetV2_run_000.ipynb` | 19m 2s | 3h 10m |

Estimated totals:

- Screws (3 notebooks): ~7h 41m
- Plants (3 notebooks): ~9h 16m
- All six notebooks: ~16h 57m

Start with screws first (public dataset) and add plants (private dataset URL) later.
These values are only estimates for planning and depend strongly on GPU/CPU, driver, and I/O speed.

## Troubleshooting

- If startup fails, confirm PyTorch and `_mtlearn` extension match the active environment.
- If datasets are not found, re-run `--list-data`.
- If memory is constrained, run smaller batches and reduce parallelism.

## License and Reproducibility Artifacts

MTLearn is distributed under `GPL-3.0-only` (`LICENSE`).
No pretrained weights/checkpoints are used; all runs train Baseline and CFP models from scratch.
