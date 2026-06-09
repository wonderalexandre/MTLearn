#!/usr/bin/env python3
"""Run the ICPR 2026 notebooks with the paper's 10 deterministic seeds."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter


@dataclass(frozen=True)
class NotebookConfig:
    """One ICPR 2026 dataset/backbone notebook configuration."""

    key: str
    label: str
    notebook: str


# Seeds generated from the master seed 42 with:
# np.random.SeedSequence(42).spawn(10)[i].generate_state(1, dtype=np.uint32)[0]
SEEDS_BY_RUN = {
    0: 2684470948,
    1: 4091952314,
    2: 233227757,
    3: 3276785861,
    4: 3644269654,
    5: 1206282609,
    6: 3543069911,
    7: 3479688010,
    8: 877132087,
    9: 1244265337,
}


CONFIGS = {
    "plant_convnet": NotebookConfig(
        key="plant_convnet",
        label="Plants / ConvNet",
        notebook="ICPR2026_plant_segmentation_ConvNet_run_000.ipynb",
    ),
    "plant_ed3": NotebookConfig(
        key="plant_ed3",
        label="Plants / ED3-NN",
        notebook="ICPR2026_plant_segmentation_ED3-NN_run_000.ipynb",
    ),
    "plant_unet": NotebookConfig(
        key="plant_unet",
        label="Plants / U-Net MobileNetV2",
        notebook="ICPR2026_plant_segmentation_U-Net-MobileNetV2_run_000.ipynb",
    ),
    "screw_convnet": NotebookConfig(
        key="screw_convnet",
        label="Screws / ConvNet",
        notebook="ICPR2026_screw_segmentation_ConvNet_run_000.ipynb",
    ),
    "screw_ed3": NotebookConfig(
        key="screw_ed3",
        label="Screws / ED3-NN",
        notebook="ICPR2026_screw_segmentation_ED3-NN_run_000.ipynb",
    ),
    "screw_unet": NotebookConfig(
        key="screw_unet",
        label="Screws / U-Net MobileNetV2",
        notebook="ICPR2026_screw_segmentation_U-Net-MobileNetV2_run_000.ipynb",
    ),
}


CONFIG_GROUPS = {
    "all": tuple(CONFIGS),
    "plants": ("plant_convnet", "plant_ed3", "plant_unet"),
    "screws": ("screw_convnet", "screw_ed3", "screw_unet"),
}


PAPER_METRICS = (
    ("AUC-ROC", "AUC-ROC"),
    ("Accuracy", "Accuracy"),
    ("Cohen's Kappa", "Cohen's Kappa"),
    ("F1-Score", "F1-Score"),
    ("Jaccard Index", "Jaccard Index"),
    ("MCC", "MCC"),
    ("Precision", "Precision"),
    ("Recall (Sensitivity)", "Sensitivity"),
    ("Specificity", "Specificity"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def notebook_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the ICPR 2026 RRPR notebooks with the 10 deterministic "
            "paper seeds generated from master seed 42."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["all"],
        choices=[*CONFIG_GROUPS.keys(), *CONFIGS.keys()],
        help=(
            "Configurations to execute. Use 'all', 'plants', or 'screws' "
            "for grouped runs."
        ),
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        type=int,
        default=sorted(SEEDS_BY_RUN),
        help="Run identifiers to execute. Defaults to 0 1 ... 9.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root output directory. Defaults to "
            "<repo>/notebooks/ICPR2026_runs."
        ),
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Optional Jupyter kernel name passed to Papermill.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configurations and deterministic paper seeds, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned executions without running Papermill.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an execution if the output notebook already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining executions if one notebook fails.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show Papermill's per-cell progress bar.",
    )
    parser.add_argument(
        "--scale-in",
        dest="scale_in",
        choices=("true", "false"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--screw-scale-in",
        dest="scale_in",
        action="store_const",
        const="true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def selected_configs(raw_configs: list[str]) -> list[NotebookConfig]:
    selected_keys: list[str] = []
    for name in raw_configs:
        if name in CONFIG_GROUPS:
            selected_keys.extend(CONFIG_GROUPS[name])
        else:
            selected_keys.append(name)

    unique_keys = list(dict.fromkeys(selected_keys))
    return [CONFIGS[name] for name in unique_keys]


def validate_run_ids(run_ids: list[int]) -> list[int]:
    invalid = [run_id for run_id in run_ids if run_id not in SEEDS_BY_RUN]
    if invalid:
        valid = " ".join(str(run_id) for run_id in sorted(SEEDS_BY_RUN))
        raise SystemExit(f"Unknown run id(s): {invalid}. Valid run ids: {valid}.")
    return run_ids


def output_notebook_path(
    output_root: Path,
    config: NotebookConfig,
    input_notebook: Path,
    run_id: int,
) -> Path:
    stem = re.sub(r"_run_\d{3}$", f"_run_{run_id:03d}", input_notebook.stem)
    return output_root / config.key / "executions" / f"{stem}.ipynb"


def output_run_dir(output_root: Path, config: NotebookConfig, run_id: int) -> Path:
    return output_root / config.key / "runs" / f"run_{run_id:03d}"


def format_duration(seconds: float) -> str:
    seconds_int = int(round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def append_summary_row(summary_path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "started_at",
        "finished_at",
        "config",
        "run_id",
        "seed",
        "status",
        "elapsed_seconds",
        "elapsed",
        "input_notebook",
        "output_notebook",
        "results_dir",
        "error",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def list_plan(configs: list[NotebookConfig], run_ids: list[int]) -> None:
    print("Recovered ICPR 2026 seeds:")
    for run_id in sorted(SEEDS_BY_RUN):
        print(f"  RUN_ID {run_id}: SEED {SEEDS_BY_RUN[run_id]}")

    print("\nSelected configurations:")
    for config in configs:
        print(f"  {config.key}: {config.label} ({config.notebook})")

    print("\nConfiguration groups:")
    for group_name, config_keys in CONFIG_GROUPS.items():
        print(f"  {group_name}: {' '.join(config_keys)}")

    print("\nSelected run ids:")
    print("  " + " ".join(str(run_id) for run_id in run_ids))


def _parse_run_id(run_dir: Path) -> int | None:
    match = re.fullmatch(r"run_(\d{3})", run_dir.name)
    if match is None:
        return None
    return int(match.group(1))


def _test_row(csv_path: Path) -> dict[str, str] | None:
    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return None
    test_rows = [row for row in rows if row.get("Evaluation Split", "Test") == "Test"]
    return test_rows[0] if test_rows else rows[-1]


def _float_value(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _fallback_dataset(config: NotebookConfig) -> str:
    return "Plants" if config.key.startswith("plant_") else "Screws"


def _fallback_model(config: NotebookConfig) -> str:
    if config.key.endswith("_convnet"):
        return "ConvNet"
    if config.key.endswith("_ed3"):
        return "ED3-NN"
    return "U-Net MobileNetV2"


def write_consolidated_paper_metrics(
    output_root: Path,
    configs: list[NotebookConfig],
) -> Path | None:
    records: list[dict[str, object]] = []
    for config in configs:
        runs_root = output_root / config.key / "runs"
        if not runs_root.exists():
            continue
        for run_dir in sorted(runs_root.glob("run_*")):
            run_id = _parse_run_id(run_dir)
            if run_id is None:
                continue
            for prefix, fallback_variant in (("base", "Baseline"), ("cfp", "CFP")):
                csv_path = run_dir / f"{prefix}.csv"
                if not csv_path.exists():
                    continue
                row = _test_row(csv_path)
                if row is None:
                    continue
                dataset = row.get("DATASET") or _fallback_dataset(config)
                model = row.get("MODEL") or _fallback_model(config)
                variant = row.get("VARIANT") or fallback_variant
                threshold_protocol = row.get("Threshold Protocol", "")
                threshold_source = row.get("Threshold Source", "")
                threshold_criterion = row.get("Threshold Criterion", "")
                for source_name, paper_name in PAPER_METRICS:
                    value = _float_value(row, source_name)
                    if value is None:
                        continue
                    records.append(
                        {
                            "dataset": dataset,
                            "config": config.key,
                            "model": model,
                            "variant": variant,
                            "metric": paper_name,
                            "run_id": run_id,
                            "seed": row.get("SEED", SEEDS_BY_RUN.get(run_id, "")),
                            "threshold_protocol": threshold_protocol,
                            "threshold_source": threshold_source,
                            "threshold_criterion": threshold_criterion,
                            "value": value,
                        }
                    )

    if not records:
        return None

    groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for record in records:
        key = (
            str(record["dataset"]),
            str(record["config"]),
            str(record["model"]),
            str(record["variant"]),
            str(record["metric"]),
            str(record["threshold_protocol"]),
            str(record["threshold_source"]),
            str(record["threshold_criterion"]),
        )
        groups.setdefault(key, []).append(record)

    metric_order = {name: index for index, (_, name) in enumerate(PAPER_METRICS)}
    variant_order = {
        "Baseline": 0,
        "base": 0,
        "without CFP": 0,
        "CFP": 1,
        "cfp": 1,
        "with CFP": 1,
    }

    output_path = output_root / "paper_table_metrics.csv"
    fieldnames = [
        "dataset",
        "config",
        "model",
        "variant",
        "metric",
        "n",
        "mean",
        "std",
        "mean_std",
        "run_ids",
        "seeds",
        "threshold_protocol",
        "threshold_source",
        "threshold_criterion",
    ]
    rows_to_write: list[dict[str, object]] = []
    for key, group_records in groups.items():
        (
            dataset,
            config_key,
            model,
            variant,
            metric,
            threshold_protocol,
            threshold_source,
            threshold_criterion,
        ) = key
        values = [float(record["value"]) for record in group_records]
        run_ids = sorted(int(record["run_id"]) for record in group_records)
        seeds = [str(record["seed"]) for record in sorted(group_records, key=lambda item: int(item["run_id"]))]
        metric_mean = mean(values)
        metric_std = stdev(values) if len(values) > 1 else 0.0
        rows_to_write.append(
            {
                "dataset": dataset,
                "config": config_key,
                "model": model,
                "variant": variant,
                "metric": metric,
                "n": len(values),
                "mean": f"{metric_mean:.9f}",
                "std": f"{metric_std:.9f}",
                "mean_std": f"{metric_mean:.3f} (±{metric_std:.3f})",
                "run_ids": " ".join(f"{run_id:03d}" for run_id in run_ids),
                "seeds": " ".join(seeds),
                "threshold_protocol": threshold_protocol,
                "threshold_source": threshold_source,
                "threshold_criterion": threshold_criterion,
            }
        )

    rows_to_write.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["config"]),
            variant_order.get(str(row["variant"]), 99),
            metric_order.get(str(row["metric"]), 99),
        )
    )
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)
    return output_path


def execute(args: argparse.Namespace) -> int:
    configs = selected_configs(args.configs)
    run_ids = validate_run_ids(args.run_ids)
    output_root = (args.output_dir or repo_root() / "notebooks" / "ICPR2026_runs")
    output_root = output_root.expanduser().resolve()

    if args.list:
        list_plan(configs, run_ids)
        return 0

    executions: list[tuple[NotebookConfig, int, int, Path, Path, Path]] = []
    for config in configs:
        input_notebook = notebook_dir() / config.notebook
        if not input_notebook.exists():
            raise SystemExit(f"Missing notebook: {input_notebook}")
        for run_id in run_ids:
            seed = SEEDS_BY_RUN[run_id]
            output_notebook = output_notebook_path(
                output_root, config, input_notebook, run_id
            )
            run_dir = output_run_dir(output_root, config, run_id)
            executions.append(
                (config, run_id, seed, input_notebook, output_notebook, run_dir)
            )

    print(f"Planned executions: {len(executions)}")
    for config, run_id, seed, input_notebook, output_notebook, run_dir in executions:
        print(
            f"- {config.key} run_{run_id:03d} seed={seed}\n"
            f"  input:  {input_notebook}\n"
            f"  output: {output_notebook}\n"
            f"  results: {run_dir}"
        )

    if args.dry_run:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "execution_summary.csv"

    try:
        import papermill as pm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Papermill is required. Install it with "
            "'python -m pip install \"papermill>=2.4\"'."
        ) from exc

    failures = 0
    for config, run_id, seed, input_notebook, output_notebook, run_dir in executions:
        if output_notebook.exists() and args.skip_existing:
            print(f"Skipping existing output: {output_notebook}")
            now = datetime.now().isoformat(timespec="seconds")
            append_summary_row(
                summary_path,
                {
                    "started_at": now,
                    "finished_at": now,
                    "config": config.key,
                    "run_id": run_id,
                    "seed": seed,
                    "status": "skipped",
                    "elapsed_seconds": 0,
                    "elapsed": "0s",
                    "input_notebook": input_notebook,
                    "output_notebook": output_notebook,
                    "results_dir": run_dir,
                    "error": "",
                },
            )
            continue

        output_notebook.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExecuting {config.key} run_{run_id:03d} (seed={seed})")
        started_at = datetime.now().isoformat(timespec="seconds")
        start = perf_counter()
        parameters = {
            "RUN_ID": run_id,
            "SEED": seed,
            "OUTPUT_DIR": str(run_dir),
        }
        if args.scale_in is not None:
            parameters["DATASET_SCALE_IN"] = args.scale_in == "true"
        try:
            pm.execute_notebook(
                input_path=str(input_notebook),
                output_path=str(output_notebook),
                parameters=parameters,
                kernel_name=args.kernel,
                progress_bar=args.progress_bar,
                request_save_on_cell_execute=False,
            )
        except Exception as exc:
            failures += 1
            elapsed_seconds = perf_counter() - start
            append_summary_row(
                summary_path,
                {
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "config": config.key,
                    "run_id": run_id,
                    "seed": seed,
                    "status": "failed",
                    "elapsed_seconds": f"{elapsed_seconds:.3f}",
                    "elapsed": format_duration(elapsed_seconds),
                    "input_notebook": input_notebook,
                    "output_notebook": output_notebook,
                    "results_dir": run_dir,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(
                f"Failed {config.key} run_{run_id:03d} after "
                f"{format_duration(elapsed_seconds)}"
            )
            if not args.continue_on_error:
                raise
            print(
                f"Execution failed for {config.key} run_{run_id:03d}; "
                "continuing because --continue-on-error was set.",
                file=sys.stderr,
            )
            continue

        elapsed_seconds = perf_counter() - start
        append_summary_row(
            summary_path,
            {
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "config": config.key,
                "run_id": run_id,
                "seed": seed,
                "status": "success",
                "elapsed_seconds": f"{elapsed_seconds:.3f}",
                "elapsed": format_duration(elapsed_seconds),
                "input_notebook": input_notebook,
                "output_notebook": output_notebook,
                "results_dir": run_dir,
                "error": "",
            },
        )
        print(
            f"Finished {config.key} run_{run_id:03d} in "
            f"{format_duration(elapsed_seconds)}"
        )

    consolidated_path = write_consolidated_paper_metrics(output_root, configs)
    if consolidated_path is not None:
        print(f"\nConsolidated paper metrics: {consolidated_path}")
    else:
        print("\nNo metric CSV files found for consolidation.")

    if failures:
        print(f"\nCompleted with {failures} failure(s).", file=sys.stderr)
        return 1

    print(f"\nCompleted. Executed notebooks are under: {output_root}")
    print(f"Execution summary: {summary_path}")
    return 0


def main() -> int:
    return execute(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
