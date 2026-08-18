#!/usr/bin/env python3
"""Run repeatable notebook validation without modifying source notebooks."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nbformat
import papermill as pm


GRADCHECK_NOTEBOOKS = (
    "notebooks/gradchecks/GradCheck_Implicit_Jacobian.ipynb",
)


@dataclass(frozen=True)
class NotebookRun:
    source: Path
    output: Path
    cwd: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate notebooks against the local mtlearn source tree. "
            "Long training notebooks are copied to a temporary smoke version."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--bindings-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the built _mtlearn extension. "
            "If omitted, the current PYTHONPATH or installed mtlearn is used."
        ),
    )
    parser.add_argument(
        "--installed-package",
        action="store_true",
        help=(
            "Validate notebooks against the installed mtlearn package instead "
            "of using the local mtlearn/python/ source tree or PYTHONPATH."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/mtlearn-notebook-validation"),
        help="Where executed notebooks and temporary smoke inputs are written.",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name used by papermill.",
    )
    parser.add_argument(
        "--skip-gradchecks",
        action="store_true",
        help="Do not run the complete gradcheck notebooks.",
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Do not run the reduced experiments notebook.",
    )
    parser.add_argument(
        "--keep-existing-output",
        action="store_true",
        help="Do not delete existing .ipynb files under the output directory before running.",
    )
    return parser.parse_args()


def configure_pythonpath(
    repo_root: Path,
    bindings_dir: Path | None,
    installed_package: bool,
) -> None:
    parts = []
    if not installed_package:
        parts.append(str(repo_root / "mtlearn" / "python"))
    if bindings_dir is not None:
        parts.append(str(bindings_dir))
    current = os.environ.get("PYTHONPATH")
    if current and not installed_package:
        parts.append(current)
    if parts:
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    else:
        os.environ.pop("PYTHONPATH", None)


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("**/*.ipynb"):
        path.unlink()


def read_notebook(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


def write_smoke_notebook(
    cells: list[nbformat.NotebookNode],
    destination: Path,
) -> Path:
    # The smoke cells replace the notebook body entirely, so build a fresh
    # notebook instead of borrowing one from the repository.
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata.setdefault(
        "kernelspec",
        {"name": "python3", "display_name": "Python 3", "language": "python"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, destination)
    return destination


def source_cell(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text)


def facade_assertion_cell(label: str) -> nbformat.NotebookNode:
    return source_cell(
        "import torch\n"
        "import mtlearn\n"
        "import numpy as np\n"
        "from mtlearn import morphology\n"
        "assert morphology.AttributeType is morphology.Attribute.Type\n"
        "assert morphology.AttributeGroup is morphology.Attribute.Group\n"
        "assert morphology.NodeIdSpace.MORPHOLOGICAL_TREE.name == 'MORPHOLOGICAL_TREE'\n"
        "assert hasattr(mtlearn.layers, 'ConnectedFilterPreprocessingLayer')\n"
        "assert not hasattr(mtlearn.layers, 'CFPLayer')\n"
        "assert not hasattr(mtlearn.layers, 'ConnectedFilterLayerWithImplicitJacobian')\n"
        "assert not hasattr(mtlearn.layers, 'ConnectedFilterSingleThresholdLayer')\n"
        "image = np.array([[1, 2], [3, 4]], dtype=np.uint8)\n"
        "tree = morphology.create_max_tree(image)\n"
        "attrs = morphology.compute_attributes(\n"
        "    tree,\n"
        "    [morphology.AttributeType.AREA, morphology.AttributeGroup.TREE_TOPOLOGY],\n"
        ")[1]\n"
        "single = morphology.Attribute.computeSingleAttribute(\n"
        "    tree,\n"
        "    morphology.AttributeType.AREA,\n"
        "    morphology.NodeIdSpace.MORPHOLOGICAL_TREE,\n"
        ")\n"
        "assert attrs.shape[0] == single.shape[0]\n"
        "layer = mtlearn.layers.ConnectedFilterPreprocessingLayer(\n"
        "    in_channels=1,\n"
        "    filter_specs=[\n"
        "        {\n"
        "            'tree_type': morphology.TreeType.MAX_TREE,\n"
        "            'attributes': (morphology.AttributeType.AREA,),\n"
        "        }\n"
        "    ],\n"
        "    device='cpu',\n"
        "    scale_mode='none',\n"
        ")\n"
        "x = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)\n"
        "y = layer(x)\n"
        "assert y.shape == (1, 1, 2, 2)\n"
        f"print('{label} ok', attrs.shape, single.shape, y.shape)"
    )


def make_screws_example_smoke(
    repo_root: Path,
    smoke_dir: Path,
    output_dir: Path,
) -> NotebookRun:
    cells = [
        source_cell(
            "import numpy as np\n"
            "import torch\n"
            "import mtlearn\n"
            "from mtlearn import morphology\n"
            "\n"
            "torch.manual_seed(42)\n"
            "dataset_dir = mtlearn.data.ensure_dataset('screws_segmentation')\n"
            "dataset = mtlearn.datasets.PairedImageDataset(\n"
            "    root_dir=str(dataset_dir),\n"
            "    grayscale_in=True,\n"
            "    invert_in=True,\n"
            "    invert_target=True,\n"
            "    num_rows=int(1324 * 0.499),\n"
            "    num_cols=int(1177 * 0.5),\n"
            "    scale_in=False,\n"
            "    suffix_in='_in',\n"
            "    suffix_target='_target',\n"
            ")\n"
            "trainset, testset = dataset.train_test_split(test_size=0.3, random_state=42)\n"
            "assert len(trainset) > 0 and len(testset) > 0\n"
            "x, y, name = trainset[0]\n"
            "assert x.ndim == 3 and y.ndim == 3\n"
            "print('screws dataset smoke ok', x.shape, y.shape, name)"
        ),
        facade_assertion_cell("screws example smoke"),
    ]
    smoke_input = smoke_dir / "screws_example_smoke.ipynb"
    write_smoke_notebook(cells, smoke_input)
    return NotebookRun(
        source=smoke_input,
        output=output_dir / "experiments" / smoke_input.name,
        cwd=repo_root,
    )


def gradcheck_runs(repo_root: Path, output_dir: Path) -> list[NotebookRun]:
    return [
        NotebookRun(
            source=repo_root / notebook,
            output=output_dir / "gradchecks" / Path(notebook).name,
            cwd=(repo_root / notebook).parent,
        )
        for notebook in GRADCHECK_NOTEBOOKS
    ]


def build_runs(args: argparse.Namespace) -> list[NotebookRun]:
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    smoke_dir = output_dir / "smoke_inputs"
    runs: list[NotebookRun] = []

    if not args.skip_gradchecks:
        runs.extend(gradcheck_runs(repo_root, output_dir))

    if not args.skip_experiments:
        runs.append(
            make_screws_example_smoke(
                repo_root,
                smoke_dir,
                output_dir,
            )
        )

    return runs


def execute_run(run: NotebookRun, kernel: str) -> None:
    run.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {run.source}")
    pm.execute_notebook(
        str(run.source),
        str(run.output),
        cwd=str(run.cwd),
        kernel_name=kernel,
        log_output=True,
        progress_bar=False,
    )
    print(f"OK  {run.output}")


def assert_no_errors(paths: Iterable[Path]) -> None:
    failed: list[str] = []
    for path in paths:
        nb = read_notebook(path)
        errors = []
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            errors.extend(
                output
                for output in cell.get("outputs", [])
                if output.get("output_type") == "error"
            )
        if errors:
            summary = "; ".join(
                f"{error.get('ename')}: {error.get('evalue')}" for error in errors[:3]
            )
            failed.append(f"{path}: {summary}")
    if failed:
        raise RuntimeError("Notebook outputs contain errors:\n" + "\n".join(failed))


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()

    if args.installed_package and args.bindings_dir:
        raise SystemExit("--installed-package cannot be combined with --bindings-dir")

    configure_pythonpath(
        repo_root,
        args.bindings_dir.resolve() if args.bindings_dir else None,
        args.installed_package,
    )
    if not args.keep_existing_output:
        clean_output_dir(output_dir)

    runs = build_runs(args)
    if not runs:
        raise SystemExit("No notebook validations selected.")

    for run in runs:
        execute_run(run, args.kernel)

    assert_no_errors(run.output for run in runs)
    print(f"Validated {len(runs)} notebook run(s). Outputs: {output_dir}")


if __name__ == "__main__":
    main()
