"""Shared helpers for the ICPR 2026 reproducibility notebooks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import random
import sys
import time


ICPR2026_DATASETS = ("screws_segmentation", "plants_segmentation")


def ensure_icpr2026_dataset(
    key: str,
    data_dir: str | Path | None = None,
    *,
    force: bool = False,
    keep_archive: bool = False,
    url: str | None = None,
) -> Path:
    """Ensure one dataset used by the ICPR 2026 notebooks is available."""
    if key not in ICPR2026_DATASETS:
        expected = ", ".join(ICPR2026_DATASETS)
        raise ValueError(f"Unknown ICPR 2026 dataset {key!r}. Expected one of: {expected}.")

    from mtlearn import data as mtlearn_data

    return mtlearn_data.ensure_dataset(
        key,
        Path(data_dir).expanduser() if data_dir is not None else None,
        force=force,
        keep_archive=keep_archive,
        url=url,
    )


def ensure_screws_data(
    data_dir: str | Path | None = None,
    *,
    force: bool = False,
    keep_archive: bool = False,
) -> Path:
    """Download or locate the public screw-segmentation dataset."""
    return ensure_icpr2026_dataset(
        "screws_segmentation",
        data_dir,
        force=force,
        keep_archive=keep_archive,
    )


def ensure_plants_data(
    data_dir: str | Path | None = None,
    *,
    force: bool = False,
    keep_archive: bool = False,
    url: str | None = None,
) -> Path:
    """Download or locate the authorized plant-segmentation review package."""
    return ensure_icpr2026_dataset(
        "plants_segmentation",
        data_dir,
        force=force,
        keep_archive=keep_archive,
        url=url,
    )


def ensure_icpr2026_data(
    datasets: list[str] | tuple[str, ...] | str = ICPR2026_DATASETS,
    data_dir: str | Path | None = None,
    *,
    force: bool = False,
    keep_archive: bool = False,
    url: str | None = None,
) -> dict[str, Path]:
    """Ensure one or more datasets used by the ICPR 2026 notebooks."""
    selected = [datasets] if isinstance(datasets, str) else list(datasets)
    if url and selected != ["plants_segmentation"]:
        raise ValueError("url can only be used when downloading plants_segmentation.")
    return {
        key: ensure_icpr2026_dataset(
            key,
            data_dir,
            force=force,
            keep_archive=keep_archive,
            url=url if key == "plants_segmentation" else None,
        )
        for key in selected
    }


def fix_randomness(seed: int = 42, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducible notebook runs."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def default_device() -> str:
    """Return the notebook device using the priority cuda, mps, then cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def split_params(model):
    """Split model parameters into CFP and non-CFP groups."""
    filter_module = model.morphological_layer
    filter_ids = set(map(id, filter_module.parameters()))
    filter_params = [p for p in model.parameters() if id(p) in filter_ids]
    backbone_params = [p for p in model.parameters() if id(p) not in filter_ids]
    return filter_params, backbone_params


class LambdaAuxScheduler:
    """Schedule the auxiliary CFP regularization weight during training."""

    def __init__(self, total_epochs, max_val=0.3, warmup=20, hold=40, decay_to=0.0):
        self.E = total_epochs
        self.max_val = max_val
        self.warmup = warmup
        self.hold = hold
        self.decay_to = decay_to
        self.last_epoch = -1
        self.value = 0.0

    def step(self, epoch=None) -> None:
        """Update the current lambda value for the given training epoch."""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        e = self.last_epoch
        if e < self.warmup:
            self.value = self.max_val * (e / max(1, self.warmup))
        elif e < self.hold:
            self.value = self.max_val
        elif e >= self.E:
            self.value = 0
        else:
            t = min((e - self.hold) / max(1, self.E - self.hold), 1.0)
            self.value = self.max_val * (1 - t) + self.decay_to * t


def _first_input_tensor(inputs):
    if isinstance(inputs, (list, tuple)):
        return inputs[0]
    return inputs


def _as_batched_image(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Unexpected image shape: {tuple(tensor.shape)}")


def train_segmentation_model(
    model,
    *,
    trainloader_cached,
    trainloader_len: int,
    pos_weight: torch.Tensor,
    device: str,
    lr: float = 0.001,
    lr_cfp: float = 0.05,
    lambda_filter: float = 1,
    num_epochs_filter: int = 0,
    num_epochs: int = 100,
    preview: bool = True,
) -> list[float]:
    """Train one notebook model and display the loss curve."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

    loss_aux = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="sum")
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    lambda_sched = LambdaAuxScheduler(
        num_epochs_filter,
        max_val=lambda_filter,
        warmup=0,
        hold=0,
        decay_to=0.0,
    )

    if num_epochs_filter != 0:
        params_cfp, params_backbone = split_params(model)
        optimizer = torch.optim.Adam(
            [
                {"params": params_backbone, "lr": lr, "weight_decay": 1e-7},
                {"params": params_cfp, "lr": lr_cfp, "weight_decay": 1e-7},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    errors = []
    model.train()
    time_begin = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        lambda_sched.step(epoch)
        lam = lambda_sched.value
        for inputs, targets in trainloader_cached:
            predicts = model(inputs)

            if epoch < num_epochs_filter:
                loss_main = loss_function(predicts, targets)
                loss_filter = loss_aux(model.h_filter, targets)
                loss = loss_main + lam * loss_filter
            else:
                loss = loss_function(predicts, targets)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        errors.append(epoch_loss / trainloader_len)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Loss: {errors[-1]:.7f}, Learning rate: {lr:.6f}")
            if epoch < num_epochs_filter:
                print(
                    f"\tLoss (main): {loss_main.item():.7f}, "
                    f"Loss (filter): {loss_filter.item():.7f}, Lambda: {lam:.4f}"
                )

        if preview and num_epochs_filter != 0 and (epoch == num_epochs - 1 or epoch % 20 == 0):
            imgs_in = _first_input_tensor(inputs).to("cpu")
            imgs_out = targets.to("cpu")
            h_filter = model.h_filter.cpu()
            imgs_pred = (torch.sigmoid(predicts).cpu().detach().numpy() >= 0.5).astype(int)

            i = np.random.randint(0, len(imgs_out))
            fig, ax = plt.subplots(1, 4, figsize=(12, 3))
            ax[0].imshow(imgs_in[i, 0].cpu().numpy(), cmap="gray")
            ax[0].set_title("Input")
            ax[1].imshow(imgs_out[i, 0].numpy(), cmap="gray")
            ax[1].set_title("Target")
            ax[2].imshow(h_filter[i, 0].detach().numpy(), cmap="gray")
            ax[2].set_title("CFilter")
            ax[3].imshow(imgs_pred[i, 0], cmap="gray")
            ax[3].set_title("Pred")
            for axis in ax:
                axis.axis("off")
            plt.tight_layout()
            plt.show()

    time_end = time.time()
    print("Finish training")
    print(f"Execution time: {(time_end - time_begin) / 60:.3f} minutes")

    plt.figure(figsize=(12, 5))
    plt.plot(errors, "-")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (mean)")
    plt.title("Loss Evolution")
    plt.show()
    return errors


def train_with_notebook_context(
    model,
    *,
    trainloader,
    trainloader_cached,
    pos_weight,
    device,
    **kwargs,
):
    """Train using the standard loader variables defined by each notebook."""
    return train_segmentation_model(
        model,
        trainloader_cached=trainloader_cached,
        trainloader_len=len(trainloader),
        pos_weight=pos_weight,
        device=device,
        **kwargs,
    )


def analyze_threshold(model, dataloader, save_path=None):
    """Compute ROC and precision-recall curves and return selected thresholds."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, desired_targets, names in dataloader:
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(desired_targets.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    best_idx_roc = np.argmax(tpr - fpr)
    best_threshold_roc = thresholds_roc[best_idx_roc]

    precision, recall, thresholds_pr = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx_pr = np.argmax(f1_scores[1:])
    best_threshold_pr = thresholds_pr[best_idx_pr - 1]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.scatter(
        fpr[best_idx_roc],
        tpr[best_idx_roc],
        c="red",
        label=f"Best Threshold = {best_threshold_roc:.2f}",
    )
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.scatter(
        recall[best_idx_pr],
        precision[best_idx_pr],
        c="red",
        label=f"Best Threshold = {best_threshold_pr:.2f}",
    )
    plt.title("Precision vs Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return best_threshold_roc, best_threshold_pr


def evaluate_model_with_metrics(model, dataset_or_loader, threshold=0.5, average="global"):
    """Evaluate binary segmentation predictions at a fixed threshold."""
    if average == "global":
        return _evaluate_global_metrics(model, dataset_or_loader, threshold)
    if average == "per_image":
        return _evaluate_per_image_metrics(model, dataset_or_loader, threshold)
    raise ValueError("average must be 'global' or 'per_image'")


def _evaluate_global_metrics(model, dataset_or_loader, threshold=0.5):
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        jaccard_score,
        matthews_corrcoef,
        precision_score,
        roc_auc_score,
    )

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y, name in dataset_or_loader:
            if x.ndim == 3:
                x = x.unsqueeze(0)
            logits = model(x)
            probs = torch.sigmoid(logits)
            probs = probs.detach().cpu().numpy().reshape(-1)
            y = y.detach().cpu().numpy().reshape(-1)

            all_probs.extend(probs.tolist())
            all_labels.extend(y.tolist())

    all_probs = np.asarray(all_probs, dtype=float)
    all_labels = np.asarray(all_labels, dtype=int)
    y_pred = (all_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "Accuracy": accuracy_score(all_labels, y_pred),
        "Precision": precision_score(all_labels, y_pred, zero_division=0),
        "Recall (Sensitivity)": sensitivity,
        "Specificity": specificity,
        "F1-Score": f1_score(all_labels, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(all_labels, all_probs),
        "Jaccard Index": jaccard_score(all_labels, y_pred, zero_division=0),
        "Cohen's Kappa": cohen_kappa_score(all_labels, y_pred),
        "MCC": matthews_corrcoef(all_labels, y_pred),
        "Threshold": float(threshold),
    }


def _evaluate_per_image_metrics(model, dataset, threshold=0.5):
    import torch
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        jaccard_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.eval()
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    roc_auc = 0
    jaccard = 0
    kappa = 0
    mcc = 0
    sensitivity = 0
    specificity = 0
    count = 0
    with torch.no_grad():
        for x, y, name in dataset:
            x = x.unsqueeze(1)
            logits = model(x)[0]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            y_pred = (probs > threshold).astype(int)
            y = y.cpu().numpy().flatten()

            accuracy += accuracy_score(y, y_pred)
            precision += precision_score(y, y_pred)
            recall += recall_score(y, y_pred)
            f1 += f1_score(y, y_pred)
            roc_auc += roc_auc_score(y, probs)
            jaccard += jaccard_score(y, y_pred)
            kappa += cohen_kappa_score(y, y_pred)
            mcc += matthews_corrcoef(y, y_pred)
            count += 1

            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            sensitivity += tp / (tp + fn)
            specificity += tn / (tn + fp)

    return {
        "Accuracy": accuracy / count,
        "Precision": precision / count,
        "Recall (Sensitivity)": sensitivity / count,
        "Specificity": specificity / count,
        "F1-Score": f1 / count,
        "AUC-ROC": roc_auc / count,
        "Jaccard Index": jaccard / count,
        "Cohen's Kappa": kappa / count,
        "MCC": mcc / count,
        "Threshold": float(threshold),
    }


def evaluate_train_test(
    model,
    trainset,
    testset,
    *,
    metric_average="global",
    batch_size=8,
):
    """Select the train-set PR threshold and return train/test metric rows."""
    import pandas as pd
    import torch

    threshold_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
    )
    best_threshold_roc, best_threshold_pr = analyze_threshold(model, threshold_loader)

    print(f"Train ROC threshold: {best_threshold_roc:.2f}")
    print(f"Train precision-recall threshold: {best_threshold_pr:.2f}")

    threshold = float(best_threshold_pr)
    train_metrics = evaluate_model_with_metrics(
        model,
        trainset,
        threshold=threshold,
        average=metric_average,
    )
    test_metrics = evaluate_model_with_metrics(
        model,
        testset,
        threshold=threshold,
        average=metric_average,
    )
    results_df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"])
    return threshold, results_df


def save_evaluation_tables(
    output_dir,
    prefix,
    results_df,
    *,
    metadata=None,
) -> None:
    """Persist the train_pr evaluation table as a CSV file."""
    if output_dir is None:
        return

    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    for stale_suffix in ("_train_pr.csv", "_thresholds.csv", "_all_thresholds.csv"):
        stale_path = output_path / f"{prefix}{stale_suffix}"
        if stale_path.exists():
            stale_path.unlink()

    out = results_df.copy()
    out.insert(0, "Evaluation Split", out.index.astype(str))
    out.insert(1, "Threshold Protocol", "train_pr")
    out.insert(2, "Threshold Source", "Train")
    out.insert(3, "Threshold Criterion", "PR")
    out = out.reset_index(drop=True)
    for key, value in metadata.items():
        out[key] = value
    out.to_csv(output_path / f"{prefix}.csv", index=False)


def show_figure3_panels(
    model_base,
    model,
    testset,
    *,
    device,
    threshold_base,
    threshold,
    sample_index=0,
) -> None:
    """Display input, target, baseline prediction, CFP output, and CFP prediction."""
    import matplotlib.pyplot as plt
    import torch

    with torch.no_grad():
        model_base.eval()
        model.eval()

        image, target, file_name = testset[sample_index]
        x = _as_batched_image(image).to(device)
        baseline_logits = model_base(x)
        cfp_logits = model(x)

        baseline_probs = torch.sigmoid(baseline_logits).detach().cpu()
        cfp_probs = torch.sigmoid(cfp_logits).detach().cpu()
        cfp_output = model.h_filter.detach().cpu()

        input_img = x.detach().cpu()[0, 0].numpy()
        target_img = target.detach().cpu().squeeze().numpy()
        baseline_pred = (baseline_probs[0, 0].numpy() >= threshold_base).astype(int)
        cfp_img = cfp_output[0, 0].numpy()
        cfp_pred = (cfp_probs[0, 0].numpy() >= threshold).astype(int)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    panels = [
        (input_img, "Input"),
        (target_img, "Target"),
        (baseline_pred, "Baseline pred"),
        (cfp_img, "CFP output"),
        (cfp_pred, "CFP pred"),
    ]
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle(str(file_name))
    plt.tight_layout()
    plt.show()


def analyze_cfp_spec_ablation(spec_info):
    """Rank CFP attributes by their effect on node-wise keep probabilities."""
    import torch

    with torch.no_grad():
        attributes = [getattr(attr, "name", str(attr)) for attr in spec_info["attributes"]]
        attrs = spec_info["norm_attrs"].detach()
        weight = spec_info["weight"].detach().view(-1)
        bias = spec_info["bias"].detach().view(())

        logits = attrs.matmul(weight) + bias
        keep_prob = torch.sigmoid(logits)
        contributions = attrs * weight.view(1, -1)

        rows = []
        for attr_index, attr_name in enumerate(attributes):
            logits_without_attr = logits - contributions[:, attr_index]
            keep_prob_without_attr = torch.sigmoid(logits_without_attr)
            delta = (keep_prob - keep_prob_without_attr).abs()
            rows.append(
                {
                    "attribute": attr_name,
                    "weight": float(weight[attr_index].cpu()),
                    "mean_abs_contribution": float(
                        contributions[:, attr_index].abs().mean().cpu()
                    ),
                    "mean_abs_delta_p": float(delta.mean().cpu()),
                }
            )
    return rows


def rank_cfp_attributes(model, dataset, *, device, sample_count=None, channel=0):
    """Aggregate CFP attribute-ablation scores over test samples."""
    import pandas as pd
    import torch

    with torch.no_grad():
        cfp_layer = model.morphological_layer
        cfp_layer.eval()
        aggregated = defaultdict(list)

        sample_total = len(dataset) if sample_count is None else min(sample_count, len(dataset))
        for sample_index in range(sample_total):
            image, _, file_name = dataset[sample_index]
            info = cfp_layer.inspect_training_sample(
                image.to(device),
                channel=channel,
                idx=sample_index,
                build_if_missing=True,
            )
            for spec_name, spec_info in info["specs"].items():
                for row in analyze_cfp_spec_ablation(spec_info):
                    key = (spec_name, row["attribute"])
                    aggregated[key].append(row["mean_abs_delta_p"])

    rows = []
    for (spec_name, attribute), values in aggregated.items():
        series = pd.Series(values, dtype=float)
        rows.append(
            {
                "spec": spec_name,
                "attribute": attribute,
                "mean_abs_delta_p": series.mean(),
                "std_abs_delta_p": series.std(),
                "n": int(series.count()),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["spec", "mean_abs_delta_p"],
        ascending=[True, False],
    ).reset_index(drop=True)
    totals = result.groupby("spec")["mean_abs_delta_p"].transform("sum")
    result["contribution_percent"] = 100.0 * result["mean_abs_delta_p"] / totals
    return result


def _list_icpr2026_datasets(data_dir: Path | None = None) -> None:
    from mtlearn import data as mtlearn_data

    root = (
        data_dir.expanduser().resolve()
        if data_dir is not None
        else mtlearn_data.default_data_dir().expanduser().resolve()
    )
    print(f"Data directory: {root}")
    for key in ICPR2026_DATASETS:
        spec = mtlearn_data.DATASETS[key]
        target = root / Path(*spec.target)
        local_path = (
            Path(os.environ[spec.local_env_var]).expanduser().resolve()
            if spec.local_env_var and os.environ.get(spec.local_env_var)
            else None
        )
        status = (
            "present"
            if (local_path and mtlearn_data.has_existing_files(local_path))
            or mtlearn_data.has_existing_files(target)
            else "missing"
        )
        print(f"{key:22s} {status:8s} {target}")
        print(f"  {spec.description}")
        if spec.url_env_var:
            print(f"  URL env var: {spec.url_env_var}")
        if spec.local_env_var:
            print(f"  Local env var: {spec.local_env_var}")
        if spec.access_note:
            print(f"  Note: {spec.access_note}")


def _parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICPR 2026 notebook helper utilities.",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=ICPR2026_DATASETS,
        help="Dataset keys to download or locate.",
    )
    parser.add_argument(
        "--download-data",
        nargs="+",
        choices=ICPR2026_DATASETS,
        help="Dataset keys to download or locate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download or locate all ICPR 2026 notebook datasets.",
    )
    parser.add_argument(
        "--list-data",
        action="store_true",
        help="List ICPR 2026 datasets and exit.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Destination data directory. Defaults to MTLEARN_DATA_DIR or ./dat.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and replace existing target directories.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip archive next to the extracted data.",
    )
    parser.add_argument(
        "--url",
        help="Authorized review-only URL for plants_segmentation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    data_dir = args.data_dir.expanduser().resolve() if args.data_dir else None

    if args.list_data:
        _list_icpr2026_datasets(data_dir)
        return 0

    selected = list(args.download_data or args.datasets)
    if args.all:
        selected = list(ICPR2026_DATASETS)

    if not selected:
        print(
            "No dataset selected. Use --list-data, --all, --download-data, or pass dataset keys.",
            file=sys.stderr,
        )
        return 2

    if args.url and selected != ["plants_segmentation"]:
        print("--url can only be used with plants_segmentation.", file=sys.stderr)
        return 2

    try:
        paths = ensure_icpr2026_data(
            selected,
            data_dir,
            force=args.force,
            keep_archive=args.keep_archive,
            url=args.url,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
