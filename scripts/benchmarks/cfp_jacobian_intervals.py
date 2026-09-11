"""Compare the former rank-based CFP adjoint, event scans, and compact scans.

Run with this checkout and its native extension on PYTHONPATH. Measurements are
operator microbenchmarks, excluding trees, attributes, transfers, and autograd.
Preparation timings measure historical traversal-order construction separately;
current compact preparation stores no traversal permutation. Event algorithms
are local references, and only the compact variant calls the production runtime.
The legacy adjoint receives cached order/num_times, matching its former layer path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import mtlearn
from mtlearn import morphology
from mtlearn.layers.cfp.runtime.tree_reconstructor import reconstruct_from_info, propagate_pixels_to_nodes


def legacy_adjoint(grad, pre, post, parent, pix, order, num_times):
    """Reference: the implementation before the interval-scan refactor."""
    base = torch.zeros(pre.numel(), dtype=grad.dtype, device=grad.device)
    base.index_add_(0, pix.long(), grad)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(pre.numel(), device=order.device)
    prefix = torch.cumsum(base[order], 0)
    prefix = torch.cat([prefix.new_zeros(1), prefix])
    counts = torch.bincount(pre, minlength=num_times)
    cumulative = torch.cumsum(counts, 0)
    time_to_rank = torch.cat([cumulative.new_zeros(1), cumulative[:-1]])
    return prefix[time_to_rank[post]] - prefix[rank]


def event_forward(signal, pre, post, pix, num_times):
    """Historical interval forward on the interleaved DFS event domain."""
    delta = torch.zeros(num_times, dtype=signal.dtype, device=signal.device)
    delta.index_add_(0, pre, signal)
    delta.index_add_(0, post, -signal)
    return torch.cumsum(delta, 0)[pre[pix.to(torch.int64)]]


def event_adjoint(grad, pre, post, pix, num_times):
    """Stage 1 reference, isolated from the compact-only production runtime."""
    grad = grad.reshape(-1)
    events = torch.zeros(num_times - 1, dtype=grad.dtype, device=grad.device)
    events.index_add_(0, pre[pix.reshape(-1).to(torch.int64)], grad)
    prefix = torch.cat([events.new_zeros(1), torch.cumsum(events, 0)])
    return prefix[post] - prefix[pre]


def event_order(pre, num_times):
    by_time = torch.full((num_times,), -1, dtype=torch.int64)
    by_time[pre] = torch.arange(pre.numel())
    order = by_time[by_time >= 0]
    return order, order


def compact_order(pre):
    order = torch.empty_like(pre)
    order[pre] = torch.arange(pre.numel())
    return order, order


def indices(pre, post, parent):
    """Build both representations outside the measured operations."""
    n = pre.numel()
    order = torch.argsort(pre)
    compact_pre = torch.empty_like(pre)
    compact_pre[order] = torch.arange(n)
    compact_post = post if int(post.max()) == n else compact_pre + (post - pre + 1) // 2
    depths = [0] * n
    parents = parent.tolist()
    for node in order.tolist():
        par = parents[node]
        if par >= 0 and par != node:
            depths[node] = depths[par] + 1
    depth = torch.tensor(depths)
    return (2 * compact_pre - depth, 2 * compact_post - depth - 1), (compact_pre, compact_post), order


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def measure(functions, device, warmups, repeats):
    # Rotate variant order to reduce systematic warm-up/thermal ordering effects.
    names = list(functions)
    values = {name: [] for name in names}
    for iteration in range(warmups + repeats):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            synchronize(device)
            start = time.perf_counter()
            result = functions[name]()
            synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000
            if iteration >= warmups:
                values[name].append(elapsed)
            del result
    return {name: {"median_ms": statistics.median(samples),
                   "q1_ms": float(np.percentile(samples, 25)),
                   "q3_ms": float(np.percentile(samples, 75)), "samples_ms": samples} for name, samples in values.items()}


def input_images(args, rng):
    """Yield scalar uint8 inputs; preserve the CFP per-channel RGB convention."""
    if args.image_dir is not None:
        from PIL import Image
        suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        paths = sorted(p for p in args.image_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
        if not paths:
            raise ValueError(f"No supported images in {args.image_dir}.")
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as source:
                if source.mode not in ("L", "RGB"):
                    raise ValueError(f"{path}: expected uint8 L or RGB, got {source.mode}.")
                pixels = np.asarray(source)
                channels = ("L",) if source.mode == "L" else ("R", "G", "B")
                for index, channel in enumerate(channels):
                    scalar = pixels if pixels.ndim == 2 else pixels[..., index]
                    metadata = {"image": path.name, "source_sha256": digest,
                                "source_mode": source.mode, "channel": channel,
                                "num_channels": len(channels), "family": "file",
                                "size": int(scalar.shape[0]), "shape": list(scalar.shape)}
                    yield metadata, np.ascontiguousarray(scalar).copy()
        return
    for size in args.sizes:
        coordinates = np.linspace(-1, 1, size)
        radius = np.maximum(np.abs(coordinates[:, None]), np.abs(coordinates[None, :]))
        images = {"noise": rng.integers(0, 256, (size, size), dtype=np.uint8),
                  "nested": np.rint((1-radius)*255).astype(np.uint8)}
        for family, img in images.items():
            yield {"size": size, "shape": [size, size], "family": family}, img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["events", "compact"], required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 256, 512])
    parser.add_argument("--image-dir", type=Path,
                        help="Use images from this directory instead of synthetic inputs; split RGB channels.")
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be non-negative and repeats must be positive")
    torch.set_num_threads(4)
    rng = np.random.default_rng(71)
    generator = torch.Generator().manual_seed(71)
    report = {"stage": args.stage, "torch": torch.__version__, "mtlearn_source": mtlearn.__file__,
              "native": mtlearn._bindings.__file__, "platform": platform.platform(),
              "threads": 4, "warmups": args.warmups, "repeats": args.repeats,
              "dtype": "float32", "seed": 71,
              "input_directory": str(args.image_dir.resolve()) if args.image_dir is not None else None,
              "color_policy": "independent native RGB channels; no resizing or grayscale conversion",
              "cases": []}
    for input_metadata, img in input_images(args, rng):
        for tree_type in ("max-tree", "min-tree"):
            tree = morphology.build_tree(img, tree_type)
            res, pre, post, parent, pix = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
            n = pre.numel()
            assert int(post.max()) == (2*n-1 if args.stage == "events" else n)
            events, compact, order = indices(pre, post, parent)
            preparation = measure({
                "original_argsort": lambda: (torch.argsort(events[0]), torch.argsort(events[0])),
                "events_linear": lambda: event_order(events[0], 2*n),
                **({"compact_linear": lambda: compact_order(compact[0])} if args.stage == "compact" else {}),
            }, "cpu", args.warmups, args.repeats)
            signal_cpu = res * torch.sigmoid(torch.randn(n, generator=generator))
            grad_cpu = torch.randn(img.size, generator=generator) / img.size
            reference_f = event_forward(signal_cpu.double(), *events, pix, 2*n)
            reference_b = event_adjoint(grad_cpu.double(), *events, pix, 2*n)
            for device in args.devices:
                if device == "mps" and not torch.backends.mps.is_available():
                    continue
                if device == "cuda" and not torch.cuda.is_available():
                    continue
                signal, grad, par, pixel, traversal = [v.to(device) for v in (signal_cpu, grad_cpu, parent, pix, order)]
                layouts = {"original_cached": events, "events_scan": events}
                if args.stage == "compact":
                    layouts["compact_scan"] = compact
                forwards, backwards, errors = {}, {}, {}
                for name, (entry, end) in layouts.items():
                    entry, end = entry.to(device), end.to(device)
                    if name == "compact_scan":
                        forwards[name] = lambda entry=entry, end=end: reconstruct_from_info(
                            signal, entry, end, pixel)
                        backwards[name] = lambda entry=entry, end=end: propagate_pixels_to_nodes(
                            grad, entry, end, pixel)
                    else:
                        forwards[name] = lambda entry=entry, end=end: event_forward(
                            signal, entry, end, pixel, 2*n)
                        if name == "original_cached":
                            backwards[name] = lambda entry=entry, end=end: legacy_adjoint(
                                grad, entry, end, par, pixel, traversal, 2*n)
                        else:
                            backwards[name] = lambda entry=entry, end=end: event_adjoint(
                                grad, entry, end, pixel, 2*n)
                    f, b = forwards[name]().cpu().double(), backwards[name]().cpu().double()
                    assert torch.isfinite(f).all() and torch.isfinite(b).all()
                    errors[name] = {"forward_max_abs": (f-reference_f).abs().max().item(),
                                    "backward_max_abs": (b-reference_b).abs().max().item()}
                row = {**input_metadata, "tree": tree_type, "nodes": n, "pixels": img.size,
                       "device": device, "event_scan_elements": 2*n, "compact_scan_elements": n+1,
                       "order_preparation_cpu": preparation,
                       "forward": measure(forwards, device, args.warmups, args.repeats),
                       "backward": measure(backwards, device, args.warmups, args.repeats),
                       "max_abs_error_vs_float64": errors}
                report["cases"].append(row)
                label = input_metadata.get("image", input_metadata["family"])
                channel = input_metadata.get("channel", "L")
                print(f'{args.stage}: {device} {label} {channel} {tree_type} {img.shape}, {n} nodes', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    with torch.no_grad():
        main()
