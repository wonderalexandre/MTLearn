from __future__ import annotations

import argparse
import gc
import json
from time import perf_counter

import torch
import torch.nn.functional as F

from mtlearn.patches import patch_grid
from mtlearn.training import backward_from_global_response


class SavedTensor:
    def __init__(self, tensor, tracker):
        self.tensor = tensor
        self.tracker = tracker
        storage = tensor.untyped_storage()
        self.key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        tracker.add(self.key)

    def __del__(self):
        self.tracker.remove(self.key)


class SavedStorageTracker:
    def __init__(self):
        self.references = {}
        self.live_bytes = 0
        self.peak_bytes = 0

    def add(self, key):
        if key not in self.references:
            self.live_bytes += key[2]
            self.references[key] = 0
        self.references[key] += 1
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)

    def remove(self, key):
        self.references[key] -= 1
        if self.references[key] == 0:
            self.live_bytes -= key[2]
            del self.references[key]

    def pack(self, tensor):
        return SavedTensor(tensor, self)



def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def run(strategy, device, size, patch_size, stride, halo):
    torch.manual_seed(12)
    producer = torch.nn.Conv2d(3, 3, 1).to(device)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.Tanh(),
        torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.Tanh(), torch.nn.Conv2d(16, 2, 1)).to(device)
    image = torch.randn(1, 3, size, size, device=device)
    target = torch.randn(1, 2, size, size, device=device)
    patches = patch_grid(size, size, patch_size, stride, halo)
    loss_fn = lambda logits, truth, weights: (logits - truth).square().mean()
    tracker = SavedStorageTracker()
    response = producer(image)
    synchronize(device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(tracker.pack, lambda saved: saved.tensor):
        if strategy == 'direct':
            losses = []
            for patch in patches:
                top, left = patch.row - halo, patch.col - halo
                bottom, right = patch.row + patch_size + halo, patch.col + patch_size + halo
                row_start, col_start = max(0, top), max(0, left)
                row_stop, col_stop = min(size, bottom), min(size, right)
                local = F.pad(response[..., row_start:row_stop, col_start:col_stop],
                    (col_start - left, right - col_stop, row_start - top, bottom - row_stop))
                core = model(local)[..., halo:halo + patch_size, halo:halo + patch_size]
                losses.append(loss_fn(core, target[..., patch.target[0], patch.target[1]], None) / len(patches))
            loss = torch.stack(losses).sum()
            loss.backward()
            value = float(loss.detach())
        else:
            result, _ = backward_from_global_response(response, model, target, patches, main_loss=loss_fn,
                                                      strategy=strategy, diagnostics=False, timing=False)
            value = result.total_loss
    synchronize(device)
    elapsed = perf_counter() - started
    gc.collect()
    return dict(strategy=strategy, device=device, torch=torch.__version__, dtype=str(image.dtype),
        shape=list(image.shape), patch_size=patch_size, stride=stride, halo=halo, patches=len(patches),
        total_loss=value, measured_seconds=elapsed,
        peak_unique_storage_bytes_referenced_by_consumer_saved_tensors=tracker.peak_bytes,
        remaining_saved_storage_bytes=tracker.live_bytes,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None), [
            p.grad.detach().cpu() for p in list(producer.parameters()) + list(model.parameters())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=24)
    parser.add_argument('--halo', type=int, default=4)
    args = parser.parse_args()
    options = (args.device, args.size, args.patch_size, args.stride, args.halo)
    run('local', *options)
    results, gradients = [], None
    for strategy in ('direct', 'local', 'global_leaf'):
        result, actual = run(strategy, *options)
        if gradients is None:
            gradients = actual
        else:
            for got, expected in zip(actual, gradients):
                torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)
        assert result['remaining_saved_storage_bytes'] == 0
        results.append(result)
    print(json.dumps({'measurement': 'Unique tensor storage referenced by autograd saved data during consumer execution; producer forward excluded; not total process or allocator memory.',
                      'warmup': 'One local pass before measurement', 'results': results}, indent=2))


if __name__ == '__main__':
    main()
