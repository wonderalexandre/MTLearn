#!/usr/bin/env python3
"""Smoke test an installed mtlearn wheel.

The test intentionally covers the public package import, the morphology facade,
the default CFP layer and explicit preparation/storage APIs on CPU, including
a spawned preparation worker. It is small enough to run once per wheel while
still catching missing native libraries, broken Torch linkage, and packaging
mistakes that plain metadata checks cannot detect.
"""

from __future__ import annotations

import argparse
import tempfile

import numpy as np
import torch

import mtlearn
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer
from mtlearn.layers.cfp import (
    CFPPreprocessor, DiskStore, MemoryStore, NullStore, PreparedBatch,
    PreparedDataset, PreparedMorphology, PreparationResult, StatisticsSnapshot,
    collate_prepared,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mtlearn release smoke test.")
    parser.add_argument(
        "--expected-version",
        required=True,
        help="Package version expected from mtlearn.__version__.",
    )
    return parser.parse_args()


def check_preparation() -> None:
    """Exercise packaged preparation, normalization, stores and CPU spawn imports."""
    layer = ConnectedFilterPreprocessingLayer(
        1, [{"tree_type": "max-tree", "attributes": (morphology.AttributeType.AREA,)}],
        device="cpu", scale_mode="dataset_zscore", clamp=None,
    )
    images = torch.arange(128, dtype=torch.float32).reshape(2, 1, 8, 8)
    dataset = torch.utils.data.TensorDataset(images, images / 255)
    snapshot = layer.fit_stats(torch.utils.data.DataLoader(dataset, batch_size=1))
    assert isinstance(snapshot, StatisticsSnapshot)
    preprocessor = CFPPreprocessor.from_layer(layer)
    assert isinstance(preprocessor.prepare_image(images[0, 0]), PreparedMorphology)
    expected = layer(images)
    expected.sum().backward()
    gradients = {name: value.grad.clone() for name, value in layer.named_parameters()}

    def check(batch):
        assert isinstance(batch, PreparedBatch)
        layer.zero_grad(set_to_none=True)
        actual = layer(batch)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.sum().backward()
        for name, value in layer.named_parameters():
            torch.testing.assert_close(value.grad, gradients[name], rtol=1e-5, atol=1e-6)

    for store in (NullStore(), MemoryStore(max_bytes=1024**2)):
        check(preprocessor.prepare_batch(images, store=store))
        store.clear()
        assert store.info()["retained_bytes"] == 0

    with tempfile.TemporaryDirectory(prefix="mtlearn-release-cache-") as cache:
        with DiskStore(cache, max_disk_bytes=8 * 1024**2) as writer:
            result = preprocessor.prepare(
                dataset, store=writer, manifest="train", source_version="smoke-v1",
                preprocessing_version="u8-v1", split="train",
                collect_stats=layer.get_statistics_contract(),
                num_workers=1, max_in_flight=1, worker_threads=1,
            )
            assert isinstance(result, PreparationResult) and result.status == "complete"
            for key, values in snapshot.statistics.items():
                for name, value in values.items():
                    torch.testing.assert_close(result.statistics.statistics[key][name], value,
                                               rtol=1e-10, atol=1e-12)
            layer.set_stats(result.statistics)
        with DiskStore(cache, readonly=True) as reader:
            prepared = PreparedDataset(reader, "train", source=dataset)
            batch, targets = next(iter(torch.utils.data.DataLoader(
                prepared, batch_size=2, num_workers=0, collate_fn=collate_prepared)))
            torch.testing.assert_close(targets, images / 255)
            check(batch)
            assert reader.info()["retained_bytes"] == 0


def main() -> int:
    args = parse_args()

    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    tree = morphology.create_max_tree(image)
    attr_index, attr_values = morphology.compute_attributes(
        tree,
        [morphology.AttributeType.AREA],
    )

    assert mtlearn.__version__ == args.expected_version
    assert attr_index["AREA"] == 0
    assert attr_values.shape[0] == tree.num_internal_node_slots

    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            }
        ],
        device="cpu",
        scale_mode="none",
        clamp=None,
    )
    output = layer(torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32))
    assert output.shape == (1, 1, 2, 2)
    check_preparation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
