#!/usr/bin/env python3
"""Smoke test an installed mtlearn wheel.

The test intentionally covers the public package import, the morphology facade,
and the default CFP layer on CPU. It is small enough to run once per wheel while
still catching missing native libraries, broken Torch linkage, and packaging
mistakes that plain metadata checks cannot detect.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import mtlearn
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mtlearn release smoke test.")
    parser.add_argument(
        "--expected-version",
        required=True,
        help="Package version expected from mtlearn.__version__.",
    )
    return parser.parse_args()


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
    assert attr_values.shape[0] == tree.numInternalNodeSlots

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
