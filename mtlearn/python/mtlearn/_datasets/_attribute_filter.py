"""Legacy morphology-generated target dataset.

``AttributeFilterDataset`` predates the current paired-image experiments. It
reads grayscale images, builds a morphology tree for each image on demand,
computes configured attributes, applies threshold criteria, and returns the
filtered image as the target. The class remains available through
``mtlearn.datasets`` for compatibility, but new experiments should generally
prefer ``PairedImageDataset`` or ``GeneratedTargetImageDataset``.
"""

from __future__ import annotations

import abc
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .. import morphology
from ._split import _split_indices


class AttributeFilterDataset(Dataset, abc.ABC):
    """Legacy dataset that builds a morphology tree and filters each image."""

    def __init__(
        self,
        root,
        tree_type,
        attributes: list,
        thresholds: dict,
        top_hat: bool = False,
        numRows: int | None = None,
        numCols: int | None = None,
        tos_interpolation=None,
        tos_infinity_seed_row: int = 0,
        tos_infinity_seed_col: int = 0,
    ):
        """Create a legacy dataset that generates targets with morphology filters.

        Args:
            root: Directory containing source images.
            tree_type: Tree type accepted by ``mtlearn.morphology.build_tree``.
            attributes: Attribute enum values used to build threshold criteria.
            thresholds: Mapping from attribute name to threshold value.
            top_hat: If true, return a top-hat residual instead of the filtered
                image directly.
            numRows: Optional output height for input resizing.
            numCols: Optional output width for input resizing.
            tos_interpolation: Tree-of-shapes interpolation mode, when relevant.
            tos_infinity_seed_row: Row seed for tree-of-shapes infinity handling.
            tos_infinity_seed_col: Column seed for tree-of-shapes infinity handling.
        """

        super().__init__()

        self.root = root
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        paths = []
        for extension in extensions:
            paths.extend(glob.glob(os.path.join(root, extension)))
        if not paths:
            raise FileNotFoundError(f"No images found in {root}")
        paths.sort()
        self.paths = paths
        self.thresholds = thresholds
        self.attributes = attributes
        self.tree_type = morphology.normalize_tree_type(tree_type)
        if self.tree_type == "tree-of-shapes":
            self.tos_interpolation = morphology.normalize_tos_interpolation(tos_interpolation)
        else:
            self.tos_interpolation = tos_interpolation
        self.tos_infinity_seed_row = int(tos_infinity_seed_row)
        self.tos_infinity_seed_col = int(tos_infinity_seed_col)
        self.numRows = numRows
        self.numCols = numCols
        self.top_hat = top_hat

    def __len__(self):
        """Return the number of source images found in ``root``."""

        return len(self.paths)

    def __getitem__(self, idx):
        """Build the tree for one image and return ``(input, target, filename)``."""

        path = self.paths[idx]
        image_u8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image_u8 is None:
            raise RuntimeError(f"Failed to read image: {path}")
        if self.numRows is not None and self.numCols is not None:
            image_u8 = cv2.resize(image_u8, (self.numCols, self.numRows))

        tree = morphology.build_tree(
            image_u8,
            self.tree_type,
            tos_interpolation=self.tos_interpolation,
            tos_infinity_seed_row=self.tos_infinity_seed_row,
            tos_infinity_seed_col=self.tos_infinity_seed_col,
        )

        attr_idx, attr_values = morphology.compute_attributes(tree, self.attributes)
        criterion = np.ones(attr_values.shape[0], dtype=bool)
        for attr_type in self.attributes:
            name = attr_type.name
            criterion = criterion & (attr_values[:, attr_idx[name]] > self.thresholds[name])

        filter_obj = morphology.create_attribute_filter(tree)
        image_out_u8 = filter_obj.filteringSubtractiveRule(criterion)
        if self.top_hat:
            if self.tree_type == "min-tree":
                image_out_u8 = image_out_u8 - image_u8
            elif self.tree_type == "max-tree":
                image_out_u8 = image_u8 - image_out_u8
            else:
                image_out_u8 = np.abs(image_out_u8 - image_u8)

        image_out = torch.from_numpy(image_out_u8).to(torch.float32).unsqueeze(0)
        image_in = torch.from_numpy(image_u8).to(torch.float32).unsqueeze(0)

        return image_in, image_out, os.path.basename(path)

    def train_test_split(self, test_size=0.25, shuffle=True, random_state=42):
        """Return ``(train_subset, test_subset)`` using stable image indices."""

        train_idx, test_idx = _split_indices(
            len(self),
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        return (
            torch.utils.data.Subset(self, train_idx.tolist()),
            torch.utils.data.Subset(self, test_idx.tolist()),
        )
