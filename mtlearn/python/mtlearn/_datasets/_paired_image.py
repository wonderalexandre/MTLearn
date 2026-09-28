"""Paired image dataset implementation for mtlearn experiments."""

from __future__ import annotations

import glob
import os

import cv2
import torch
from torch.utils.data import Dataset

from ._image_ops import (
    DEFAULT_IMAGE_EXTENSIONS,
    basename,
    invert_image,
    normalize_extensions,
    normalize_resize_shape,
    read_image,
    resize_image,
    scale_tensor,
    to_channel_first_tensor,
    validate_spatial_shape,
)
from ._split import _split_indices


class PairedImageDataset(Dataset):
    """Read matched input/target image pairs from one directory.

    Samples are returned as ``(input_tensor, target_tensor, filename)`` with
    channel-first tensors. Resize is optional; when configured, both input and
    target images are resized to ``(num_rows, num_cols)``. Without resize, the
    input and target spatial shapes must already match.
    """

    @classmethod
    def from_pairs(cls, pairs, **kwargs):
        """Read explicit ``(sample_id, input_path, target_path)`` records.

        IDs must be unique, nonempty strings. ``ordering="provided"`` preserves
        record order; ``"textual"`` sorts strings. ``"numeric"`` sorts decimal
        IDs numerically and normalizes them, so ``"01"`` and ``"1"`` collide.
        Indexing checks file existence without decoding images.

        Reader options shared with :meth:`from_folders` and :meth:`from_suffixes`:

        * ``grayscale_in`` and ``grayscale_target`` default to True. False reads
          RGB; ordinary reading converts to uint8 and discards alpha channels.
        * ``strict_grayscale_uint8=True`` requires native 2D uint8 input and
          target arrays, with both grayscale flags enabled.
        * ``binary_target=True`` requires a native 2D uint8 target whose values
          belong to either {0, 1} or {0, 255}. It maps positive values to one,
          requires grayscale_target=True and rejects invert_target=True.
        * ``invert_in`` and ``invert_target`` default to False and apply
          ``255 - image`` before resizing.
        * ``num_rows`` and ``num_cols`` default to None. Supply both to resize
          inputs with area interpolation and targets with nearest interpolation.
          Input and target spatial sizes must match before resizing.
        * ``dtype`` defaults to torch.float32. ``scale_in`` and ``scale_out``
          default to True and divide converted tensors by 255; binary targets
          bypass scale_out. Integer tensors can promote to a floating dtype.

        Return a PairedImageDataset with ``sample_ids``, immutable ``records``
        (sample_id, input_path and target_path fields), and ``pairs`` containing
        only input/target paths. Items are ``(input, target, sample_id)`` on CPU
        with shapes ``(C, H, W)`` and ``(C_target, H, W)``. The returned reader
        provides ``get_config()`` and ``get_preprocessing_contract()``. The latter
        returns copied ``input`` and ``target`` mappings describing decoding,
        channel order, alpha and orientation policies, inversion, resize,
        interpolation, binarization and scaling. ``dtype`` records conversion;
        ``output_dtype`` records the effective dtype after scaling. Native file
        shape and dtype are recorded separately by :func:`audit_image_pairs`.
        """
        from ._explicit_pairs import ExplicitPairedImageDataset
        return ExplicitPairedImageDataset(pairs, **kwargs)

    @classmethod
    def from_folders(cls, input_dir, target_dir, *, ordering="textual", unmatched="error",
                     extensions=DEFAULT_IMAGE_EXTENSIONS, **kwargs):
        """Match direct files in two directories by filename stem.

        ``ordering="textual"`` preserves textual IDs; ``"numeric"`` normalizes
        decimal IDs. Duplicate IDs within a role always fail. ``unmatched`` is
        error, warn or ignore and controls incomplete pairs. Missing counterparts
        are reported in ``missing_input_ids`` and ``missing_target_ids``.
        Extensions are case-normalized. Reader options follow :meth:`from_pairs`.

        Configuration reconstruction rescans the directories and checks the
        ordered ID digest. Use :meth:`from_manifest` for a saved selection that
        must ignore unrelated new files.
        """
        from ._explicit_pairs import ExplicitPairedImageDataset, folder_pairs
        pairs, missing = folder_pairs(input_dir, target_dir, ordering=ordering,
                                      unmatched=unmatched, extensions=extensions)
        dataset = ExplicitPairedImageDataset(pairs, **kwargs)
        from pathlib import Path
        dataset._folder_config = dict(input_dir=str(Path(input_dir).expanduser().resolve()),
            target_dir=str(Path(target_dir).expanduser().resolve()), ordering=ordering,
            unmatched=unmatched, extensions=list(extensions))
        dataset.missing_input_ids = tuple(missing["missing_input_ids"])
        dataset.missing_target_ids = tuple(missing["missing_target_ids"])
        return dataset

    @classmethod
    def from_suffixes(cls, root_dir, *, suffix_in="_in", suffix_target="_target",
                      prefix_in="", prefix_target="", extensions=(".png", ".jpg", ".pgm"),
                      ordering="textual", unmatched="error", **reader_options):
        """Build strict pairs from literal filename prefixes and suffixes.

        Index direct files in textual ID order without decoding. IDs are nonempty and
        preserved exactly. Suffixes must be distinct and nonempty; a file matching both
        roles or duplicate IDs within a role always fails. ``unmatched`` accepts error,
        warn or ignore for incomplete pairs only. Report exclusions through
        ``missing_input_ids`` and ``missing_target_ids``. Extensions are case-normalized.

        Reader options follow :meth:`from_pairs`. Configuration roundtrips rescan this
        source and check its ordered ID digest. To freeze a historical selection
        against unrelated added files, use :meth:`from_manifest`.
        """
        from ._explicit_pairs import ExplicitPairedImageDataset
        from ._suffix_pairs import suffix_pairs
        pairs, missing, config = suffix_pairs(root_dir, suffix_in=suffix_in,
            suffix_target=suffix_target, prefix_in=prefix_in, prefix_target=prefix_target,
            extensions=extensions, ordering=ordering, unmatched=unmatched)
        dataset = ExplicitPairedImageDataset(pairs, **reader_options)
        dataset._suffix_config = config
        dataset.missing_input_ids = tuple(missing["missing_input_ids"])
        dataset.missing_target_ids = tuple(missing["missing_target_ids"])
        return dataset

    @classmethod
    def from_manifest(cls, path, *, roots, split=None, expected_fingerprint=None):
        """Reconstruct a nonempty saved selection with the manifest's exact reader.

        Resolve only saved records, optionally in one split's order. The compact source
        configuration stores the manifest path, expected manifest fingerprint, roots
        and split rather than embedding every pair. Worker recreation rejects changed
        manifest contracts. Root locations may be changed explicitly for relocation.

        This loads metadata and checks source existence/containment, without decoding
        or verifying source hashes. Call ``SplitManifest.validate_files`` explicitly
        before execution. Use ``subsets_from_ids(..., allow_empty=True)`` when empty
        views are needed. The dataset's effective reader must remain unchanged before
        exporting its configuration.
        """
        from ._manifest_dataset import manifest_dataset
        return manifest_dataset(path, roots=roots, split=split,
                                expected_fingerprint=expected_fingerprint)

    def __init__(
        self,
        root_dir: str,
        num_rows: int | None = None,
        num_cols: int | None = None,
        *,
        grayscale_in: bool = True,
        grayscale_target: bool = True,
        invert_in: bool = False,
        invert_target: bool = False,
        extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
        dtype: torch.dtype = torch.float32,
        scale_in: bool = True,
        scale_out: bool = True,
        prefix_in: str = "",
        prefix_target: str = "",
        suffix_in: str = "_in",
        suffix_target: str = "_target",
    ):
        """Create a dataset from image pairs stored in one directory."""

        super().__init__()
        self.root_dir = os.fspath(root_dir)
        self.resize_shape = normalize_resize_shape(num_rows, num_cols)
        self.num_rows = None if self.resize_shape is None else self.resize_shape[0]
        self.num_cols = None if self.resize_shape is None else self.resize_shape[1]
        self.grayscale_in = bool(grayscale_in)
        self.grayscale_target = bool(grayscale_target)
        self.invert_in = bool(invert_in)
        self.invert_target = bool(invert_target)
        self.extensions = normalize_extensions(extensions)
        self.dtype = dtype
        self.scale_in = bool(scale_in)
        self.scale_out = bool(scale_out)
        self.prefix_in = str(prefix_in)
        self.prefix_target = str(prefix_target)
        self.suffix_in = str(suffix_in)
        self.suffix_target = str(suffix_target)

        self.pairs = self._scan_pairs()
        if not self.pairs:
            raise RuntimeError(
                f"No input/target image pairs found in {self.root_dir} "
                f"with suffixes {self.suffix_in!r}/{self.suffix_target!r} "
                f"and extensions {self.extensions}."
            )

    def __len__(self):
        """Return the number of matched input/target pairs."""

        return len(self.pairs)

    def __getitem__(self, idx: int):
        """Return one matched image pair as ``(input, target, filename)``."""

        input_path, target_path = self.pairs[idx]

        image_in = read_image(input_path, grayscale=self.grayscale_in)
        image_target = read_image(target_path, grayscale=self.grayscale_target)

        if self.invert_in:
            image_in = invert_image(image_in)
        if self.invert_target:
            image_target = invert_image(image_target)

        image_in = resize_image(
            image_in,
            self.resize_shape,
            interpolation=cv2.INTER_AREA,
        )
        image_target = resize_image(
            image_target,
            self.resize_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        validate_spatial_shape(
            image_target,
            image_in.shape[:2],
            name="target image",
            path=target_path,
        )

        tensor_in = to_channel_first_tensor(image_in, dtype=self.dtype)
        tensor_target = to_channel_first_tensor(image_target, dtype=self.dtype)

        tensor_in = scale_tensor(tensor_in, enabled=self.scale_in)
        tensor_target = scale_tensor(tensor_target, enabled=self.scale_out)

        return tensor_in, tensor_target, basename(input_path)

    def _scan_pairs(self) -> list[tuple[str, str]]:
        """Find input files and match each one to the first existing target."""

        pairs = []
        suffix_in_len = len(self.suffix_in)
        for extension in self.extensions:
            pattern = os.path.join(
                self.root_dir,
                f"{self.prefix_in}*{self.suffix_in}{extension}",
            )
            for input_path in glob.glob(pattern):
                base = basename(input_path)
                if not base.startswith(self.prefix_in) or not base.endswith(
                    self.suffix_in + extension
                ):
                    continue
                stem = base[len(self.prefix_in) : -suffix_in_len - len(extension)]
                target_path = None
                for target_extension in self.extensions:
                    candidate = f"{self.prefix_target}{stem}{self.suffix_target}{target_extension}"
                    candidate_path = os.path.join(self.root_dir, candidate)
                    if os.path.exists(candidate_path):
                        target_path = candidate_path
                        break
                if target_path is not None:
                    pairs.append((input_path, target_path))

        pairs.sort(key=lambda pair: pair[0])
        return pairs

    def train_test_split(self, test_size=0.25, shuffle=True, random_state=42):
        """Return ``(train_subset, test_subset)`` using stable pair indices."""

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
