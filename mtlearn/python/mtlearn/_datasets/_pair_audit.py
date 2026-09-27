"""Inspect original image files separately from effective dataset preprocessing."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2

from ._dataset_contracts import (
    _hash_file, _json_copy, _portable_path, _preprocessing,
    _reader_options, _roots, _signature,
)
from ._sample_ids import _ids


@dataclass(frozen=True, init=False)
class PairAudit:
    """Hold detached metadata from one explicit audit of original files.

    ``sample_ids`` preserves source order. ``sources`` records named relative
    paths, SHA-256, native shape and dtype, and optional callback metadata.
    ``preprocessing`` describes the effective reader separately. Public containers
    are defensive copies. In-process file signatures detect ordinary source
    changes before creating a manifest; they are excluded from content fingerprints.
    Obtain this object with :func:`audit_image_pairs`. Native metadata describes
    OpenCV IMREAD_UNCHANGED output, before the dataset reader's transformations.
    """
    _json: str
    _signatures: tuple

    def __init__(self, data, signatures=()):
        object.__setattr__(self, '_json', json.dumps(_json_copy(data), ensure_ascii=False, allow_nan=False))
        object.__setattr__(self, '_signatures', tuple(signatures))

    @property
    def sample_ids(self):
        """Return audited textual IDs in source order as a tuple."""
        return tuple(self.to_dict()['sample_ids'])

    @property
    def sources(self):
        """Return copied input/target file references and callback metadata by ID."""
        return self.to_dict()['sources']

    @property
    def preprocessing(self):
        """Return a copy of the effective input and target reader contracts."""
        return self.to_dict()['preprocessing']

    def to_dict(self):
        """Return JSON-compatible audit data without local file signatures."""
        return json.loads(self._json)

    def _check_unchanged(self):
        for path, signature in self._signatures:
            if _signature(path) != signature:
                raise RuntimeError(f'Audited file changed; repeat the audit: {path}')


def audit_image_pairs(dataset, roots, validate_sample=None):
    """Audit explicit image pairs with streamed hashes and one decoded pair at a time.

    Args:
        dataset: Dataset returned by an explicit PairedImageDataset factory.
            Its records and effective reader options define the audited contract.
        roots: Mapping containing ``input`` and ``target`` directories. Every file
            must resolve within its role's root, including through symbolic links.
        validate_sample: Optional callback ``(record, input_array, target_array)``.
            Arrays are read-only native OpenCV IMREAD_UNCHANGED results: color
            channels are BGR/BGRA, not the reader's RGB tensors. Return None or
            a JSON mapping with finite numeric values; raise to reject a sample. The callback
            must not retain arrays or mutate sources or reader configuration.

    Returns:
        A PairAudit. Native input/target spatial sizes must match; binary masks,
        grayscale, uint8 and matching channel counts are not imposed by the audit.

    Hashing uses bounded blocks. Only one decoded pair is retained by this function.
    File signatures are rechecked across the audit and when constructing a manifest;
    a detected change requires a fresh audit. This is a snapshot check, not a lock
    against later filesystem changes. No audit is added to dataset item access.
    """
    reader = _reader_options(dataset)
    sample_ids = _ids(dataset.sample_ids, 'sample_ids', nonempty=True)
    roots = _roots(roots)
    if not {'input', 'target'} <= roots.keys():
        raise ValueError('Audit roots must include input and target')
    if validate_sample is not None and not callable(validate_sample):
        raise TypeError('validate_sample must be callable or None')
    sources, signatures = {}, []
    if len(dataset.records) != len(sample_ids):
        raise ValueError('Pair records and sample_ids disagree')
    for sample_id, record in zip(sample_ids, dataset.records):
        if record.sample_id != sample_id:
            raise ValueError('Pair record order differs from sample_ids')
        values, arrays = {}, []
        for role, filename in (('input', record.input_path), ('target', record.target_path)):
            path = Path(filename).expanduser().resolve()
            try:
                relative = _portable_path(path.relative_to(roots[role]).as_posix())
            except ValueError as exc:
                raise ValueError(f'Sample {sample_id!r} {role}: file is outside its root: {path}') from exc
            try:
                signature = _signature(path)
                digest = _hash_file(path)
                array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if array is None or array.ndim not in (2, 3) or not array.size:
                    raise ValueError('Could not decode a nonempty image')
            except (OSError, ValueError, RuntimeError) as exc:
                raise type(exc)(f'Sample {sample_id!r} {role}: {exc}') from exc
            values[role] = dict(root=role, path=relative, sha256=digest,
                                shape=list(array.shape), dtype=str(array.dtype))
            array.setflags(write=False)
            arrays.append(array)
            signatures.append((str(path), signature))
        if arrays[0].shape[:2] != arrays[1].shape[:2]:
            raise ValueError(f'Sample {sample_id!r}: native input/target spatial shapes differ')
        metadata = None
        if validate_sample is not None:
            try:
                metadata = validate_sample(record, *arrays)
            except Exception as exc:
                raise ValueError(f'Sample {sample_id!r} validation: {exc}') from exc
        if metadata is not None and type(metadata) is not dict:
            raise TypeError(f'Sample {sample_id!r}: validation metadata must be a JSON mapping or None')
        values['metadata'] = {} if metadata is None else _json_copy(metadata)
        sources[sample_id] = values
        del arrays, array
    audit = PairAudit(dict(sample_ids=list(sample_ids), sources=sources, reader=reader,
                           preprocessing=_preprocessing(reader)), signatures)
    audit._check_unchanged()
    if _reader_options(dataset) != reader:
        raise RuntimeError('Reader configuration changed during audit')
    return audit
