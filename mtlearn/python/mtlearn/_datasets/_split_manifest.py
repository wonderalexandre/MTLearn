"""Versioned image partitions with portable sources and independent fingerprints."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from ._dataset_contracts import (
    _canonical, _fingerprint, _hash_file, _json_copy, _load_json, _portable_path,
    _preprocessing, _reader_options, _resolve_file, _roots, _validate_reader,
)
from ._pair_audit import PairAudit
from ._split_ids import _validate


_FIELDS = {'format', 'schema_version', 'sample_ids', 'sources', 'splits', 'groups',
           'policy', 'preprocessing', 'reader', 'provenance'}


def _validated(data):
    data = _json_copy(data)
    if not isinstance(data, dict) or set(data) != _FIELDS:
        raise ValueError('Manifest must contain exactly the versioned schema fields')
    if data['format'] != 'mtlearn.dataset-manifest' or type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError('Unsupported dataset manifest format or schema_version')
    policy = data['policy']
    if not isinstance(policy, dict) or set(policy) != {'require_complete', 'require_complete_groups', 'allow_empty'}:
        raise ValueError('Manifest policy must declare all split validation options')
    report, splits, groups = _validate(data['sample_ids'], data['splits'], data['groups'], **policy)
    data['sample_ids'] = list(report.sample_ids)
    data['splits'] = {name: list(values) for name, values in splits.items()}
    data['groups'] = groups
    if not isinstance(data['provenance'], dict):
        raise ValueError('Manifest provenance must be a JSON mapping')
    sources = data['sources']
    if not isinstance(sources, dict) or set(sources) != set(report.sample_ids):
        raise ValueError('Manifest sources must match the complete sample_ids universe')
    for sample_id, source in sources.items():
        if not isinstance(source, dict) or set(source) != {'input', 'target', 'metadata'}:
            raise ValueError(f'Sample {sample_id!r}: expected input, target and metadata')
        if not isinstance(source['metadata'], dict):
            raise ValueError(f'Sample {sample_id!r}: metadata must be a JSON mapping')
        for role in ('input', 'target'):
            value = source[role]
            if not isinstance(value, dict) or set(value) != {'root', 'path', 'sha256', 'shape', 'dtype'}:
                raise ValueError(f'Sample {sample_id!r} {role}: invalid file reference')
            if not isinstance(value['root'], str) or not value['root']:
                raise ValueError(f'Sample {sample_id!r} {role}: root must be a nonempty name')
            _portable_path(value['path'])
            if not isinstance(value['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', value['sha256']):
                raise ValueError(f'Sample {sample_id!r} {role}: invalid lowercase SHA-256')
            shape = value['shape']
            if not isinstance(shape, list) or len(shape) not in (2, 3) or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError(f'Sample {sample_id!r} {role}: invalid native shape')
            try:
                dtype = np.dtype(value['dtype'])
                if not isinstance(value['dtype'], str) or dtype.kind not in 'buif':
                    raise ValueError('Expected a real image dtype')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Sample {sample_id!r} {role}: invalid native dtype') from exc
        if source['input']['shape'][:2] != source['target']['shape'][:2]:
            raise ValueError(f'Sample {sample_id!r}: native spatial shapes differ')
    data['reader'] = _validate_reader(data['reader'])
    if _canonical(data['preprocessing']) != _canonical(_preprocessing(data['reader'])):
        raise ValueError('Preprocessing contract differs from the effective reader configuration')
    return data


@dataclass(frozen=True, init=False)
class SplitManifest:
    """Store a validated, portable image-selection document with defensive copies.

    The format is ``mtlearn.dataset-manifest``, schema version 1. It records the
    complete audited universe, splits, optional groups, policies, native source
    metadata, effective reader options, preprocessing contracts and provenance.
    Construct from :meth:`from_audit`, :meth:`load`, or a complete schema mapping.

    File references use canonical relative POSIX paths and named roots. Root
    locations are supplied at execution time. Loading validates structure without
    reading images. Call :meth:`validate_files` explicitly before using a saved
    selection, and :meth:`validate_preprocessing` for an independently built reader.
    """
    _json: str

    def __init__(self, data):
        object.__setattr__(self, '_json', json.dumps(_validated(data), ensure_ascii=False, allow_nan=False))

    @classmethod
    def from_audit(cls, audit, split_ids, groups=None, *, preprocessing=None,
                   require_complete=True, require_complete_groups=True, allow_empty=False,
                   provenance=None):
        """Create a manifest from audited sources and validated split membership.

        Preprocessing is derived from the audited reader. An optional supplied contract
        must match it exactly. Split policies follow :func:`validate_split_ids`, with
        complete coverage enabled by default. Optional provenance is a JSON mapping,
        for example a seed, algorithm name or decoder version; it is excluded from
        fingerprints. Reuse an audit for several partitions only while its files remain
        unchanged. A reduced selection that cuts a group requires
        ``require_complete_groups=False``; preserve the full manifest separately.
        """
        if not isinstance(audit, PairAudit):
            raise TypeError('audit must be a PairAudit')
        audit._check_unchanged()
        data = audit.to_dict()
        if preprocessing is not None and _canonical(preprocessing) != _canonical(data['preprocessing']):
            raise ValueError('Preprocessing differs from the audited reader')
        report, splits, groups = _validate(data['sample_ids'], split_ids, groups,
            require_complete, require_complete_groups, allow_empty)
        data.update(format='mtlearn.dataset-manifest', schema_version=1,
            splits={name: list(values) for name, values in splits.items()}, groups=groups,
            policy=dict(require_complete=require_complete, require_complete_groups=require_complete_groups,
                        allow_empty=allow_empty), provenance={} if provenance is None else provenance)
        return cls(data)

    def to_dict(self):
        """Return a detached, JSON-compatible copy of the validated schema."""
        return json.loads(self._json)

    @property
    def sample_ids(self):
        """Return the complete audited universe in source order as a tuple."""
        return tuple(self.to_dict()['sample_ids'])

    @property
    def splits(self):
        """Return copied split names and ordered sample-ID lists."""
        return self.to_dict()['splits']

    @property
    def groups(self):
        """Return a copied sample-to-group mapping, or None when grouping is off."""
        return self.to_dict()['groups']

    @property
    def preprocessing(self):
        """Return copied effective reader contracts for input and target."""
        return self.to_dict()['preprocessing']

    def _selection(self, data, split):
        if split is None:
            return data['sample_ids']
        if not isinstance(split, str) or split not in data['splits']:
            raise ValueError(f'Unknown split {split!r}')
        return data['splits'][split]

    def fingerprint(self, kind='manifest', *, split=None):
        """Hash a validated identity component, optionally for one named split.

        ``input`` and ``target`` hash ID/file-hash pairs sorted by textual ID. They
        exclude paths and selection order. ``splits`` hashes ordered membership,
        selected groups and policies. ``preprocessing_input`` and
        ``preprocessing_target`` describe each role's effective reader operations.
        ``manifest`` includes all schema fields except provenance, including relative
        paths and audit metadata, and guards compact worker reconstruction.

        Absolute roots and timestamps in provenance are excluded. JSON map order and
        formatting do not affect hashes; list order does. File hashes identify original
        bytes, not decoded tensors or morphological cache entries. A split-scoped hash
        excludes other splits' sources and groups. Never use a whole-manifest hash as
        an indiscriminate morphology-cache key.
        """
        data = self.to_dict()
        ids = self._selection(data, split)
        if kind in ('input', 'target'):
            value = [[sample_id, data['sources'][sample_id][kind]['sha256']] for sample_id in sorted(ids)]
        elif kind == 'splits':
            groups = None if data['groups'] is None else {sample_id: data['groups'][sample_id] for sample_id in ids}
            value = dict(splits=data['splits'] if split is None else {split: ids}, groups=groups,
                         policy=data['policy'])
        elif kind in ('preprocessing_input', 'preprocessing_target'):
            value = data['preprocessing'][kind.removeprefix('preprocessing_')]
        elif kind == 'manifest':
            value = {key: item for key, item in data.items() if key != 'provenance'}
            if split is not None:
                value.update(sample_ids=list(ids), sources={sample_id: data['sources'][sample_id] for sample_id in ids},
                    splits={split: list(ids)}, groups=None if data['groups'] is None else
                    {sample_id: data['groups'][sample_id] for sample_id in ids})
        else:
            raise ValueError(f'Unknown fingerprint kind {kind!r}')
        return _fingerprint(f'mtlearn.dataset-manifest.v1.{kind}', value)

    def fingerprints(self, *, split=None):
        """Return all six identity components for the universe or a selected split.

        Preprocessing components are shared across splits; content and membership are
        restricted when ``split`` is supplied. No files are opened or hashed here.
        """
        return {kind: self.fingerprint(kind, split=split) for kind in
                ('input', 'target', 'splits', 'preprocessing_input', 'preprocessing_target', 'manifest')}

    def validate_preprocessing(self, dataset_or_contract):
        """Check an explicit reader, its Subset, or an effective contract mapping.

        Return a successful role report or raise ValueError naming changed roles.
        Checking file hashes alone cannot detect inversion, scaling or resize changes.
        """
        from torch.utils.data import Subset
        while isinstance(dataset_or_contract, Subset):
            dataset_or_contract = dataset_or_contract.dataset
        current = (_json_copy(dataset_or_contract) if isinstance(dataset_or_contract, dict)
                   else _preprocessing(_reader_options(dataset_or_contract)))
        expected = self.preprocessing
        if _canonical(current) != _canonical(expected):
            roles = [role for role in ('input', 'target') if _canonical(current.get(role)) != _canonical(expected[role])]
            raise ValueError(f'Preprocessing differs for roles: {roles}')
        return {'valid': True, 'roles': ['input', 'target']}

    def pairs(self, *, roots, split=None):
        """Resolve saved records to explicit ``(sample_id, input_path, target_path)`` tuples.

        The saved order is preserved and directories are not rescanned for new samples.
        Check existence and containment, rejecting symbolic links outside their roots.
        This operation does not decode or verify file hashes; use ``validate_files``
        for that explicit preflight. A selected empty split returns an empty list.
        """
        roots = _roots(roots)
        data = self.to_dict()
        result = []
        for sample_id in self._selection(data, split):
            paths = []
            for role in ('input', 'target'):
                try:
                    paths.append(str(_resolve_file(data['sources'][sample_id][role], roots)))
                except (ValueError, OSError) as exc:
                    raise ValueError(f'Sample {sample_id!r} {role}: {exc}') from exc
            result.append((sample_id, *paths))
        return result

    def validate_files(self, *, roots, split=None, raise_on_error=True):
        """Verify saved original bytes without decoding or modifying image files.

        Return checked sample IDs, a file count and issues identifying sample and role.
        By default divergences raise ValueError. With ``raise_on_error=False``, collect
        file-level issues in the report. Invalid root configuration still raises.
        Validation must be repeated if sources can change after this snapshot.
        """
        if type(raise_on_error) is not bool:
            raise TypeError('raise_on_error must be a boolean')
        roots = _roots(roots)
        data = self.to_dict()
        ids = self._selection(data, split)
        issues, checked = [], 0
        for sample_id in ids:
            for role in ('input', 'target'):
                checked += 1
                reference = data['sources'][sample_id][role]
                try:
                    path = _resolve_file(reference, roots)
                    if _hash_file(path) != reference['sha256']:
                        raise ValueError('SHA-256 differs from the audited file')
                except (OSError, ValueError, RuntimeError) as exc:
                    issues.append(dict(sample_id=sample_id, role=role, error=str(exc)))
        report = dict(valid=not issues, sample_ids=list(ids), checked_files=checked, issues=issues)
        if issues and raise_on_error:
            raise ValueError(f'Manifest file validation failed: {issues}')
        return report

    def save(self, path):
        """Write JSON through a unique temporary file and atomic replacement.

        The parent directory must exist. A failed replacement preserves the previous
        file and removes the temporary file. The manifest itself remains unchanged.
        """
        path = Path(path).expanduser()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                    prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path):
        """Load and validate schema and contracts without reading source images.

        Reject unknown schemas, duplicate JSON keys, nonfinite numbers, invalid paths
        and inconsistent IDs, splits or reader contracts. Integer readers with scaling
        require the saved effective floating dtype to match the current Torch default.
        No legacy schema is inferred.
        """
        return cls(_load_json(path))
