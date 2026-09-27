"""Validate ordered image partitions and keep acquisition groups disjoint."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Optional

import numpy as np
from torch.utils.data import Subset

from ._sample_ids import _ids, _resolve_sample_ids


def _named_mapping(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f'{name} must be a mapping')
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f'{name} keys must be nonempty strings')
    return dict(value)


def _groups(sample_ids, groups):
    if groups is None:
        return None
    groups = _named_mapping(groups, 'groups')
    missing, unknown = set(sample_ids) - groups.keys(), groups.keys() - set(sample_ids)
    if missing or unknown:
        raise ValueError(f'Group mapping differs from the source: missing={sorted(missing)}, unknown={sorted(unknown)}')
    if any(not isinstance(value, str) or not value for value in groups.values()):
        raise ValueError('group IDs must be nonempty strings')
    return groups


@dataclass(frozen=True)
class SplitValidationReport:
    """Describe the checked universe, split counts and unassigned sample IDs.

    Counts distinguish samples from groups. ``group_counts`` is None when grouping
    is disabled. ``checks`` identifies enforced policies; a false policy does not
    claim that the data violate it. Containers returned by ``to_dict`` are copies.
    """
    sample_ids: tuple[str, ...]
    sample_counts: Mapping[str, int]
    group_counts: Optional[Mapping[str, int]]
    unassigned_ids: tuple[str, ...]
    checks: Mapping[str, bool]

    def to_dict(self):
        """Return a JSON-compatible copy of the universe, counts and checks."""
        return dict(sample_ids=list(self.sample_ids), sample_counts=dict(self.sample_counts),
                    group_counts=None if self.group_counts is None else dict(self.group_counts),
                    unassigned_ids=list(self.unassigned_ids), checks=dict(self.checks))


def _validate(sample_ids, split_ids, groups, require_complete, require_complete_groups, allow_empty):
    for name, value in (('require_complete', require_complete),
                        ('require_complete_groups', require_complete_groups), ('allow_empty', allow_empty)):
        if type(value) is not bool:
            raise TypeError(f'{name} must be a boolean')
    sample_ids = _ids(sample_ids, 'sample_ids', nonempty=True)
    splits = {name: _ids(values, f'split {name!r}', nonempty=not allow_empty)
              for name, values in _named_mapping(split_ids, 'split_ids').items()}
    if not splits or not any(splits.values()):
        raise ValueError('At least one split must select samples')
    groups = _groups(sample_ids, groups)
    universe, owners, group_owners = set(sample_ids), {}, {}
    for name, values in splits.items():
        for sample_id in values:
            if sample_id not in universe:
                raise ValueError(f'Split {name!r} contains unknown ID {sample_id!r}')
            if sample_id in owners:
                raise ValueError(f'ID {sample_id!r} occurs in splits {owners[sample_id]!r} and {name!r}')
            owners[sample_id] = name
            if groups is not None:
                group = groups[sample_id]
                if group in group_owners and group_owners[group] != name:
                    raise ValueError(f'Group {group!r} occurs in different splits {group_owners[group]!r} and {name!r}')
                group_owners[group] = name
    unassigned = tuple(sample_id for sample_id in sample_ids if sample_id not in owners)
    if require_complete and unassigned:
        raise ValueError(f'Unassigned source IDs: {list(unassigned)}')
    if groups is not None and require_complete_groups:
        partial = [sample_id for sample_id in unassigned if groups[sample_id] in group_owners]
        if partial:
            raise ValueError(f'Selected groups are incomplete; missing IDs: {partial}')
    report = SplitValidationReport(
        sample_ids, MappingProxyType({name: len(values) for name, values in splits.items()}),
        None if groups is None else MappingProxyType({
            name: len({groups[sample_id] for sample_id in values}) for name, values in splits.items()}),
        unassigned, MappingProxyType(dict(unique_ids=True, known_ids=True,
            disjoint_splits=True, complete_coverage=require_complete,
            group_separation=groups is not None,
            complete_groups=groups is not None and require_complete_groups,
            nonempty_splits=not allow_empty)),
    )
    return report, splits, groups


def validate_split_ids(sample_ids, split_ids, groups=None, *, require_complete=False,
                       require_complete_groups=True, allow_empty=False):
    """Validate split membership without reading files or dataset items.

    Args:
        sample_ids: Ordered nonempty source universe of unique nonempty strings.
            IDs are opaque; ``"01"`` and ``"1"`` remain distinct.
        split_ids: Mapping from nonempty split names to ordered ID iterables.
            No ID may occur twice within or across splits.
        groups: Optional complete mapping from source IDs to nonempty string
            group IDs. None disables grouping; an empty mapping does not.
        require_complete: Require every source ID to be assigned.
        require_complete_groups: Require every member of a selected group to be
            assigned, relative to the supplied universe. Groups never cross splits.
        allow_empty: Permit individual empty splits. At least one must be nonempty.

    Returns:
        A SplitValidationReport with source order and unassigned IDs preserved.

    Raises:
        TypeError: An argument has an unsupported container or policy type.
        ValueError: IDs, groups, coverage or split membership are inconsistent.
    """
    return _validate(sample_ids, split_ids, groups, require_complete, require_complete_groups, allow_empty)[0]


def subsets_from_ids(dataset, split_ids, *, sample_ids=None, groups=None,
                     require_complete=False, require_complete_groups=True, allow_empty=False):
    """Return Subset views in exactly the requested split and sample order.

    Use explicit ``sample_ids`` or resolve ``dataset.sample_ids`` and nested Subset
    indices. No call to ``__getitem__`` discovers identities. Repeated source
    positions are invalid for partitioning; training samplers may still repeat
    positions. Remaining arguments follow :func:`validate_split_ids`.

    The original dataset and each item's return type are preserved without copying
    tensors. A dataset without identity metadata requires explicit sample IDs.
    """
    ids = _resolve_sample_ids(dataset, sample_ids)
    _, splits, _ = _validate(ids, split_ids, groups, require_complete, require_complete_groups, allow_empty)
    positions = {sample_id: index for index, sample_id in enumerate(ids)}
    return {name: Subset(dataset, [positions[sample_id] for sample_id in values])
            for name, values in splits.items()}


def split_ids_by_group(sample_ids, groups, group_counts, seed=42):
    """Partition all source groups using an isolated NumPy RandomState.

    Args:
        sample_ids: Ordered unique nonempty string IDs.
        groups: Complete sample-ID to group-ID mapping.
        group_counts: Mapping of split names to positive integer group counts.
            Counts must sum to all groups. Mapping order assigns permuted groups.
        seed: Integer in ``[0, 2**32)``; booleans are rejected.

    Returns:
        Split names mapped to ordered ID lists. Group names are sorted textually
        before permutation; source order is preserved within each chosen group.

    No global Python, NumPy or Torch random state changes. Counts refer to groups,
    not images or percentages. Persist the returned IDs to identify the selection.
    """
    sample_ids = _ids(sample_ids, 'sample_ids', nonempty=True)
    groups = _groups(sample_ids, groups)
    if groups is None:
        raise ValueError('groups must be provided for group splitting')
    counts = _named_mapping(group_counts, 'group_counts')
    if not counts or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in counts.values()):
        raise ValueError('group_counts must contain positive integers excluding booleans')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32), excluding booleans')
    by_group = defaultdict(list)
    for sample_id in sample_ids:
        by_group[groups[sample_id]].append(sample_id)
    names = sorted(by_group)
    if sum(counts.values()) != len(names):
        raise ValueError('group_counts must sum to the number of source groups')
    order = np.random.RandomState(int(seed)).permutation(len(names))
    result, start = {}, 0
    for split, count in counts.items():
        result[split] = [sample_id for index in order[start:start + count]
                         for sample_id in by_group[names[index]]]
        start += count
    return result
