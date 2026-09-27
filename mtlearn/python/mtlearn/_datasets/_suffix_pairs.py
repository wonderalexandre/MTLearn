"""Index literal filename suffixes without decoding images or resolving ambiguities."""

from pathlib import Path
import warnings

from ._image_ops import normalize_extensions


def suffix_pairs(root_dir, *, suffix_in='_in', suffix_target='_target', prefix_in='',
                 prefix_target='', extensions=('.png', '.jpg', '.pgm'),
                 ordering='textual', unmatched='error'):
    if ordering != 'textual':
        raise ValueError("from_suffixes requires ordering='textual'")
    if unmatched not in ('error', 'warn', 'ignore'):
        raise ValueError("unmatched must be 'error', 'warn', or 'ignore'")
    for name, value in (('suffix_in', suffix_in), ('suffix_target', suffix_target),
                        ('prefix_in', prefix_in), ('prefix_target', prefix_target)):
        if not isinstance(value, str) or any(char in value for char in ('/', '\\', '\x00')):
            raise ValueError(f'{name} must be a literal filename fragment')
    if not suffix_in or not suffix_target or suffix_in == suffix_target:
        raise ValueError('Input and target suffixes must be nonempty and distinct')
    if isinstance(extensions, (str, bytes)):
        raise TypeError('extensions must be a sequence of extension strings')
    extensions = tuple(extensions)
    if not extensions or any(not isinstance(ext, str) or not ext for ext in extensions):
        raise ValueError('extensions must contain nonempty strings')
    extensions = normalize_extensions(extensions)
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    indexed = {'input': {}, 'target': {}}
    patterns = (('input', prefix_in, suffix_in), ('target', prefix_target, suffix_target))
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        matches = [(role, prefix, suffix) for role, prefix, suffix in patterns
                   if path.stem.startswith(prefix) and path.stem.endswith(suffix)]
        if len(matches) > 1:
            raise ValueError(f'File matches both input and target roles: {path}')
        if not matches:
            continue
        role, prefix, suffix = matches[0]
        sample_id = path.stem[len(prefix):-len(suffix)]
        if not sample_id:
            raise ValueError(f'Empty sample ID for {role}: {path}')
        if sample_id in indexed[role]:
            raise ValueError(f'Duplicate {role} ID {sample_id!r}: {indexed[role][sample_id]} and {path}')
        indexed[role][sample_id] = str(path.resolve())
    inputs, targets = indexed['input'], indexed['target']
    missing = dict(missing_input_ids=sorted(targets.keys() - inputs.keys()),
                   missing_target_ids=sorted(inputs.keys() - targets.keys()))
    if any(missing.values()):
        message = f'Unmatched image pairs in {root}: {missing}'
        if unmatched == 'error':
            raise ValueError(message)
        if unmatched == 'warn':
            warnings.warn(message, UserWarning, stacklevel=3)
    pairs = [(sample_id, inputs[sample_id], targets[sample_id]) for sample_id in sorted(inputs.keys() & targets.keys())]
    if not pairs:
        raise ValueError(f'No matched image pairs in {root}')
    config = dict(root_dir=str(root), suffix_in=suffix_in, suffix_target=suffix_target,
                  prefix_in=prefix_in, prefix_target=prefix_target, extensions=list(extensions),
                  ordering=ordering, unmatched=unmatched)
    return pairs, missing, config
