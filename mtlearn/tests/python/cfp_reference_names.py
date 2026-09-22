import re


def canonical_reference_names(value):
    if isinstance(value, str):
        return re.sub(r"(?<![A-Z0-9_])BOX_WIDTH(?![A-Z0-9_])", "BOUNDING_BOX_WIDTH", value)
    if isinstance(value, dict):
        return {canonical_reference_names(key): canonical_reference_names(item)
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(canonical_reference_names(item) for item in value)
    return value
