"""Versioned, primitive-only identities for local persistent preparation."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .... import morphology
from ...._native import load_bindings
from ..specs import FeatureSpec, TreeSpec

FORMAT_VERSION = 4
PREPARATION_SEMANTICS = "cfp-cpu-u8-topology-attributes-node-u32-compact-preorder-v4"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def implementation_identity():
    # Binary identity is deliberately conservative across backend builds. The
    # semantic version must change when Python preparation/quantization changes.
    return {"semantics": PREPARATION_SEMANTICS,
            "native_sha256": digest_file(Path(load_bindings().__file__))}


def tree_config(spec):
    return {"tree_type": getattr(spec.tree_type, "value", spec.tree_type),
            "tree_enum": isinstance(spec.tree_type, morphology.TreeType),
            "tos_interpolation": None if spec.tos_interpolation is None else spec.tos_interpolation.name,
            "tos_infinity_seed_row": spec.tos_infinity_seed_row,
            "tos_infinity_seed_col": spec.tos_infinity_seed_col}


def tree_from_config(config):
    config = dict(config)
    enum = config.pop("tree_enum")
    if enum:
        config["tree_type"] = morphology.TreeType(config["tree_type"])
    interpolation = config["tos_interpolation"]
    if interpolation is not None:
        config["tos_interpolation"] = getattr(morphology.ToSInterpolation, interpolation)
    return TreeSpec(**config)


def preprocessor_config(preprocessor):
    from .cfp_preprocessor import CFPPreprocessor
    if type(preprocessor) is not CFPPreprocessor or preprocessor.morphology is not morphology:
        raise ValueError("Persistent preparation currently requires the standard CFPPreprocessor implementation.")
    return {"format_version": FORMAT_VERSION, "implementation": dict(implementation_identity()),
            "attribute_dtype": preprocessor.attribute_dtype.name,
            "trees": [{"tree": tree_config(spec),
                       "attributes": [a.name for a in preprocessor.features[key].attributes]}
                      for key, spec in preprocessor.tree_specs.items()]}


def preprocessor_from_config(config):
    from .cfp_preprocessor import CFPPreprocessor
    if config["format_version"] != FORMAT_VERSION or config["implementation"] != implementation_identity():
        raise ValueError("Incompatible persistent preparation implementation or format.")
    trees, features = {}, {}
    for entry in config["trees"]:
        tree = tree_from_config(entry["tree"])
        trees[tree.cache_key()] = tree
        features[tree.cache_key()] = FeatureSpec(tuple(getattr(morphology.AttributeType, a) for a in entry["attributes"]))
    return CFPPreprocessor(tree_specs=trees, features=features, attribute_dtype=config["attribute_dtype"])


def image_identity(preprocessor, image, tree_key, *, pixel_digest=None):
    config = preprocessor_config(preprocessor)
    return {"format_version": FORMAT_VERSION, "implementation": config["implementation"],
            "pixels_sha256": pixel_digest or hashlib.sha256(memoryview(image)).hexdigest(),
            "shape": list(image.shape), "tree": tree_config(preprocessor.tree_specs[tree_key]),
            "attributes": [a.name for a in preprocessor.features[tree_key].attributes],
            "attribute_dtype": config["attribute_dtype"]}


def identity_key(identity):
    return hashlib.sha256(canonical_json(identity).encode("utf8")).hexdigest()


def validate_identity(identity):
    expected = {"format_version", "implementation", "pixels_sha256", "shape", "tree", "attributes", "attribute_dtype"}
    if set(identity) != expected or identity["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported preparation identity format.")
    if identity["implementation"] != implementation_identity():
        raise ValueError("Preparation backend/semantics do not match this runtime.")
    digest = identity["pixels_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid pixel fingerprint.")
    if len(identity["shape"]) != 2 or any(type(n) is not int or n <= 0 for n in identity["shape"]):
        raise ValueError("Invalid persistent image shape.")
    if identity["attribute_dtype"] not in ("float32", "float64"):
        raise ValueError("Invalid persistent attribute dtype.")
    tree_from_config(identity["tree"])
    attributes = identity["attributes"]
    if not attributes or len(set(attributes)) != len(attributes):
        raise ValueError("Invalid persistent feature order.")
    for attr in attributes:
        getattr(morphology.AttributeType, attr)
    return identity_key(identity)
