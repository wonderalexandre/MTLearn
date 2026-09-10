"""Restricted tensor serialization and scorer-independent scalar summaries."""
import hashlib

import torch

from .... import morphology
from ..normalization import AttributeNormalizer
from ..preparation._identity import canonical_json, tree_config, tree_from_config, validate_identity
from ..preparation.prepared_morphology import PreparedMorphology
from ..specs import FeatureSpec


def summarize(prepared):
    summaries = {}
    zscore = AttributeNormalizer("dataset_zscore")
    minmax = AttributeNormalizer("dataset_minmax01")
    for attr, values in prepared.raw_attributes.items():
        moments = zscore.summarize(attr.name, values.view(-1))[attr.name]
        bounds = minmax.summarize(attr.name, values.view(-1))[attr.name]
        summaries[attr.name] = {key: value.item() for key, value in (moments | bounds).items()}
    # Reject non-finite summaries before publishing a reusable artifact.
    canonical_json(summaries)
    return summaries


def match_prepared(identity, prepared):
    validate_identity(identity)
    if (identity["tree"] != tree_config(prepared.tree_spec)
            or identity["shape"] != list(prepared.image_shape)
            or identity["attributes"] != [a.name for a in prepared.feature_spec.attributes]
            or any(str(t.dtype) != "torch." + identity["attribute_dtype"] for t in prepared.raw_attributes.values())):
        raise ValueError("Prepared morphology does not match the persistent identity.")


def tensor_digest(tensor):
    return hashlib.sha256(memoryview(tensor.detach().numpy())).hexdigest()


def pack(identity, prepared):
    prepared.validate(full=True)
    match_prepared(identity, prepared)
    info = {key: value for key, value in prepared.info.items() if key != "tree_type"}
    attributes = {a.name: tensor for a, tensor in prepared.raw_attributes.items()}
    hashes = {key: tensor_digest(tensor) for key, tensor in info.items() if torch.is_tensor(tensor)}
    hashes.update({"attr:" + key: tensor_digest(tensor) for key, tensor in attributes.items()})
    summary = summarize(prepared)
    return {"format_version": 2, "identity": identity, "info": info, "attributes": attributes,
            "summary": summary, "tensor_sha256": hashes,
            "summary_sha256": hashlib.sha256(canonical_json(summary).encode()).hexdigest()}


def unpack(data, *, full=False):
    expected = {"format_version", "identity", "info", "attributes", "summary", "tensor_sha256", "summary_sha256"}
    if not isinstance(data, dict) or set(data) != expected or data["format_version"] != 2:
        raise ValueError("Unsupported persistent morphology format.")
    identity = data["identity"]
    validate_identity(identity)
    tree = tree_from_config(identity["tree"])
    prepared = PreparedMorphology(tree, FeatureSpec(tuple(getattr(morphology.AttributeType, a)
                                                         for a in identity["attributes"])),
        dict(data["info"], tree_type=tree.tree_type),
        {getattr(morphology.AttributeType, a): t for a, t in data["attributes"].items()})
    match_prepared(identity, prepared)
    prepared.validate(full=full)
    summary = data["summary"]
    if set(summary) != set(identity["attributes"]):
        raise ValueError("Persistent summaries do not match the attributes.")
    for values in summary.values():
        if set(values) != {"count", "sum", "sumsq", "amin", "amax"} or values["count"] != prepared.num_nodes:
            raise ValueError("Invalid persistent node statistics.")
        if type(values["count"]) is not int or values["amin"] > values["amax"] or values["sumsq"] < 0:
            raise ValueError("Invalid persistent scalar statistics.")
    if hashlib.sha256(canonical_json(summary).encode()).hexdigest() != data["summary_sha256"]:
        raise ValueError("Persistent summary checksum mismatch.")
    if full:
        actual = {key: tensor_digest(t) for key, t in data["info"].items() if torch.is_tensor(t)}
        actual.update({"attr:" + key: tensor_digest(t) for key, t in data["attributes"].items()})
        if actual != data["tensor_sha256"]:
            raise ValueError("Persistent tensor checksum mismatch.")
    return prepared, summary
