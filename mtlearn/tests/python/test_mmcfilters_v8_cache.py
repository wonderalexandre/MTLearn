import gzip
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore, PreparedDataset, collate_prepared
from mtlearn.layers.cfp.storage import _disk_format


FIXTURES = Path(__file__).parent / "fixtures/mmcfilters_v8"


def forbid(*args, **kwargs):
    raise AssertionError("The v1.3.0 cache must be reused without recomputing or summarizing entries")


@pytest.fixture(scope="module", params=["float32", "float64"])
def reference(request):
    path = FIXTURES / f"v1.3.0-{request.param}-cache.pt.gz"
    with gzip.open(path, "rb") as source:
        result = torch.load(source, weights_only=True, map_location="cpu")
    result["cache_files"] = {name: value.numpy().tobytes() for name, value in result["cache_files"].items()}
    return result


def test_legacy_attribute_ordinals_are_unchanged(reference):
    assert reference["source_revision"] == "caa614ca92239b71ae13c15f333328e3ecf35961"
    assert reference["mmcfilters_revision"] == "c8c226ce945fe4038350c5e7003e0f55a9008f2c"
    assert len(reference["enum_values"]) == 133
    for name, value in reference["enum_values"].items():
        assert int(getattr(morphology.AttributeType, name)) == value
    assert int(morphology.AttributeType.GRAY_LEVEL) == 132
    assert int(morphology.AttributeType.AXIS_ORIENTATION_SIGNED) == 133
    assert int(morphology.AttributeType.FILLED_AXIS_ORIENTATION_SIGNED) == 134
    assert int(morphology.AttributeType.RAW_INERTIA) == 135


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_real_v130_disk_cache_reuse_preserves_payloads_outputs_and_gradients(reference, tmp_path, monkeypatch, device):
    if device == "mps" and reference["preprocessor_config"]["attribute_dtype"] == "float64":
        pytest.skip("MPS does not support float64")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    for name, content in reference["cache_files"].items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    source = TensorDataset(reference["images"], reference["targets"])
    layer = Layer.from_config(reference["layer_config"], device=device)
    layer.load_state_dict(reference["layer_state"])
    preprocessor = CFPPreprocessor.from_config(reference["preprocessor_config"])
    assert preprocessor.get_config() == reference["preprocessor_config"]
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
    monkeypatch.setattr(_disk_format, "summarize", forbid)
    options = reference["options"]
    result = preprocessor.prepare_or_reuse(path=tmp_path, mode="reuse",
        collect_stats=layer.get_statistics_contract(), **options)
    assert result.status == "complete"
    assert result.prepared_entries == result.repaired_entries == 0
    layer.set_stats(result.statistics)
    with DiskStore(tmp_path, readonly=True) as store:
        dataset = PreparedDataset(store, "train", source)
        batch, targets = collate_prepared([dataset[index] for index in range(len(dataset))])
        for sample, expected in zip(batch.samples, reference["attributes"]):
            assert set(sample[0]) == set(expected)
            for tree_key, tree in sample[0].items():
                assert len(tree.raw_attributes) == 133
                for attribute, values in tree.raw_attributes.items():
                    torch.testing.assert_close(values, expected[tree_key][attribute.name], rtol=0, atol=0)
        output = layer(batch)
        loss = (output / 255).square().mean()
        loss.backward()
        tolerance = dict(rtol=1e-4, atol=1e-5) if device == "mps" else dict(rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(output.cpu(), reference["output"], **tolerance)
        torch.testing.assert_close(loss.cpu(), reference["loss"], **tolerance)
        for key, parameter in layer.named_parameters():
            torch.testing.assert_close(parameter.grad.cpu(), reference["gradients"][key], **tolerance)
        del batch, dataset, tree, sample, values
    for name, content in reference["cache_files"].items():
        assert (tmp_path / name).read_bytes() == content


def test_new_preparation_matches_all_legacy_cached_attributes(reference):
    preprocessor = CFPPreprocessor.from_config(reference["preprocessor_config"])
    batch = preprocessor.prepare_batch(reference["images"])
    tolerance = dict(rtol=1e-6, atol=1e-7) if preprocessor.attribute_dtype.name == "float32" else dict(rtol=1e-12, atol=1e-12)
    for sample, expected in zip(batch.samples, reference["attributes"]):
        for tree_key, tree in sample[0].items():
            for attribute, values in tree.raw_attributes.items():
                torch.testing.assert_close(values, expected[tree_key][attribute.name], **tolerance)
