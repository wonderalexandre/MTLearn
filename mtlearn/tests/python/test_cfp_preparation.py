"""P1 contracts: independent preparation, streaming fit and frozen statistics."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import weakref

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import mtlearn
from mtlearn import morphology

pytestmark = pytest.mark.integration
if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

from mtlearn.layers import ConnectedFilterPreprocessingLayer, load_checkpoint
from mtlearn.layers.cfp import CFPPreprocessor, PreparedMorphology, StatisticsSnapshot
from mtlearn.layers.cfp.normalization import AttributeNormalizer
from mtlearn.layers.cfp.normalization import attribute_statistics

FIXTURES = Path(__file__).parent / "fixtures/cfp_cache_p0"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.fixture(scope="module")
def baseline():
    return torch.load(FIXTURES / "baseline.pt", weights_only=True, map_location="cpu")


def loader(images, *, batch_size=2, **kwargs):
    return DataLoader(TensorDataset(images, torch.zeros(len(images))), batch_size=batch_size, **kwargs)


def model(*, mode="dataset_clipped_zscore01", tree="max-tree", scoring="linear_sigmoid", channels=1):
    return ConnectedFilterPreprocessingLayer(channels, [{"tree_type": tree,
        "attributes": [morphology.AttributeType.AREA, morphology.AttributeType.GRAY_HEIGHT],
        "scoring": {"kind": scoring}}], scale_mode=mode, clamp=12)


def assert_stats(actual, expected):
    assert actual.keys() == expected.keys()
    for key, values in expected.items():
        assert actual[key].keys() == values.keys()
        for name, value in values.items():
            assert actual[key][name].device.type == "cpu"
            torch.testing.assert_close(actual[key][name], value.cpu(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", MANIFEST["cases"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_streaming_fit_matches_frozen_statistics_outputs_and_gradients(baseline, name, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    case = baseline["cases"][name]
    layer, _ = load_checkpoint(FIXTURES / case["checkpoint"],
        lambda configs: ConnectedFilterPreprocessingLayer.from_config(configs[""], device=device), device=device)
    snapshot = layer.fit_stats(loader(baseline["images"][:3]))
    assert isinstance(snapshot, StatisticsSnapshot) and snapshot.sample_count == 3
    assert layer.cached_sample_count() == 0 and layer._stats_frozen
    assert_stats(snapshot.statistics, case["stats"])
    assert_stats(layer.get_stats().statistics, case["stats"])
    output = layer(baseline["images"][3:])
    loss = ((output / 255.0 - baseline["targets"][3:].to(device)) ** 2).mean()
    loss.backward()
    tolerance = MANIFEST["cpu_mps_tolerance" if device == "mps" else "same_device_tolerance"]
    torch.testing.assert_close(output.cpu(), case["output"], **tolerance)
    torch.testing.assert_close(loss.cpu(), case["loss"], **tolerance)
    for param_name, parameter in layer.named_parameters():
        torch.testing.assert_close(parameter.grad.cpu(), case["gradients"][param_name], **tolerance)
    assert_stats(layer.get_stats().statistics, case["stats"])


@pytest.mark.parametrize("mode", ["none", "dataset_minmax01", "dataset_zscore", "dataset_clipped_zscore01"])
def test_snapshots_combine_and_share_between_scorers_without_aliasing(baseline, mode):
    linear, mlp = model(mode=mode), model(mode=mode, scoring="mlp")
    images = baseline["images"][:3]
    whole = linear.fit_stats(loader(images))
    first = linear.fit_stats(loader(images[:2]))
    second = linear.fit_stats(loader(images[2:]))
    combined = StatisticsSnapshot.combine(iter([first, second]))
    assert combined.sample_count == 3
    assert_stats(combined.statistics, whole.statistics)
    mlp.set_stats(combined)
    assert_stats(mlp.get_stats().statistics, whole.statistics)
    # Public exports and the installed model have separate scalar storage.
    exported = combined.statistics
    for values in exported.values():
        for value in values.values():
            value.zero_()
    assert_stats(mlp.get_stats().statistics, whole.statistics)
    assert_stats(combined.statistics, whole.statistics)
    assert mlp.cached_sample_count() == 0
    with pytest.raises(TypeError):
        combined.contract["eps"] = 42


@pytest.mark.parametrize("tree", ["max-tree", "min-tree", "tree-of-shapes"])
@pytest.mark.parametrize("constant", [False, True])
def test_prepared_schema_full_validation_and_scorer_independence(baseline, tree, constant):
    layer = model(tree=tree)
    prep = CFPPreprocessor.from_layer(layer)
    layer_ref = weakref.ref(layer)
    del layer
    gc.collect()  # The legacy layer/provider can have an internal reference cycle.
    assert layer_ref() is None  # No bound model/normalizer/scorer survives extraction.
    image = torch.zeros((1, 1)) if constant else baseline["images"][0, 0]
    prepared = prep.prepare_image(image, input_id="sample/channel")
    assert isinstance(prepared, PreparedMorphology)
    prepared.validate(full=True)
    assert prepared.image_shape == tuple(image.shape)
    assert prepared.input_id == "sample/channel"
    assert not hasattr(prep, "normalizer") and not hasattr(prep, "device")
    assert all(value.device.type == "cpu" and not value.requires_grad for value in prepared.raw_attributes.values())
    assert "norm_attrs" not in prepared.info
    with pytest.raises(TypeError):
        prepared.info["num_rows"] = 2
    with pytest.raises(ValueError, match="wrong length"):
        replace(prepared, info={**prepared.info, "node_of_pixel": torch.zeros(image.numel()+1, dtype=torch.uint32)})
    with pytest.raises(ValueError, match="dtype"):
        replace(prepared, info={**prepared.info, "parent": prepared.info["parent"].int()})
    broken = replace(prepared, info={**prepared.info, "node_of_pixel": torch.full_like(prepared.info["node_of_pixel"], 2**32 - 1)})
    with pytest.raises(ValueError, match="invalid node"):
        broken.validate(full=True)


def test_prepared_byte_accounting_deduplicates_views(baseline):
    prepared = CFPPreprocessor.from_layer(model()).prepare_image(baseline["images"][0, 0])
    attrs = list(prepared.raw_attributes)
    first = prepared.raw_attributes[attrs[0]]
    second = prepared.raw_attributes[attrs[1]]
    shared = replace(prepared, raw_attributes={attrs[0]: first, attrs[1]: first.view_as(first)})
    assert shared.nbytes == prepared.nbytes - second.untyped_storage().nbytes()


def test_fit_releases_each_payload_and_does_not_grow_legacy_cache(baseline, monkeypatch):
    layer = model()
    previous = []
    visits = 0
    original = CFPPreprocessor.prepare_u8
    def tracked(self, *args, **kwargs):
        nonlocal visits
        assert all(reference() is None for reference in previous)
        prepared = original(self, *args, **kwargs)
        previous[:] = [weakref.ref(prepared), weakref.ref(next(iter(prepared.raw_attributes.values())))]
        visits += 1
        return prepared
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", tracked)
    images = baseline["images"][:1].repeat(12, 1, 1, 1)
    snapshot = layer.fit_stats(loader(images, batch_size=1))
    assert visits == 12 and snapshot.sample_count == 12
    assert all(reference() is None for reference in previous)
    assert layer.cached_sample_count() == 0


def test_fit_obeys_sampler_multiplicity_drop_last_and_replaces_previous_fit(baseline):
    layer = model()
    images = baseline["images"][:3]
    actual = layer.fit_stats(loader(images, sampler=[2, 0, 2], drop_last=True))
    expected = model().fit_stats(loader(images[[2, 0]]))
    assert actual.sample_count == 2
    assert_stats(actual.statistics, expected.statistics)
    repeated = layer.fit_stats(loader(images, sampler=[2, 0, 2], drop_last=True))
    assert_stats(repeated.statistics, expected.statistics)
    duplicate = layer.fit_stats(loader(images[[0, 0]]))
    single = model().fit_stats(loader(images[[0]]))
    for key in single.statistics:
        assert duplicate.statistics[key]["count"] == 2 * single.statistics[key]["count"]


def test_failed_empty_or_incompatible_fit_keeps_previous_state_and_cache(baseline):
    layer = model()
    cached = layer.build_dataloader_cached(loader(baseline["images"][:3]))
    before = layer.get_stats()
    before_count = layer.cached_sample_count()
    before_epoch = layer._stats_epoch
    good = (baseline["images"][:1], None)
    bad = (torch.full_like(baseline["images"][:1], float("nan")), None)
    with pytest.raises(ValueError, match="finite"):
        layer.fit_stats(iter([good, bad]))
    with pytest.raises(ValueError, match="non-empty"):
        layer.fit_stats(iter([]))
    incompatible = model(mode="dataset_minmax01").fit_stats(loader(baseline["images"][:3]))
    with pytest.raises(ValueError, match="incompatible"):
        layer.set_stats(incompatible)
    assert layer._stats_epoch == before_epoch and layer.cached_sample_count() == before_count
    assert_stats(layer.get_stats().statistics, before.statistics)
    indexed_input, _ = next(iter(cached))
    assert torch.isfinite(layer(indexed_input)).all()


def test_new_fit_invalidates_existing_normalized_cache_without_rebuilding(baseline, monkeypatch):
    layer = model()
    cached = layer.build_dataloader_cached(loader(baseline["images"][:2]))
    new_stats = model().fit_stats(loader(baseline["images"][2:3]))
    layer.set_stats(new_stats)
    assert layer.cached_sample_count() == 2
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached morphology must not be rebuilt after a statistics change")
    monkeypatch.setattr(layer._tree_payload_provider, "prepare_u8", forbidden)
    indexed_input, _ = next(iter(cached))
    cached_output = layer(indexed_input)
    other = model()
    other.load_state_dict(layer.state_dict())
    torch.testing.assert_close(cached_output, other(baseline["images"][:2]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_normalization_matches_formula_and_reuses_moments_across_dtypes(dtype, monkeypatch):
    normalizer = AttributeNormalizer(clipped_zscore_radius=2.0, clipped_zscore_floor=0.1)
    fit = torch.tensor([1., 3., 5.])
    normalizer.update("x", fit)
    calls = []
    original = attribute_statistics._zscore_moments
    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(attribute_statistics, "_zscore_moments", measured)
    normalizer.freeze()
    normalizer.prepare_constants(dtype=dtype)
    values = torch.tensor([-10., 2., 100.], dtype=dtype)
    std = torch.sqrt(torch.tensor(8/3, dtype=torch.float64)).to(dtype)
    expected = 0.1 + 0.9 * ((torch.clamp((values - 3) / std, -2, 2) + 2) / 4)
    for _ in range(3):
        torch.testing.assert_close(normalizer.normalize("x", values), expected)
    normalizer.prepare_constants(dtype=torch.float64 if dtype == torch.float32 else torch.float32)
    assert len(calls) == 1
    normalizer.unfreeze()
    normalizer.update("x", torch.tensor([20.]))
    normalizer.freeze()
    normalizer.normalize("x", values)
    assert len(calls) == 2


def test_training_does_not_update_statistics_or_recompute_moments(baseline, monkeypatch):
    layer = model()
    snapshot = layer.fit_stats(loader(baseline["images"][:3]))
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen training must not update or recompute global statistics")
    monkeypatch.setattr(AttributeNormalizer, "update", forbidden)
    monkeypatch.setattr(AttributeNormalizer, "merge", forbidden)
    monkeypatch.setattr(attribute_statistics, "_zscore_moments", forbidden)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        layer(baseline["images"][3:]).mean().backward()
        optimizer.step()
    assert_stats(layer.get_stats().statistics, snapshot.statistics)


def test_streaming_multichannel_shared_attributes_and_multiple_trees_match_legacy(baseline):
    specs = [{"tree_type": "max-tree", "attributes": [morphology.AttributeType.AREA]},
             {"tree_type": "max-tree", "attributes": [morphology.AttributeType.AREA, morphology.AttributeType.GRAY_HEIGHT]},
             {"tree_type": "min-tree", "attributes": [morphology.AttributeType.AREA]}]
    images = baseline["images"][:3].repeat(1, 2, 1, 1)
    legacy = ConnectedFilterPreprocessingLayer(2, specs)
    streamed = ConnectedFilterPreprocessingLayer(2, specs)
    legacy.build_dataloader_cached(loader(images))
    assert_stats(streamed.fit_stats(loader(images)).statistics, legacy.get_stats().statistics)
    assert streamed.cached_sample_count() == 0


def test_invalid_snapshot_and_summary_are_rejected_before_mutation(baseline):
    layer = model()
    good = layer.fit_stats(loader(baseline["images"][:3]))
    invalid = good.statistics
    key = next(iter(invalid))
    invalid[key]["count"] = torch.tensor(0)
    with pytest.raises(ValueError, match="positive"):
        StatisticsSnapshot(good.contract, invalid)
    normalizer = AttributeNormalizer()
    normalizer.merge(good.statistics)
    before = normalizer.stats_epoch
    with pytest.raises(ValueError, match="positive"):
        normalizer.merge(invalid)
    assert normalizer.stats_epoch == before
    assert_stats(normalizer.ds_stats, good.statistics)
    with pytest.raises(ValueError, match="incompatible"):
        StatisticsSnapshot.combine([good, model(mode="none").fit_stats(loader(baseline["images"][:3]))])


@pytest.mark.parametrize("factory", [False, True])
def test_checkpoint_target_device_applies_to_cpu_constructed_layer(baseline, factory):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    case = baseline["cases"]["max-tree_linear_dataset_clipped_zscore01"]
    target = (lambda configs: ConnectedFilterPreprocessingLayer.from_config(configs[""], device="cpu")) if factory else ConnectedFilterPreprocessingLayer.from_config(case["config"], device="cpu")
    layer, checkpoint = load_checkpoint(FIXTURES / case["checkpoint"], target, device="mps")
    assert layer.device.type == "mps" and next(layer.parameters()).device.type == "mps"
    output = layer(baseline["images"][3:])
    torch.testing.assert_close(output.cpu(), case["output"], **MANIFEST["cpu_mps_tolerance"])
    assert_stats(layer.get_stats().statistics, case["stats"])


def test_changing_normalization_policy_discards_previous_constants():
    normalizer = AttributeNormalizer()
    values = torch.tensor([1., 2., 3.])
    normalizer.update("x", values)
    normalizer.freeze()
    for radius in (1., 2., 3.):
        normalizer.clipped_zscore_radius = radius
        output = normalizer.normalize("x", values)
        expected = attribute_statistics._normalize_dataset_clipped_zscore01(
            normalizer.ds_stats, normalizer.eps, "x", values, clipped_zscore_radius=radius)
        torch.testing.assert_close(output, expected)
        assert len(normalizer._constants) == 1


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_restored_frozen_statistics_are_ready_before_forward(baseline, monkeypatch, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    case = baseline["cases"]["max-tree_linear_dataset_clipped_zscore01"]
    layer, _ = load_checkpoint(FIXTURES / case["checkpoint"],
        lambda configs: ConnectedFilterPreprocessingLayer.from_config(configs[""], device=device), device=device)
    layer.load_stats(FIXTURES / "max-tree_linear_dataset_clipped_zscore01.stats.pt")
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen checkpoint/statistics loading must prepare constants before forward")
    monkeypatch.setattr(attribute_statistics, "_zscore_moments", forbidden)
    output = layer(baseline["images"][3:])
    torch.testing.assert_close(output.cpu(), case["output"], **MANIFEST["cpu_mps_tolerance"])


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_checkpoint_restores_parameterless_custom_scorer(baseline, monkeypatch, tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    from mtlearn.layers import save_checkpoint
    from mtlearn.layers.cfp import SCORING_MODEL_REGISTRY, ScoringModel
    class ConstantScorer(ScoringModel):
        def __init__(self, num_features):
            super().__init__()
            self.num_features = num_features
        def forward(self, features, tree_info=None, context=None, **kwargs):
            return torch.ones(features.size(0), dtype=features.dtype, device=features.device)
        def to_config(self):
            return {"kind": "p1_constant"}
    monkeypatch.setattr(SCORING_MODEL_REGISTRY, "_factories", dict(SCORING_MODEL_REGISTRY._factories))
    SCORING_MODEL_REGISTRY.register("p1_constant", ConstantScorer)
    layer = model(mode="none", scoring="p1_constant")
    assert not list(layer.parameters())
    layer.build_dataloader_cached(loader(baseline["images"][:1]))
    expected = layer(baseline["images"][:1])
    path = tmp_path / "constant.pt"
    save_checkpoint(path, layer)
    restored, _ = load_checkpoint(path, layer, device=device)
    assert restored.device.type == device
    torch.testing.assert_close(restored(baseline["images"][:1]).cpu(), expected, **MANIFEST["cpu_mps_tolerance"])
