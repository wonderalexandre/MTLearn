"""P2: prepared execution, bounded ownership and immutable P0 references."""
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch

import mtlearn
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer, load_checkpoint
from mtlearn.layers.cfp import CFPPreprocessor, PreparedBatch, MemoryStore, NullStore, FeatureSpec
from mtlearn.layers.cfp.normalization import AttributeNormalizer

pytestmark = pytest.mark.integration
if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

FIXTURES = Path(__file__).parent / "fixtures/cfp_cache_p0_mmcfilters_v5_2_0"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
AREA, HEIGHT = morphology.AttributeType.AREA, morphology.AttributeType.GRAY_LEVEL_HEIGHT


@pytest.fixture(scope="module")
def baseline():
    return torch.load(FIXTURES / "baseline.pt", weights_only=True, map_location="cpu")


def model(*, channels=1, scoring="linear_sigmoid", mode="none", attrs=(AREA, HEIGHT), tree="max-tree", **kwargs):
    return Layer(channels, [{"tree_type": tree, "attributes": attrs,
                            "scoring": {"kind": scoring}}], scale_mode=mode, **kwargs)


def forbid(*a, **kw):
    raise AssertionError("Prepared consumption must not prepare trees or update statistics")


def first(batch):
    return next(iter(batch.samples[0][0].values()))


def assert_stats(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        for name in expected[key]:
            torch.testing.assert_close(actual[key][name], expected[key][name], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", MANIFEST["cases"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("policy", ["temporary", "ram", "oversize"])
def test_prepared_outputs_and_gradients_match_p0_after_eviction(baseline, name, device, policy, monkeypatch):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    case = baseline["cases"][name]
    layer, _ = load_checkpoint(FIXTURES / case["checkpoint"],
        lambda configs: Layer.from_config(configs[""], device=device), device=device)
    prep = CFPPreprocessor.from_layer(layer)
    store = NullStore() if policy == "temporary" else MemoryStore(16 * 1024**2 if policy == "ram" else 0)
    images = baseline["images"][3:]
    batch = prep.prepare_batch(images, store=store, sample_ids=("100", "101"))
    if policy == "ram":
        pointers = [first(batch).info["node_of_pixel"].data_ptr()]
        batch = prep.prepare_batch(images, store=store)
        assert store.info()["hits"] == 2
        assert first(batch).info["node_of_pixel"].data_ptr() == pointers[0]
    elif policy == "oversize":
        assert store.info()["oversized_entries"] == 2
        assert store.info()["retained_bytes"] == 0
    batch_ref = weakref.ref(batch)
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
    for method in ("update", "summarize", "merge"):
        monkeypatch.setattr(AttributeNormalizer, method, forbid)
    output = layer.forward_prepared(batch)
    store.clear()
    del batch
    assert batch_ref() is not None and store.info()["active_bytes"] > 0
    assert store.info()["retained_bytes"] == 0
    loss = ((output / 255 - baseline["targets"][3:].to(device)) ** 2).mean()
    loss.backward()
    assert batch_ref() is None and store.info()["active_bytes"] == 0
    tol = MANIFEST["cpu_mps_tolerance" if device == "mps" else "same_device_tolerance"]
    torch.testing.assert_close(output.cpu(), case["output"], **tol)
    torch.testing.assert_close(loss.cpu(), case["loss"], **tol)
    for key, parameter in layer.named_parameters():
        torch.testing.assert_close(parameter.grad.cpu(), case["gradients"][key], **tol)
    assert_stats(layer.get_stats().statistics, case["stats"])
    assert layer.cached_sample_count() == 0
    assert layer.get_parameter_contract() == case["contracts"]["parameter_contract"]
    assert layer.get_inference_contract() == case["contracts"]["inference_contract"]


def test_default_skips_hashing_and_does_not_retain(baseline, monkeypatch):
    prep = CFPPreprocessor.from_layer(model())
    monkeypatch.setattr(hashlib, "sha256", forbid)
    batch = prep.prepare_batch(baseline["images"][:1])
    ref = weakref.ref(first(batch))
    del batch
    assert ref() is None


@pytest.mark.parametrize("budget", [-1, True, 1.5, "100", None])
def test_memory_budget_must_be_explicit_nonnegative_integer(budget):
    with pytest.raises(ValueError, match="max_bytes"):
        MemoryStore(budget)


def test_lru_counts_shared_storages_and_drops_only_store_owner(baseline):
    prep = CFPPreprocessor.from_layer(model())
    a = prep.prepare_image(baseline["images"][0, 0])
    b = prep.prepare_image(baseline["images"][1, 0])
    # Count the full backing allocation behind contiguous views, across entries.
    views = replace(a, raw_attributes={key: value.view_as(value) for key, value in a.raw_attributes.items()})
    c = prep.prepare_image(baseline["images"][2, 0])
    store = MemoryStore(a.nbytes + max(b.nbytes, c.nbytes))
    one = store.put("a", a)
    two = store.put("views", views)
    assert store.info()["retained_bytes"] == a.nbytes
    assert store.info()["active_bytes"] == a.nbytes
    three = store.put("b", b)
    assert store.info()["retained_bytes"] == a.nbytes + b.nbytes
    del one, two, three
    assert store.info()["active_bytes"] == 0
    borrowed = store.get("a")  # views becomes least-recently-used.
    store.put("c", c)
    assert store.get("views") is None and store.get("b") is None
    assert store.get("a") is not None
    assert store.info()["evictions"] == 2
    store.clear()
    assert store.info()["retained_bytes"] == 0
    assert store.info()["active_not_retained_bytes"] == a.nbytes
    assert torch.equal(borrowed.info["residues"], a.info["residues"])
    del borrowed
    assert store.info()["live_bytes"] == 0


def test_replacement_and_oversize_do_not_corrupt_accounting(baseline):
    prep = CFPPreprocessor.from_layer(model())
    prepared = prep.prepare_image(baseline["images"][0, 0])
    store = MemoryStore(prepared.nbytes)
    store.put("one", prepared)
    store.put("one", prepared)
    assert store.info()["retained_bytes"] == prepared.nbytes
    larger = replace(prepared, raw_attributes={k: torch.cat([v, v]).flatten()[:v.numel()].reshape_as(v)
                                              for k, v in prepared.raw_attributes.items()})
    assert larger.nbytes > prepared.nbytes
    borrowed = store.put("large", larger)
    assert store.info()["entries"] == 1 and store.get("one") is not None
    assert store.info()["oversized_entries"] == 1
    assert store.info()["live_bytes"] > store.info()["retained_bytes"]
    with pytest.raises(TypeError):
        store.put([], prepared)
    with pytest.raises(TypeError):
        store.put("bad", torch.zeros(1))
    assert store.info()["entries"] == 1
    del borrowed


def test_retention_is_bounded_as_number_of_images_grows(baseline):
    prep = CFPPreprocessor.from_layer(model())
    prototype = prep.prepare_image(baseline["images"][0, 0])
    budget = 2 * prototype.nbytes
    store = MemoryStore(budget)
    for index in range(80):
        image = torch.full((1, 1, 6, 7), index, dtype=torch.uint8)
        batch = prep.prepare_batch(image, store=store)
        assert store.info()["retained_bytes"] <= budget
        assert store.info()["active_entries"] == 1
        del batch
        assert store.info()["active_entries"] == 0
    assert store.info()["evictions"] > 0 and store.info()["entries"] < 80


def test_pixel_identity_ignores_indices_but_rejects_changed_content_and_options(baseline):
    layer = model()
    prep = CFPPreprocessor.from_layer(layer)
    store = MemoryStore(1024**2)
    image = baseline["images"][:1]
    batch = prep.prepare_batch(image, store=store, sample_ids=("old",))
    again = prep.prepare_batch(image.clone(), store=store, sample_ids=("renamed",))
    assert first(batch).info["tpre"].data_ptr() == first(again).info["tpre"].data_ptr()
    assert again.sample_key(0, 0) == "renamed_0"
    # Reorder dataset access; the same index must not authorize reuse.
    changed = prep.prepare_batch(1 - image, store=store, sample_ids=("old",))
    assert first(changed).info["tpre"].data_ptr() != first(batch).info["tpre"].data_ptr()
    prep.prepare_batch(image[:, :, :3, :3], store=store)
    for other in (model(attrs=(AREA,)), model(tree="min-tree"), model(attribute_dtype=np.float64)):
        CFPPreprocessor.from_layer(other).prepare_batch(image, store=store)
    assert store.info()["hits"] == 1 and store.info()["misses"] == 6
    # Canonical uint8 equality, independent of the source numeric representation.
    uint8 = torch.tensor([[[[0, 128], [255, 64]]]], dtype=torch.uint8)
    prep.prepare_batch(uint8, store=store)
    prep.prepare_batch(uint8.float() / 255, store=store)
    assert store.info()["hits"] == 2


def test_linear_and_mlp_share_preparation_but_keep_parameters_and_stats_independent(baseline, monkeypatch):
    linear = model(mode="dataset_zscore")
    mlp = model(mode="dataset_zscore", scoring="mlp")
    snapshot = linear.fit_stats([baseline["images"][:3]])
    mlp.set_stats(snapshot)
    batch = CFPPreprocessor.from_layer(linear).prepare_batch(baseline["images"][3:])
    before = {key: value.clone() for key, value in first(batch).raw_attributes.items()}
    direct = {"linear": linear(baseline["images"][3:]), "mlp": mlp(baseline["images"][3:])}
    mlp_params = {key: p.detach().clone() for key, p in mlp.named_parameters()}
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
    for name, layer in (("linear", linear), ("mlp", mlp)):
        torch.testing.assert_close(layer(batch), direct[name])
    optimizer = torch.optim.SGD(linear.parameters(), lr=1e-3)
    linear(batch).mean().backward()
    optimizer.step()
    for key, p in mlp.named_parameters():
        torch.testing.assert_close(p, mlp_params[key])
    for key, value in before.items():
        torch.testing.assert_close(first(batch).raw_attributes[key], value)
    # New frozen constants affect consumption without changing raw preparation.
    monkeypatch.undo()
    linear.fit_stats([baseline["images"][1:2]])
    expected = linear(baseline["images"][3:])
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
    torch.testing.assert_close(linear(batch), expected)
    assert_stats(mlp.get_stats().statistics, snapshot.statistics)


@pytest.mark.parametrize("scoring", ["linear_sigmoid", "mlp"])
def test_multichannel_multitree_extensions_predict_and_inspect(baseline, scoring, monkeypatch):
    specs = [{"name": f"spec{i}", "tree_type": tree, "attributes": attrs,
              "scoring": {"kind": scoring}, "constraints": ["preserve_root"],
              "regularizers": [{"kind": kind, "weight": 0.25}]}
             for i, (tree, attrs, kind) in enumerate([
                 ("max-tree", (AREA,), "edge_score_monotonicity"),
                 ("min-tree", (HEIGHT, AREA), "attribute_order_score_monotonicity"),
                 ("max-tree", (AREA, HEIGHT), "path_score_monotonicity")])]
    layer = Layer(2, specs, scale_mode="dataset_clipped_zscore01")
    images = torch.cat((baseline["images"], 1 - baseline["images"]), dim=1)
    layer.fit_stats([images[:3]])
    images = images[3:]
    direct, penalty = layer(images), layer.regularization_penalty(images)
    (direct.square().mean() + penalty).backward()
    gradients = {key: p.grad.clone() for key, p in layer.named_parameters()}
    layer.zero_grad(set_to_none=True)
    expected_predict = layer.predict(images, score_sharpness=4)
    expected_inspect = layer.inspect_training_sample(images[1], channel=1)
    prep = CFPPreprocessor.from_layer(layer)
    batch = prep.prepare_batch(images, sample_ids=("test-a", "test-b"))
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
    contexts = []
    # Capture small context metadata without retaining its tensor payloads.
    def capture(module, args, kwargs):
        c = kwargs["context"]
        contexts.append((c.sample_key, c.batch_index, c.channel_index, c.spec_index, c.mode, c.is_training,
                         c.attribute_names, c.image_shape, c.score_sharpness))
        assert c.raw_attributes and c.normalized_attributes
    hooks = [scorer.register_forward_pre_hook(capture, with_kwargs=True) for scorer in layer._scoring_models.values()]
    actual, actual_penalty = layer(batch), layer.regularization_penalty(batch)
    assert actual.shape == (2, 6, 6, 7)
    torch.testing.assert_close(actual, direct)
    torch.testing.assert_close(actual_penalty, penalty)
    (actual.square().mean() + actual_penalty).backward()
    for key, p in layer.named_parameters():
        torch.testing.assert_close(p.grad, gradients[key])
    assert [(c[0], c[3]) for c in contexts[:12]] == [
        (f"test-{sample}_{ch}", spec) for sample in ("a", "b") for ch in range(2) for spec in range(3)]
    assert {c[4] for c in contexts} == {"forward", "regularization_penalty"}
    torch.testing.assert_close(layer.predict(batch, score_sharpness=4), expected_predict)
    assert layer.training and layer._score_sharpness_override is None and layer._active_context is None
    assert all(not c[5] and c[-1] == 4 for c in contexts[-12:])
    actual_inspect = layer.inspect_prepared_sample(batch, batch_index=1, channel=1)
    for key, spec in actual_inspect["specs"].items():
        for field in ("base_attrs", "norm_attrs", "altitude_increments"):
            torch.testing.assert_close(spec[field], expected_inspect["specs"][key][field])
    single = PreparedBatch((batch.samples[1],), ("test-b",))
    assert layer.inspect_training_sample(single, channel=1)["specs"].keys() == actual_inspect["specs"].keys()
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("lifetime", ["backward", "retain_graph", "discard", "no_grad"])
def test_active_handles_follow_autograd_lifetime(baseline, lifetime):
    layer = model()
    store = MemoryStore(1024**2)
    batch = CFPPreprocessor.from_layer(layer).prepare_batch(baseline["images"][:1], store=store)
    ref = weakref.ref(batch)
    with torch.set_grad_enabled(lifetime != "no_grad"):
        output = layer(batch)
    store.clear()
    del batch
    if lifetime == "no_grad":
        assert ref() is None
    else:
        assert ref() is not None
        if lifetime == "retain_graph":
            output.sum().backward(retain_graph=True)
            assert ref() is not None and store.info()["active_bytes"] > 0
            layer.zero_grad(set_to_none=True)
        if lifetime in ("backward", "retain_graph"):
            output.sum().backward()
        else:
            del output
            gc.collect()
        assert ref() is None
    assert store.info()["live_bytes"] == 0


def test_prepared_validation_and_frozen_stats(baseline):
    layer = model()
    prep = CFPPreprocessor.from_layer(layer)
    batch = prep.prepare_batch(baseline["images"][:1])
    prepared = first(batch)
    with pytest.raises(TypeError):
        batch.samples[0][0][prepared.tree_key] = prepared
    with pytest.raises(ValueError, match="spatial shape"):
        PreparedBatch(((dict(batch.samples[0][0]),),
                       ({prepared.tree_key: prep.prepare_image(torch.zeros(3, 3))},)))
    with pytest.raises(ValueError, match="channel count"):
        PreparedBatch((batch.samples[0], batch.samples[0] * 2))
    for samples in ((), ((),), (({},),), (({"wrong": prepared},),)):
        with pytest.raises(ValueError):
            PreparedBatch(samples)
    for ids in ((), (1,), ("a", "b")):
        with pytest.raises(ValueError, match="sample_ids"):
            prep.prepare_batch(baseline["images"][:1], sample_ids=ids)
    for incompatible in (model(channels=2), model(tree="min-tree"), model(attribute_dtype=np.float64),
                         model(attrs=(AREA, HEIGHT, morphology.AttributeType.VOLUME))):
        with pytest.raises(ValueError):
            incompatible(batch)
    subset = model(attrs=(AREA,))
    torch.testing.assert_close(subset(batch), subset(baseline["images"][:1]))
    layer = model(mode="dataset_zscore")
    with pytest.raises(RuntimeError, match="statistics"):
        layer(batch)
    layer.fit_stats([baseline["images"][:3]])
    layer.unfreeze_ds_stats()
    with pytest.raises(RuntimeError, match="frozen statistics"):
        layer(batch)
    with pytest.raises(TypeError, match="PreparedBatch"):
        layer.forward_prepared(baseline["images"])
    with pytest.raises(IndexError):
        model().inspect_prepared_sample(batch, batch_index=1)


def test_prepared_checkpoint_roundtrip_has_no_storage_dependency(baseline, tmp_path):
    from mtlearn.layers import save_checkpoint
    layer = model(mode="dataset_zscore")
    layer.fit_stats([baseline["images"][:3]])
    store = MemoryStore(1024**2)
    batch = CFPPreprocessor.from_layer(layer).prepare_batch(baseline["images"][3:], store=store)
    output = layer(batch)
    output.mean().backward()
    expected_keys = layer.state_dict().keys()
    saved = save_checkpoint(tmp_path / "prepared.pt", layer)
    assert saved["model_state_dict"].keys() == expected_keys
    del batch, store, layer
    restored, _ = load_checkpoint(tmp_path / "prepared.pt", lambda configs: Layer.from_config(configs[""]))
    assert restored.cached_sample_count() == 0
    torch.testing.assert_close(restored(baseline["images"][3:]), output)


def test_two_consumers_keep_independent_autograd_ownership(baseline):
    linear, mlp = model(), model(scoring="mlp")
    store = NullStore()
    batch = CFPPreprocessor.from_layer(linear).prepare_batch(baseline["images"][:1], store=store)
    linear_output, mlp_output = linear(batch), mlp(batch)
    del batch
    assert store.info()["active_bytes"] > 0
    linear_output.sum().backward()
    assert store.info()["active_bytes"] > 0
    mlp_output.sum().backward()
    assert store.info()["active_bytes"] == 0


def test_prepared_double_precision_matches_direct(baseline):
    layer = model(attribute_dtype=np.float64, mode="dataset_zscore").double()
    layer.fit_stats([baseline["images"][:3]])
    images = baseline["images"][3:]
    batch = CFPPreprocessor.from_layer(layer).prepare_batch(images)
    direct = layer(images)
    direct.square().mean().backward()
    expected = [p.grad.clone() for p in layer.parameters()]
    layer.zero_grad(set_to_none=True)
    actual = layer(batch)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, direct, rtol=1e-12, atol=1e-12)
    for p, gradient in zip(layer.parameters(), expected):
        torch.testing.assert_close(p.grad, gradient, rtol=1e-12, atol=1e-12)


def test_prepared_exception_resets_context_and_prediction_mode(baseline):
    layer = model()
    batch = CFPPreprocessor.from_layer(layer).prepare_batch(baseline["images"][:1])
    def fail(*args):
        raise RuntimeError("scoring failed")
    handle = next(iter(layer._scoring_models.values())).register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="scoring failed"):
        layer.predict(batch)
    assert layer._active_context is None and layer._score_sharpness_override is None and layer.training
    handle.remove()
    assert torch.isfinite(layer(batch)).all()
