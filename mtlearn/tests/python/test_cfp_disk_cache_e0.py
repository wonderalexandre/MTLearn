import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import mtlearn
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer
from mtlearn.layers.cfp import CFPPreprocessor, MemoryStore, collate_prepared

pytestmark = pytest.mark.integration
if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

FIXTURES = Path(__file__).parent / "fixtures/cfp_disk_cache_e0"
CASES = [(path, scorer, tree) for path in ("memory", "legacy")
         for scorer in ("linear_sigmoid", "mlp")
         for tree in ("max-tree", "min-tree", "tree-of-shapes")]


def copy_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: copy_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(copy_state(item) for item in value)
    return value


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state().clone(),
            "numpy": (state[0], state[1].tolist(), *state[2:])}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))


def capture_case(path, scorer, tree, device="cpu"):
    previous_rng, previous_threads = rng_state(), torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        images = torch.randint(0, 256, (5, 1, 6, 7), dtype=torch.uint8,
                               generator=torch.Generator().manual_seed(1050))
        targets = (images >= 173).float()
        train = TensorDataset(images[:3], targets[:3])
        test = TensorDataset(images[3:], targets[3:])
        record = {"path": path, "scorer": scorer, "tree": tree,
                  "images": images, "targets": targets, "before_model_rng": rng_state()}
        layer = Layer(1, [{"name": "area_height", "tree_type": tree,
                         "attributes": [morphology.AttributeType.AREA,
                                        morphology.AttributeType.GRAY_LEVEL_HEIGHT],
                         "scoring": {"kind": scorer, **({"hidden_units": 8, "activation": "tanh"}
                                                      if scorer == "mlp" else {})}}],
                      scale_mode="dataset_clipped_zscore01", device=device)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01, weight_decay=1e-7)
        record.update(config=layer.get_config(), before_preparation_rng=rng_state(),
                      before_preparation_state=copy_state(layer.state_dict()),
                      before_preparation_optimizer=copy_state(optimizer.state_dict()))
        if path == "memory":
            preprocessor = CFPPreprocessor.from_layer(layer)
            store = MemoryStore(32 * 1024**2)
            result = preprocessor.prepare(train, store=store,
                collect_stats=layer.get_statistics_contract(), sample_ids=["0", "1", "2"])
            layer.set_stats(result.statistics)
            train_items, test_items = [], []
            raw = {}
            for source, offset, items in ((train, 0, train_items), (test, 100, test_items)):
                for i in range(len(source)):
                    image, target = source[i]
                    batch = preprocessor.prepare_batch(image[None], store=store, sample_ids=[str(i + offset)])
                    items.append((batch, target))
                    raw[str(i + offset)] = {
                        key: {"info": copy_state({k: v for k, v in prepared.info.items() if k != "tree_type"}),
                              "attributes": {a.name: copy_state(v) for a, v in prepared.raw_attributes.items()}}
                        for key, prepared in batch.samples[0][0].items()}
            train_loader = DataLoader(train_items, batch_size=2, shuffle=False, collate_fn=collate_prepared)
            test_loader = DataLoader(test_items, batch_size=2, shuffle=False, collate_fn=collate_prepared)
            assert store.info()["entries"] == 5 and store.info()["oversized_entries"] == 0
            record["raw"] = raw
        else:
            train_loader = layer.build_dataloader_cached(DataLoader(train, batch_size=2, shuffle=True))
            train_statistics = copy_state(layer.get_stats().statistics)
            test_loader = layer.build_dataloader_cached_fixed_stats(
                DataLoader(test, batch_size=2, shuffle=False), index_offset=100)
            assert_nested(layer.get_stats().statistics, train_statistics, rtol=0, atol=0)
            assert layer.cached_sample_count() == 5
            record["raw"] = {
                key: {tree_key: {"info": copy_state({k: v for k, v in payload["info"].items()
                                                      if k != "tree_type"}),
                                "attributes": {a.name: copy_state(v) for a, v in payload["base_attrs"].items()}}
                      for tree_key, payload in layer._tree_payload_cache.sample_payloads(key).items()}
                for key in layer._tree_payload_cache.sample_keys()}
        record["after_preparation_rng"] = rng_state()
        record["statistics"] = copy_state(layer.get_stats().statistics)
        record["statistics_contract"] = dict(layer.get_stats().contract)
        layer.init_identity(p0=0.995)
        record["training_start_state"] = copy_state(layer.state_dict())
        record["training_start_rng"] = rng_state()
        record["training_start_optimizer"] = copy_state(optimizer.state_dict())
        record["steps"] = []
        for epoch in range(2):
            for batch, target in train_loader:
                ids = batch.sample_ids if path == "memory" else tuple(str(i) for i in batch[1].tolist())
                optimizer.zero_grad(set_to_none=True)
                output = layer(batch)
                loss = ((output / 255.0 - target.to(device)) ** 2).mean()
                loss.backward()
                step = {"epoch": epoch, "ids": ids, "target": copy_state(target),
                        "output": copy_state(output), "loss": copy_state(loss),
                        "gradients": {k: copy_state(p.grad) for k, p in layer.named_parameters()}}
                optimizer.step()
                step.update(state=copy_state(layer.state_dict()), optimizer=copy_state(optimizer.state_dict()),
                            rng=rng_state())
                record["steps"].append(step)
        layer.eval()
        record["evaluation"] = []
        with torch.no_grad():
            for batch, target in test_loader:
                ids = batch.sample_ids if path == "memory" else tuple(str(i) for i in batch[1].tolist())
                record["evaluation"].append({"ids": ids, "target": copy_state(target),
                                               "output": copy_state(layer(batch))})
        assert_nested(layer.get_stats().statistics, record["statistics"], rtol=0, atol=0)
        record.update(final_rng=rng_state(), final_state=copy_state(layer.state_dict()),
                      final_optimizer=copy_state(optimizer.state_dict()))
        return record
    finally:
        restore_rng(previous_rng)
        torch.set_num_threads(previous_threads)


def assert_nested(actual, expected, *, rtol=1e-5, atol=1e-6):
    if torch.is_tensor(expected):
        actual = actual.detach().cpu()
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        if expected.is_floating_point():
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        else:
            assert actual.numpy().tobytes() == expected.numpy().tobytes()
    elif isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_nested(a, e, rtol=rtol, atol=atol)
    else:
        assert actual == expected


def reference(path, scorer, tree):
    return torch.load(FIXTURES / f"{path}_{scorer}_{tree}.pt", map_location="cpu", weights_only=True)


def test_e0_fixture_integrity():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert len(manifest["files"]) == len(CASES)
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("path,scorer,tree", CASES)
def test_ram_against_e0_before_disk_evolution(path, scorer, tree):
    assert_nested(capture_case(path, scorer, tree), reference(path, scorer, tree))


@pytest.mark.parametrize("path,scorer,tree", CASES)
def test_e0_frozen_statistics_and_weights_without_preparation_fit(path, scorer, tree):
    expected = reference(path, scorer, tree)
    previous = rng_state()
    try:
        layer = Layer.from_config(expected["config"], device="cpu")
        layer.load_state_dict(expected["final_state"], strict=True)
        layer.eval()
        with torch.no_grad():
            output = layer(expected["images"][3:])
        assert_nested(output, expected["evaluation"][0]["output"])
        assert_nested(layer.get_stats().statistics, expected["statistics"], rtol=0, atol=0)
    finally:
        restore_rng(previous)
