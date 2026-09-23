import json

import numpy as np
import pytest
import torch

import mtlearn
from mtlearn import _native
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore, PreparedDataset
from mtlearn.layers.cfp.preparation import _identity
from test_cfp_disk_cache import QUOTA, forbid, model, prepare


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("workers", [0, 1])
def test_cache_reuse_across_binary_and_package_versions(tmp_path, monkeypatch, legacy, workers):
    preprocessor = CFPPreprocessor.from_layer(model())
    source = [torch.tensor([[[0, 1], [2, 3]]], dtype=torch.uint8)]
    current = _identity.implementation_identity
    with monkeypatch.context() as previous:
        if legacy:
            previous.setattr(_identity, "implementation_identity", lambda: dict(current(), native_sha256="a" * 64))
        previous.setattr(mtlearn, "__version__", "1.2.0")
        with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
            prepare(preprocessor, source, store)
            config = json.loads(store._db.execute("SELECT config FROM manifests").fetchone()[0])
            entries = [tuple(row) for row in store._db.execute("SELECT * FROM entries")]
            expected = PreparedDataset(store, "train", source)[0]
            expected_attributes = next(iter(expected.samples[0][0].values())).raw_attributes
            expected_attributes = {key: value.clone() for key, value in expected_attributes.items()}
            del expected
    files = {path: path.read_bytes() for path in tmp_path.rglob("*.pt")}
    assert files
    monkeypatch.setattr(mtlearn, "__version__", "1.2.1")
    monkeypatch.setattr(_native, "load_bindings", forbid)
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
    assert _identity.implementation_identity() == {"semantics": _identity.PREPARATION_SEMANTICS}
    assert CFPPreprocessor.from_config(config).get_config() == config
    for readonly in (True, False):
        with DiskStore(tmp_path, readonly=readonly, max_disk_bytes=None if readonly else QUOTA) as store:
            batch = PreparedDataset(store, "train", source)[0]
            actual = next(iter(batch.samples[0][0].values())).raw_attributes
            for key in expected_attributes:
                torch.testing.assert_close(actual[key], expected_attributes[key])
            del batch, actual
            if not readonly:
                assert prepare(preprocessor, source, store, num_workers=workers).status == "complete"
            assert [tuple(row) for row in store._db.execute("SELECT * FROM entries")] == entries
    result = preprocessor.prepare_or_reuse(path=tmp_path, manifest="train", source_version="source-v1",
        preprocessing_version="uint8-v1", mode="reuse")
    assert result.status == "complete"
    assert result.prepared_entries == 0
    assert {path: path.read_bytes() for path in files} == files


@pytest.mark.parametrize("field,value", [("semantics", "incompatible-preparation"), ("unknown", "value")])
def test_incompatible_implementation_rejected_at_all_boundaries(tmp_path, field, value):
    preprocessor = CFPPreprocessor.from_layer(model())
    config = preprocessor.get_config()
    config["implementation"][field] = value
    with pytest.raises(ValueError, match="Incompatible"):
        CFPPreprocessor.from_config(config)
    identity = _identity.image_identity(preprocessor, np.array([[0, 1]], dtype=np.uint8),
                                       next(iter(preprocessor.tree_specs)))
    identity["implementation"][field] = value
    with pytest.raises(ValueError, match="backend/semantics"):
        _identity.validate_identity(identity)
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        store._db.execute("UPDATE metadata SET value=? WHERE key='contract'", (_identity.canonical_json({
            "format_version": _identity.FORMAT_VERSION, "implementation": config["implementation"]}),))
        store._db.commit()
    with pytest.raises(ValueError, match="incompatible"):
        DiskStore(tmp_path, readonly=True)


@pytest.mark.parametrize("workers", [0, 1])
def test_resume_legacy_cache_preserves_original_keys(tmp_path, monkeypatch, workers):
    preprocessor = CFPPreprocessor.from_layer(model())
    source = [torch.tensor([[[0, 1], [2, 3]]], dtype=torch.uint8),
              torch.tensor([[[3, 2], [0, 1]]], dtype=torch.uint8)]
    current = _identity.implementation_identity
    with monkeypatch.context() as previous:
        previous.setattr(_identity, "implementation_identity", lambda: dict(current(), native_sha256="b" * 64))
        with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
            result = prepare(preprocessor, source, store,
                cancel=lambda: store._db.execute("SELECT count(*) FROM samples").fetchone()[0] == 1)
            assert result.status == "cancelled"
            config = json.loads(store._db.execute("SELECT config FROM manifests").fetchone()[0])
            key = store._db.execute("SELECT key FROM entries").fetchone()[0]
    result = preprocessor.prepare_or_reuse(source, path=tmp_path, manifest="train", source_version="source-v1",
        preprocessing_version="uint8-v1", mode="prepare_missing", max_disk_bytes=QUOTA, num_workers=workers)
    assert result.status == "complete"
    with DiskStore(tmp_path, readonly=True) as store:
        assert json.loads(store._db.execute("SELECT config FROM manifests").fetchone()[0]) == config
        assert store.get(key) is not None
        dataset = PreparedDataset(store, "train", source)
        assert len(dataset) == 2
        for index in range(len(dataset)):
            assert dataset[index].shape == (1, 1, 2, 2)
        for row in store._db.execute("SELECT identity FROM entries"):
            assert json.loads(row[0])["implementation"] == config["implementation"]
