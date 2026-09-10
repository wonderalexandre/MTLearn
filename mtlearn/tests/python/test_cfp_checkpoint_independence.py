"""Old checkpoints must not acquire a dependency on prepared SSD storage."""
import sqlite3

import pytest
from mtlearn.layers import ConnectedFilterPreprocessingLayer, load_checkpoint
from mtlearn.layers.cfp import DiskStore
from test_cfp_cache_baseline import (
    CASE_NAMES, FIXTURES, assert_outputs_and_gradients, reference,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("name", CASE_NAMES)
def test_legacy_checkpoint_without_prepared_storage(reference, name, device, tmp_path, monkeypatch):
    import torch
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("Checkpoint restoration and direct execution must not access prepared storage")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DiskStore, "__init__", forbidden)
    monkeypatch.setattr(DiskStore, "get", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    case = reference["cases"][name]
    model, _ = load_checkpoint(
        FIXTURES / case["checkpoint"],
        lambda configs: ConnectedFilterPreprocessingLayer.from_config(configs[""], device=device),
        device=device,
    )
    assert_outputs_and_gradients(reference, case, model, device, "direct")
    assert model.cached_sample_count() == 0
    assert not list(tmp_path.iterdir())
