"""Compact pixel ownership maps preserve reconstruction and persistent values."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore
from mtlearn.layers.cfp.preparation._identity import canonical_json, implementation_identity
from mtlearn.layers.cfp.runtime.tree_reconstructor import TreeReconstructor


def prepared_image(tree_type="max-tree"):
    layer = Layer(1, [{"tree_type": tree_type,
                       "attributes": (morphology.AttributeType.AREA,)}])
    prep = CFPPreprocessor.from_layer(layer)
    image = np.array([[0, 4, 2], [3, 1, 4]], dtype=np.uint8)
    return prep, image, prep.prepare_image(image)


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree", "tree-of-shapes"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_compact_map_matches_signed_outputs_and_gradients(tree_type, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    _, _, prepared = prepared_image(tree_type)
    prepared.validate(full=True)
    compact = {k: v.to(device) if torch.is_tensor(v) else v for k, v in prepared.info.items()}
    assert compact["node_of_pixel"].dtype == torch.uint32
    signed = dict(compact, node_of_pixel=compact["node_of_pixel"].to(torch.int64))
    assert compact["node_of_pixel"].nbytes * 2 == signed["node_of_pixel"].nbytes
    results = []
    for info in (compact, signed):
        signal = torch.linspace(-1, 1, prepared.num_nodes, device=device, requires_grad=True)
        output = TreeReconstructor.apply(signal, info)
        (output.square().sum()).backward()
        results.append((output.detach(), signal.grad))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mmap", [False, True])
def test_disk_retains_uint32_and_all_values(tmp_path, mmap):
    prep, image, prepared = prepared_image()
    with DiskStore(tmp_path, max_disk_bytes=1024**2) as store:
        batch = prep.prepare_batch(torch.from_numpy(image)[None, None], store=store)
    with DiskStore(tmp_path, readonly=True, mmap=mmap) as store:
        batch = prep.prepare_batch(torch.from_numpy(image)[None, None], store=store)
        assert store.info()["disk_hits"] == 1
    files = list((tmp_path / "entries").glob("*.pt"))
    data = torch.load(files[0], weights_only=True, mmap=mmap)
    assert data["format_version"] == 2
    pixel_map = data["info"]["node_of_pixel"]
    assert pixel_map.dtype == torch.uint32
    assert pixel_map.nbytes == image.size * 4
    assert torch.equal(pixel_map, prepared.info["node_of_pixel"])


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_unsigned_range_survives_transfer_and_serialization(tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    # Include values above the signed 32-bit range to detect reinterpretation.
    expected = torch.tensor([0, 2**31 - 1, 2**31, 2**32 - 1], dtype=torch.int64)
    path = tmp_path / "range.pt"
    torch.save(expected.to(torch.uint32), path)
    actual = torch.load(path, weights_only=True).to(device).to(torch.int64).cpu()
    assert torch.equal(actual, expected)


def test_old_store_rejected_without_rewriting_metadata(tmp_path):
    with DiskStore(tmp_path, max_disk_bytes=1024**2) as store:
        old = canonical_json({"format_version": 1, "implementation": implementation_identity()})
        store._db.execute("UPDATE metadata SET value=? WHERE key='contract'", (old,))
        store._db.commit()
    with pytest.raises(ValueError, match="incompatible"):
        DiskStore(tmp_path, readonly=True)
    import sqlite3
    with sqlite3.connect(tmp_path / "manifest.sqlite3") as db:
        assert db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()[0] == old


def test_unsigned_out_of_tree_range_rejected():
    _, _, prepared = prepared_image()
    bad = replace(prepared, info=dict(prepared.info, node_of_pixel=torch.full_like(
        prepared.info["node_of_pixel"], 2**32 - 1)))
    with pytest.raises(ValueError, match="invalid node"):
        bad.validate(full=True)
