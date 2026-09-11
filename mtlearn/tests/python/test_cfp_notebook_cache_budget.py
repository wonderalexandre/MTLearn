"""Capacity and resumption checks for the notebook's full-dataset SSD profile."""
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore
from test_cfp_notebook_streaming import NOTEBOOK, definitions, nbformat

GIB = 1024**3


def helpers(tmp_path):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = dict(Path=Path, shutil=shutil, DiskStore=DiskStore, torch=torch,
                     CACHE_DIR=tmp_path, CACHE_MIN_FREE_BYTES=100 * GIB)
    definitions(notebook.cells[7].source, namespace)
    definitions(notebook.cells[11].source, namespace)
    return namespace


def test_insufficient_capacity_stops_without_creating_cache(tmp_path, monkeypatch):
    ns = helpers(tmp_path)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=571 * GIB))
    target = tmp_path / 'uncreated-cache'
    with pytest.raises(RuntimeError, match='Free at least 139.00 GiB'):
        ns['check_cache_capacity'](target, quota_bytes=610 * GIB, reserve_bytes=100 * GIB)
    assert not target.exists()


def test_resume_credits_existing_tensor_files_and_accepts_exact_capacity(tmp_path, monkeypatch):
    layer = ConnectedFilterPreprocessingLayer(
        1, [{'tree_type': 'max-tree', 'attributes': (morphology.AttributeType.AREA,)}],
        scale_mode='none', device='cpu',
    )
    quota, reserve = 1024**2, 1024**2
    with DiskStore(tmp_path, max_disk_bytes=quota) as store:
        batch = CFPPreprocessor.from_layer(layer).prepare_batch(
            torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4), store=store,
        )
        stored = store.info()['disk_bytes']
        del batch
    assert stored > 0
    ns = helpers(tmp_path)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=quota - stored + reserve))
    report = ns['check_cache_capacity'](tmp_path, quota_bytes=quota, reserve_bytes=reserve)
    assert report['remaining_quota_gib'] * GIB == quota - stored
    assert report['existing_cache_gib'] * GIB == stored
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=quota - stored + reserve - 1))
    with pytest.raises(RuntimeError, match='Insufficient SSD space'):
        ns['check_cache_capacity'](tmp_path, quota_bytes=quota, reserve_bytes=reserve)
    with pytest.raises(RuntimeError, match='already exceed'):
        ns['check_cache_capacity'](tmp_path, quota_bytes=stored - 1, reserve_bytes=0)


@pytest.mark.parametrize('free,stop', [(100 * GIB, False), (100 * GIB - 1, True)])
def test_reserve_stops_preparation_and_incomplete_split_cannot_start_training(tmp_path, monkeypatch, free, stop):
    ns = helpers(tmp_path)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=free))
    assert ns['cache_reserve_reached']() is stop
    ns['require_complete_preparation'](SimpleNamespace(status='complete'))
    with pytest.raises(RuntimeError, match='before completion'):
        ns['require_complete_preparation'](SimpleNamespace(status='cancelled'))
