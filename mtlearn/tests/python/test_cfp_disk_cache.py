"""P3: persistent identities, interrupted preparation and manifest consumption."""
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import weakref

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

import mtlearn
from mtlearn._native import load_bindings
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer, load_checkpoint
from mtlearn.layers.cfp import (CFPPreprocessor, DiskStore, MemoryStore, PreparedDataset,
                               PreparedBatch, collate_prepared)
from mtlearn.layers.cfp.preparation._identity import (image_identity, identity_key, implementation_identity)
from mtlearn.layers.cfp.storage import _disk_format

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parent / "fixtures/cfp_cache_p0_mmcfilters_v5_2_0"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
ROOT = Path(__file__).resolve().parents[3]
AREA, HEIGHT = morphology.AttributeType.AREA, morphology.AttributeType.GRAY_LEVEL_HEIGHT
QUOTA = 32 * 1024**2


@pytest.fixture(scope="module")
def baseline():
    return torch.load(FIXTURES / "baseline.pt", map_location="cpu", weights_only=True)


def model(*, mode="dataset_clipped_zscore01", scoring="linear_sigmoid", attrs=(AREA, HEIGHT), dtype=np.float32):
    return Layer(1, [{"tree_type": "max-tree", "attributes": attrs, "scoring": {"kind": scoring}}],
                 scale_mode=mode, attribute_dtype=dtype)


def source(baseline, selection=slice(0, 3)):
    return TensorDataset(baseline["images"][selection], baseline["targets"][selection])


def prepare(preprocessor, dataset, store, **kw):
    options = dict(manifest="train", source_version="source-v1", preprocessing_version="uint8-v1", split="train")
    options.update(kw)
    return preprocessor.prepare(dataset, store=store, **options)


def forbid(*a, **kw):
    raise AssertionError("An existing complete preparation must not be rebuilt or summarized again")


def assert_stats(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        for name, value in expected[key].items():
            torch.testing.assert_close(actual[key][name], value.cpu(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", MANIFEST["cases"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("mmap", [True, False])
@pytest.mark.parametrize("validation", ["always", "session"])
def test_ssd_statistics_outputs_and_gradients_match_p0(baseline, tmp_path, name, device, mmap, validation, monkeypatch):
    if validation == "session" and sys.platform == "win32":
        pytest.skip("Session validation requires POSIX shared locks")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    case = baseline["cases"][name]
    layer, _ = load_checkpoint(FIXTURES / case["checkpoint"],
        lambda configs: Layer.from_config(configs[""], device=device), device=device)
    prep = CFPPreprocessor.from_layer(layer)
    with DiskStore(tmp_path, max_disk_bytes=QUOTA, mmap=mmap) as store:
        result = prepare(prep, source(baseline), store, collect_stats=layer.get_statistics_contract())
        assert result.status == "complete" and result.sample_count == 3
        assert result.statistics.sample_count == 3 and result.statistics_id
        assert_stats(result.statistics.statistics, case["stats"])
        layer.set_stats(result.statistics)
        prepare(prep, source(baseline, slice(3, None)), store, manifest="test", split="evaluation")
        assert store.info()["retained_bytes"] == 0
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
    monkeypatch.setattr(_disk_format, "summarize", forbid)
    with DiskStore(tmp_path, readonly=True, mmap=mmap, validation=validation,
                   immutable=validation == "session") as reopened:
        dataset = PreparedDataset(reopened, "test", source(baseline, slice(3, None)))
        if validation == "session":
            for index in range(len(dataset)):
                prepared_sample = dataset[index]
                del prepared_sample
        batch, targets = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_prepared)))
        if validation == "session":
            assert reopened.counters()["validation_hits"] == len(dataset)
        output = layer(batch)
        reference = weakref.ref(batch)
        del batch
        reopened.clear()
        assert reference() is not None
        loss = ((output / 255 - targets.to(device)) ** 2).mean()
        loss.backward()
        assert reference() is None and reopened.info()["active_bytes"] == 0
        tolerance = MANIFEST["cpu_mps_tolerance" if device == "mps" else "same_device_tolerance"]
        torch.testing.assert_close(output.cpu(), case["output"], **tolerance)
        torch.testing.assert_close(loss.cpu(), case["loss"], **tolerance)
        for key, parameter in layer.named_parameters():
            torch.testing.assert_close(parameter.grad.cpu(), case["gradients"][key], **tolerance)
        assert layer.cached_sample_count() == 0
        with pytest.raises(ValueError, match="train manifest"):
            reopened.statistics("test", layer.get_statistics_contract())


@pytest.mark.parametrize("mode", ["none", "dataset_minmax01", "dataset_zscore", "dataset_clipped_zscore01"])
def test_resume_recombines_unique_logical_samples_with_content_multiplicity(baseline, tmp_path, monkeypatch, mode):
    layer = model(mode=mode)
    dataset = Subset(source(baseline), [2, 0, 2])
    expected = layer.fit_stats(DataLoader(dataset, batch_size=2))
    prep = CFPPreprocessor.from_layer(layer)
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        options = dict(sample_ids=("repeat-a", "zero", "repeat-b"), collect_stats=layer.get_statistics_contract())
        result = prepare(prep, dataset, store, **options)
        assert store.info()["disk_entries"] == 2
        assert_stats(result.statistics.statistics, expected.statistics)
        monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
        monkeypatch.setattr(_disk_format, "summarize", forbid)
        resumed = prepare(prep, dataset, store, **options)
        assert resumed.statistics_id == result.statistics_id
        assert_stats(resumed.statistics.statistics, expected.statistics)
        assert store._db.execute("SELECT COUNT(*) FROM contributions").fetchone()[0] == 3
        prepared = PreparedDataset(store, "train", dataset)
        # Subset order and explicit sampler preserve targets and logical IDs.
        view = Subset(prepared, [2, 0, 1])
        batches = list(DataLoader(view, batch_size=2, sampler=[2, 0, 1], collate_fn=collate_prepared))
        assert [i for batch, _ in batches for i in batch.sample_ids] == ["zero", "repeat-b", "repeat-a"]
        torch.testing.assert_close(torch.cat([targets for _, targets in batches]),
                                   torch.stack([dataset[i][1] for i in [1, 2, 0]]))


def test_cancel_and_failure_resume_without_double_counting(baseline, tmp_path, monkeypatch):
    layer, dataset = model(), source(baseline)
    prep = CFPPreprocessor.from_layer(layer)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls == 3
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        result = prepare(prep, dataset, store, cancel=cancel, collect_stats=layer.get_statistics_contract())
        assert result.status == "cancelled" and result.sample_count == 2 and result.statistics is None
        with pytest.raises(RuntimeError, match="cancelled"):
            PreparedDataset(store, "train")
        assert store._db.execute("SELECT COUNT(*) FROM contributions").fetchone()[0] == 2
    original = CFPPreprocessor.prepare_u8
    built = []
    def count_build(self, *a, **kw):
        built.append(1)
        return original(self, *a, **kw)
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", count_build)
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        completed = prepare(prep, dataset, store, collect_stats=layer.get_statistics_contract())
        assert len(built) == 1 and completed.sample_count == 3
        assert store._db.execute("SELECT COUNT(*) FROM contributions").fetchone()[0] == 3
        expected = layer.fit_stats(DataLoader(dataset, batch_size=1))
        assert_stats(completed.statistics.statistics, expected.statistics)


@pytest.mark.parametrize("stage", ["before_publish", "after_publish"])
def test_atomic_failure_intervals_are_recoverable(baseline, tmp_path, monkeypatch, stage):
    layer, dataset = model(), source(baseline, slice(0, 1))
    prep = CFPPreprocessor.from_layer(layer)
    import mtlearn.layers.cfp.storage.disk_store as disk_module
    store = DiskStore(tmp_path, max_disk_bytes=QUOTA)
    if stage == "before_publish":
        monkeypatch.setattr(disk_module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("injected publication failure")))
    else:
        monkeypatch.setattr(store, "_register_valid", lambda *args: (_ for _ in ()).throw(OSError("injected registration failure")))
    with pytest.raises(OSError, match="injected"):
        prepare(prep, dataset, store)
    assert store.info()["disk_entries"] == 0
    assert not list((tmp_path / "entries").glob("*.tmp"))
    store.close()
    monkeypatch.undo()
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as reopened:
        if stage == "after_publish":
            assert reopened.info()["disk_entries"] == 1
            monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
        completed = prepare(prep, dataset, reopened, collect_stats=layer.get_statistics_contract())
        assert completed.status == "complete"
        assert reopened._db.execute("SELECT COUNT(*) FROM contributions").fetchone()[0] == 1


def test_orphan_file_is_validated_and_registered(baseline, tmp_path, monkeypatch):
    prep = CFPPreprocessor.from_layer(model())
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        batch = prep.prepare_batch(baseline["images"][:1], store=store)
        del batch
        with store._db:
            store._db.execute("DELETE FROM entries")
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        assert store.info()["disk_entries"] == 1
        monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
        assert prep.prepare_batch(baseline["images"][:1], store=store).shape == (1, 1, 6, 7)


def test_corrupt_and_missing_files_are_rejected_or_repaired(baseline, tmp_path):
    layer, dataset = model(), source(baseline, slice(0, 1))
    prep = CFPPreprocessor.from_layer(layer)
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        result = prepare(prep, dataset, store, collect_stats=layer.get_statistics_contract())
        layer.set_stats(result.statistics)
        expected = layer(baseline["images"][:1])
        path = next((tmp_path / "entries").glob("*.pt"))
        data = bytearray(path.read_bytes());data[len(data)//2] ^= 1;path.write_bytes(data)
        with pytest.raises(ValueError, match="checksum"):
            PreparedDataset(store, "train")[0]
        # A live source permits deterministic repair; stats are never updated.
        actual, _ = PreparedDataset(store, "train", dataset)[0]
        torch.testing.assert_close(layer(actual), expected)
        assert_stats(store.statistics("train", layer.get_statistics_contract()).statistics, result.statistics.statistics)
        del actual
        path.unlink()
        store.clear()
        with pytest.raises(FileNotFoundError):
            PreparedDataset(store, "train")[0]
        actual, _ = PreparedDataset(store, "train", dataset)[0]
        torch.testing.assert_close(layer(actual), expected)


def test_orphan_embedded_checksum_and_incomplete_file_fail_validation(baseline, tmp_path):
    prep = CFPPreprocessor.from_layer(model())
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        prep.prepare_batch(baseline["images"][:1], store=store)
        path = next((tmp_path / "entries").glob("*.pt"))
        data = torch.load(path, weights_only=True)
        data["attributes"]["AREA"][0] += 1
        torch.save(data, path)
        with store._db:
            store._db.execute("DELETE FROM entries")
        recovery = store.recover()
        assert recovery["failed"] == 1 and store.info()["disk_entries"] == 0
        path.write_bytes(b"incomplete archive")
        assert store.recover()["failed"] == 1


def test_quota_preserves_completed_files_and_resumes(baseline, tmp_path):
    layer, dataset = model(), source(baseline)
    prep = CFPPreprocessor.from_layer(layer)
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as initial:
        prep.prepare_batch(baseline["images"][:1], store=initial)
        size = initial.info()["disk_bytes"]
    original = next((tmp_path / "entries").glob("*.pt"))
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    with DiskStore(tmp_path, max_disk_bytes=size + 512) as limited:
        with pytest.raises(OSError, match="quota exceeded"):
            prepare(prep, dataset, limited)
        assert limited.info()["disk_bytes"] <= size + 512
        assert limited.info()["disk_entries"] == 1
        assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
        assert not list((tmp_path / "entries").glob("*.tmp"))
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as resumed:
        result = prepare(prep, dataset, resumed, collect_stats=layer.get_statistics_contract())
        assert result.status == "complete" and resumed.info()["disk_entries"] == 3
        assert result.statistics.sample_count == 3


def test_identity_versions_source_validation_and_eval_isolation(baseline, tmp_path):
    layer, dataset = model(), source(baseline)
    prep = CFPPreprocessor.from_layer(layer)
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        prepare(prep, dataset, store)
        for options in ({"source_version": "v2"}, {"preprocessing_version": "v2"}, {"split": "evaluation"}):
            with pytest.raises(ValueError, match="contract/source version"):
                prepare(prep, dataset, store, **options)
        changed = TensorDataset(1 - baseline["images"][:3], baseline["targets"][:3])
        with pytest.raises(ValueError, match="pixels/shape changed"):
            PreparedDataset(store, "train", changed)[0]
        with pytest.raises(ValueError, match="Source content/order changed"):
            prepare(prep, changed, store)
        prepare(prep, dataset, store)  # restore complete state after source mismatch
        with pytest.raises(ValueError, match="split='train'"):
            prepare(prep, dataset, store, manifest="eval", split="evaluation", collect_stats=layer.get_statistics_contract())
        prepare(prep, dataset, store, manifest="eval", split="evaluation")
        with pytest.raises(ValueError, match="train manifest"):
            store.statistics("eval", layer.get_statistics_contract())
        assert store.info()["disk_entries"] == 3
        with pytest.raises(ValueError, match="Statistical contract"):
            store.statistics("train", model(attrs=(AREA,)).get_statistics_contract())


def test_single_writer_readers_and_closed_handles(baseline, tmp_path):
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as writer:
        prepare(CFPPreprocessor.from_layer(model()), source(baseline), writer)
        with pytest.raises(RuntimeError, match="writer is already active"):
            DiskStore(tmp_path, max_disk_bytes=QUOTA)
        with DiskStore(tmp_path, readonly=True) as reader:
            assert PreparedDataset(reader, "train")[0].shape == (1, 1, 6, 7)
            with pytest.raises(RuntimeError, match="readonly"):
                reader.recover()
    with pytest.raises(RuntimeError, match="closed"):
        writer.info()
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as reopened:
        assert reopened.info()["disk_entries"] == 3


@pytest.mark.parametrize("option", [{}, {"max_disk_bytes":-1}, {"max_disk_bytes":True}, {"max_disk_bytes":1.2}])
def test_disk_quota_must_be_explicit(tmp_path, option):
    with pytest.raises(ValueError):
        DiskStore(tmp_path, **option)


def test_statistical_preparation_without_persistence_retains_no_samples(baseline, monkeypatch):
    layer = model()
    expected = layer.fit_stats(DataLoader(source(baseline), batch_size=1))
    prep = CFPPreprocessor.from_layer(layer)
    references = []
    original = prep.prepare_u8
    def checked(*args, **kwargs):
        assert all(ref() is None for ref in references)
        prepared = original(*args, **kwargs)
        references.extend([weakref.ref(prepared), weakref.ref(next(iter(prepared.raw_attributes.values())))])
        return prepared
    monkeypatch.setattr(prep, "prepare_u8", checked)
    result = prep.prepare(source(baseline), collect_stats=layer.get_statistics_contract())
    assert result.store_path is None and result.manifest is None
    assert_stats(result.statistics.statistics, expected.statistics)


def test_true_process_interruption_and_new_process_reuse(baseline, tmp_path):
    script = tmp_path / "process_probe.py"
    store_path = tmp_path / "store"
    script.write_text('''import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "scripts/benchmarks"))
import cfp_cache_p0 as ref
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != "_mtlearn_editable"]
sys.path[:0] = [str(Path(sys.argv[1]) / "mtlearn/python"), sys.argv[4]]
import torch
ref.Layer = __import__("mtlearn.layers", fromlist=["ConnectedFilterPreprocessingLayer"]).ConnectedFilterPreprocessingLayer
ref.torch = torch
ref.np = __import__("numpy")
ref.morphology = __import__("mtlearn", fromlist=["morphology"]).morphology
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore, PreparedDataset
from torch.utils.data import TensorDataset
import torch
baseline=torch.load(Path(sys.argv[1])/"mtlearn/tests/python/fixtures/cfp_cache_p0_mmcfilters_v5_2_0/baseline.pt",weights_only=True)
source=TensorDataset(baseline["images"][:1],baseline["targets"][:1])
if sys.argv[3]=="crash":
    store=DiskStore(sys.argv[2],max_disk_bytes=32*1024**2)
    def crash(data, handle):
        handle.write(b"partial")
        os._exit(42)
    torch.save=crash
    CFPPreprocessor.from_layer(ref.layer("linear")).prepare(source,store=store,manifest="train",
        source_version="source-v1",preprocessing_version="uint8-v1")
else:
    def forbidden(*a,**kw): raise AssertionError("new session rebuilt a tree")
    CFPPreprocessor.prepare_u8=forbidden
    with DiskStore(sys.argv[2],readonly=True) as store:
        assert PreparedDataset(store,"train",source)[0][0].shape==(1,1,6,7)
        print("REUSED_IN_NEW_PROCESS")
''')
    child = subprocess.run([sys.executable, str(script), str(ROOT), str(store_path), "crash", str(Path(load_bindings().__file__).resolve().parent)], capture_output=True, text=True)
    assert child.returncode == 42, child.stderr
    assert list((store_path / "entries").glob("*.tmp"))
    with DiskStore(store_path, max_disk_bytes=QUOTA) as store:
        assert not list((store_path / "entries").glob("*.tmp"))
        # Match the descriptor used by the child; there is no live child model.
        layer=Layer(1,[{"name":"baseline","tree_type":"max-tree","attributes":(AREA,HEIGHT),
                       "score_sharpness":1,"scoring":{"kind":"linear_sigmoid"}}],clamp=12)
        prepare(CFPPreprocessor.from_layer(layer),source(baseline,slice(0,1)),store)
    child = subprocess.run([sys.executable, str(script), str(ROOT), str(store_path), "reuse", str(Path(load_bindings().__file__).resolve().parent)], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert "REUSED_IN_NEW_PROCESS" in child.stdout


def test_empty_database_creation_recovers_and_incompatible_version_rejects(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    sqlite3.connect(tmp_path / "manifest.sqlite3").close()
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        assert store.info()["disk_entries"] == 0
        with store._db:
            store._db.execute("UPDATE metadata SET value=? WHERE key='contract'", ('{"format_version":999}',))
    with pytest.raises(ValueError, match="incompatible"):
        DiskStore(tmp_path, readonly=True)


def test_persistent_fingerprint_changes_with_dtype_features_tree_and_shape(baseline, tmp_path):
    standard = model()
    configs = [standard, model(attrs=(AREA,)), model(dtype=np.float64),
               Layer(1,[{"tree_type":"min-tree","attributes":(AREA,HEIGHT)}])]
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        for layer in configs:
            CFPPreprocessor.from_layer(layer).prepare_batch(baseline["images"][:1],store=store)
        prep = CFPPreprocessor.from_layer(standard)
        prep.prepare_batch(baseline["images"][:1,:,:3,:3],store=store)
        assert store.info()["disk_entries"]==5
        # Labels and input numeric representation do not change canonical pixels.
        pixels=torch.tensor([[[[0,128],[255,64]]]],dtype=torch.uint8)
        prep.prepare_batch(pixels,store=store,sample_ids=("a",))
        prep.prepare_batch(pixels.float()/255,store=store,sample_ids=("b",))
        assert store.info()["disk_entries"]==6


def test_linear_and_mlp_reuse_disk_entries_and_frozen_statistics(baseline, tmp_path, monkeypatch):
    from mtlearn.layers.cfp.normalization import AttributeNormalizer
    linear, mlp = model(), model(scoring="mlp")
    prep=CFPPreprocessor.from_layer(linear)
    with DiskStore(tmp_path,max_disk_bytes=QUOTA,max_ram_bytes=2048) as store:
        result=prepare(prep,source(baseline),store,collect_stats=linear.get_statistics_contract())
        for layer in (linear,mlp):
            layer.set_stats(result.statistics)
        direct=[layer(baseline["images"][:1]).detach() for layer in (linear,mlp)]
        monkeypatch.setattr(CFPPreprocessor,"prepare_u8",forbid)
        monkeypatch.setattr(_disk_format,"summarize",forbid)
        monkeypatch.setattr(AttributeNormalizer,"update",forbid)
        monkeypatch.setattr(AttributeNormalizer,"merge",forbid)
        batch=CFPPreprocessor.from_layer(mlp).prepare_batch(baseline["images"][:1],store=store)
        before={attr:t.clone() for attr,t in next(iter(batch.samples[0][0].values())).raw_attributes.items()}
        for layer, expected in zip((linear,mlp),direct):
            response=layer(batch)
            torch.testing.assert_close(response,expected)
            response.mean().backward()
            assert layer.cached_sample_count()==0
            assert_stats(layer.get_stats().statistics,result.statistics.statistics)
        for attr,t in next(iter(batch.samples[0][0].values())).raw_attributes.items():
            torch.testing.assert_close(t,before[attr])
        assert store.info()["retained_bytes"]<=2048


def test_duplicate_logical_ids_fail_without_duplicate_contributions(baseline,tmp_path):
    with DiskStore(tmp_path,max_disk_bytes=QUOTA) as store:
        with pytest.raises(ValueError,match="unique and stable"):
            prepare(CFPPreprocessor.from_layer(model()),source(baseline),store,sample_ids=("dup","dup","last"))
        assert store._db.execute("SELECT COUNT(*) FROM contributions").fetchone()[0]==1
        with pytest.raises(RuntimeError,match="failed"):
            store.statistics("train",model().get_statistics_contract())


def test_corrupt_entry_is_never_consumed_by_readonly_reader(baseline,tmp_path):
    with DiskStore(tmp_path,max_disk_bytes=QUOTA) as store:
        prepare(CFPPreprocessor.from_layer(model()),source(baseline,slice(0,1)),store)
    path=next((tmp_path/"entries").glob("*.pt"))
    path.write_bytes(b"corrupt")
    with DiskStore(tmp_path,readonly=True) as reader:
        with pytest.raises(ValueError,match="checksum"):
            PreparedDataset(reader,"train",source(baseline,slice(0,1)))[0]
    assert path.read_bytes()==b"corrupt"


def test_sample_binding_is_transactional_across_channels_and_trees(baseline,tmp_path,monkeypatch):
    layer=Layer(2,[{"tree_type":tree,"attributes":(AREA,HEIGHT)} for tree in ("max-tree","min-tree")])
    image=torch.cat((baseline["images"][0],baseline["images"][1]),dim=0)
    dataset=[(image,torch.zeros_like(image))]
    expected=layer.fit_stats(DataLoader(dataset,batch_size=1))
    prep=CFPPreprocessor.from_layer(layer)
    with DiskStore(tmp_path,max_disk_bytes=QUOTA) as store:
        def deny_binding(action,table,*args):
            return sqlite3.SQLITE_DENY if action==sqlite3.SQLITE_INSERT and table=="contributions" else sqlite3.SQLITE_OK
        store._db.set_authorizer(deny_binding)
        with pytest.raises(sqlite3.DatabaseError):
            prepare(prep,dataset,store)
        store._db.set_authorizer(None)
        assert store._db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]==0
        assert store._db.execute("SELECT COUNT(*) FROM contributions").fetchone()[0]==0
        assert store.info()["disk_entries"]==4
        monkeypatch.setattr(CFPPreprocessor,"prepare_u8",forbid)
        result=prepare(prep,dataset,store,collect_stats=layer.get_statistics_contract())
        assert result.sample_count==1
        assert_stats(result.statistics.statistics,expected.statistics)
        batch,target=PreparedDataset(store,"train",dataset)[0]
        layer.set_stats(result.statistics)
        assert layer(batch).shape==(1,4,6,7)
        assert store._db.execute("SELECT COUNT(*) FROM contributions").fetchone()[0]==4


def test_variable_image_shapes_use_separate_batches(baseline,tmp_path):
    dataset=[(baseline["images"][0],baseline["targets"][0]),
             (baseline["images"][1,:,:4],baseline["targets"][1,:,:4])]
    with DiskStore(tmp_path,max_disk_bytes=QUOTA) as store:
        prepare(CFPPreprocessor.from_layer(model()),dataset,store)
        prepared=PreparedDataset(store,"train",dataset)
        assert prepared[0][0].shape==(1,1,6,7)
        assert prepared[1][0].shape==(1,1,4,7)
        with pytest.raises(ValueError,match="spatial shape"):
            next(iter(DataLoader(prepared,batch_size=2,collate_fn=collate_prepared)))


def test_unlabelled_source_and_float64_roundtrip(baseline,tmp_path):
    layer=model(dtype=np.float64).double()
    images=baseline["images"][:3]
    expected=layer.fit_stats([images])
    reference=layer(images)
    reference.square().mean().backward()
    gradients=[p.grad.clone() for p in layer.parameters()]
    layer.zero_grad(set_to_none=True)
    with DiskStore(tmp_path,max_disk_bytes=QUOTA) as store:
        result=prepare(CFPPreprocessor.from_layer(layer),images,store,collect_stats=layer.get_statistics_contract())
        assert_stats(result.statistics.statistics,expected.statistics)
        dataset=PreparedDataset(store,"train",images)
        batch=next(iter(DataLoader(dataset,batch_size=3,collate_fn=collate_prepared)))
        assert isinstance(batch,PreparedBatch)
        actual=layer(batch)
        actual.square().mean().backward()
        torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-12)
        for p,gradient in zip(layer.parameters(),gradients):
            torch.testing.assert_close(p.grad,gradient,rtol=1e-12,atol=1e-12)


def test_resume_revalidates_ssd_even_when_ram_has_valid_buffers(baseline,tmp_path,monkeypatch):
    layer=model()
    with DiskStore(tmp_path,max_disk_bytes=QUOTA,max_ram_bytes=QUOTA) as store:
        prep=CFPPreprocessor.from_layer(layer)
        first=prepare(prep,source(baseline),store,collect_stats=layer.get_statistics_contract())
        assert store.info()["retained_bytes"]>0
        path=next((tmp_path/"entries").glob("*.pt"))
        data=bytearray(path.read_bytes());data[len(data)//2]^=1;path.write_bytes(data)
        original=CFPPreprocessor.prepare_u8
        builds=[]
        def build(self,*a,**kw):
            builds.append(1)
            return original(self,*a,**kw)
        monkeypatch.setattr(CFPPreprocessor,"prepare_u8",build)
        resumed=prepare(prep,source(baseline),store,collect_stats=layer.get_statistics_contract())
        assert len(builds)==1
        assert resumed.statistics_id==first.statistics_id
        assert_stats(resumed.statistics.statistics,first.statistics.statistics)


def test_checksum_binds_the_validated_bytes_before_publication(baseline,tmp_path,monkeypatch):
    import mtlearn.layers.cfp.storage.disk_store as disk_module
    replace=disk_module.os.replace
    def corrupt_after_publish(src,dst):
        replace(src,dst)
        data=torch.load(dst,weights_only=True)
        data["attributes"]["AREA"][0]+=1
        torch.save(data,dst)
    with DiskStore(tmp_path,max_disk_bytes=QUOTA) as store:
        monkeypatch.setattr(disk_module.os,"replace",corrupt_after_publish)
        CFPPreprocessor.from_layer(model()).prepare_batch(baseline["images"][:1],store=store)
        key=store._db.execute("SELECT key FROM entries").fetchone()[0]
        with pytest.raises(ValueError,match="checksum"):
            store.get(key)


@pytest.mark.parametrize("group_name", ["DIST_TRANSF", "DIST_TRANSF_EXACT", "FILLED_SHAPE"])
@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree", "tree-of-shapes"])
@pytest.mark.parametrize("device, dtype", [("cpu", np.float32), ("cpu", np.float64), ("mps", np.float32)])
def test_distance_and_filled_shape_groups_preserve_cached_outputs_and_gradients(
    tmp_path, monkeypatch, group_name, tree_type, dtype, device
):
    from mtlearn.layers.cfp.preparation._identity import preprocessor_config, preprocessor_from_config

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    images = torch.zeros(2, 1, 7, 7)
    images[:, :, 1:6, 1:6] = 200
    images[1, 0, 2, 2] = 0
    if tree_type == "min-tree":
        images = 255 - images
    layer = Layer(1, [{"tree_type": tree_type, "attributes": [getattr(morphology.AttributeGroup, group_name)]}],
                  scale_mode="dataset_minmax01", attribute_dtype=dtype, device=device)
    layer.fit_stats(DataLoader(TensorDataset(images), batch_size=2))
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(0.01)
    direct = layer(images)
    probe = torch.linspace(-0.5, 0.75, direct.numel(), device=device).reshape_as(direct)
    (direct * probe).sum().backward()
    gradients = {name: parameter.grad.detach().clone() for name, parameter in layer.named_parameters()}
    assert any(torch.count_nonzero(gradient) for gradient in gradients.values())
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())

    restored = Layer.from_config(layer.get_config(), device=device)
    restored.load_state_dict(layer.state_dict())
    prep = CFPPreprocessor.from_layer(restored)
    prep = preprocessor_from_config(preprocessor_config(prep))
    expected_attributes = set(morphology.expand_attribute_group(getattr(morphology.AttributeGroup, group_name)))
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        batch = prep.prepare_batch(images, store=store)
        for sample in batch.samples:
            for prepared in sample[0].values():
                assert set(prepared.raw_attributes) == expected_attributes
        del batch
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
    with DiskStore(tmp_path, readonly=True) as store:
        batch = prep.prepare_batch(images, store=store)
        cached = restored(batch)
        (cached * probe).sum().backward()
    torch.testing.assert_close(cached, direct)
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name])
