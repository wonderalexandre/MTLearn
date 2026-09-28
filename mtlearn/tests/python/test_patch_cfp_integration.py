import copy
import gc
import weakref

import pytest
import torch
from torch.utils.data import TensorDataset

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer
from mtlearn.layers.cfp import CFPPreprocessor, MemoryStore, build_prepared_dataloader
from mtlearn.layers.cfp.normalization import AttributeNormalizer
from mtlearn.patches import patch_grid
from mtlearn.training import backward_from_global_response
from test_patch_backward import direct_objective, squared_loss, assert_gradients

pytestmark = pytest.mark.integration


def make_layer(scoring, device='cpu'):
    specs = [{"tree_type": tree, "attributes": [morphology.AttributeType.AREA, morphology.AttributeType.GRAY_LEVEL_HEIGHT],
              "scoring": {"kind": scoring, **({"hidden_units": [4], "activation": "tanh"} if scoring == "mlp" else {})},
              "regularizers": [{"kind": "edge_score_monotonicity", "weight": 0.3}]}
             for tree in ('max-tree', 'min-tree')]
    return Layer(2, specs, scale_mode='dataset_clipped_zscore01', device=device, clamp=None)


def forbidden(*args, **kwargs):
    raise AssertionError('Prepared patch consumption must not prepare trees or update statistics')


@pytest.mark.parametrize('scoring', ['linear_sigmoid', 'mlp'])
@pytest.mark.parametrize('strategy', ['local', 'global_leaf'])
@pytest.mark.parametrize('device', ['cpu', 'mps', 'cuda'])
def test_prepared_cfp_full_objective_and_lifetime_match_reference(scoring, strategy, device, monkeypatch):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(32)
    producer = make_layer(scoring, device)
    images = torch.rand(2, 2, 7, 9)
    producer.fit_stats([images])
    actual_producer = Layer.from_config(producer.get_config(), device=device)
    actual_producer.load_state_dict(producer.state_dict())
    actual_producer.set_stats(producer.get_stats())
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 3, 3, padding=1), torch.nn.Tanh()).to(device)
    actual_model = copy.deepcopy(model)
    target = torch.rand(2, 3, 7, 9, device=device)
    patches = patch_grid(7, 9, 3, 2, 1)
    prep = CFPPreprocessor.from_layer(producer)
    expected_batch = prep.prepare_batch(images)
    store = MemoryStore(1024**2)
    actual_batch = prep.prepare_batch(images, store=store)
    before_stats = copy.deepcopy(actual_producer.get_stats().statistics)
    monkeypatch.setattr(CFPPreprocessor, 'prepare_u8', forbidden)
    for name in ('update', 'summarize', 'merge'):
        monkeypatch.setattr(AttributeNormalizer, name, forbidden)
    expected_response = producer.forward_prepared(expected_batch) / 255
    expected_response.retain_grad()
    auxiliary = lambda z, y: z.square().mean()
    loss, _, _ = direct_objective(expected_response, model, target, patches, squared_loss,
        auxiliary_loss=auxiliary, auxiliary_weight=0.2, parameter_loss=producer.regularization_penalty(expected_batch))
    loss.backward()
    calls = []
    original_forward = actual_producer._forward_executor.forward
    def counted_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)
    monkeypatch.setattr(actual_producer._forward_executor, 'forward', counted_forward)
    actual_response = actual_producer.forward_prepared(actual_batch) / 255
    penalty = actual_producer.regularization_penalty(actual_batch)
    batch_ref = weakref.ref(actual_batch)
    store.clear()
    del actual_batch
    def require_alive(*args):
        assert batch_ref() is not None and store.info()['active_bytes'] > 0
    hook = actual_model.register_forward_pre_hook(require_alive)
    result, gradient = backward_from_global_response(actual_response, actual_model, target, patches,
        main_loss=squared_loss, auxiliary_loss=auxiliary, auxiliary_weight=0.2, parameter_loss=penalty,
        strategy=strategy, diagnostics=True, diagnostic_parameters=actual_producer.parameters())
    hook.remove()
    gc.collect()
    assert calls == [1]
    assert batch_ref() is None and store.info()['active_bytes'] == 0
    torch.testing.assert_close(gradient, expected_response.grad, rtol=2e-5, atol=2e-6)
    assert result.total_loss == pytest.approx(float(loss.detach()), rel=2e-5, abs=2e-6)
    assert_gradients(actual_producer, producer, rtol=2e-5, atol=2e-6)
    assert_gradients(actual_model, model, rtol=2e-5, atol=2e-6)
    for key, values in before_stats.items():
        for name, value in values.items():
            torch.testing.assert_close(actual_producer.get_stats().statistics[key][name], value, rtol=0, atol=0)


def test_prepared_reader_keeps_batch_alive_until_patch_backward_finishes(tmp_path, monkeypatch):
    layer = make_layer('linear_sigmoid')
    images = torch.rand(2, 2, 6, 7)
    target = torch.rand(2, 1, 6, 7)
    layer.fit_stats([images])
    source = TensorDataset(images, target)
    CFPPreprocessor.from_layer(layer).prepare_or_reuse(source, path=tmp_path, manifest='train',
        source_version='patch-test', preprocessing_version='v1', mode='prepare_missing', max_disk_bytes=1024**2)
    with build_prepared_dataloader(tmp_path, 'train', source=source, batch_size=2, prefetch=False) as reader:
        iterator = iter(reader)
        batch, truth = next(iterator)
        reference = weakref.ref(batch)
        response = layer.forward_prepared(batch) / 255
        penalty = layer.regularization_penalty(batch)
        del batch
        assert iterator.close()
        monkeypatch.setattr(CFPPreprocessor, 'prepare_u8', forbidden)
        model = torch.nn.Conv2d(4, 1, 3, padding=1)
        backward_from_global_response(response, model, truth, patch_grid(6, 7, 3, 2, 1),
            main_loss=squared_loss, parameter_loss=penalty)
        gc.collect()
        assert reference() is None
        del truth
        gc.collect()
        assert reader.counters()['active_bytes'] == 0
