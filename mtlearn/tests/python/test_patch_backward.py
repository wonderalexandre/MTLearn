import copy
import gc
import weakref

import pytest
import torch
import torch.nn.functional as F

from mtlearn.patches import Patch, patch_grid
from mtlearn.training import backward_from_global_response


def direct_objective(response, model, target, patches, main_loss, *, weights=None, auxiliary_loss=None, auxiliary_weight=0.0, parameter_loss=None):
    values = []
    for patch in patches:
        side = patch.size + 2 * patch.halo
        padded = F.pad(response, (patch.halo,) * 4)
        local = padded[..., patch.row:patch.row + side, patch.col:patch.col + side]
        output = model(local)
        core = output[..., patch.halo:patch.halo + patch.size, patch.halo:patch.halo + patch.size]
        truth = target[..., patch.row:patch.row + patch.size, patch.col:patch.col + patch.size]
        local_weights = None if weights is None else weights[..., patch.row:patch.row + patch.size, patch.col:patch.col + patch.size]
        values.append(main_loss(core, truth, local_weights) / len(patches))
    main = torch.stack(values).sum()
    loss = main
    aux = None if auxiliary_loss is None else auxiliary_loss(response, target)
    if auxiliary_weight:
        loss = loss + auxiliary_weight * aux
    if parameter_loss is not None:
        loss = loss + parameter_loss
    return loss, main, aux


def squared_loss(logits, target, weights):
    errors = (logits - target).square()
    return errors.mean() if weights is None else (errors * weights).mean()


def assert_gradients(actual, expected, **tolerances):
    for got, want in zip(actual.parameters(), expected.parameters()):
        if want.grad is None:
            assert got.grad is None
        else:
            torch.testing.assert_close(got.grad, want.grad, **tolerances)


@pytest.mark.parametrize("strategy", ["local", "global_leaf"])
@pytest.mark.parametrize("auxiliary_weight", [0.0, 0.3])
@pytest.mark.parametrize("regularized", [False, True])
@pytest.mark.parametrize("device,dtype", [("cpu", torch.float64), ("mps", torch.float32), ("cuda", torch.float32)])
def test_full_objective_gradients_and_adamw_update_match_reference(strategy, auxiliary_weight, regularized, device, dtype):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(41)
    producer = torch.nn.Conv2d(3, 2, 1).to(device=device, dtype=dtype)
    model = torch.nn.Sequential(torch.nn.ReLU(inplace=True), torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.Tanh(), torch.nn.Conv2d(4, 3, 1)).to(device=device, dtype=dtype)
    actual_producer, actual_model = copy.deepcopy(producer), copy.deepcopy(model)
    image = torch.randn(2, 3, 7, 9, device=device, dtype=dtype)
    target = torch.randn_like(image)
    weights = torch.randint(0, 2, image.shape, device=device).to(dtype)
    weights[..., :3, :3] = 0
    patches = patch_grid(7, 9, 3, 2, 2)
    optimizers = [torch.optim.AdamW(list(p.parameters()) + list(m.parameters()), lr=0.01, weight_decay=0.1)
                  for p, m in ((producer, model), (actual_producer, actual_model))]
    for step in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for parameter in list(producer.parameters()) + list(model.parameters()) + list(actual_producer.parameters()) + list(actual_model.parameters()):
            parameter.grad = torch.full_like(parameter, 0.125)
        response = producer(image)
        response.retain_grad()
        auxiliary = lambda z, y: z.square().mean()
        penalty = (0.02 * sum(p.square().sum() for p in list(producer.parameters()) + list(model.parameters()))) if regularized else None
        loss, main, aux = direct_objective(response, model, target, patches, squared_loss,
            weights=weights, auxiliary_loss=auxiliary, auxiliary_weight=auxiliary_weight, parameter_loss=penalty)
        loss.backward()
        actual_response = actual_producer(image)
        actual_response.retain_grad()
        actual_penalty = (0.02 * sum(p.square().sum() for p in list(actual_producer.parameters()) + list(actual_model.parameters()))) if regularized else None
        result, gradient = backward_from_global_response(actual_response, actual_model, target, iter(patches),
            main_loss=squared_loss, pixel_weights=weights, auxiliary_loss=auxiliary,
            auxiliary_weight=auxiliary_weight, parameter_loss=actual_penalty,
            strategy=strategy, diagnostic_parameters=actual_producer.parameters(), diagnostics=True)
        tolerances = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-12)
        assert result.total_loss == pytest.approx(float(loss.detach()), **dict(rel=tolerances['rtol'], abs=tolerances['atol']))
        assert result.main_loss == pytest.approx(float(main.detach()), rel=tolerances['rtol'], abs=tolerances['atol'])
        assert result.aux_loss == pytest.approx(float(aux.detach()), rel=tolerances['rtol'], abs=tolerances['atol'])
        assert result.timings == {} and result.patches == len(patches)
        assert not gradient.requires_grad
        torch.testing.assert_close(gradient, response.grad, **tolerances)
        torch.testing.assert_close(actual_response.grad, response.grad, **tolerances)
        assert_gradients(actual_producer, producer, **tolerances)
        assert_gradients(actual_model, model, **tolerances)
        for optimizer in optimizers:
            optimizer.step()
        for actual, expected in ((actual_producer, producer), (actual_model, model)):
            for got, want in zip(actual.parameters(), expected.parameters()):
                torch.testing.assert_close(got, want, **tolerances)


class IndependentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))

    def forward(self, image):
        return self.bias.expand(image.shape)


@pytest.mark.parametrize("strategy", ["local", "global_leaf"])
@pytest.mark.parametrize("optimizer_kind", ["adamw", "momentum"])
@pytest.mark.parametrize("connected_zero", [False, True])
def test_disconnected_and_zero_gradients_have_distinct_optimizer_behavior(strategy, optimizer_kind, connected_zero):
    producer = torch.nn.Conv2d(1, 1, 1).double()
    model = IndependentModel()
    actual_producer, actual_model = copy.deepcopy(producer), copy.deepcopy(model)
    factories = {'adamw': lambda parameters: torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.1),
                 'momentum': lambda parameters: torch.optim.SGD(parameters, lr=0.1, momentum=0.9, weight_decay=0.1)}
    optimizers = [factories[optimizer_kind](list(p.parameters()) + list(m.parameters()))
                  for p, m in ((producer, model), (actual_producer, actual_model))]
    image = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    patches = patch_grid(4, 4, 2, 2)
    for optimizer, p, m in zip(optimizers, (producer, actual_producer), (model, actual_model)):
        (p(image).sum() + m(image).sum()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    def consumer(m):
        return lambda x: m(x) + x * 0 if connected_zero else m(x)
    expected, _, _ = direct_objective(producer(image), consumer(model), image, patches, squared_loss)
    expected.backward()
    result, gradient = backward_from_global_response(actual_producer(image), consumer(actual_model), image, patches,
        main_loss=squared_loss, strategy=strategy, diagnostics=True, diagnostic_parameters=actual_producer.parameters())
    assert torch.count_nonzero(gradient) == 0
    assert result.total_loss == pytest.approx(float(expected.detach()))
    assert_gradients(actual_producer, producer)
    for parameter in actual_producer.parameters():
        assert (parameter.grad is not None) == connected_zero
    for optimizer in optimizers:
        optimizer.step()
    for got, want in zip(actual_producer.parameters(), producer.parameters()):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("strategy", ["local", "global_leaf"])
@pytest.mark.parametrize("freeze", ["producer", "model"])
def test_frozen_module_matches_direct_reference(strategy, freeze):
    torch.manual_seed(4)
    producer = torch.nn.Conv2d(1, 1, 1).double()
    model = torch.nn.Conv2d(1, 1, 3, padding=1).double()
    (producer if freeze == 'producer' else model).requires_grad_(False)
    actual_producer, actual_model = copy.deepcopy(producer), copy.deepcopy(model)
    image = torch.randn(1, 1, 5, 6, dtype=torch.float64)
    patches = patch_grid(5, 6, 3, 2, 1)
    loss, _, _ = direct_objective(producer(image), model, image, patches, squared_loss)
    loss.backward()
    calls = []
    def auxiliary(z, y):
        calls.append(1)
        return z.square().mean()
    result, gradient = backward_from_global_response(actual_producer(image), actual_model, image, patches,
        main_loss=squared_loss, auxiliary_loss=auxiliary, strategy=strategy)
    assert calls == [1] and result.aux_loss is not None
    assert (gradient is None) == (freeze == 'producer')
    assert_gradients(actual_producer, producer, rtol=1e-12, atol=1e-12)
    assert_gradients(actual_model, model, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("strategy", ["local", "global_leaf"])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_dropout_batchnorm_and_rng_match_sequential_reference(strategy, diagnostics):
    torch.manual_seed(123)
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 3, 3, padding=1), torch.nn.BatchNorm2d(3),
                                torch.nn.Dropout(0.3), torch.nn.Conv2d(3, 1, 1)).double()
    actual = copy.deepcopy(model)
    response = torch.randn(2, 1, 7, 8, dtype=torch.float64, requires_grad=True)
    actual_response = response.detach().clone().requires_grad_()
    target = torch.randn_like(response)
    patches = patch_grid(7, 8, 3, 2, 1)
    state = torch.get_rng_state()
    loss, _, _ = direct_objective(response, model, target, patches, squared_loss)
    loss.backward()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(state)
    result, gradient = backward_from_global_response(actual_response, actual, target, patches,
        main_loss=squared_loss, strategy=strategy, diagnostics=diagnostics)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert model.training and actual.training
    assert_gradients(actual, model, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(gradient, response.grad, rtol=1e-10, atol=1e-12)
    for got, want in zip(actual.buffers(), model.buffers()):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert bool(result.diagnostics) == diagnostics


def test_partial_coverage_gradients_include_halo_but_not_external_pixels():
    response = torch.ones(1, 1, 8, 9, requires_grad=True)
    model = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
    model.weight.data.fill_(1)
    patch = Patch(2, 3, 2, 1)
    _, gradient = backward_from_global_response(response, model, torch.zeros_like(response), [patch], main_loss=squared_loss)
    assert gradient[..., 1:5, 2:6].gt(0).all()
    gradient[..., 1:5, 2:6] = 0
    assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("invalid", ['target_grad', 'weights_grad', 'negative_weights', 'patch', 'empty', 'weight_nan', 'parameter_loss_response'])
def test_preflight_errors_preserve_existing_gradients(invalid):
    model = torch.nn.Conv2d(1, 1, 1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    response = torch.ones(1, 1, 4, 4, requires_grad=True)
    target = torch.ones_like(response)
    patches = patch_grid(4, 4, 2, 2)
    kwargs = {}
    if invalid == 'target_grad': target.requires_grad_()
    if invalid == 'weights_grad': kwargs['pixel_weights'] = torch.ones_like(target, requires_grad=True)
    if invalid == 'negative_weights': kwargs['pixel_weights'] = -torch.ones_like(target)
    if invalid == 'patch': patches.append(Patch(3, 3, 2))
    if invalid == 'empty': patches = []
    if invalid == 'weight_nan': kwargs['auxiliary_weight'] = float('nan')
    if invalid == 'parameter_loss_response': kwargs['parameter_loss'] = response.sum()
    with pytest.raises((ValueError, FloatingPointError)):
        backward_from_global_response(response, model, target, patches, main_loss=squared_loss, **kwargs)
    assert response.grad is None
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)


@pytest.mark.parametrize("loss", [lambda x, y, w: 1., lambda x, y, w: x.sum().detach(),
                                  lambda x, y, w: x.mean() * float('nan'), lambda x, y, w: x])
def test_invalid_losses_report_patch_and_stage(loss):
    response = torch.ones(1, 1, 4, 4, requires_grad=True)
    with pytest.raises((ValueError, FloatingPointError), match='patch 0 model output/main_loss'):
        backward_from_global_response(response, torch.nn.Identity(), response.detach(), patch_grid(4, 4, 2, 2), main_loss=loss)


def test_differentiable_zero_and_integer_class_targets():
    response = torch.randn(2, 3, 4, 4, requires_grad=True)
    target = torch.randint(0, 3, (2, 1, 4, 4))
    result, gradient = backward_from_global_response(response, torch.nn.Identity(), target, patch_grid(4, 4, 2, 2),
        main_loss=lambda x, y, w: F.cross_entropy(x, y[:, 0]) * 0)
    assert result.total_loss == 0 and torch.count_nonzero(gradient) == 0
    assert response.grad is not None


@pytest.mark.parametrize("strategy", ['local', 'global_leaf'])
def test_local_saved_activations_are_released_between_patches_and_calls(strategy):
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.Tanh(), torch.nn.Conv2d(4, 1, 1))
    parameter_ids = {id(p) for p in model.parameters()}
    references, alive = [], []
    def pack(tensor):
        if id(tensor) not in parameter_ids and tensor is not response:
            references.append(weakref.ref(tensor))
        return tensor
    def before(*_):
        gc.collect()
        alive.append(sum(ref() is not None for ref in references))
    hook = model.register_forward_pre_hook(before)
    for _ in range(3):
        response = torch.randn(1, 1, 7, 8, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            backward_from_global_response(response, model, torch.zeros_like(response), patch_grid(7, 8, 3, 2, 1), main_loss=squared_loss, strategy=strategy)
        model.zero_grad(set_to_none=True)
    hook.remove()
    assert max(alive) == 0
    gc.collect()
    assert not any(ref() is not None for ref in references)


def test_timing_is_independent_of_diagnostics(monkeypatch):
    from mtlearn.training import _patch_diagnostics
    calls = []
    monkeypatch.setattr(_patch_diagnostics, '_synchronize', lambda device: calls.append(device))
    for diagnostics, timing in ((True, False), (False, True)):
        calls.clear()
        response = torch.randn(1, 1, 4, 4, requires_grad=True)
        result, _ = backward_from_global_response(response, torch.nn.Identity(), torch.zeros_like(response),
            patch_grid(4, 4, 2, 2), main_loss=squared_loss, diagnostics=diagnostics, timing=timing)
        assert bool(calls) == timing and bool(result.timings) == timing
        assert bool(result.diagnostics) == diagnostics
        assert all(value >= 0 for value in result.timings.values())


@pytest.mark.parametrize("strategy", ['local', 'global_leaf'])
def test_parameter_loss_and_response_share_upstream_graph(strategy):
    producer = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    trunk = producer.sigmoid()
    response = trunk.square().expand(1, 1, 4, 4)
    penalty = trunk.square() * 0.3
    target = torch.zeros_like(response)
    patches = patch_grid(4, 4, 2, 2)
    expected, _, _ = direct_objective(response, torch.nn.Identity(), target, patches, squared_loss, parameter_loss=penalty)
    expected.backward()
    actual_producer = producer.detach().clone().requires_grad_()
    actual_trunk = actual_producer.sigmoid()
    result, _ = backward_from_global_response(actual_trunk.square().expand_as(response), torch.nn.Identity(), target, patches,
        main_loss=squared_loss, parameter_loss=actual_trunk.square() * 0.3, strategy=strategy,
        diagnostics=True, diagnostic_parameters=(actual_producer,))
    torch.testing.assert_close(actual_producer.grad, producer.grad, rtol=1e-12, atol=1e-12)
    assert result.total_loss == pytest.approx(float(expected.detach()))


def test_auxiliary_and_diagnostics_preserve_existing_retained_response_gradient():
    parameter = torch.tensor(2., requires_grad=True)
    response = (parameter * torch.ones(1, 1, 4, 4)).sin()
    response.retain_grad()
    response.grad = torch.full_like(response, 5.)
    _, gradient = backward_from_global_response(response, torch.nn.Identity(), torch.zeros_like(response),
        patch_grid(4, 4, 2, 2), main_loss=squared_loss, auxiliary_loss=lambda z, y: z.square().mean(),
        auxiliary_weight=0.2, diagnostics=True, diagnostic_parameters=(parameter,))
    torch.testing.assert_close(response.grad, gradient + 5)


def test_parameter_loss_only_updates_its_parameters_when_response_disconnected():
    producer = torch.tensor(2., requires_grad=True)
    regularized = torch.tensor(3., requires_grad=True)
    model = IndependentModel()
    response = producer.expand(1, 1, 4, 4).double()
    _, gradient = backward_from_global_response(response, model, torch.zeros_like(response), patch_grid(4, 4, 2, 2),
        main_loss=squared_loss, parameter_loss=regularized.square(), diagnostics=True, diagnostic_parameters=(producer, regularized))
    assert producer.grad is None
    torch.testing.assert_close(regularized.grad, torch.tensor(6.))
    assert torch.count_nonzero(gradient) == 0
