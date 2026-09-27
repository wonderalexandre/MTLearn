import pytest
import torch
import torch.nn.functional as F

from mtlearn.patches import Patch, patch_grid, reconstruct_logits


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("halo", [0, 2, 12])
def test_identity_reconstruction_is_exact_for_representable_values(dtype, halo):
    image = torch.arange(2 * 3 * 7 * 9, dtype=dtype).reshape(2, 3, 7, 9)
    actual = reconstruct_logits(torch.nn.Identity(), image, iter(patch_grid(7, 9, 3, 2, halo)))
    torch.testing.assert_close(actual, image, rtol=0, atol=0)
    assert actual.dtype == image.dtype and not actual.requires_grad


def test_multichannel_reconstruction_matches_independent_average_and_calls_once_per_patch():
    torch.manual_seed(3)
    model = torch.nn.Conv2d(3, 2, 3, padding=1).double().train()
    image = torch.randn(2, 3, 7, 9, dtype=torch.float64)
    patches = patch_grid(7, 9, 3, 2, 2)
    expected = torch.zeros(2, 2, 7, 9, dtype=torch.float64)
    coverage = torch.zeros(7, 9, dtype=torch.int64)
    with torch.no_grad():
        for patch in patches:
            side = patch.size + 2 * patch.halo
            local = F.pad(image, (patch.halo,) * 4)[..., patch.row:patch.row + side, patch.col:patch.col + side]
            logits = model(local)[..., patch.halo:patch.halo + patch.size, patch.halo:patch.halo + patch.size]
            expected[..., patch.target[0], patch.target[1]] += logits
            coverage[patch.target] += 1
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(torch.is_grad_enabled()))
    actual = reconstruct_logits(model, image, patches)
    handle.remove()
    torch.testing.assert_close(actual, expected / coverage, rtol=1e-12, atol=1e-12)
    assert calls == [False] * len(patches) and model.training
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("patches", [[], [Patch(0, 0, 2)], [Patch(0, 0, 5)]])
def test_invalid_coverage_is_rejected_before_forward(patches):
    def model(_):
        raise AssertionError("forward must not run")
    with pytest.raises(ValueError):
        reconstruct_logits(model, torch.zeros(1, 1, 4, 4), patches)


@pytest.mark.parametrize("bad", [lambda x: (x,), lambda x: x[:, :, :-1], lambda x: x[:0], lambda x: x * float("nan")])
def test_invalid_model_output_has_patch_context(bad):
    with pytest.raises((ValueError, TypeError, FloatingPointError), match="patch 0 reconstruction"):
        reconstruct_logits(bad, torch.zeros(1, 1, 4, 4), patch_grid(4, 4, 2, 2))


def test_output_dtype_is_inferred_and_must_remain_consistent():
    image = torch.ones(1, 1, 4, 4)
    patches = patch_grid(4, 4, 2, 2)
    assert reconstruct_logits(lambda x: x.double(), image, patches).dtype == torch.float64
    calls = []
    def varying(x):
        calls.append(None)
        return x if len(calls) == 1 else x.double()
    with pytest.raises(ValueError, match="patch 1.*dtype"):
        reconstruct_logits(varying, image, patches)
