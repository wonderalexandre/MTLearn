import pytest
import torch

from mtlearn.layers.cfp.runtime.tree_reconstructor import (
    TreeReconstructionFunction, propagate_pixels_to_nodes, reconstruct_from_info,
)

pytestmark = pytest.mark.integration


def require_device(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_interval_reconstruction_matches_supports_and_adjoint(device, dtype):
    require_device(device)
    pre = torch.tensor([1, 3, 0, 2, 0], device=device)
    post = torch.tensor([3, 4, 4, 3, 0], device=device)
    owners = torch.tensor([3, 0, 1, 2, 3], dtype=torch.uint32, device=device)
    supports = torch.tensor([[1, 1, 0, 0, 1], [0, 0, 1, 0, 0],
                             [1, 1, 1, 1, 1], [1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0]], dtype=dtype)
    signal = torch.tensor([.25, -1., 2., .5, .125], dtype=dtype, device=device, requires_grad=True)
    probe = torch.tensor([.25, -.5, 1.25, -.25, .75], dtype=dtype, device=device)
    output = TreeReconstructionFunction.apply(signal, pre, post, owners, 1, 5)
    assert output.device == signal.device and output.dtype == dtype
    torch.testing.assert_close(output.cpu().flatten(), supports.T @ signal.detach().cpu(), rtol=1e-5, atol=1e-6)
    output.backward(probe.reshape(1, 5))
    torch.testing.assert_close(signal.grad.cpu(), supports @ probe.cpu(), rtol=1e-5, atol=1e-6)
    assert signal.grad.device == signal.device
    assert signal.grad[-1] == 0


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("cpu_threads", [1, 2])
def test_repeated_endpoints_and_pixel_owners_reproduce_cpu_order(device, cpu_threads):
    require_device(device)
    previous_threads = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(cpu_threads)
        generator = torch.Generator().manual_seed(413)
        count = 4096
        pre_cpu = torch.randperm(count, generator=generator)
        post_cpu = torch.full((count,), count)
        owners_cpu = torch.randint(count, (65536,), generator=generator).to(torch.uint32)
        signal_cpu = torch.rand(count, generator=generator) * .25 - .125
        probe_cpu = torch.rand(65536, generator=generator) - .5
        expected_output = reconstruct_from_info(signal_cpu, pre_cpu, post_cpu, owners_cpu)
        expected_gradient = propagate_pixels_to_nodes(probe_cpu, pre_cpu, post_cpu, owners_cpu)
        pre, post, owners, signal, probe = [value.to(device) for value in
            (pre_cpu, post_cpu, owners_cpu, signal_cpu, probe_cpu)]
        for _ in range(5):
            output = reconstruct_from_info(signal, pre, post, owners)
            gradient = propagate_pixels_to_nodes(probe, pre, post, owners)
            torch.testing.assert_close(output.cpu(), expected_output, rtol=0, atol=0)
            torch.testing.assert_close(gradient.cpu(), expected_gradient, rtol=0, atol=0)
            assert output.device == signal.device and gradient.device == probe.device
        assert torch.are_deterministic_algorithms_enabled() == deterministic_before
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only_before
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_reconstruction_retained_graph_and_second_derivative(device):
    require_device(device)
    pre = torch.tensor([1, 0, 2], device=device)
    post = torch.tensor([2, 3, 3], device=device)
    owners = torch.tensor([0, 1, 2, 2], dtype=torch.uint32, device=device)
    supports = torch.tensor([[1., 0., 0., 0.], [1., 1., 1., 1.], [0., 0., 1., 1.]])
    signal = torch.tensor([.5, -.25, .125], device=device, requires_grad=True)
    output = TreeReconstructionFunction.apply(signal, pre, post, owners, 2, 2)
    first, = torch.autograd.grad(output.square().sum(), signal, create_graph=True, retain_graph=True)
    repeated, = torch.autograd.grad(output.square().sum(), signal, retain_graph=True)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    second, = torch.autograd.grad(first.sum(), signal)
    torch.testing.assert_close(first.cpu(), 2 * supports @ (supports.T @ signal.detach().cpu()), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(second.cpu(), 2 * supports @ supports.T @ torch.ones(3), rtol=1e-5, atol=1e-6)
