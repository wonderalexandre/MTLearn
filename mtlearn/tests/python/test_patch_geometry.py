import pytest
import torch
import torch.nn.functional as F

from mtlearn.patches import Patch, add_patch_gradient, input_patch, patch_grid, supervised_logits


@pytest.mark.parametrize("shape,size,stride,halo", [((9, 11), 4, 3, 2), ((4, 4), 4, 4, 0), ((7, 8), 3, 3, 12)])
def test_grid_covers_domain_in_row_major_order(shape, size, stride, halo):
    patches = patch_grid(*shape, size, stride, halo)
    positions = [(patch.row, patch.col) for patch in patches]
    assert positions == sorted(set(positions))
    assert positions[0] == (0, 0)
    assert positions[-1] == (shape[0] - size, shape[1] - size)
    coverage = torch.zeros(shape, dtype=torch.int64)
    for patch in patches:
        assert patch.size == size and patch.halo == halo
        coverage[patch.target] += 1
    assert (coverage > 0).all()


@pytest.mark.parametrize("field,value", [("row", -1), ("col", True), ("size", 0), ("size", 2.5), ("halo", -1), ("halo", False)])
def test_manual_patch_rejects_invalid_values(field, value):
    values = dict(row=0, col=0, size=2, halo=1)
    values[field] = value
    with pytest.raises(ValueError, match=field):
        Patch(**values)


@pytest.mark.parametrize("args", [(4, 4, 5, 1), (4, 4, 2, 3), (True, 4, 2, 1), (4, 4, 2, 1.0), (0, 4, 2, 1)])
def test_grid_rejects_invalid_dimensions(args):
    with pytest.raises(ValueError):
        patch_grid(*args)


@pytest.mark.parametrize("patch", [Patch(0, 0, 3), Patch(0, 0, 3, 9), Patch(2, 4, 3, 2)])
def test_extraction_and_adjoint_match_independent_padding(patch):
    torch.manual_seed(8)
    image = torch.randn(2, 3, 7, 5, dtype=torch.float64).transpose(-1, -2).requires_grad_()
    actual, source, local = input_patch(image, patch)
    side = patch.size + 2 * patch.halo
    expected = F.pad(image, (patch.halo,) * 4)[..., patch.row:patch.row + side, patch.col:patch.col + side]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    vector = torch.randn_like(actual)
    expected_gradient = torch.autograd.grad((expected * vector).sum(), image)[0]
    gradient = torch.zeros_like(image)
    add_patch_gradient(gradient, vector, source, local)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close((actual * vector).sum(), (image * gradient).sum(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(supervised_logits(actual, patch), image[..., patch.target[0], patch.target[1]])


def test_adjoint_sums_overlapping_halos():
    image = torch.zeros(2, 3, 6, 7, dtype=torch.float64, requires_grad=True)
    actual = torch.zeros_like(image)
    expected_loss = image.sum() * 0
    for patch in patch_grid(6, 7, 3, 2, 2):
        local_input, source, local = input_patch(image, patch)
        add_patch_gradient(actual, torch.ones_like(local_input), source, local)
        side = patch.size + 2 * patch.halo
        expected_loss = expected_loss + F.pad(image, (patch.halo,) * 4)[..., patch.row:patch.row + side, patch.col:patch.col + side].sum()
    torch.testing.assert_close(actual, torch.autograd.grad(expected_loss, image)[0])


@pytest.mark.parametrize("source,local", [
    ((slice(-1, 2), slice(0, 2)), (slice(0, 3), slice(0, 2))),
    ((slice(0, 3), slice(0, 2)), (slice(0, 2), slice(0, 2))),
    ((slice(0, 5), slice(0, 2)), (slice(0, 5), slice(0, 2))),
    ((slice(0, 2, 2), slice(0, 2)), (slice(0, 2), slice(0, 2))),
    ((slice(None, 2), slice(0, 2)), (slice(0, 2), slice(0, 2))),
])
def test_adjoint_validates_slices_before_mutation(source, local):
    gradient = torch.zeros(1, 2, 4, 4)
    with pytest.raises(ValueError):
        add_patch_gradient(gradient, torch.ones_like(gradient), source, local)
    assert torch.count_nonzero(gradient) == 0


def test_extraction_and_logits_validate_geometry():
    with pytest.raises(ValueError, match="within"):
        input_patch(torch.zeros(1, 1, 4, 4), Patch(2, 0, 3))
    with pytest.raises(ValueError, match="BCHW"):
        input_patch(torch.zeros(3, 4, 4), Patch(0, 0, 3))
    with pytest.raises(ValueError, match="spatial"):
        supervised_logits(torch.zeros(1, 1, 3, 3), Patch(0, 0, 3, 1))
