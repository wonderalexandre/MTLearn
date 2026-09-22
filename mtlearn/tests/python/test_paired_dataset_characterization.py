import inspect
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mtlearn.datasets import PairedImageDataset, _split_indices


def write(path, image):
    assert cv2.imwrite(str(path), image)


@pytest.mark.parametrize('grayscale', [True, False])
@pytest.mark.parametrize('scale', [True, False])
@pytest.mark.parametrize('resize', [None, (3, 2)])
def test_legacy_pixels_order_scales_resize_and_rng(tmp_path, grayscale, scale, resize):
    image = np.arange(24, dtype=np.uint8).reshape(4, 6) * 10
    rgb = np.stack((image, 255-image, image//2), axis=-1)
    for stem in ['2', '10', '1']:
        write(tmp_path / f'x{stem}_in.png', rgb)
        write(tmp_path / f'y{stem}_out.png', image)
    write(tmp_path / 'x0_in.png', image)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    dataset = PairedImageDataset(tmp_path, *(resize or (None, None)), prefix_in='x',
        prefix_target='y', suffix_target='_out', grayscale_in=grayscale, scale_in=scale,
        scale_out=scale, invert_target=True, dtype=torch.float64)
    assert [Path(p[0]).name for p in dataset.pairs] == ['x10_in.png', 'x1_in.png', 'x2_in.png']
    x, y, name = dataset[1]
    raw = cv2.imread(str(tmp_path / name), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if not grayscale:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    target = 255-image
    if resize:
        raw = cv2.resize(raw, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)
    expected = torch.tensor(raw, dtype=torch.float64)
    expected = expected[None] if grayscale else expected.permute(2, 0, 1)
    target = torch.tensor(target, dtype=torch.float64)[None]
    torch.testing.assert_close(x, expected/(255 if scale else 1), rtol=0, atol=0)
    torch.testing.assert_close(y, target/(255 if scale else 1), rtol=0, atol=0)
    assert name == 'x1_in.png'
    train, test = dataset.train_test_split(test_size=1, random_state=42)
    assert train.indices == [1, 2] and test.indices == [0]
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert np.random.get_state()[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


def test_legacy_extension_priority_duplicates_missing_and_resize(tmp_path):
    image = np.full((3, 4), 128, np.uint8)
    write(tmp_path/'1_in.png', image)
    write(tmp_path/'1_in.bmp', image)
    write(tmp_path/'1_target.png', image)
    write(tmp_path/'1_target.bmp', np.ones((2, 2), np.uint8))
    dataset = PairedImageDataset(tmp_path, extensions=('.bmp', '.png'))
    assert len(dataset) == 2
    assert all(Path(p[1]).suffix == '.bmp' for p in dataset.pairs)
    with pytest.raises(ValueError, match='spatial shape'):
        dataset[0]
    dataset = PairedImageDataset(tmp_path, 2, 2, extensions=('.bmp', '.png'))
    assert dataset[0][0].shape == dataset[0][1].shape == (1, 2, 2)
    assert dataset[-1][2] == '1_in.png'


def test_legacy_signature_and_split():
    assert str(inspect.signature(PairedImageDataset)).startswith("(root_dir: 'str', num_rows: 'int | None' = None, num_cols: 'int | None' = None, *, grayscale_in: 'bool' = True")
    train, test = _split_indices(10, test_size=.3, random_state=42)
    assert train.tolist() == [0, 7, 2, 9, 4, 3, 6]
    assert test.tolist() == [8, 1, 5]
    train, test = _split_indices(10, test_size=3, shuffle=False)
    assert train.tolist() == list(range(7)) and test.tolist() == [7, 8, 9]
