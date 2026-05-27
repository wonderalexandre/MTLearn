import pytest

import mtlearn

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Python dependency unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology
from mtlearn.layers import (
    ConnectedFilterPreprocessingLayer,
    ConnectedFilterPreprocessingLayerWithCPUTreeTraversal,
    ConnectedFilterPreprocessingLayerWithExplicitJacobian,
    collect_cfp_configs,
    load_checkpoint,
    save_checkpoint,
)

pytestmark = pytest.mark.integration


def _tiny_dataset_loader(batch_size=2):
    x = torch.tensor(
        [
            [[[2, 2, 0], [2, 5, 0], [3, 3, 1]]],
            [[[1, 0, 1], [4, 4, 2], [0, 2, 2]]],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _single_area_layer(layer_cls=ConnectedFilterPreprocessingLayer, *, scale_mode="minmax01"):
    if layer_cls is ConnectedFilterPreprocessingLayer:
        return layer_cls(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                }
            ],
            device="cpu",
            scale_mode=scale_mode,
            beta_f=1.0,
            clamp=None,
        )
    return layer_cls(
        in_channels=1,
        attributes_spec=[(morphology.AttributeType.AREA,)],
        tree_type="max-tree",
        device="cpu",
        scale_mode=scale_mode,
        beta_f=1.0,
        clamp_logits=False,
    )


class _TinyBackbone(torch.nn.Module):
    def __init__(self, cfp_layer):
        super().__init__()
        self.cfp = cfp_layer
        self.head = torch.nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.head(self.cfp(x))


class _TwoCfpBackbone(torch.nn.Module):
    def __init__(self, first_cfp, second_cfp):
        super().__init__()
        self.pre = first_cfp
        self.branch = torch.nn.Module()
        self.branch.cfp = second_cfp


def _named_primary_layer(name, attribute, *, scale_mode="minmax01"):
    return ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "name": name,
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (attribute,),
            }
        ],
        device="cpu",
        scale_mode=scale_mode,
        beta_f=1.0,
        clamp=None,
    )


def test_build_dataloader_cached_populates_primary_layer_cache():
    layer = _single_area_layer(scale_mode="minmax01")
    loader = _tiny_dataset_loader()

    cached_loader = layer.build_dataloader_cached(loader)

    assert layer._stats_frozen is True
    assert set(layer._tree_info) == {"0_0", "1_0"}
    assert set(layer._base_attrs) == {"0_0", "1_0"}
    assert set(layer._norm_attrs) == {"0_0", "1_0"}
    assert layer._stats_epoch > 0
    assert all(epoch == layer._stats_epoch for epoch in layer._norm_epoch_by_key.values())

    ((x, idx), y) = next(iter(cached_loader))
    assert x.shape == (2, 1, 3, 3)
    assert idx.tolist() == [0, 1]
    assert y.tolist() == [0, 1]


def test_cached_forward_matches_uncached_forward_with_same_parameters():
    x, _ = next(iter(_tiny_dataset_loader()))
    cached_layer = _single_area_layer(scale_mode="none")
    plain_layer = _single_area_layer(scale_mode="none")
    plain_layer.load_state_dict(cached_layer.state_dict())

    cached_loader = cached_layer.build_dataloader_cached(_tiny_dataset_loader())
    cached_input, _ = next(iter(cached_loader))

    y_cached = cached_layer(cached_input)
    y_plain = plain_layer(x)

    assert y_cached.shape == y_plain.shape
    assert torch.allclose(y_cached, y_plain)


def test_freeze_and_unfreeze_dataset_stats_controls_stat_updates():
    layer = _single_area_layer(scale_mode="minmax01")
    attr = morphology.AttributeType.AREA
    stat_key = layer._stat_key(layer.filter_specs[0].tree_key, attr)

    layer._update_ds_stats(stat_key, torch.tensor([2.0, 4.0]))
    initial_epoch = layer._stats_epoch
    initial_min = layer._ds_stats[stat_key]["amin"].clone()

    layer.freeze_ds_stats()
    layer._update_ds_stats(stat_key, torch.tensor([0.0, 10.0]))

    assert layer._stats_epoch == initial_epoch
    assert torch.equal(layer._ds_stats[stat_key]["amin"], initial_min)

    layer.unfreeze_ds_stats()
    layer._update_ds_stats(stat_key, torch.tensor([0.0, 10.0]))

    assert layer._stats_epoch == initial_epoch + 1
    assert layer._ds_stats[stat_key]["amin"].item() == 0.0


def test_refresh_cached_normalization_uses_latest_stats():
    layer = _single_area_layer(scale_mode="minmax01")
    layer.build_dataloader_cached(_tiny_dataset_loader())

    attr = morphology.AttributeType.AREA
    tree_key = layer.filter_specs[0].tree_key
    stat_key = layer._stat_key(tree_key, attr)
    before = layer._norm_attrs["0_0"][tree_key][attr].clone()

    layer._ds_stats[stat_key]["amin"] = torch.tensor(0.0)
    layer._ds_stats[stat_key]["amax"] = torch.tensor(100.0)
    layer._stats_epoch += 1
    layer.refresh_cached_normalization()

    after = layer._norm_attrs["0_0"][tree_key][attr]
    assert not torch.allclose(after, before)
    assert layer._norm_epoch_by_key["0_0"] == layer._stats_epoch


@pytest.mark.parametrize(
    "layer_cls",
    [
        ConnectedFilterPreprocessingLayer,
        ConnectedFilterPreprocessingLayerWithExplicitJacobian,
        ConnectedFilterPreprocessingLayerWithCPUTreeTraversal,
    ],
)
def test_load_stats_roundtrip_after_save(tmp_path, layer_cls):
    source = _single_area_layer(layer_cls, scale_mode="minmax01")
    source.build_dataloader_cached(_tiny_dataset_loader())
    stats_path = tmp_path / "stats.pt"

    source.save_stats(str(stats_path))

    target = _single_area_layer(layer_cls, scale_mode="minmax01")
    target.load_stats(str(stats_path))

    assert target._ds_stats.keys() == source._ds_stats.keys()
    attr = morphology.AttributeType.AREA
    if layer_cls is ConnectedFilterPreprocessingLayer:
        attr = source._stat_key(source.filter_specs[0].tree_key, attr)
    assert torch.equal(target._ds_stats[attr]["amin"], source._ds_stats[attr]["amin"])
    assert torch.equal(target._ds_stats[attr]["amax"], source._ds_stats[attr]["amax"])


def test_primary_layer_state_dict_extra_state_roundtrip_inside_backbone(tmp_path):
    source_cfp = _single_area_layer(ConnectedFilterPreprocessingLayer, scale_mode="minmax01")
    source_cfp.build_dataloader_cached(_tiny_dataset_loader())
    source_model = _TinyBackbone(source_cfp)
    with torch.no_grad():
        source_model.cfp._weights["spec_000"].fill_(0.25)
        source_model.cfp._biases["spec_000"].fill_(-0.5)
        source_model.head.weight.fill_(2.0)
        source_model.head.bias.fill_(0.75)

    checkpoint_path = tmp_path / "model_state.pt"
    torch.save(source_model.state_dict(), checkpoint_path)
    loaded_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert "cfp._extra_state" in loaded_state

    target_cfp = ConnectedFilterPreprocessingLayer.from_config(source_cfp.get_config(), device="cpu")
    target_model = _TinyBackbone(target_cfp)
    target_model.load_state_dict(loaded_state)

    stat_key = source_cfp._stat_key(source_cfp.filter_specs[0].tree_key, morphology.AttributeType.AREA)
    assert target_model.cfp.get_config() == source_cfp.get_config()
    assert target_model.cfp._stats_frozen is True
    assert torch.equal(target_model.cfp._ds_stats[stat_key]["amin"], source_cfp._ds_stats[stat_key]["amin"])
    assert torch.equal(target_model.cfp._ds_stats[stat_key]["amax"], source_cfp._ds_stats[stat_key]["amax"])
    assert torch.equal(target_model.cfp._weights["spec_000"], source_model.cfp._weights["spec_000"])
    assert torch.equal(target_model.head.weight, source_model.head.weight)


def test_primary_layer_state_dict_rejects_incompatible_extra_state():
    source = _single_area_layer(ConnectedFilterPreprocessingLayer, scale_mode="minmax01")
    incompatible = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MIN_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            }
        ],
        device="cpu",
        scale_mode="minmax01",
    )

    with pytest.raises(RuntimeError, match="checkpoint weight contract is incompatible"):
        incompatible.load_state_dict(source.state_dict())


def test_checkpoint_helpers_roundtrip_model_with_cfp_factory(tmp_path):
    source_cfp = _single_area_layer(ConnectedFilterPreprocessingLayer, scale_mode="minmax01")
    source_cfp.build_dataloader_cached(_tiny_dataset_loader())
    source_model = _TinyBackbone(source_cfp)
    with torch.no_grad():
        source_model.cfp._weights["spec_000"].fill_(0.25)
        source_model.cfp._biases["spec_000"].fill_(-0.5)
        source_model.head.weight.fill_(2.0)
        source_model.head.bias.fill_(0.75)

    checkpoint_path = tmp_path / "model_checkpoint.pt"
    saved_payload = save_checkpoint(
        checkpoint_path,
        source_model,
    )

    assert saved_payload["cfp_configs"] == {"cfp": source_model.cfp.get_config()}
    assert "optimizer_state_dict" not in saved_payload

    def model_factory(cfp_configs):
        cfp_layer = ConnectedFilterPreprocessingLayer.from_config(cfp_configs["cfp"], device="cpu")
        return _TinyBackbone(cfp_layer)

    loaded_model, checkpoint = load_checkpoint(checkpoint_path, model_factory, device="cpu")

    stat_key = source_cfp._stat_key(source_cfp.filter_specs[0].tree_key, morphology.AttributeType.AREA)
    assert "optimizer_state_dict" not in checkpoint
    assert loaded_model.cfp.get_config() == source_model.cfp.get_config()
    assert torch.equal(loaded_model.cfp._ds_stats[stat_key]["amin"], source_model.cfp._ds_stats[stat_key]["amin"])
    assert torch.equal(loaded_model.cfp._weights["spec_000"], source_model.cfp._weights["spec_000"])
    assert torch.equal(loaded_model.head.weight, source_model.head.weight)


def test_checkpoint_helpers_load_existing_model(tmp_path):
    source_cfp = _single_area_layer(ConnectedFilterPreprocessingLayer, scale_mode="minmax01")
    source_model = _TinyBackbone(source_cfp)
    checkpoint_path = tmp_path / "model_checkpoint.pt"
    save_checkpoint(checkpoint_path, source_model)

    target_cfp = ConnectedFilterPreprocessingLayer.from_config(source_cfp.get_config(), device="cpu")
    target_model = _TinyBackbone(target_cfp)

    loaded_model, _ = load_checkpoint(
        checkpoint_path,
        target_model,
        device="cpu",
    )

    assert loaded_model is target_model


def test_checkpoint_helpers_load_no_arg_factory(tmp_path):
    source_cfp = _single_area_layer(ConnectedFilterPreprocessingLayer, scale_mode="minmax01")
    source_model = _TinyBackbone(source_cfp)
    with torch.no_grad():
        source_model.cfp._weights["spec_000"].fill_(0.4)
        source_model.head.bias.fill_(0.2)
    checkpoint_path = tmp_path / "model_checkpoint.pt"
    save_checkpoint(checkpoint_path, source_model)

    def model_factory():
        return _TinyBackbone(_single_area_layer(ConnectedFilterPreprocessingLayer, scale_mode="minmax01"))

    loaded_model, _ = load_checkpoint(checkpoint_path, model_factory, device="cpu")

    assert torch.equal(loaded_model.cfp._weights["spec_000"], source_model.cfp._weights["spec_000"])
    assert torch.equal(loaded_model.head.bias, source_model.head.bias)


def test_checkpoint_helpers_roundtrip_multiple_cfps(tmp_path):
    first = _named_primary_layer("area_filter", morphology.AttributeType.AREA)
    second = _named_primary_layer("gray_filter", morphology.AttributeType.GRAY_HEIGHT)
    first.build_dataloader_cached(_tiny_dataset_loader())
    second.build_dataloader_cached(_tiny_dataset_loader())
    source_model = _TwoCfpBackbone(first, second)
    with torch.no_grad():
        source_model.pre._weights["area_filter"].fill_(0.3)
        source_model.branch.cfp._weights["gray_filter"].fill_(-0.4)

    checkpoint_path = tmp_path / "multi_cfp_checkpoint.pt"
    payload = save_checkpoint(checkpoint_path, source_model)

    assert set(payload["cfp_configs"]) == {"pre", "branch.cfp"}
    assert payload["cfp_configs"]["pre"] == source_model.pre.get_config()
    assert payload["cfp_configs"]["branch.cfp"] == source_model.branch.cfp.get_config()

    def model_factory():
        return _TwoCfpBackbone(
            _named_primary_layer("area_filter", morphology.AttributeType.AREA),
            _named_primary_layer("gray_filter", morphology.AttributeType.GRAY_HEIGHT),
        )

    loaded_model, _ = load_checkpoint(checkpoint_path, model_factory, device="cpu")

    first_stat_key = first._stat_key(first.filter_specs[0].tree_key, morphology.AttributeType.AREA)
    second_stat_key = second._stat_key(second.filter_specs[0].tree_key, morphology.AttributeType.GRAY_HEIGHT)
    assert torch.equal(loaded_model.pre._weights["area_filter"], source_model.pre._weights["area_filter"])
    assert torch.equal(loaded_model.branch.cfp._weights["gray_filter"], source_model.branch.cfp._weights["gray_filter"])
    assert torch.equal(loaded_model.pre._ds_stats[first_stat_key]["amin"], source_model.pre._ds_stats[first_stat_key]["amin"])
    assert torch.equal(
        loaded_model.branch.cfp._ds_stats[second_stat_key]["amin"],
        source_model.branch.cfp._ds_stats[second_stat_key]["amin"],
    )


def test_checkpoint_helpers_do_not_accept_optimizer_state(tmp_path):
    model = _TinyBackbone(_single_area_layer(ConnectedFilterPreprocessingLayer, scale_mode="none"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    with pytest.raises(TypeError, match="optimizer"):
        save_checkpoint(tmp_path / "model_checkpoint.pt", model, optimizer=optimizer)

    checkpoint_path = tmp_path / "model_checkpoint.pt"
    save_checkpoint(checkpoint_path, model)
    with pytest.raises(TypeError, match="optimizer"):
        load_checkpoint(checkpoint_path, model, optimizer=optimizer)


def test_collect_cfp_configs_discovers_primary_layers_only():
    model = torch.nn.Module()
    model.cfp = _single_area_layer(ConnectedFilterPreprocessingLayer, scale_mode="none")
    model.legacy = _single_area_layer(ConnectedFilterPreprocessingLayerWithExplicitJacobian, scale_mode="none")

    configs = collect_cfp_configs(model)

    assert list(configs) == ["cfp"]
    assert configs["cfp"] == model.cfp.get_config()
