# PyTorch Integration

The primary CFP layer is a normal `torch.nn.Module`: its weights and biases are
learnable parameters, it participates in `state_dict`, and it can be composed
with standard PyTorch modules. Tree construction and attribute computation stay
outside autograd.

## Use CFP in a Model

```python
import torch
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer


class SmallModel(torch.nn.Module):
    def __init__(self, *, cfp_scale_mode="hybrid", cfp_device="cpu"):
        super().__init__()
        self.cfp = ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "name": "area",
                    "tree_type": "max-tree",
                    "attributes": morphology.AttributeType.AREA,
                },
                {
                    "name": "shape",
                    "tree_type": "tree-of-shapes",
                    "attributes": morphology.AttributeGroup.SHAPE,
                },
            ],
            scale_mode=cfp_scale_mode,
            device=cfp_device,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(self.cfp.out_channels, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.head(self.cfp(x))
```

The CFP output channel count is `in_channels * len(filter_specs)`. Use
`self.cfp.out_channels` when wiring the next layer.

## Cache Dataset Statistics

For `scale_mode="hybrid"`, build a cached DataLoader before training. The
wrapped loader yields `((x, idx), y)` so the CFP layer can reuse tree payloads
by stable dataset index.

```python
from torch.utils.data import DataLoader

model = SmallModel()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
train_loader_cached = model.cfp.build_dataloader_cached(train_loader)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for (x, idx), target in train_loader_cached:
        optimizer.zero_grad(set_to_none=True)
        logits = model((x, idx))
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
```

If you do not use the cached loader, pass ordinary tensors to the model. This
is useful for quick checks, but direct forward passes do not update hybrid
dataset statistics.

```python
debug_model = SmallModel(cfp_scale_mode="minmax01")
logits = debug_model(torch.rand(2, 1, 32, 32))
```

## Device and dtype

The layer stores trainable parameters and CFP tensors on its `device`.
Morphology-tree construction runs in the native CPU backend, so image tensors
are copied to CPU and converted to uint8 before tree construction.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=filter_specs,
    device=device,
)
```

For full models, pass the same device into the CFP constructor and then move
the surrounding model as usual.

```python
model = SmallModel(cfp_device=device).to(device)
```

Keep inputs in the image range expected by the conversion helper:

- tensors with max value `<= 1.5` are treated as normalized images in `[0, 1]`
  and scaled to uint8;
- other tensors are cast directly to uint8.

## Checkpoints

Use `mtlearn.layers.save_checkpoint` and `load_checkpoint` for models that
contain CFP layers. The helpers save ordinary PyTorch weights plus CFP configs
that describe the meaning and shape of CFP parameters.

```python
from mtlearn.layers import load_checkpoint, save_checkpoint

save_checkpoint("model.pt", model)


def model_factory(cfp_configs):
    model = SmallModel()
    if "cfp" in cfp_configs:
        model.cfp = ConnectedFilterPreprocessingLayer.from_config(
            cfp_configs["cfp"],
            device=device,
        )
    return model


loaded_model, checkpoint = load_checkpoint("model.pt", model_factory)
```

You can also pass an already constructed model:

```python
loaded_model, checkpoint = load_checkpoint("model.pt", SmallModel())
```

## Inference

Use `predict` on the CFP layer when you want hard-gate-like behavior during
evaluation.

```python
model.eval()
with torch.no_grad():
    features = model.cfp.predict(x, beta_f=1000.0)
    logits = model.head(features)
```

If the model was trained with cached hybrid statistics, load stats or restore a
checkpoint before inference.

```python
model.cfp.save_stats("cfp-stats.pt")
model.cfp.load_stats("cfp-stats.pt")
```

## Debugging Training

Inspect one sample when loss is unstable or CFP outputs look wrong.

```python
sample, target = train_dataset[0]
report = model.cfp.inspect_training_sample(sample, idx=0)

for name, spec_report in report["specs"].items():
    print(name)
    print("raw:", spec_report["base_attrs"].shape)
    print("norm:", spec_report["norm_attrs"].shape)
    print("weight:", spec_report["weight"].detach())
    print("bias:", spec_report["bias"].detach())
```

Common issues:

- `scale_mode="hybrid"` without cached or loaded stats raises at forward time.
- Reordered unnamed specs can make old checkpoints incompatible.
- Inputs outside `[0, 1]` may be cast to uint8 directly.
- Tree construction is CPU-side preprocessing, so very large batches can spend
  most time outside GPU kernels unless caches are used.
