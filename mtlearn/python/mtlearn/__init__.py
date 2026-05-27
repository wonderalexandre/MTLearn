"""Python interface for mtlearn."""

from importlib import import_module

try:
    # Torch must be imported before the native module so the compiled bindings
    # can use its Tensor converters.
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - explicit runtime failure
    raise ImportError("mtlearn requires torch to load its compiled bindings") from exc


from . import morphology
from . import layers
from ._native import load_bindings

_bindings = load_bindings()

WITH_TORCH = getattr(_bindings, "WITH_TORCH", False)

ConnectedFilterPreprocessingTreeTensors = getattr(_bindings, "ConnectedFilterPreprocessingTreeTensors", None)
ConnectedFilterPreprocessingTreeTraversal = getattr(_bindings, "ConnectedFilterPreprocessingTreeTraversal", None)


__all__ = [
    "WITH_TORCH",
    "data",
    "datasets",
    "layers",
    "morphology",
    "ConnectedFilterPreprocessingTreeTensors",
    "ConnectedFilterPreprocessingTreeTraversal",
]

__version__ = "1.0.7"


def __getattr__(name: str):
    """Load optional public submodules only when users ask for them."""

    if name in {"data", "datasets"}:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
