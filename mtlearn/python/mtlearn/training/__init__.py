"""First-order training through local patches of a global response."""

from ._patch_backward import PatchBackwardResult, backward_from_global_response

__all__ = ["PatchBackwardResult", "backward_from_global_response"]
