"""Compatibility shim for the production CFP layer.

The implementation lives in ``mtlearn.layers.cfp.connected_filter_preprocessing_layer``.
This module preserves the historical import path.
"""

from .cfp.connected_filter_preprocessing_layer import (
    CFPLayer,
    CFPValuation,
    ConnectedFilterPreprocessingImplicitJacobianFunction,
    ConnectedFilterPreprocessingLayer,
)

__all__ = [
    "CFPValuation",
    "ConnectedFilterPreprocessingImplicitJacobianFunction",
    "ConnectedFilterPreprocessingLayer",
    "CFPLayer",
]
