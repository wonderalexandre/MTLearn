"""CFP training regularizers."""

from .base import Regularizer
from .monotone_scores import MonotoneScoresRegularizer

__all__ = [
    "MonotoneScoresRegularizer",
    "Regularizer",
]
