"""Extensible connected-filter preprocessing components."""

from .constraints import PreserveRootConstraint, ScoreConstraint
from .regularization import (
    AttributeOrderScoreMonotonicityRegularizer,
    EdgeScoreMonotonicityRegularizer,
    PathScoreMonotonicityRegularizer,
    Regularizer,
)
from .scoring import (
    LinearSigmoidScorer,
    MLPScorer,
    ScoringModel,
)
from .preparation import CFPPreprocessor, PreparedMorphology, PreparedBatch, PreparationResult, DiskPreparationResult, PreparedDataset, collate_prepared
from .storage import NullStore, MemoryStore, DiskStore
from .preparation import PreparedDataLoader, PreparedDataLoadingError, build_prepared_dataloader
from .preparation import calibrate_disk_cache
from .normalization.statistics_snapshot import StatisticsSnapshot
from .specs import FeatureSpec, FilterSpec, SpecRegistry, TreeSpec
from .component_registries import (
    REGULARIZER_REGISTRY,
    SCORE_CONSTRAINT_REGISTRY,
    SCORING_MODEL_REGISTRY,
)
from .connected_filter_preprocessing_layer import (
    ConnectedFilterPreprocessingLayer,
)

__all__ = [
    "AttributeOrderScoreMonotonicityRegularizer",
    "ConnectedFilterPreprocessingLayer",
    "CFPPreprocessor",
    "PreparedMorphology",
    "PreparedBatch",
    "NullStore",
    "MemoryStore",
    "DiskStore",
    "PreparationResult",
    "DiskPreparationResult",
    "PreparedDataLoader",
    "PreparedDataLoadingError",
    "build_prepared_dataloader",
    "calibrate_disk_cache",
    "PreparedDataset",
    "collate_prepared",
    "StatisticsSnapshot",
    "FeatureSpec",
    "FilterSpec",
    "LinearSigmoidScorer",
    "MLPScorer",
    "EdgeScoreMonotonicityRegularizer",
    "PathScoreMonotonicityRegularizer",
    "PreserveRootConstraint",
    "REGULARIZER_REGISTRY",
    "Regularizer",
    "SCORE_CONSTRAINT_REGISTRY",
    "SCORING_MODEL_REGISTRY",
    "ScoreConstraint",
    "ScoringModel",
    "SpecRegistry",
    "TreeSpec",
]
