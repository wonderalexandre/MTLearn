"""Extensible connected-filter preprocessing components.

This package is the migration target for the production
``ConnectedFilterPreprocessingLayer``. The first refactoring step keeps the
existing layer as the runtime implementation and exposes small extension
interfaces for future scoring models, constraints, and regularizers.
"""

from .constraints import PreserveRootConstraint, ScoreConstraint
from .normalization import AttributeNormalizer, StatsSerializer
from .regularization import (
    AttributeOrderScoreMonotonicityRegularizer,
    EdgeScoreMonotonicityRegularizer,
    PathScoreMonotonicityRegularizer,
    Regularizer,
)
from .runtime import (
    BatchInput,
    BatchInputNormalizer,
    CachedDataLoaderBuilder,
    CFPCacheInputError,
    CFPContext,
    ConnectedFilterPreprocessingImplicitJacobianFunction,
    ForwardExecutor,
    TrainingSampleInspector,
    TreePayloadCache,
    TreePayloadProvider,
    TreeReconstructionFunction,
    TreeReconstructor,
    validate_cfp_cache_batch_x,
)
from .scoring import (
    LegacyLinearParameterInitializer,
    LinearSigmoidScorer,
    MLPScorer,
    ScoringModel,
)
from .serialization import ConfigDeserializer, ConfigSerializer, PersistentStateManager
from .specs import FeatureSpec, FilterSpec, SpecRegistry, TreeSpec
from .component_registries import (
    REGULARIZER_REGISTRY,
    SCORE_CONSTRAINT_REGISTRY,
    SCORING_MODEL_REGISTRY,
)
from .connected_filter_preprocessing_layer import (
    CFPLayer,
    ConnectedFilterPreprocessingLayer,
)

__all__ = [
    "AttributeNormalizer",
    "AttributeOrderScoreMonotonicityRegularizer",
    "BatchInput",
    "BatchInputNormalizer",
    "CachedDataLoaderBuilder",
    "CFPCacheInputError",
    "CFPContext",
    "CFPLayer",
    "ConfigDeserializer",
    "ConfigSerializer",
    "ConnectedFilterPreprocessingImplicitJacobianFunction",
    "ConnectedFilterPreprocessingLayer",
    "FeatureSpec",
    "FilterSpec",
    "ForwardExecutor",
    "LegacyLinearParameterInitializer",
    "LinearSigmoidScorer",
    "MLPScorer",
    "EdgeScoreMonotonicityRegularizer",
    "PathScoreMonotonicityRegularizer",
    "PersistentStateManager",
    "PreserveRootConstraint",
    "REGULARIZER_REGISTRY",
    "Regularizer",
    "SCORE_CONSTRAINT_REGISTRY",
    "SCORING_MODEL_REGISTRY",
    "ScoreConstraint",
    "ScoringModel",
    "SpecRegistry",
    "StatsSerializer",
    "TreePayloadCache",
    "TreePayloadProvider",
    "TreeReconstructionFunction",
    "TreeReconstructor",
    "TrainingSampleInspector",
    "TreeSpec",
    "validate_cfp_cache_batch_x",
]
