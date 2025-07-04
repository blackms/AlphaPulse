"""Online learning module for real-time model adaptation."""

from alpha_pulse.ml.online.online_learner import (
    BaseOnlineLearner,
    OnlineLearnerEnsemble,
    AdaptiveLearningController,
    OnlineDataPoint,
    LearningState
)

from alpha_pulse.ml.online.incremental_models import (
    IncrementalSGD,
    IncrementalNaiveBayes,
    IncrementalPassiveAggressive,
    HoeffdingTree,
    AdaptiveRandomForest,
    OnlineGradientBoosting
)

from alpha_pulse.ml.online.adaptive_algorithms import (
    AdaptiveLearningRateScheduler,
    AdaptiveOptimizer,
    MultiArmedBandit,
    AdaptiveMetaLearner,
    AdaptiveStrategy
)

from alpha_pulse.ml.online.concept_drift_detector import (
    ConceptDriftDetector,
    DriftPoint,
    ADWIN,
    DDM,
    PageHinkley,
    KSWIN
)

from alpha_pulse.ml.online.memory_manager import (
    MemoryManager,
    SlidingWindowBuffer,
    ReservoirSampler,
    MemoryItem
)

from alpha_pulse.ml.online.streaming_validation import (
    StreamingValidator,
    PrequentialEvaluator,
    StabilityTracker,
    AnomalyDetector,
    StreamingCrossValidator
)

from alpha_pulse.ml.online.online_learning_service import (
    OnlineLearningService
)

from alpha_pulse.ml.online.online_model import (
    OnlineLearningSession,
    DriftEvent,
    ModelCheckpoint,
    StreamingMetrics,
    OnlineDataPointModel,
    StreamingBatch,
    LearningSessionRequest,
    LearningSessionResponse,
    PredictionRequest,
    PredictionResponse,
    DriftDetectionAlert,
    LearningMetrics,
    ModelUpdateNotification,
    OnlineLearningConfig
)

__all__ = [
    # Base classes
    'BaseOnlineLearner',
    'OnlineLearnerEnsemble',
    'AdaptiveLearningController',
    'OnlineDataPoint',
    'LearningState',
    
    # Incremental models
    'IncrementalSGD',
    'IncrementalNaiveBayes',
    'IncrementalPassiveAggressive',
    'HoeffdingTree',
    'AdaptiveRandomForest',
    'OnlineGradientBoosting',
    
    # Adaptive algorithms
    'AdaptiveLearningRateScheduler',
    'AdaptiveOptimizer',
    'MultiArmedBandit',
    'AdaptiveMetaLearner',
    'AdaptiveStrategy',
    
    # Drift detection
    'ConceptDriftDetector',
    'DriftPoint',
    'ADWIN',
    'DDM',
    'PageHinkley',
    'KSWIN',
    
    # Memory management
    'MemoryManager',
    'SlidingWindowBuffer',
    'ReservoirSampler',
    'MemoryItem',
    
    # Validation
    'StreamingValidator',
    'PrequentialEvaluator',
    'StabilityTracker',
    'AnomalyDetector',
    'StreamingCrossValidator',
    
    # Service
    'OnlineLearningService',
    
    # Models
    'OnlineLearningSession',
    'DriftEvent',
    'ModelCheckpoint',
    'StreamingMetrics',
    'OnlineDataPointModel',
    'StreamingBatch',
    'LearningSessionRequest',
    'LearningSessionResponse',
    'PredictionRequest',
    'PredictionResponse',
    'DriftDetectionAlert',
    'LearningMetrics',
    'ModelUpdateNotification',
    'OnlineLearningConfig'
]