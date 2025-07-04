"""Machine Learning module for AlphaPulse."""

# Import online learning components
from alpha_pulse.ml.online import (
    # Core classes
    BaseOnlineLearner,
    OnlineLearnerEnsemble,
    AdaptiveLearningController,
    OnlineDataPoint,
    
    # Models
    IncrementalSGD,
    IncrementalNaiveBayes,
    IncrementalPassiveAggressive,
    HoeffdingTree,
    AdaptiveRandomForest,
    OnlineGradientBoosting,
    
    # Algorithms
    AdaptiveLearningRateScheduler,
    AdaptiveOptimizer,
    MultiArmedBandit,
    AdaptiveMetaLearner,
    
    # Drift Detection
    ConceptDriftDetector,
    ADWIN,
    DDM,
    PageHinkley,
    KSWIN,
    
    # Utilities
    MemoryManager,
    StreamingValidator,
    PrequentialEvaluator,
    
    # Service
    OnlineLearningService
)

__all__ = [
    # Online Learning
    'BaseOnlineLearner',
    'OnlineLearnerEnsemble',
    'AdaptiveLearningController',
    'OnlineDataPoint',
    'IncrementalSGD',
    'IncrementalNaiveBayes',
    'IncrementalPassiveAggressive',
    'HoeffdingTree',
    'AdaptiveRandomForest',
    'OnlineGradientBoosting',
    'AdaptiveLearningRateScheduler',
    'AdaptiveOptimizer',
    'MultiArmedBandit',
    'AdaptiveMetaLearner',
    'ConceptDriftDetector',
    'ADWIN',
    'DDM',
    'PageHinkley',
    'KSWIN',
    'MemoryManager',
    'StreamingValidator',
    'PrequentialEvaluator',
    'OnlineLearningService'
]