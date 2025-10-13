"""Explanation result models for explainable AI."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np
from enum import Enum


class ExplanationType(Enum):
    """Types of explanations."""
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    COUNTERFACTUAL = "counterfactual"
    ATTENTION = "attention"
    DECISION_TREE = "decision_tree"


class ExplanationLevel(Enum):
    """Level of explanation detail."""
    GLOBAL = "global"
    LOCAL = "local"
    COHORT = "cohort"


@dataclass
class FeatureContribution:
    """Individual feature contribution to prediction."""
    feature_name: str
    value: float
    contribution: float
    baseline_value: Optional[float] = None
    interaction_effects: Optional[Dict[str, float]] = None
    confidence_interval: Optional[tuple] = None


@dataclass
class ExplanationResult:
    """Result of model explanation."""
    explanation_id: str
    model_id: str
    prediction_id: str
    timestamp: datetime
    explanation_type: ExplanationType
    explanation_level: ExplanationLevel
    
    # Core explanation data
    feature_contributions: List[FeatureContribution]
    base_value: float
    prediction_value: float
    
    # Additional metadata
    confidence_score: Optional[float] = None
    computation_time: Optional[float] = None
    method_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Visualization data
    visualization_data: Optional[Dict[str, Any]] = None
    
    # Regulatory compliance
    audit_trail: Dict[str, Any] = field(default_factory=dict)
    compliance_notes: Optional[str] = None
    
    def get_top_features(self, n: int = 10) -> List[FeatureContribution]:
        """Get top n features by absolute contribution."""
        return sorted(
            self.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )[:n]
    
    def get_positive_features(self) -> List[FeatureContribution]:
        """Get features with positive contributions."""
        return [f for f in self.feature_contributions if f.contribution > 0]
    
    def get_negative_features(self) -> List[FeatureContribution]:
        """Get features with negative contributions."""
        return [f for f in self.feature_contributions if f.contribution < 0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "explanation_id": self.explanation_id,
            "model_id": self.model_id,
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp.isoformat(),
            "explanation_type": self.explanation_type.value,
            "explanation_level": self.explanation_level.value,
            "feature_contributions": [
                {
                    "feature_name": fc.feature_name,
                    "value": fc.value,
                    "contribution": fc.contribution,
                    "baseline_value": fc.baseline_value,
                    "interaction_effects": fc.interaction_effects,
                    "confidence_interval": fc.confidence_interval
                }
                for fc in self.feature_contributions
            ],
            "base_value": self.base_value,
            "prediction_value": self.prediction_value,
            "confidence_score": self.confidence_score,
            "computation_time": self.computation_time,
            "method_parameters": self.method_parameters,
            "visualization_data": self.visualization_data,
            "audit_trail": self.audit_trail,
            "compliance_notes": self.compliance_notes
        }


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation showing minimal changes for different outcome."""
    original_features: Dict[str, float]
    counterfactual_features: Dict[str, float]
    original_prediction: float
    counterfactual_prediction: float
    
    # Changes required
    feature_changes: Dict[str, tuple]  # (old_value, new_value)
    change_distance: float
    
    # Validity and quality metrics
    is_valid: bool
    validity_constraints: List[str]
    quality_score: float
    
    # Additional metadata
    generation_method: str
    computation_time: float
    num_features_changed: int
    
    def get_minimal_changes(self) -> Dict[str, tuple]:
        """Get only the features that changed."""
        return {
            feature: change
            for feature, change in self.feature_changes.items()
            if change[0] != change[1]
        }


@dataclass
class AttentionExplanation:
    """Attention-based explanation for neural networks."""
    layer_name: str
    attention_weights: np.ndarray
    input_tokens: Optional[List[str]] = None
    output_tokens: Optional[List[str]] = None
    
    # Aggregated attention scores
    feature_attention_scores: Dict[str, float] = field(default_factory=dict)
    temporal_attention_pattern: Optional[np.ndarray] = None
    
    # Visualization helpers
    attention_matrix: Optional[np.ndarray] = None
    head_contributions: Optional[Dict[int, float]] = None


@dataclass
class GlobalExplanation:
    """Global model explanation aggregating multiple local explanations."""
    model_id: str
    timestamp: datetime
    num_samples: int

    # Aggregated feature importance
    global_feature_importance: Dict[str, float]
    feature_interaction_importance: Dict[tuple, float]

    # Feature statistics
    feature_value_distributions: Dict[str, Dict[str, float]]
    feature_contribution_distributions: Dict[str, Dict[str, float]]

    # Model behavior patterns
    decision_rules: List[str]

    # Performance attribution (required fields - must come before optional fields)
    performance_by_feature: Dict[str, float]
    error_attribution: Dict[str, float]

    # Optional fields with defaults (must come after required fields)
    behavior_clusters: Optional[List[Dict[str, Any]]] = None

    # Bias and fairness metrics
    bias_metrics: Dict[str, float] = field(default_factory=dict)
    fairness_indicators: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ExplanationComparison:
    """Comparison between multiple explanations."""
    explanation_ids: List[str]
    comparison_timestamp: datetime
    
    # Feature contribution differences
    contribution_differences: Dict[str, List[float]]
    consistency_scores: Dict[str, float]
    
    # Method agreement
    method_agreement_matrix: np.ndarray
    consensus_features: List[str]
    disputed_features: List[str]
    
    # Statistical tests
    statistical_significance: Dict[str, float]
    correlation_metrics: Dict[str, float]