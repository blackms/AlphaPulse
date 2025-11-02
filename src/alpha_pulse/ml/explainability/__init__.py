"""Explainability module for ML models."""
from .shap_explainer import SHAPExplainer, SHAPConfig
from .lime_explainer import (
    LIMETabularExplainer, LIMETimeSeriesExplainer, LIMETextExplainer, LIMEConfig
)
from .feature_importance import FeatureImportanceAnalyzer, FeatureImportanceConfig
from .decision_trees import DecisionTreeSurrogate, SurrogateTreeConfig
from .explanation_aggregator import ExplanationAggregator, AggregationConfig
from .explainer import ModelExplainer, ExplanationResult

__all__ = [
    "SHAPExplainer",
    "SHAPConfig",
    "LIMETabularExplainer",
    "LIMETimeSeriesExplainer",
    "LIMETextExplainer",
    "LIMEConfig",
    "FeatureImportanceAnalyzer",
    "FeatureImportanceConfig",
    "DecisionTreeSurrogate",
    "SurrogateTreeConfig",
    "ExplanationAggregator",
    "AggregationConfig",
    "ModelExplainer",
    "ExplanationResult"
]