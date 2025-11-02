"""
Unified Model Explainer Interface.

This module provides a unified interface for explaining ML model predictions
using various explainability techniques (SHAP, LIME, feature importance).
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger

from .shap_explainer import SHAPExplainer, SHAPConfig
from .lime_explainer import LIMETabularExplainer, LIMEConfig
from .feature_importance import FeatureImportanceAnalyzer, FeatureImportanceConfig
from .explanation_aggregator import ExplanationAggregator, AggregationConfig


@dataclass
class ExplanationResult:
    """Result of model explanation."""
    method: str
    feature_importance: Dict[str, float]
    local_explanation: Optional[Dict[str, Any]] = None
    global_explanation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelExplainer:
    """
    Unified interface for explaining ML model predictions.

    This class provides a simple interface to generate explanations using
    multiple explainability techniques and aggregate the results.

    Example:
        >>> explainer = ModelExplainer(model=my_model, method='shap')
        >>> result = explainer.explain(X_test)
        >>> print(result.feature_importance)
    """

    def __init__(
        self,
        model: Any,
        method: str = 'shap',
        feature_names: Optional[List[str]] = None,
        background_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize model explainer.

        Args:
            model: Trained ML model to explain
            method: Explanation method ('shap', 'lime', 'importance', 'all')
            feature_names: Optional list of feature names
            background_data: Optional background dataset for SHAP
        """
        self.model = model
        self.method = method.lower()
        self.feature_names = feature_names
        self.background_data = background_data

        # Initialize explainers based on method
        self._shap_explainer: Optional[SHAPExplainer] = None
        self._lime_explainer: Optional[LIMETabularExplainer] = None
        self._importance_analyzer: Optional[FeatureImportanceAnalyzer] = None
        self._aggregator: Optional[ExplanationAggregator] = None

        self._initialize_explainers()

        logger.info(f"Initialized ModelExplainer with method: {method}")

    def _initialize_explainers(self) -> None:
        """Initialize explainers based on selected method."""
        try:
            if self.method in ['shap', 'all']:
                config = SHAPConfig()
                self._shap_explainer = SHAPExplainer(
                    model=self.model,
                    config=config,
                    background_data=self.background_data
                )

            if self.method in ['lime', 'all']:
                config = LIMEConfig()
                self._lime_explainer = LIMETabularExplainer(
                    model=self.model,
                    config=config,
                    feature_names=self.feature_names
                )

            if self.method in ['importance', 'all']:
                config = FeatureImportanceConfig()
                self._importance_analyzer = FeatureImportanceAnalyzer(
                    model=self.model,
                    config=config
                )

            if self.method == 'all':
                config = AggregationConfig()
                self._aggregator = ExplanationAggregator(config=config)

        except Exception as e:
            logger.warning(f"Failed to initialize some explainers: {e}")

    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_index: Optional[int] = None
    ) -> ExplanationResult:
        """
        Generate explanation for predictions.

        Args:
            X: Input data to explain
            instance_index: Optional specific instance to explain

        Returns:
            ExplanationResult containing feature importances and explanations
        """
        try:
            if self.method == 'shap' and self._shap_explainer:
                return self._explain_with_shap(X, instance_index)
            elif self.method == 'lime' and self._lime_explainer:
                return self._explain_with_lime(X, instance_index)
            elif self.method == 'importance' and self._importance_analyzer:
                return self._explain_with_importance(X)
            elif self.method == 'all':
                return self._explain_with_all(X, instance_index)
            else:
                raise ValueError(f"Unknown or uninitialized method: {self.method}")

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            raise

    def _explain_with_shap(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_index: Optional[int] = None
    ) -> ExplanationResult:
        """Generate SHAP-based explanation."""
        if self._shap_explainer is None:
            raise ValueError("SHAP explainer not initialized")

        # Generate SHAP values
        shap_values = self._shap_explainer.explain(X)

        # Extract feature importance
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Calculate global importance (mean absolute SHAP values)
        if len(shap_values.shape) > 1:
            importance = np.abs(shap_values).mean(axis=0)
        else:
            importance = np.abs(shap_values)

        feature_importance = dict(zip(feature_names, importance))

        # Local explanation for specific instance
        local_explanation = None
        if instance_index is not None and len(shap_values.shape) > 1:
            local_explanation = {
                'shap_values': dict(zip(feature_names, shap_values[instance_index])),
                'instance_index': instance_index
            }

        return ExplanationResult(
            method='shap',
            feature_importance=feature_importance,
            local_explanation=local_explanation,
            global_explanation={'shap_values': shap_values}
        )

    def _explain_with_lime(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_index: Optional[int] = None
    ) -> ExplanationResult:
        """Generate LIME-based explanation."""
        if self._lime_explainer is None:
            raise ValueError("LIME explainer not initialized")

        # LIME is primarily for local explanations
        if instance_index is None:
            instance_index = 0  # Default to first instance

        if isinstance(X, pd.DataFrame):
            instance = X.iloc[instance_index].values
        else:
            instance = X[instance_index]

        # Generate LIME explanation
        explanation = self._lime_explainer.explain_instance(instance)

        # Extract feature importance from LIME
        feature_importance = dict(explanation)

        local_explanation = {
            'lime_values': explanation,
            'instance_index': instance_index
        }

        return ExplanationResult(
            method='lime',
            feature_importance=feature_importance,
            local_explanation=local_explanation
        )

    def _explain_with_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> ExplanationResult:
        """Generate feature importance-based explanation."""
        if self._importance_analyzer is None:
            raise ValueError("Feature importance analyzer not initialized")

        # Calculate feature importance
        importance = self._importance_analyzer.calculate_importance(X)

        return ExplanationResult(
            method='importance',
            feature_importance=importance,
            global_explanation={'importance_scores': importance}
        )

    def _explain_with_all(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_index: Optional[int] = None
    ) -> ExplanationResult:
        """Generate explanation using all available methods."""
        results = []

        if self._shap_explainer:
            try:
                results.append(self._explain_with_shap(X, instance_index))
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        if self._lime_explainer:
            try:
                results.append(self._explain_with_lime(X, instance_index))
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")

        if self._importance_analyzer:
            try:
                results.append(self._explain_with_importance(X))
            except Exception as e:
                logger.warning(f"Importance analysis failed: {e}")

        if not results:
            raise RuntimeError("All explanation methods failed")

        # Aggregate results
        if self._aggregator and len(results) > 1:
            aggregated = self._aggregator.aggregate([r.feature_importance for r in results])
            feature_importance = aggregated
        else:
            # Simple average if aggregator not available
            all_importances = [r.feature_importance for r in results]
            feature_importance = {}
            for importance_dict in all_importances:
                for feature, value in importance_dict.items():
                    feature_importance[feature] = feature_importance.get(feature, 0) + value

            # Average the values
            for feature in feature_importance:
                feature_importance[feature] /= len(all_importances)

        return ExplanationResult(
            method='aggregated',
            feature_importance=feature_importance,
            local_explanation={'methods': [r.method for r in results]},
            global_explanation={'individual_results': results},
            metadata={'num_methods': len(results)}
        )

    def get_top_features(self, n: int = 10) -> List[tuple]:
        """
        Get top N most important features.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        # This requires running explain first
        logger.warning("get_top_features requires calling explain() first")
        return []

    def plot_explanation(self, explanation: ExplanationResult) -> None:
        """
        Plot explanation results (placeholder for visualization).

        Args:
            explanation: ExplanationResult to visualize
        """
        logger.info(f"Plotting {explanation.method} explanation")
        logger.info(f"Top features: {sorted(explanation.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]}")
