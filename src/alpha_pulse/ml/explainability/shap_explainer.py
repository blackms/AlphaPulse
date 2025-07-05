"""SHAP (SHapley Additive exPlanations) implementation for model interpretability."""
import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import logging
from dataclasses import dataclass
import warnings
from sklearn.base import BaseEstimator

from ...models.explanation_result import (
    ExplanationResult, ExplanationType, ExplanationLevel,
    FeatureContribution, GlobalExplanation
)


logger = logging.getLogger(__name__)


@dataclass
class SHAPConfig:
    """Configuration for SHAP explainer."""
    max_samples: int = 100
    background_samples: int = 100
    check_additivity: bool = True
    link: str = "identity"  # or "logit" for classification
    feature_perturbation: str = "interventional"  # or "tree_path_dependent"
    model_output: str = "raw"  # or "probability", "log_loss"
    n_jobs: int = -1


class SHAPExplainer:
    """SHAP explainer for various model types."""
    
    def __init__(self, model: Any, config: Optional[SHAPConfig] = None):
        """Initialize SHAP explainer.
        
        Args:
            model: The model to explain
            config: SHAP configuration
        """
        self.model = model
        self.config = config or SHAPConfig()
        self.explainer = None
        self.background_data = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Set up appropriate SHAP explainer based on model type."""
        model_type = type(self.model).__name__
        
        # Tree-based models
        if hasattr(self.model, "tree_") or model_type in [
            "XGBRegressor", "XGBClassifier", "LGBMRegressor", "LGBMClassifier",
            "RandomForestRegressor", "RandomForestClassifier",
            "GradientBoostingRegressor", "GradientBoostingClassifier"
        ]:
            self._setup_tree_explainer()
        
        # Neural networks
        elif model_type in ["Sequential", "Model", "Module"]:
            self._setup_deep_explainer()
        
        # Linear models
        elif model_type in [
            "LinearRegression", "LogisticRegression", "Ridge", "Lasso",
            "ElasticNet", "SGDRegressor", "SGDClassifier"
        ]:
            self._setup_linear_explainer()
        
        # Generic model
        else:
            self._setup_kernel_explainer()
        
        logger.info(f"Set up {type(self.explainer).__name__} for {model_type}")
    
    def _setup_tree_explainer(self):
        """Set up TreeExplainer for tree-based models."""
        try:
            self.explainer = shap.TreeExplainer(
                self.model,
                feature_perturbation=self.config.feature_perturbation,
                model_output=self.config.model_output,
                check_additivity=self.config.check_additivity
            )
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}. Falling back to KernelExplainer.")
            self._setup_kernel_explainer()
    
    def _setup_deep_explainer(self):
        """Set up DeepExplainer for neural networks."""
        if self.background_data is None:
            raise ValueError("DeepExplainer requires background data")
        
        try:
            self.explainer = shap.DeepExplainer(
                self.model,
                self.background_data[:self.config.background_samples]
            )
        except Exception as e:
            logger.warning(f"DeepExplainer failed: {e}. Falling back to GradientExplainer.")
            self._setup_gradient_explainer()
    
    def _setup_gradient_explainer(self):
        """Set up GradientExplainer for neural networks."""
        if self.background_data is None:
            raise ValueError("GradientExplainer requires background data")
        
        self.explainer = shap.GradientExplainer(
            self.model,
            self.background_data[:self.config.background_samples]
        )
    
    def _setup_linear_explainer(self):
        """Set up LinearExplainer for linear models."""
        try:
            if hasattr(self.model, "coef_"):
                # For sklearn linear models
                self.explainer = shap.LinearExplainer(
                    self.model,
                    self.background_data,
                    feature_perturbation=self.config.feature_perturbation
                )
            else:
                self._setup_kernel_explainer()
        except Exception as e:
            logger.warning(f"LinearExplainer failed: {e}. Falling back to KernelExplainer.")
            self._setup_kernel_explainer()
    
    def _setup_kernel_explainer(self):
        """Set up KernelExplainer as fallback for any model."""
        if self.background_data is None:
            logger.warning("KernelExplainer without background data may be slow")
        
        # Create prediction function wrapper
        if hasattr(self.model, "predict_proba"):
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict
        
        # Use background data or create synthetic
        background = self.background_data[:self.config.background_samples] \
            if self.background_data is not None else shap.sample(100)
        
        self.explainer = shap.KernelExplainer(
            predict_fn,
            background,
            link=self.config.link
        )
    
    def set_background_data(self, data: Union[np.ndarray, pd.DataFrame]):
        """Set background data for explainers that need it.
        
        Args:
            data: Background dataset for explanations
        """
        self.background_data = data
        if isinstance(self.explainer, (shap.DeepExplainer, shap.GradientExplainer, shap.KernelExplainer)):
            self._setup_explainer()  # Reinitialize with background data
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.DataFrame],
        prediction_id: str,
        feature_names: Optional[List[str]] = None
    ) -> ExplanationResult:
        """Explain a single instance prediction.
        
        Args:
            instance: Single instance to explain
            prediction_id: ID of the prediction
            feature_names: Names of features
            
        Returns:
            ExplanationResult with SHAP values
        """
        start_time = datetime.now()
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # Handle multi-output models
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take first output
        
        # Ensure 1D array for single instance
        if shap_values.ndim > 1:
            shap_values = shap_values[0]
        
        # Get feature names
        if feature_names is None:
            if hasattr(instance, "columns"):
                feature_names = list(instance.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(len(shap_values))]
        
        # Get instance values
        instance_values = instance.values[0] if hasattr(instance, "values") else instance[0]
        
        # Create feature contributions
        feature_contributions = []
        for i, (name, value, shap_val) in enumerate(zip(feature_names, instance_values, shap_values)):
            contribution = FeatureContribution(
                feature_name=name,
                value=float(value),
                contribution=float(shap_val),
                baseline_value=float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else None
            )
            feature_contributions.append(contribution)
        
        # Calculate base and prediction values
        base_value = float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else 0.0
        prediction_value = base_value + sum(shap_values)
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        return ExplanationResult(
            explanation_id=f"shap_{prediction_id}_{datetime.now().timestamp()}",
            model_id=str(id(self.model)),
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            explanation_type=ExplanationType.SHAP,
            explanation_level=ExplanationLevel.LOCAL,
            feature_contributions=feature_contributions,
            base_value=base_value,
            prediction_value=prediction_value,
            computation_time=computation_time,
            method_parameters={
                "explainer_type": type(self.explainer).__name__,
                "config": self.config.__dict__
            }
        )
    
    def explain_batch(
        self,
        instances: Union[np.ndarray, pd.DataFrame],
        prediction_ids: List[str],
        feature_names: Optional[List[str]] = None
    ) -> List[ExplanationResult]:
        """Explain multiple instances.
        
        Args:
            instances: Batch of instances to explain
            prediction_ids: IDs of predictions
            feature_names: Names of features
            
        Returns:
            List of ExplanationResults
        """
        results = []
        for i, (instance, pred_id) in enumerate(zip(instances, prediction_ids)):
            # Reshape single instance
            if isinstance(instances, np.ndarray):
                instance = instance.reshape(1, -1)
            else:
                instance = instances.iloc[[i]]
            
            result = self.explain_instance(instance, pred_id, feature_names)
            results.append(result)
        
        return results
    
    def compute_global_importance(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ) -> GlobalExplanation:
        """Compute global feature importance using SHAP.
        
        Args:
            data: Dataset to compute global importance on
            feature_names: Names of features
            sample_size: Number of samples to use
            
        Returns:
            GlobalExplanation with aggregated SHAP values
        """
        start_time = datetime.now()
        
        # Sample data if needed
        if sample_size and len(data) > sample_size:
            indices = np.random.choice(len(data), sample_size, replace=False)
            if isinstance(data, pd.DataFrame):
                data = data.iloc[indices]
            else:
                data = data[indices]
        
        # Compute SHAP values for all samples
        shap_values = self.explainer.shap_values(data)
        
        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Get feature names
        if feature_names is None:
            if hasattr(data, "columns"):
                feature_names = list(data.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
        
        # Compute global feature importance (mean absolute SHAP values)
        global_importance = {}
        for i, name in enumerate(feature_names):
            global_importance[name] = float(np.mean(np.abs(shap_values[:, i])))
        
        # Compute feature value distributions
        feature_value_distributions = {}
        for i, name in enumerate(feature_names):
            values = data.iloc[:, i] if hasattr(data, "iloc") else data[:, i]
            feature_value_distributions[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            }
        
        # Compute feature contribution distributions
        feature_contribution_distributions = {}
        for i, name in enumerate(feature_names):
            contributions = shap_values[:, i]
            feature_contribution_distributions[name] = {
                "mean": float(np.mean(contributions)),
                "std": float(np.std(contributions)),
                "min": float(np.min(contributions)),
                "max": float(np.max(contributions)),
                "median": float(np.median(contributions))
            }
        
        # Compute feature interactions (top pairs)
        interaction_importance = {}
        if hasattr(self.explainer, "shap_interaction_values"):
            try:
                interaction_values = self.explainer.shap_interaction_values(data[:100])  # Limit for performance
                for i in range(len(feature_names)):
                    for j in range(i+1, len(feature_names)):
                        interaction_key = (feature_names[i], feature_names[j])
                        interaction_importance[interaction_key] = float(
                            np.mean(np.abs(interaction_values[:, i, j]))
                        )
            except Exception as e:
                logger.warning(f"Could not compute interaction values: {e}")
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        return GlobalExplanation(
            model_id=str(id(self.model)),
            timestamp=datetime.now(),
            num_samples=len(data),
            global_feature_importance=global_importance,
            feature_interaction_importance=interaction_importance,
            feature_value_distributions=feature_value_distributions,
            feature_contribution_distributions=feature_contribution_distributions,
            decision_rules=[],  # SHAP doesn't directly provide rules
            performance_by_feature={},  # To be computed separately
            error_attribution={}  # To be computed separately
        )
    
    def get_waterfall_data(self, explanation: ExplanationResult) -> Dict[str, Any]:
        """Get data for SHAP waterfall plot.
        
        Args:
            explanation: ExplanationResult from SHAP
            
        Returns:
            Dictionary with waterfall plot data
        """
        # Sort features by absolute contribution
        sorted_features = sorted(
            explanation.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )
        
        return {
            "base_value": explanation.base_value,
            "features": [fc.feature_name for fc in sorted_features],
            "values": [fc.value for fc in sorted_features],
            "contributions": [fc.contribution for fc in sorted_features],
            "prediction": explanation.prediction_value
        }
    
    def get_force_plot_data(self, explanation: ExplanationResult) -> Dict[str, Any]:
        """Get data for SHAP force plot.
        
        Args:
            explanation: ExplanationResult from SHAP
            
        Returns:
            Dictionary with force plot data
        """
        positive_features = []
        negative_features = []
        
        for fc in explanation.feature_contributions:
            feature_data = {
                "name": fc.feature_name,
                "value": fc.value,
                "contribution": fc.contribution
            }
            
            if fc.contribution > 0:
                positive_features.append(feature_data)
            else:
                negative_features.append(feature_data)
        
        # Sort by absolute contribution
        positive_features.sort(key=lambda x: x["contribution"], reverse=True)
        negative_features.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        
        return {
            "base_value": explanation.base_value,
            "prediction": explanation.prediction_value,
            "positive_features": positive_features,
            "negative_features": negative_features
        }