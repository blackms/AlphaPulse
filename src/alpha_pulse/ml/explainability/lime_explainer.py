"""LIME (Local Interpretable Model-agnostic Explanations) implementation."""
import lime
import lime.lime_tabular
import lime.lime_text
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline

from ...models.explanation_result import (
    ExplanationResult, ExplanationType, ExplanationLevel,
    FeatureContribution
)


logger = logging.getLogger(__name__)


@dataclass
class LIMEConfig:
    """Configuration for LIME explainer."""
    num_features: int = 10
    num_samples: int = 5000
    kernel_width: Optional[float] = None
    kernel: Optional[Callable] = None
    verbose: bool = False
    mode: str = "regression"  # or "classification"
    sample_around_instance: bool = True
    discretize_continuous: bool = True
    discretizer: str = "quartile"  # or "decile", "entropy"
    feature_selection: str = "auto"  # or "forward_selection", "lasso_path", "none"
    random_state: int = 42


class LIMETabularExplainer:
    """LIME explainer for tabular data."""
    
    def __init__(
        self,
        training_data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        config: Optional[LIMEConfig] = None
    ):
        """Initialize LIME tabular explainer.
        
        Args:
            training_data: Training data for the explainer
            feature_names: Names of features
            class_names: Names of classes (for classification)
            config: LIME configuration
        """
        self.config = config or LIMEConfig()
        self.training_data = training_data
        
        # Get feature names
        if feature_names is None:
            if hasattr(training_data, "columns"):
                self.feature_names = list(training_data.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(training_data.shape[1])]
        else:
            self.feature_names = feature_names
        
        self.class_names = class_names
        
        # Convert to numpy if pandas
        if isinstance(training_data, pd.DataFrame):
            training_array = training_data.values
        else:
            training_array = training_data
        
        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_array,
            feature_names=self.feature_names,
            class_names=self.class_names,
            verbose=self.config.verbose,
            mode=self.config.mode,
            discretize_continuous=self.config.discretize_continuous,
            discretizer=self.config.discretizer,
            sample_around_instance=self.config.sample_around_instance,
            random_state=self.config.random_state,
            kernel_width=self.config.kernel_width,
            kernel=self.config.kernel,
            feature_selection=self.config.feature_selection
        )
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.DataFrame],
        predict_fn: Callable,
        prediction_id: str,
        model_id: str,
        num_features: Optional[int] = None
    ) -> ExplanationResult:
        """Explain a single instance using LIME.
        
        Args:
            instance: Instance to explain
            predict_fn: Prediction function (model.predict or model.predict_proba)
            prediction_id: ID of the prediction
            model_id: ID of the model
            num_features: Number of features to include in explanation
            
        Returns:
            ExplanationResult with LIME explanation
        """
        start_time = datetime.now()
        
        # Convert to numpy array
        if isinstance(instance, pd.DataFrame):
            instance_array = instance.values[0]
        else:
            instance_array = instance.flatten()
        
        # Get number of features
        num_features = num_features or self.config.num_features
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            instance_array,
            predict_fn,
            num_features=num_features,
            num_samples=self.config.num_samples
        )
        
        # Extract feature contributions
        feature_contributions = []
        feature_weights = explanation.as_list()
        
        # Create a mapping of feature names to values
        feature_values = {}
        for i, name in enumerate(self.feature_names):
            feature_values[name] = float(instance_array[i])
        
        # Process LIME output
        for feature_desc, weight in feature_weights:
            # LIME returns feature descriptions like "feature_name > value"
            # Extract the feature name
            feature_name = None
            for fname in self.feature_names:
                if fname in feature_desc:
                    feature_name = fname
                    break
            
            if feature_name:
                contribution = FeatureContribution(
                    feature_name=feature_name,
                    value=feature_values.get(feature_name, 0.0),
                    contribution=float(weight),
                    baseline_value=None  # LIME doesn't provide baseline
                )
                feature_contributions.append(contribution)
        
        # Get prediction value
        prediction = predict_fn(instance_array.reshape(1, -1))
        if isinstance(prediction, np.ndarray):
            prediction_value = float(prediction[0])
        else:
            prediction_value = float(prediction)
        
        # Calculate base value (intercept)
        base_value = explanation.intercept[0] if hasattr(explanation, 'intercept') else 0.0
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        # Get local model score
        local_model_score = explanation.score if hasattr(explanation, 'score') else None
        
        return ExplanationResult(
            explanation_id=f"lime_{prediction_id}_{datetime.now().timestamp()}",
            model_id=model_id,
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            explanation_type=ExplanationType.LIME,
            explanation_level=ExplanationLevel.LOCAL,
            feature_contributions=feature_contributions,
            base_value=float(base_value),
            prediction_value=prediction_value,
            confidence_score=local_model_score,
            computation_time=computation_time,
            method_parameters={
                "num_features": num_features,
                "num_samples": self.config.num_samples,
                "mode": self.config.mode,
                "feature_selection": self.config.feature_selection
            },
            visualization_data={
                "lime_explanation": explanation,
                "feature_weights": feature_weights
            }
        )
    
    def explain_batch(
        self,
        instances: Union[np.ndarray, pd.DataFrame],
        predict_fn: Callable,
        prediction_ids: List[str],
        model_id: str,
        num_features: Optional[int] = None
    ) -> List[ExplanationResult]:
        """Explain multiple instances.
        
        Args:
            instances: Batch of instances to explain
            predict_fn: Prediction function
            prediction_ids: IDs of predictions
            model_id: ID of the model
            num_features: Number of features to include
            
        Returns:
            List of ExplanationResults
        """
        results = []
        
        for i, pred_id in enumerate(prediction_ids):
            if isinstance(instances, pd.DataFrame):
                instance = instances.iloc[[i]]
            else:
                instance = instances[i:i+1]
            
            result = self.explain_instance(
                instance, predict_fn, pred_id, model_id, num_features
            )
            results.append(result)
        
        return results
    
    def get_feature_importance_stability(
        self,
        instance: Union[np.ndarray, pd.DataFrame],
        predict_fn: Callable,
        n_runs: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Assess stability of LIME explanations across multiple runs.
        
        Args:
            instance: Instance to explain
            predict_fn: Prediction function
            n_runs: Number of runs to assess stability
            
        Returns:
            Dictionary with stability metrics for each feature
        """
        # Convert instance
        if isinstance(instance, pd.DataFrame):
            instance_array = instance.values[0]
        else:
            instance_array = instance.flatten()
        
        # Collect explanations across runs
        all_explanations = []
        for _ in range(n_runs):
            explanation = self.explainer.explain_instance(
                instance_array,
                predict_fn,
                num_features=len(self.feature_names),
                num_samples=self.config.num_samples
            )
            
            # Extract weights as dictionary
            weights = {}
            for feature_desc, weight in explanation.as_list():
                for fname in self.feature_names:
                    if fname in feature_desc:
                        weights[fname] = weight
                        break
            
            all_explanations.append(weights)
        
        # Compute stability metrics
        stability_metrics = {}
        for feature in self.feature_names:
            values = [exp.get(feature, 0.0) for exp in all_explanations]
            
            stability_metrics[feature] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "cv": float(np.std(values) / (np.mean(values) + 1e-10))  # Coefficient of variation
            }
        
        return stability_metrics


class LIMETimeSeriesExplainer:
    """LIME explainer for time series data."""
    
    def __init__(
        self,
        window_size: int,
        feature_names: Optional[List[str]] = None,
        config: Optional[LIMEConfig] = None
    ):
        """Initialize LIME time series explainer.
        
        Args:
            window_size: Size of the time window
            feature_names: Names of features
            config: LIME configuration
        """
        self.window_size = window_size
        self.feature_names = feature_names
        self.config = config or LIMEConfig()
    
    def explain_instance(
        self,
        time_series: np.ndarray,
        predict_fn: Callable,
        prediction_id: str,
        model_id: str,
        time_index: int
    ) -> ExplanationResult:
        """Explain a time series prediction at a specific time point.
        
        Args:
            time_series: Time series data
            predict_fn: Prediction function
            prediction_id: ID of the prediction
            model_id: ID of the model
            time_index: Time index to explain
            
        Returns:
            ExplanationResult with time series explanation
        """
        start_time = datetime.now()
        
        # Extract window around time_index
        start_idx = max(0, time_index - self.window_size)
        end_idx = min(len(time_series), time_index + 1)
        window = time_series[start_idx:end_idx]
        
        # Create segments (each time step is a feature)
        num_segments = len(window)
        
        # Create a perturber function
        def segment_perturber(segments_mask):
            """Perturb time series by masking segments."""
            perturbed_series = []
            
            for mask in segments_mask:
                series_copy = window.copy()
                # Mask segments (set to mean value)
                for i, m in enumerate(mask):
                    if m == 0:  # Segment is "off"
                        series_copy[i] = np.mean(window)
                
                perturbed_series.append(series_copy)
            
            return np.array(perturbed_series)
        
        # Create a wrapped predict function
        def wrapped_predict(perturbed_windows):
            """Predict on perturbed windows."""
            predictions = []
            for window in perturbed_windows:
                # Reconstruct full series with perturbed window
                full_series = time_series.copy()
                full_series[start_idx:end_idx] = window
                pred = predict_fn(full_series.reshape(1, -1))
                predictions.append(pred[0] if isinstance(pred, np.ndarray) else pred)
            
            return np.array(predictions)
        
        # Use LIME with custom perturbation
        explainer = lime.lime_base.LimeBase(
            kernel_width=self.config.kernel_width,
            kernel=self.config.kernel,
            verbose=self.config.verbose,
            random_state=self.config.random_state
        )
        
        # Generate explanation
        explanation = explainer.explain_instance_with_data(
            segment_perturber,
            wrapped_predict,
            np.ones(num_segments),  # All segments "on" for original
            num_samples=self.config.num_samples,
            distance_metric='euclidean'
        )
        
        # Extract feature contributions
        feature_contributions = []
        for i, (_, weight) in enumerate(explanation[0]):
            time_step = start_idx + i
            feature_name = f"t-{time_index - time_step}" if time_step < time_index else "t"
            
            contribution = FeatureContribution(
                feature_name=feature_name,
                value=float(window[i]),
                contribution=float(weight)
            )
            feature_contributions.append(contribution)
        
        # Get prediction
        prediction_value = float(predict_fn(time_series.reshape(1, -1))[0])
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        return ExplanationResult(
            explanation_id=f"lime_ts_{prediction_id}_{datetime.now().timestamp()}",
            model_id=model_id,
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            explanation_type=ExplanationType.LIME,
            explanation_level=ExplanationLevel.LOCAL,
            feature_contributions=feature_contributions,
            base_value=0.0,  # LIME doesn't provide base value for custom explainers
            prediction_value=prediction_value,
            computation_time=computation_time,
            method_parameters={
                "window_size": self.window_size,
                "time_index": time_index,
                "num_samples": self.config.num_samples
            }
        )


class LIMETextExplainer:
    """LIME explainer for text data (e.g., news sentiment)."""
    
    def __init__(self, config: Optional[LIMEConfig] = None):
        """Initialize LIME text explainer.
        
        Args:
            config: LIME configuration
        """
        self.config = config or LIMEConfig()
        self.explainer = lime.lime_text.LimeTextExplainer(
            class_names=None,  # Will be set when explaining
            verbose=self.config.verbose,
            random_state=self.config.random_state,
            kernel=self.config.kernel,
            kernel_width=self.config.kernel_width
        )
    
    def explain_instance(
        self,
        text: str,
        predict_fn: Callable,
        prediction_id: str,
        model_id: str,
        num_features: Optional[int] = None
    ) -> ExplanationResult:
        """Explain text classification/sentiment.
        
        Args:
            text: Text to explain
            predict_fn: Prediction function (takes list of texts)
            prediction_id: ID of the prediction
            model_id: ID of the model
            num_features: Number of features (words) to include
            
        Returns:
            ExplanationResult with text explanation
        """
        start_time = datetime.now()
        
        num_features = num_features or self.config.num_features
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=self.config.num_samples
        )
        
        # Extract word contributions
        feature_contributions = []
        for word, weight in explanation.as_list():
            contribution = FeatureContribution(
                feature_name=word,
                value=1.0,  # Word presence
                contribution=float(weight)
            )
            feature_contributions.append(contribution)
        
        # Get prediction
        prediction = predict_fn([text])
        if isinstance(prediction, np.ndarray):
            if prediction.ndim > 1:
                prediction_value = float(prediction[0, 0])
            else:
                prediction_value = float(prediction[0])
        else:
            prediction_value = float(prediction)
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        return ExplanationResult(
            explanation_id=f"lime_text_{prediction_id}_{datetime.now().timestamp()}",
            model_id=model_id,
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            explanation_type=ExplanationType.LIME,
            explanation_level=ExplanationLevel.LOCAL,
            feature_contributions=feature_contributions,
            base_value=0.0,
            prediction_value=prediction_value,
            computation_time=computation_time,
            method_parameters={
                "num_features": num_features,
                "num_samples": self.config.num_samples,
                "text_length": len(text.split())
            },
            visualization_data={
                "original_text": text,
                "highlighted_text": explanation.as_html()
            }
        )