"""Explainability service for integrating XAI features into trading system."""
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json
from pathlib import Path
import pickle

from ..models.explanation_result import FeatureContribution

from ..ml.explainability.shap_explainer import SHAPExplainer, SHAPConfig
from ..ml.explainability.lime_explainer import (
    LIMETabularExplainer, LIMETimeSeriesExplainer, LIMETextExplainer, LIMEConfig
)
from ..ml.explainability.feature_importance import (
    FeatureImportanceAnalyzer, FeatureImportanceConfig
)
from ..ml.explainability.decision_trees import (
    DecisionTreeSurrogate, SurrogateTreeConfig
)
from ..ml.explainability.explanation_aggregator import (
    ExplanationAggregator, AggregationConfig
)
from ..models.explanation_result import (
    ExplanationResult, ExplanationType, ExplanationLevel,
    GlobalExplanation, CounterfactualExplanation
)
from ..utils.visualization_utils import ExplainabilityVisualizer
from ..database.models import Base
from sqlalchemy import Column, String, JSON, Float, DateTime, Integer, Boolean
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base


logger = logging.getLogger(__name__)


# Database model for storing explanations
class ExplanationRecord(Base):
    """Database model for explanation records."""
    __tablename__ = "explanations"
    
    id = Column(Integer, primary_key=True)
    explanation_id = Column(String, unique=True, index=True)
    model_id = Column(String, index=True)
    prediction_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    explanation_type = Column(String)
    explanation_level = Column(String)
    feature_contributions = Column(JSON)
    base_value = Column(Float)
    prediction_value = Column(Float)
    confidence_score = Column(Float, nullable=True)
    computation_time = Column(Float, nullable=True)
    metadata = Column(JSON)
    compliance_notes = Column(String, nullable=True)


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability service."""
    enable_shap: bool = True
    enable_lime: bool = True
    enable_feature_importance: bool = True
    enable_surrogate: bool = True
    enable_counterfactual: bool = True
    
    # Method-specific configs
    shap_config: Optional[SHAPConfig] = None
    lime_config: Optional[LIMEConfig] = None
    importance_config: Optional[FeatureImportanceConfig] = None
    surrogate_config: Optional[SurrogateTreeConfig] = None
    aggregation_config: Optional[AggregationConfig] = None
    
    # Storage settings
    store_explanations: bool = True
    explanation_cache_size: int = 1000
    visualization_output_dir: str = "./explanations/visualizations"
    
    # Performance settings
    async_explanations: bool = True
    max_concurrent_explanations: int = 5
    explanation_timeout: float = 60.0  # seconds
    
    # Compliance settings
    require_explanation_for_trades: bool = False
    min_confidence_threshold: float = 0.6
    audit_all_decisions: bool = True


class ExplainabilityService:
    """Service for managing model explainability."""
    
    def __init__(
        self,
        config: ExplainabilityConfig,
        db_session: Optional[AsyncSession] = None
    ):
        """Initialize explainability service.
        
        Args:
            config: Explainability configuration
            db_session: Database session for storing explanations
        """
        self.config = config
        self.db_session = db_session
        
        # Initialize components
        self.shap_explainers = {}
        self.lime_explainers = {}
        self.importance_analyzer = FeatureImportanceAnalyzer(config.importance_config)
        self.visualizer = ExplainabilityVisualizer()
        
        # Cache for explanations
        self.explanation_cache = {}
        
        # Create output directory
        Path(config.visualization_output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized ExplainabilityService")
    
    async def explain_prediction(
        self,
        model: Any,
        instance: Union[np.ndarray, pd.DataFrame],
        prediction_id: str,
        model_id: str,
        feature_names: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> Dict[str, ExplanationResult]:
        """Generate explanations for a prediction using multiple methods.
        
        Args:
            model: The model that made the prediction
            instance: The input instance
            prediction_id: ID of the prediction
            model_id: ID of the model
            feature_names: Names of features
            methods: Methods to use (None = all enabled)
            background_data: Background data for explainers
            
        Returns:
            Dictionary of explanation results by method
        """
        explanations = {}
        
        # Determine which methods to use
        if methods is None:
            methods = []
            if self.config.enable_shap:
                methods.append("shap")
            if self.config.enable_lime:
                methods.append("lime")
            if self.config.enable_surrogate:
                methods.append("surrogate")
        
        # Generate explanations
        if self.config.async_explanations:
            # Async generation
            tasks = []
            
            if "shap" in methods:
                tasks.append(self._explain_with_shap_async(
                    model, instance, prediction_id, model_id, 
                    feature_names, background_data
                ))
            
            if "lime" in methods:
                tasks.append(self._explain_with_lime_async(
                    model, instance, prediction_id, model_id,
                    feature_names, background_data
                ))
            
            if "surrogate" in methods:
                tasks.append(self._explain_with_surrogate_async(
                    model, instance, prediction_id, model_id,
                    feature_names, background_data
                ))
            
            # Run tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            method_names = [t.name for t in tasks]
            for method, result in zip(method_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Error in {method} explanation: {result}")
                else:
                    explanations[method] = result
        
        else:
            # Sync generation
            if "shap" in methods:
                try:
                    explanations["shap"] = self._explain_with_shap(
                        model, instance, prediction_id, model_id,
                        feature_names, background_data
                    )
                except Exception as e:
                    logger.error(f"SHAP explanation failed: {e}")
            
            if "lime" in methods:
                try:
                    explanations["lime"] = self._explain_with_lime(
                        model, instance, prediction_id, model_id,
                        feature_names, background_data
                    )
                except Exception as e:
                    logger.error(f"LIME explanation failed: {e}")
            
            if "surrogate" in methods:
                try:
                    explanations["surrogate"] = self._explain_with_surrogate(
                        model, instance, prediction_id, model_id,
                        feature_names, background_data
                    )
                except Exception as e:
                    logger.error(f"Surrogate explanation failed: {e}")
        
        # Store explanations if configured
        if self.config.store_explanations and self.db_session:
            await self._store_explanations(explanations)
        
        # Cache explanations
        self.explanation_cache[prediction_id] = explanations
        
        # Limit cache size
        if len(self.explanation_cache) > self.config.explanation_cache_size:
            # Remove oldest entries
            oldest_keys = list(self.explanation_cache.keys())[
                :len(self.explanation_cache) - self.config.explanation_cache_size
            ]
            for key in oldest_keys:
                del self.explanation_cache[key]
        
        return explanations
    
    def _explain_with_shap(
        self,
        model: Any,
        instance: Union[np.ndarray, pd.DataFrame],
        prediction_id: str,
        model_id: str,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> ExplanationResult:
        """Generate SHAP explanation."""
        # Get or create SHAP explainer
        if model_id not in self.shap_explainers:
            explainer = SHAPExplainer(model, self.config.shap_config)
            if background_data is not None:
                explainer.set_background_data(background_data)
            self.shap_explainers[model_id] = explainer
        else:
            explainer = self.shap_explainers[model_id]
        
        # Generate explanation
        return explainer.explain_instance(instance, prediction_id, feature_names)
    
    async def _explain_with_shap_async(
        self,
        model: Any,
        instance: Union[np.ndarray, pd.DataFrame],
        prediction_id: str,
        model_id: str,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> ExplanationResult:
        """Generate SHAP explanation asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._explain_with_shap,
            model, instance, prediction_id, model_id,
            feature_names, background_data
        )
    
    def _explain_with_lime(
        self,
        model: Any,
        instance: Union[np.ndarray, pd.DataFrame],
        prediction_id: str,
        model_id: str,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> ExplanationResult:
        """Generate LIME explanation."""
        # Get or create LIME explainer
        if model_id not in self.lime_explainers:
            if background_data is None:
                raise ValueError("LIME requires background/training data")
            
            explainer = LIMETabularExplainer(
                background_data,
                feature_names,
                config=self.config.lime_config
            )
            self.lime_explainers[model_id] = explainer
        else:
            explainer = self.lime_explainers[model_id]
        
        # Get prediction function
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            predict_fn = model.predict
        
        # Generate explanation
        return explainer.explain_instance(
            instance, predict_fn, prediction_id, model_id
        )
    
    async def _explain_with_lime_async(
        self,
        model: Any,
        instance: Union[np.ndarray, pd.DataFrame],
        prediction_id: str,
        model_id: str,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> ExplanationResult:
        """Generate LIME explanation asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._explain_with_lime,
            model, instance, prediction_id, model_id,
            feature_names, background_data
        )
    
    def _explain_with_surrogate(
        self,
        model: Any,
        instance: Union[np.ndarray, pd.DataFrame],
        prediction_id: str,
        model_id: str,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> ExplanationResult:
        """Generate surrogate tree explanation."""
        if background_data is None:
            raise ValueError("Surrogate model requires training data")
        
        # Create and fit surrogate
        surrogate = DecisionTreeSurrogate(model, self.config.surrogate_config)
        surrogate.fit(background_data, feature_names)
        
        # Get instance explanation
        instance_exp = surrogate.explain_instance(instance)
        
        # Convert to ExplanationResult
        feature_contributions = []
        for step in instance_exp["decision_path"][:-1]:  # Exclude leaf
            # Create contribution for each decision
            contribution = FeatureContribution(
                feature_name=step["feature"],
                value=step["value"],
                contribution=0.0,  # Surrogate doesn't provide direct contributions
                baseline_value=step["threshold"]
            )
            feature_contributions.append(contribution)
        
        return ExplanationResult(
            explanation_id=f"surrogate_{prediction_id}_{datetime.now().timestamp()}",
            model_id=model_id,
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            explanation_type=ExplanationType.DECISION_TREE,
            explanation_level=ExplanationLevel.LOCAL,
            feature_contributions=feature_contributions,
            base_value=0.0,
            prediction_value=instance_exp["surrogate_prediction"],
            confidence_score=surrogate.fidelity_score,
            method_parameters={
                "fidelity_score": surrogate.fidelity_score,
                "tree_depth": self.config.surrogate_config.max_depth
            },
            visualization_data=instance_exp
        )
    
    async def _explain_with_surrogate_async(
        self,
        model: Any,
        instance: Union[np.ndarray, pd.DataFrame],
        prediction_id: str,
        model_id: str,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> ExplanationResult:
        """Generate surrogate explanation asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._explain_with_surrogate,
            model, instance, prediction_id, model_id,
            feature_names, background_data
        )
    
    async def explain_global_behavior(
        self,
        model: Any,
        data: Union[np.ndarray, pd.DataFrame],
        model_id: str,
        feature_names: Optional[List[str]] = None
    ) -> GlobalExplanation:
        """Generate global explanation of model behavior.
        
        Args:
            model: The model to explain
            data: Dataset to analyze
            model_id: ID of the model
            feature_names: Names of features
            
        Returns:
            GlobalExplanation result
        """
        # Use feature importance analyzer
        global_exp = self.importance_analyzer.analyze_global_importance(
            model, data, data, feature_names, method="all"
        )
        
        # Store if configured
        if self.config.store_explanations and self.db_session:
            await self._store_global_explanation(global_exp)
        
        return global_exp
    
    async def generate_counterfactual(
        self,
        model: Any,
        instance: Union[np.ndarray, pd.DataFrame],
        desired_outcome: float,
        model_id: str,
        feature_names: Optional[List[str]] = None,
        feature_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> CounterfactualExplanation:
        """Generate counterfactual explanation.
        
        Args:
            model: The model
            instance: Original instance
            desired_outcome: Desired prediction value
            model_id: ID of the model
            feature_names: Names of features
            feature_constraints: Valid ranges for features
            
        Returns:
            CounterfactualExplanation
        """
        aggregator = ExplanationAggregator(self.config.aggregation_config)
        
        counterfactual = aggregator.generate_counterfactual(
            instance, model, desired_outcome,
            feature_names, feature_ranges=feature_constraints
        )
        
        return counterfactual
    
    def aggregate_explanations(
        self,
        explanations: Union[List[ExplanationResult], Dict[str, ExplanationResult]]
    ) -> ExplanationResult:
        """Aggregate multiple explanations.
        
        Args:
            explanations: List or dict of explanations
            
        Returns:
            Aggregated explanation
        """
        aggregator = ExplanationAggregator(self.config.aggregation_config)
        
        if isinstance(explanations, dict):
            explanations = list(explanations.values())
        
        for exp in explanations:
            aggregator.add_explanation(exp)
        
        return aggregator.aggregate()
    
    def compare_methods(
        self,
        explanations: Dict[str, ExplanationResult]
    ) -> Dict[str, Any]:
        """Compare different explanation methods.
        
        Args:
            explanations: Dictionary of explanations by method
            
        Returns:
            Comparison results
        """
        aggregator = ExplanationAggregator(self.config.aggregation_config)
        comparison = aggregator.compare_explanations(list(explanations.values()))
        
        return {
            "comparison": comparison,
            "consensus_features": comparison.consensus_features,
            "disputed_features": comparison.disputed_features,
            "mean_agreement": comparison.correlation_metrics["mean_correlation"]
        }
    
    async def visualize_explanation(
        self,
        explanation: ExplanationResult,
        output_name: str,
        plot_types: List[str] = ["feature_importance"]
    ) -> Dict[str, str]:
        """Create visualizations for an explanation.
        
        Args:
            explanation: Explanation to visualize
            output_name: Base name for output files
            plot_types: Types of plots to create
            
        Returns:
            Dictionary of plot type to file path
        """
        output_paths = {}
        base_path = Path(self.config.visualization_output_dir) / output_name
        
        for plot_type in plot_types:
            if plot_type == "feature_importance":
                fig = self.visualizer.plot_feature_importance(
                    explanation,
                    save_path=f"{base_path}_importance.png"
                )
                output_paths["feature_importance"] = f"{base_path}_importance.png"
            
            elif plot_type == "shap_waterfall" and explanation.explanation_type == ExplanationType.SHAP:
                fig = self.visualizer.plot_shap_waterfall(
                    explanation,
                    save_path=f"{base_path}_waterfall.html"
                )
                output_paths["shap_waterfall"] = f"{base_path}_waterfall.html"
            
            elif plot_type == "lime" and explanation.explanation_type == ExplanationType.LIME:
                fig = self.visualizer.plot_lime_explanation(
                    explanation,
                    save_path=f"{base_path}_lime.png"
                )
                output_paths["lime"] = f"{base_path}_lime.png"
        
        return output_paths
    
    async def check_explanation_compliance(
        self,
        explanation: ExplanationResult
    ) -> Dict[str, Any]:
        """Check if explanation meets compliance requirements.
        
        Args:
            explanation: Explanation to check
            
        Returns:
            Compliance check results
        """
        compliance_results = {
            "is_compliant": True,
            "issues": [],
            "warnings": []
        }
        
        # Check confidence threshold
        if explanation.confidence_score and explanation.confidence_score < self.config.min_confidence_threshold:
            compliance_results["warnings"].append(
                f"Confidence score {explanation.confidence_score:.3f} below threshold {self.config.min_confidence_threshold}"
            )
        
        # Check for required features
        feature_names = [fc.feature_name for fc in explanation.feature_contributions]
        
        # Add more compliance checks as needed
        
        # Update compliance status
        if compliance_results["issues"]:
            compliance_results["is_compliant"] = False
        
        return compliance_results
    
    async def _store_explanations(self, explanations: Dict[str, ExplanationResult]):
        """Store explanations in database."""
        if not self.db_session:
            return
        
        for method, explanation in explanations.items():
            record = ExplanationRecord(
                explanation_id=explanation.explanation_id,
                model_id=explanation.model_id,
                prediction_id=explanation.prediction_id,
                timestamp=explanation.timestamp,
                explanation_type=explanation.explanation_type.value,
                explanation_level=explanation.explanation_level.value,
                feature_contributions=[
                    {
                        "feature_name": fc.feature_name,
                        "value": fc.value,
                        "contribution": fc.contribution
                    }
                    for fc in explanation.feature_contributions
                ],
                base_value=explanation.base_value,
                prediction_value=explanation.prediction_value,
                confidence_score=explanation.confidence_score,
                computation_time=explanation.computation_time,
                metadata=explanation.method_parameters,
                compliance_notes=explanation.compliance_notes
            )
            
            self.db_session.add(record)
        
        await self.db_session.commit()
    
    async def _store_global_explanation(self, explanation: GlobalExplanation):
        """Store global explanation in database."""
        if not self.db_session:
            return
        
        # Convert to a storable format
        record = ExplanationRecord(
            explanation_id=f"global_{explanation.model_id}_{explanation.timestamp.timestamp()}",
            model_id=explanation.model_id,
            prediction_id="global",
            timestamp=explanation.timestamp,
            explanation_type="global_importance",
            explanation_level="global",
            feature_contributions=[
                {
                    "feature_name": feature,
                    "value": 0.0,
                    "contribution": importance
                }
                for feature, importance in explanation.global_feature_importance.items()
            ],
            base_value=0.0,
            prediction_value=0.0,
            metadata={
                "num_samples": explanation.num_samples,
                "feature_interactions": [
                    {"features": list(k), "importance": v}
                    for k, v in list(explanation.feature_interaction_importance.items())[:10]
                ]
            }
        )
        
        self.db_session.add(record)
        await self.db_session.commit()
    
    def get_cached_explanation(self, prediction_id: str) -> Optional[Dict[str, ExplanationResult]]:
        """Get cached explanation if available.
        
        Args:
            prediction_id: ID of the prediction
            
        Returns:
            Cached explanations or None
        """
        return self.explanation_cache.get(prediction_id)