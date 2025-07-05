"""Aggregator for combining multiple explanation methods."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from scipy.stats import spearmanr, kendalltau
import json

from ...models.explanation_result import (
    ExplanationResult, ExplanationType, ExplanationLevel,
    FeatureContribution, ExplanationComparison, CounterfactualExplanation
)


logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """Configuration for explanation aggregation."""
    methods: List[str] = None  # Methods to include (None = all)
    weights: Dict[str, float] = None  # Method weights (None = equal)
    consensus_threshold: float = 0.7  # Agreement threshold
    top_k_features: int = 10
    normalize_scores: bool = True
    aggregation_method: str = "weighted_mean"  # or "median", "rank_fusion"


class ExplanationAggregator:
    """Aggregates explanations from multiple methods."""
    
    def __init__(self, config: Optional[AggregationConfig] = None):
        """Initialize explanation aggregator.
        
        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()
        self.explanations = []
    
    def add_explanation(self, explanation: ExplanationResult):
        """Add an explanation to aggregate.
        
        Args:
            explanation: Explanation result to add
        """
        self.explanations.append(explanation)
    
    def aggregate(self) -> ExplanationResult:
        """Aggregate all added explanations.
        
        Returns:
            Aggregated ExplanationResult
        """
        if not self.explanations:
            raise ValueError("No explanations to aggregate")
        
        # Filter by methods if specified
        if self.config.methods:
            filtered = [
                exp for exp in self.explanations
                if exp.explanation_type.value in self.config.methods
            ]
            if not filtered:
                logger.warning("No explanations match specified methods")
                filtered = self.explanations
        else:
            filtered = self.explanations
        
        # Group by prediction ID
        prediction_groups = {}
        for exp in filtered:
            if exp.prediction_id not in prediction_groups:
                prediction_groups[exp.prediction_id] = []
            prediction_groups[exp.prediction_id].append(exp)
        
        # Aggregate each prediction
        aggregated_results = []
        for pred_id, exp_group in prediction_groups.items():
            aggregated = self._aggregate_group(exp_group)
            aggregated_results.append(aggregated)
        
        # Return single result or create meta-result
        if len(aggregated_results) == 1:
            return aggregated_results[0]
        else:
            return self._create_meta_result(aggregated_results)
    
    def _aggregate_group(self, explanations: List[ExplanationResult]) -> ExplanationResult:
        """Aggregate a group of explanations for the same prediction."""
        # Collect all feature contributions
        feature_contributions_by_method = {}
        for exp in explanations:
            method = exp.explanation_type.value
            feature_contributions_by_method[method] = {
                fc.feature_name: fc for fc in exp.feature_contributions
            }
        
        # Get all unique features
        all_features = set()
        for contributions in feature_contributions_by_method.values():
            all_features.update(contributions.keys())
        
        # Aggregate contributions for each feature
        aggregated_contributions = []
        for feature in all_features:
            aggregated_contrib = self._aggregate_feature_contribution(
                feature, feature_contributions_by_method
            )
            if aggregated_contrib:
                aggregated_contributions.append(aggregated_contrib)
        
        # Sort by absolute contribution
        aggregated_contributions.sort(
            key=lambda x: abs(x.contribution),
            reverse=True
        )
        
        # Take top k features
        if self.config.top_k_features:
            aggregated_contributions = aggregated_contributions[:self.config.top_k_features]
        
        # Use first explanation as template
        template = explanations[0]
        
        # Calculate aggregated base and prediction values
        base_values = [exp.base_value for exp in explanations if exp.base_value is not None]
        prediction_values = [exp.prediction_value for exp in explanations]
        
        aggregated_base = np.mean(base_values) if base_values else 0.0
        aggregated_prediction = np.mean(prediction_values)
        
        # Compute confidence based on agreement
        confidence = self._compute_confidence(explanations)
        
        return ExplanationResult(
            explanation_id=f"aggregated_{template.prediction_id}_{datetime.now().timestamp()}",
            model_id=template.model_id,
            prediction_id=template.prediction_id,
            timestamp=datetime.now(),
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,  # Generic type
            explanation_level=template.explanation_level,
            feature_contributions=aggregated_contributions,
            base_value=float(aggregated_base),
            prediction_value=float(aggregated_prediction),
            confidence_score=confidence,
            method_parameters={
                "aggregation_config": self.config.__dict__,
                "methods_used": [exp.explanation_type.value for exp in explanations],
                "num_explanations": len(explanations)
            }
        )
    
    def _aggregate_feature_contribution(
        self,
        feature_name: str,
        contributions_by_method: Dict[str, Dict[str, FeatureContribution]]
    ) -> Optional[FeatureContribution]:
        """Aggregate contribution for a single feature across methods."""
        # Collect contributions
        contributions = []
        values = []
        weights = []
        
        for method, all_contributions in contributions_by_method.items():
            if feature_name in all_contributions:
                fc = all_contributions[feature_name]
                contributions.append(fc.contribution)
                values.append(fc.value)
                
                # Get method weight
                if self.config.weights and method in self.config.weights:
                    weights.append(self.config.weights[method])
                else:
                    weights.append(1.0)
        
        if not contributions:
            return None
        
        # Normalize scores if requested
        if self.config.normalize_scores:
            max_abs = max(abs(c) for c in contributions)
            if max_abs > 0:
                contributions = [c / max_abs for c in contributions]
        
        # Aggregate based on method
        if self.config.aggregation_method == "weighted_mean":
            weights_array = np.array(weights)
            weights_array = weights_array / weights_array.sum()
            aggregated_contribution = np.sum(np.array(contributions) * weights_array)
        elif self.config.aggregation_method == "median":
            aggregated_contribution = np.median(contributions)
        elif self.config.aggregation_method == "rank_fusion":
            # Rank-based aggregation
            ranks = self._compute_ranks(contributions)
            aggregated_contribution = self._fuse_ranks(ranks, weights)
        else:
            aggregated_contribution = np.mean(contributions)
        
        # Use most common value (or mean)
        feature_value = np.mean(values) if values else 0.0
        
        return FeatureContribution(
            feature_name=feature_name,
            value=float(feature_value),
            contribution=float(aggregated_contribution),
            confidence_interval=(float(np.min(contributions)), float(np.max(contributions)))
        )
    
    def _compute_ranks(self, scores: List[float]) -> List[int]:
        """Compute ranks from scores (higher score = lower rank)."""
        sorted_indices = np.argsort([-abs(s) for s in scores])
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(scores))
        return ranks.tolist()
    
    def _fuse_ranks(self, ranks: List[int], weights: List[float]) -> float:
        """Fuse ranks using weighted Borda count."""
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()
        
        # Lower rank is better, so invert
        max_rank = max(ranks)
        inverted_ranks = [max_rank - r for r in ranks]
        
        # Weighted sum
        fused_score = np.sum(np.array(inverted_ranks) * weights_array)
        
        # Normalize to [-1, 1] range
        return (fused_score / max_rank) * 2 - 1
    
    def _compute_confidence(self, explanations: List[ExplanationResult]) -> float:
        """Compute confidence based on agreement between explanations."""
        if len(explanations) < 2:
            return 1.0
        
        # Compare top features across explanations
        top_features_sets = []
        for exp in explanations:
            top_features = exp.get_top_features(self.config.top_k_features)
            top_feature_names = [fc.feature_name for fc in top_features]
            top_features_sets.append(set(top_feature_names))
        
        # Compute Jaccard similarity
        similarities = []
        for i in range(len(top_features_sets)):
            for j in range(i + 1, len(top_features_sets)):
                intersection = len(top_features_sets[i] & top_features_sets[j])
                union = len(top_features_sets[i] | top_features_sets[j])
                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)
        
        # Average similarity as confidence
        confidence = np.mean(similarities) if similarities else 0.5
        
        return float(confidence)
    
    def _create_meta_result(self, results: List[ExplanationResult]) -> ExplanationResult:
        """Create a meta-result from multiple aggregated results."""
        # Aggregate all feature contributions
        all_contributions = {}
        for result in results:
            for fc in result.feature_contributions:
                if fc.feature_name not in all_contributions:
                    all_contributions[fc.feature_name] = []
                all_contributions[fc.feature_name].append(fc.contribution)
        
        # Create averaged contributions
        meta_contributions = []
        for feature, contribs in all_contributions.items():
            meta_contributions.append(FeatureContribution(
                feature_name=feature,
                value=0.0,  # Unknown in meta-result
                contribution=float(np.mean(contribs))
            ))
        
        # Sort by importance
        meta_contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        return ExplanationResult(
            explanation_id=f"meta_{datetime.now().timestamp()}",
            model_id=results[0].model_id,
            prediction_id="meta",
            timestamp=datetime.now(),
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            explanation_level=ExplanationLevel.GLOBAL,
            feature_contributions=meta_contributions[:self.config.top_k_features],
            base_value=float(np.mean([r.base_value for r in results])),
            prediction_value=float(np.mean([r.prediction_value for r in results])),
            confidence_score=float(np.mean([r.confidence_score for r in results if r.confidence_score])),
            method_parameters={
                "type": "meta_aggregation",
                "num_results": len(results)
            }
        )
    
    def compare_explanations(
        self,
        explanations: Optional[List[ExplanationResult]] = None
    ) -> ExplanationComparison:
        """Compare multiple explanations for consistency.
        
        Args:
            explanations: Explanations to compare (uses stored if None)
            
        Returns:
            ExplanationComparison result
        """
        if explanations is None:
            explanations = self.explanations
        
        if len(explanations) < 2:
            raise ValueError("Need at least 2 explanations to compare")
        
        # Extract feature contributions
        feature_matrices = []
        all_features = set()
        
        for exp in explanations:
            feature_dict = {fc.feature_name: fc.contribution for fc in exp.feature_contributions}
            feature_matrices.append(feature_dict)
            all_features.update(feature_dict.keys())
        
        # Create contribution matrix
        all_features = sorted(list(all_features))
        contribution_matrix = np.zeros((len(explanations), len(all_features)))
        
        for i, feature_dict in enumerate(feature_matrices):
            for j, feature in enumerate(all_features):
                contribution_matrix[i, j] = feature_dict.get(feature, 0.0)
        
        # Compute agreement matrix
        agreement_matrix = np.zeros((len(explanations), len(explanations)))
        for i in range(len(explanations)):
            for j in range(i, len(explanations)):
                # Compute correlation
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    corr, _ = spearmanr(contribution_matrix[i], contribution_matrix[j])
                    agreement_matrix[i, j] = agreement_matrix[j, i] = corr
        
        # Find consensus and disputed features
        contribution_differences = {}
        consistency_scores = {}
        
        for j, feature in enumerate(all_features):
            contributions = contribution_matrix[:, j]
            non_zero = contributions[contributions != 0]
            
            if len(non_zero) > 1:
                # Compute consistency
                std_dev = np.std(non_zero)
                mean_abs = np.mean(np.abs(non_zero))
                consistency = 1.0 - (std_dev / (mean_abs + 1e-10))
                consistency_scores[feature] = float(consistency)
                
                # Store differences
                contribution_differences[feature] = contributions.tolist()
            else:
                consistency_scores[feature] = 0.0
                contribution_differences[feature] = contributions.tolist()
        
        # Identify consensus and disputed features
        consensus_threshold = self.config.consensus_threshold
        consensus_features = [
            f for f, score in consistency_scores.items()
            if score >= consensus_threshold
        ]
        disputed_features = [
            f for f, score in consistency_scores.items()
            if score < consensus_threshold and score > 0
        ]
        
        # Statistical significance tests
        statistical_significance = {}
        for feature in all_features:
            contributions = [fm.get(feature, 0.0) for fm in feature_matrices]
            # Simple variance test
            variance = np.var(contributions)
            statistical_significance[feature] = float(1.0 / (1.0 + variance))
        
        return ExplanationComparison(
            explanation_ids=[exp.explanation_id for exp in explanations],
            comparison_timestamp=datetime.now(),
            contribution_differences=contribution_differences,
            consistency_scores=consistency_scores,
            method_agreement_matrix=agreement_matrix,
            consensus_features=consensus_features,
            disputed_features=disputed_features,
            statistical_significance=statistical_significance,
            correlation_metrics={
                "mean_correlation": float(np.mean(agreement_matrix[np.triu_indices_from(agreement_matrix, k=1)])),
                "min_correlation": float(np.min(agreement_matrix[np.triu_indices_from(agreement_matrix, k=1)]))
            }
        )
    
    def generate_counterfactual(
        self,
        instance: Union[np.ndarray, pd.DataFrame],
        model: Any,
        desired_outcome: float,
        feature_names: Optional[List[str]] = None,
        max_changes: int = 5,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> CounterfactualExplanation:
        """Generate counterfactual explanation.
        
        Args:
            instance: Original instance
            model: Model to use for predictions
            desired_outcome: Desired prediction outcome
            feature_names: Names of features
            max_changes: Maximum number of features to change
            feature_ranges: Valid ranges for features
            
        Returns:
            CounterfactualExplanation
        """
        start_time = datetime.now()
        
        # Convert instance to array
        if isinstance(instance, pd.DataFrame):
            instance_array = instance.values[0]
            if feature_names is None:
                feature_names = list(instance.columns)
        else:
            instance_array = instance.flatten()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(instance_array))]
        
        # Get original prediction
        original_pred = model.predict(instance_array.reshape(1, -1))[0]
        
        # Simple greedy approach for counterfactual generation
        counterfactual = instance_array.copy()
        changes = {}
        
        # Try changing each feature
        feature_impacts = []
        for i, feature in enumerate(feature_names):
            # Skip if no range specified
            if feature_ranges and feature in feature_ranges:
                min_val, max_val = feature_ranges[feature]
            else:
                # Use heuristic range
                min_val = instance_array[i] * 0.5
                max_val = instance_array[i] * 1.5
            
            # Test impact of changing feature
            test_instance = instance_array.copy()
            
            # Try moving towards desired outcome
            if desired_outcome > original_pred:
                test_instance[i] = max_val
            else:
                test_instance[i] = min_val
            
            test_pred = model.predict(test_instance.reshape(1, -1))[0]
            impact = abs(test_pred - original_pred)
            
            feature_impacts.append((i, feature, impact, test_instance[i]))
        
        # Sort by impact
        feature_impacts.sort(key=lambda x: x[2], reverse=True)
        
        # Apply top changes
        for i, (idx, feature, impact, new_value) in enumerate(feature_impacts[:max_changes]):
            old_value = counterfactual[idx]
            counterfactual[idx] = new_value
            changes[feature] = (float(old_value), float(new_value))
            
            # Check if we've reached desired outcome
            current_pred = model.predict(counterfactual.reshape(1, -1))[0]
            if abs(current_pred - desired_outcome) < 0.1:  # Threshold
                break
        
        # Final prediction
        final_pred = model.predict(counterfactual.reshape(1, -1))[0]
        
        # Compute distance
        change_distance = np.linalg.norm(counterfactual - instance_array)
        
        # Validate counterfactual
        is_valid = abs(final_pred - desired_outcome) < 0.1
        validity_constraints = []
        if not is_valid:
            validity_constraints.append(f"Target not reached: {final_pred:.3f} vs {desired_outcome:.3f}")
        
        # Quality score (inverse of normalized distance)
        max_distance = np.linalg.norm(instance_array)
        quality_score = 1.0 - (change_distance / (max_distance + 1e-10))
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        return CounterfactualExplanation(
            original_features={f: float(v) for f, v in zip(feature_names, instance_array)},
            counterfactual_features={f: float(v) for f, v in zip(feature_names, counterfactual)},
            original_prediction=float(original_pred),
            counterfactual_prediction=float(final_pred),
            feature_changes=changes,
            change_distance=float(change_distance),
            is_valid=is_valid,
            validity_constraints=validity_constraints,
            quality_score=float(quality_score),
            generation_method="greedy_search",
            computation_time=computation_time,
            num_features_changed=len(changes)
        )