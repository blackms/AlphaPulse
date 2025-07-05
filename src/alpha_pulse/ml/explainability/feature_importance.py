"""Feature importance analysis for model interpretability."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import logging
from dataclasses import dataclass
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ...models.explanation_result import (
    GlobalExplanation, FeatureContribution
)


logger = logging.getLogger(__name__)


@dataclass
class FeatureImportanceConfig:
    """Configuration for feature importance analysis."""
    n_repeats: int = 10
    random_state: int = 42
    n_jobs: int = -1
    scoring: Optional[str] = None  # Auto-detect based on model
    sample_size: Optional[int] = None
    importance_threshold: float = 0.01
    max_features_to_test: Optional[int] = None
    use_test_set: bool = True


class FeatureImportanceAnalyzer:
    """Comprehensive feature importance analysis."""
    
    def __init__(self, config: Optional[FeatureImportanceConfig] = None):
        """Initialize feature importance analyzer.
        
        Args:
            config: Configuration for analysis
        """
        self.config = config or FeatureImportanceConfig()
    
    def analyze_global_importance(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
        method: str = "all"
    ) -> GlobalExplanation:
        """Analyze global feature importance using multiple methods.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data
            feature_names: Names of features
            method: Method(s) to use ("all", "permutation", "drop_column", "model_based")
            
        Returns:
            GlobalExplanation with comprehensive importance metrics
        """
        start_time = datetime.now()
        
        # Get feature names
        if feature_names is None:
            if hasattr(X, "columns"):
                feature_names = list(X.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Sample if needed
        if self.config.sample_size and len(X) > self.config.sample_size:
            indices = np.random.RandomState(self.config.random_state).choice(
                len(X), self.config.sample_size, replace=False
            )
            X_sample = X.iloc[indices] if hasattr(X, "iloc") else X[indices]
            y_sample = y.iloc[indices] if hasattr(y, "iloc") else y[indices]
        else:
            X_sample = X
            y_sample = y
        
        # Initialize results
        importance_results = {}
        
        # Permutation importance
        if method in ["all", "permutation"]:
            perm_importance = self._compute_permutation_importance(
                model, X_sample, y_sample, feature_names
            )
            importance_results["permutation"] = perm_importance
        
        # Drop column importance
        if method in ["all", "drop_column"]:
            drop_importance = self._compute_drop_column_importance(
                model, X_sample, y_sample, feature_names
            )
            importance_results["drop_column"] = drop_importance
        
        # Model-based importance (if available)
        if method in ["all", "model_based"]:
            model_importance = self._get_model_based_importance(model, feature_names)
            if model_importance:
                importance_results["model_based"] = model_importance
        
        # Aggregate importance scores
        aggregated_importance = self._aggregate_importance_scores(importance_results)
        
        # Compute feature statistics
        feature_stats = self._compute_feature_statistics(X_sample, feature_names)
        
        # Compute feature interactions
        interaction_importance = self._compute_feature_interactions(
            model, X_sample, y_sample, feature_names
        )
        
        # Compute performance attribution
        performance_attribution = self._compute_performance_attribution(
            model, X_sample, y_sample, feature_names, aggregated_importance
        )
        
        return GlobalExplanation(
            model_id=str(id(model)),
            timestamp=datetime.now(),
            num_samples=len(X_sample),
            global_feature_importance=aggregated_importance,
            feature_interaction_importance=interaction_importance,
            feature_value_distributions=feature_stats["distributions"],
            feature_contribution_distributions=feature_stats["contribution_stats"],
            decision_rules=self._extract_decision_rules(model, feature_names),
            performance_by_feature=performance_attribution,
            error_attribution=self._compute_error_attribution(
                model, X_sample, y_sample, feature_names
            )
        )
    
    def _compute_permutation_importance(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute permutation importance."""
        # Determine scoring function
        if self.config.scoring:
            scoring = self.config.scoring
        else:
            # Auto-detect based on model type
            if hasattr(model, "predict_proba"):
                scoring = "neg_log_loss"
            else:
                scoring = "neg_mean_squared_error"
        
        # Compute permutation importance
        result = permutation_importance(
            model, X, y,
            n_repeats=self.config.n_repeats,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            scoring=scoring
        )
        
        # Create importance dictionary
        importance_dict = {}
        for i, name in enumerate(feature_names):
            importance_dict[name] = float(result.importances_mean[i])
        
        return importance_dict
    
    def _compute_drop_column_importance(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute importance by dropping each column."""
        # Get baseline performance
        baseline_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            baseline_score = -np.mean(np.log(model.predict_proba(X)[np.arange(len(y)), y]))
        else:
            baseline_score = mean_squared_error(y, baseline_pred)
        
        importance_dict = {}
        
        # Test each feature
        for i, feature_name in enumerate(feature_names):
            # Create data with feature dropped
            X_dropped = X.copy() if hasattr(X, "copy") else np.copy(X)
            
            if hasattr(X_dropped, "iloc"):
                # Pandas DataFrame
                X_dropped.iloc[:, i] = X_dropped.iloc[:, i].mean()
            else:
                # NumPy array
                X_dropped[:, i] = np.mean(X_dropped[:, i])
            
            # Compute performance drop
            dropped_pred = model.predict(X_dropped)
            if hasattr(model, "predict_proba"):
                dropped_score = -np.mean(np.log(model.predict_proba(X_dropped)[np.arange(len(y)), y]))
            else:
                dropped_score = mean_squared_error(y, dropped_pred)
            
            # Importance is the performance degradation
            importance_dict[feature_name] = float(abs(dropped_score - baseline_score))
        
        return importance_dict
    
    def _get_model_based_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Get model-based feature importance if available."""
        importance_dict = None
        
        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importance_dict = {}
            for i, name in enumerate(feature_names):
                importance_dict[name] = float(model.feature_importances_[i])
        
        # Linear models
        elif hasattr(model, "coef_"):
            importance_dict = {}
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef[0]  # For multi-output, take first
            
            for i, name in enumerate(feature_names):
                importance_dict[name] = float(abs(coef[i]))
        
        return importance_dict
    
    def _aggregate_importance_scores(
        self,
        importance_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate importance scores from multiple methods."""
        if not importance_results:
            return {}
        
        # Get all features
        all_features = set()
        for scores in importance_results.values():
            all_features.update(scores.keys())
        
        # Aggregate scores
        aggregated = {}
        for feature in all_features:
            scores = []
            for method_scores in importance_results.values():
                if feature in method_scores:
                    # Normalize scores within each method
                    max_score = max(method_scores.values())
                    if max_score > 0:
                        normalized_score = method_scores[feature] / max_score
                        scores.append(normalized_score)
            
            if scores:
                aggregated[feature] = float(np.mean(scores))
        
        return aggregated
    
    def _compute_feature_statistics(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Compute feature value and contribution statistics."""
        distributions = {}
        contribution_stats = {}
        
        for i, name in enumerate(feature_names):
            if hasattr(X, "iloc"):
                values = X.iloc[:, i]
            else:
                values = X[:, i]
            
            # Value distribution
            distributions[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75))
            }
            
            # Placeholder for contribution stats (would be filled by SHAP/LIME)
            contribution_stats[name] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0
            }
        
        return {
            "distributions": distributions,
            "contribution_stats": contribution_stats
        }
    
    def _compute_feature_interactions(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: List[str],
        top_k: int = 10
    ) -> Dict[Tuple[str, str], float]:
        """Compute feature interaction importance."""
        interactions = {}
        
        # Only compute for top features to limit computational cost
        importance_scores = self._compute_permutation_importance(model, X, y, feature_names)
        top_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_feature_names = [f[0] for f in top_features]
        top_feature_indices = [feature_names.index(f) for f in top_feature_names]
        
        # Test pairwise interactions
        for i, (feat1_idx, feat1_name) in enumerate(zip(top_feature_indices, top_feature_names)):
            for feat2_idx, feat2_name in zip(top_feature_indices[i+1:], top_feature_names[i+1:]):
                # Permute both features together
                X_permuted = X.copy() if hasattr(X, "copy") else np.copy(X)
                
                # Shuffle both features
                perm_indices = np.random.RandomState(self.config.random_state).permutation(len(X))
                if hasattr(X_permuted, "iloc"):
                    X_permuted.iloc[:, feat1_idx] = X_permuted.iloc[perm_indices, feat1_idx]
                    X_permuted.iloc[:, feat2_idx] = X_permuted.iloc[perm_indices, feat2_idx]
                else:
                    X_permuted[:, feat1_idx] = X_permuted[perm_indices, feat1_idx]
                    X_permuted[:, feat2_idx] = X_permuted[perm_indices, feat2_idx]
                
                # Compute performance drop
                if hasattr(model, "predict_proba"):
                    baseline_score = accuracy_score(y, model.predict(X))
                    permuted_score = accuracy_score(y, model.predict(X_permuted))
                else:
                    baseline_score = mean_squared_error(y, model.predict(X))
                    permuted_score = mean_squared_error(y, model.predict(X_permuted))
                
                interaction_importance = abs(permuted_score - baseline_score)
                
                # Subtract individual importances to get interaction effect
                individual_sum = importance_scores[feat1_name] + importance_scores[feat2_name]
                interaction_effect = max(0, interaction_importance - individual_sum)
                
                interactions[(feat1_name, feat2_name)] = float(interaction_effect)
        
        return interactions
    
    def _compute_performance_attribution(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: List[str],
        feature_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """Attribute model performance to features."""
        performance_attribution = {}
        
        # Get baseline performance
        baseline_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            baseline_score = accuracy_score(y, baseline_pred)
        else:
            baseline_score = 1.0 / (1.0 + mean_squared_error(y, baseline_pred))
        
        # Weight performance by feature importance
        total_importance = sum(feature_importance.values())
        
        for feature, importance in feature_importance.items():
            if total_importance > 0:
                performance_attribution[feature] = float(
                    baseline_score * (importance / total_importance)
                )
            else:
                performance_attribution[feature] = 0.0
        
        return performance_attribution
    
    def _compute_error_attribution(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Attribute prediction errors to features."""
        error_attribution = {}
        
        # Get predictions and errors
        predictions = model.predict(X)
        errors = np.abs(y - predictions)
        
        # For each feature, compute correlation with error
        for i, feature in enumerate(feature_names):
            if hasattr(X, "iloc"):
                feature_values = X.iloc[:, i]
            else:
                feature_values = X[:, i]
            
            # Compute correlation between feature and error
            correlation = np.corrcoef(feature_values, errors)[0, 1]
            error_attribution[feature] = float(abs(correlation))
        
        return error_attribution
    
    def _extract_decision_rules(
        self,
        model: Any,
        feature_names: List[str],
        max_rules: int = 20
    ) -> List[str]:
        """Extract interpretable decision rules from model."""
        rules = []
        
        # For tree-based models, extract rules
        if hasattr(model, "estimators_"):  # Ensemble
            # Use first few trees
            for i, tree in enumerate(model.estimators_[:3]):
                if hasattr(tree, "tree_"):
                    tree_rules = self._extract_rules_from_tree(
                        tree.tree_, feature_names, max_depth=3
                    )
                    rules.extend(tree_rules[:5])  # Limit rules per tree
                    
                if len(rules) >= max_rules:
                    break
        
        elif hasattr(model, "tree_"):  # Single tree
            rules = self._extract_rules_from_tree(
                model.tree_, feature_names, max_depth=4
            )[:max_rules]
        
        return rules
    
    def _extract_rules_from_tree(
        self,
        tree,
        feature_names: List[str],
        max_depth: int = 3
    ) -> List[str]:
        """Extract rules from a single decision tree."""
        rules = []
        
        def recurse(node, depth, rule_prefix=""):
            if depth > max_depth:
                return
            
            if tree.feature[node] != -2:  # Not a leaf
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                # Left child (<=)
                left_rule = f"{rule_prefix}({feature} <= {threshold:.3f})"
                if tree.feature[tree.children_left[node]] == -2:  # Left child is leaf
                    value = tree.value[tree.children_left[node]]
                    rules.append(f"IF {left_rule} THEN {value[0][0]:.3f}")
                else:
                    recurse(tree.children_left[node], depth + 1, f"{left_rule} AND ")
                
                # Right child (>)
                right_rule = f"{rule_prefix}({feature} > {threshold:.3f})"
                if tree.feature[tree.children_right[node]] == -2:  # Right child is leaf
                    value = tree.value[tree.children_right[node]]
                    rules.append(f"IF {right_rule} THEN {value[0][0]:.3f}")
                else:
                    recurse(tree.children_right[node], depth + 1, f"{right_rule} AND ")
        
        recurse(0, 0)
        return rules
    
    def compute_mutual_information(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute mutual information between features and target."""
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        # Get feature names
        if feature_names is None:
            if hasattr(X, "columns"):
                feature_names = list(X.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Determine if classification or regression
        if len(np.unique(y)) < 10:  # Assume classification
            mi_scores = mutual_info_classif(X, y, random_state=self.config.random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=self.config.random_state)
        
        # Create dictionary
        mi_dict = {}
        for name, score in zip(feature_names, mi_scores):
            mi_dict[name] = float(score)
        
        return mi_dict