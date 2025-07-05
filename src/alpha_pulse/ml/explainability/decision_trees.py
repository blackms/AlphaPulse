"""Decision tree surrogate models for model interpretability."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import graphviz
from sklearn.tree import export_graphviz, export_text
import json

from ...models.explanation_result import ExplanationType, ExplanationLevel


logger = logging.getLogger(__name__)


@dataclass
class SurrogateTreeConfig:
    """Configuration for surrogate decision tree."""
    max_depth: int = 5
    min_samples_split: int = 20
    min_samples_leaf: int = 10
    max_features: Optional[Union[int, float, str]] = None
    random_state: int = 42
    ccp_alpha: float = 0.0  # Complexity parameter for pruning
    max_leaf_nodes: Optional[int] = None
    splitter: str = "best"  # or "random"
    criterion: Optional[str] = None  # Auto-detect based on task


class DecisionTreeSurrogate:
    """Decision tree surrogate model for interpretability."""
    
    def __init__(
        self,
        original_model: Any,
        config: Optional[SurrogateTreeConfig] = None
    ):
        """Initialize decision tree surrogate.
        
        Args:
            original_model: The complex model to approximate
            config: Configuration for surrogate tree
        """
        self.original_model = original_model
        self.config = config or SurrogateTreeConfig()
        self.surrogate_tree = None
        self.fidelity_score = None
        self.feature_names = None
        self.is_classifier = hasattr(original_model, "predict_proba")
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'DecisionTreeSurrogate':
        """Fit surrogate tree to approximate the original model.
        
        Args:
            X: Training data
            feature_names: Names of features
            sample_weight: Sample weights for training
            
        Returns:
            Self for chaining
        """
        # Get feature names
        if feature_names is None:
            if hasattr(X, "columns"):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Get predictions from original model
        if self.is_classifier:
            y_original = self.original_model.predict(X)
            # Also get probabilities for soft targets
            y_proba = self.original_model.predict_proba(X)
        else:
            y_original = self.original_model.predict(X)
        
        # Determine criterion
        if self.config.criterion is None:
            criterion = "gini" if self.is_classifier else "squared_error"
        else:
            criterion = self.config.criterion
        
        # Create and fit surrogate tree
        if self.is_classifier:
            self.surrogate_tree = DecisionTreeClassifier(
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                random_state=self.config.random_state,
                ccp_alpha=self.config.ccp_alpha,
                max_leaf_nodes=self.config.max_leaf_nodes,
                splitter=self.config.splitter,
                criterion=criterion
            )
        else:
            self.surrogate_tree = DecisionTreeRegressor(
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                random_state=self.config.random_state,
                ccp_alpha=self.config.ccp_alpha,
                max_leaf_nodes=self.config.max_leaf_nodes,
                splitter=self.config.splitter,
                criterion=criterion
            )
        
        # Fit the surrogate
        self.surrogate_tree.fit(X, y_original, sample_weight=sample_weight)
        
        # Compute fidelity score
        self.fidelity_score = self._compute_fidelity(X, y_original)
        
        logger.info(f"Fitted surrogate tree with fidelity score: {self.fidelity_score:.4f}")
        
        return self
    
    def _compute_fidelity(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_original: np.ndarray
    ) -> float:
        """Compute fidelity of surrogate to original model."""
        y_surrogate = self.surrogate_tree.predict(X)
        
        if self.is_classifier:
            # Accuracy for classification
            fidelity = accuracy_score(y_original, y_surrogate)
        else:
            # R² for regression
            fidelity = r2_score(y_original, y_surrogate)
        
        return float(fidelity)
    
    def explain_global(self) -> Dict[str, Any]:
        """Generate global explanation from surrogate tree.
        
        Returns:
            Dictionary with tree structure and rules
        """
        if self.surrogate_tree is None:
            raise ValueError("Surrogate tree not fitted yet")
        
        # Extract tree structure
        tree_structure = self._extract_tree_structure()
        
        # Extract decision rules
        decision_rules = self._extract_decision_rules()
        
        # Get feature importance from tree
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_importance[name] = float(self.surrogate_tree.feature_importances_[i])
        
        # Compute tree statistics
        tree_stats = self._compute_tree_statistics()
        
        return {
            "tree_structure": tree_structure,
            "decision_rules": decision_rules,
            "feature_importance": feature_importance,
            "fidelity_score": self.fidelity_score,
            "tree_statistics": tree_stats,
            "feature_names": self.feature_names
        }
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Explain a single instance using the surrogate tree.
        
        Args:
            instance: Instance to explain
            
        Returns:
            Dictionary with decision path and contribution
        """
        if self.surrogate_tree is None:
            raise ValueError("Surrogate tree not fitted yet")
        
        # Get instance as array
        if isinstance(instance, pd.DataFrame):
            instance_array = instance.values[0]
        else:
            instance_array = instance.flatten()
        
        # Get decision path
        decision_path = self._get_decision_path(instance_array)
        
        # Get prediction from both models
        original_pred = self.original_model.predict(instance_array.reshape(1, -1))[0]
        surrogate_pred = self.surrogate_tree.predict(instance_array.reshape(1, -1))[0]
        
        # Get leaf node information
        leaf_id = self.surrogate_tree.apply(instance_array.reshape(1, -1))[0]
        leaf_stats = self._get_leaf_statistics(leaf_id)
        
        return {
            "decision_path": decision_path,
            "original_prediction": float(original_pred),
            "surrogate_prediction": float(surrogate_pred),
            "prediction_difference": float(abs(original_pred - surrogate_pred)),
            "leaf_id": int(leaf_id),
            "leaf_statistics": leaf_stats,
            "feature_values": {
                name: float(value) 
                for name, value in zip(self.feature_names, instance_array)
            }
        }
    
    def _extract_tree_structure(self) -> Dict[str, Any]:
        """Extract the tree structure as a dictionary."""
        tree = self.surrogate_tree.tree_
        
        def recurse(node_id):
            if tree.feature[node_id] == -2:  # Leaf node
                return {
                    "type": "leaf",
                    "value": float(tree.value[node_id][0][0]),
                    "samples": int(tree.n_node_samples[node_id]),
                    "impurity": float(tree.impurity[node_id])
                }
            else:
                return {
                    "type": "split",
                    "feature": self.feature_names[tree.feature[node_id]],
                    "threshold": float(tree.threshold[node_id]),
                    "samples": int(tree.n_node_samples[node_id]),
                    "impurity": float(tree.impurity[node_id]),
                    "left": recurse(tree.children_left[node_id]),
                    "right": recurse(tree.children_right[node_id])
                }
        
        return recurse(0)
    
    def _extract_decision_rules(self, max_rules: int = 50) -> List[str]:
        """Extract interpretable decision rules from the tree."""
        tree = self.surrogate_tree.tree_
        rules = []
        
        def recurse(node_id, rule_prefix=""):
            if tree.feature[node_id] == -2:  # Leaf node
                value = tree.value[node_id][0][0]
                samples = tree.n_node_samples[node_id]
                
                if self.is_classifier:
                    rule = f"IF {rule_prefix[:-5]} THEN class = {int(value)} (samples: {samples})"
                else:
                    rule = f"IF {rule_prefix[:-5]} THEN value = {value:.3f} (samples: {samples})"
                
                if rule_prefix:  # Don't add empty rules
                    rules.append(rule)
            else:
                feature = self.feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                
                # Left branch (<=)
                left_rule = f"{rule_prefix}{feature} <= {threshold:.3f} AND "
                recurse(tree.children_left[node_id], left_rule)
                
                # Right branch (>)
                right_rule = f"{rule_prefix}{feature} > {threshold:.3f} AND "
                recurse(tree.children_right[node_id], right_rule)
        
        recurse(0)
        
        # Sort by sample count (implicit in tree traversal order)
        return rules[:max_rules]
    
    def _compute_tree_statistics(self) -> Dict[str, Any]:
        """Compute statistics about the surrogate tree."""
        tree = self.surrogate_tree.tree_
        
        # Count nodes
        n_leaves = np.sum(tree.feature == -2)
        n_splits = np.sum(tree.feature != -2)
        
        # Compute depth statistics
        depths = self._compute_node_depths()
        
        # Feature usage
        feature_usage = {}
        for i, name in enumerate(self.feature_names):
            usage_count = np.sum(tree.feature == i)
            feature_usage[name] = int(usage_count)
        
        return {
            "n_leaves": int(n_leaves),
            "n_splits": int(n_splits),
            "total_nodes": int(n_leaves + n_splits),
            "max_depth": int(np.max(depths)),
            "avg_depth": float(np.mean(depths)),
            "feature_usage": feature_usage
        }
    
    def _compute_node_depths(self) -> np.ndarray:
        """Compute depth of each node in the tree."""
        tree = self.surrogate_tree.tree_
        n_nodes = tree.node_count
        depths = np.zeros(n_nodes)
        
        def recurse(node_id, depth):
            depths[node_id] = depth
            if tree.feature[node_id] != -2:  # Not a leaf
                recurse(tree.children_left[node_id], depth + 1)
                recurse(tree.children_right[node_id], depth + 1)
        
        recurse(0, 0)
        return depths
    
    def _get_decision_path(self, instance: np.ndarray) -> List[Dict[str, Any]]:
        """Get the decision path for an instance."""
        tree = self.surrogate_tree.tree_
        path = []
        
        node_id = 0
        while tree.feature[node_id] != -2:  # Not a leaf
            feature_idx = tree.feature[node_id]
            feature_name = self.feature_names[feature_idx]
            threshold = tree.threshold[node_id]
            feature_value = instance[feature_idx]
            
            if feature_value <= threshold:
                decision = "<="
                next_node = tree.children_left[node_id]
            else:
                decision = ">"
                next_node = tree.children_right[node_id]
            
            path.append({
                "node_id": int(node_id),
                "feature": feature_name,
                "value": float(feature_value),
                "threshold": float(threshold),
                "decision": decision,
                "samples": int(tree.n_node_samples[node_id])
            })
            
            node_id = next_node
        
        # Add leaf information
        path.append({
            "node_id": int(node_id),
            "type": "leaf",
            "prediction": float(tree.value[node_id][0][0]),
            "samples": int(tree.n_node_samples[node_id])
        })
        
        return path
    
    def _get_leaf_statistics(self, leaf_id: int) -> Dict[str, Any]:
        """Get statistics for a specific leaf node."""
        tree = self.surrogate_tree.tree_
        
        return {
            "samples": int(tree.n_node_samples[leaf_id]),
            "value": float(tree.value[leaf_id][0][0]),
            "impurity": float(tree.impurity[leaf_id]),
            "percentage_of_data": float(tree.n_node_samples[leaf_id] / tree.n_node_samples[0])
        }
    
    def visualize_tree(
        self,
        output_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        filled: bool = True,
        rounded: bool = True,
        precision: int = 2
    ) -> str:
        """Generate visualization of the surrogate tree.
        
        Args:
            output_path: Path to save visualization (if None, returns DOT string)
            feature_names: Feature names to use
            filled: Whether to fill nodes with colors
            rounded: Whether to use rounded corners
            precision: Decimal precision for numbers
            
        Returns:
            DOT format string
        """
        if self.surrogate_tree is None:
            raise ValueError("Surrogate tree not fitted yet")
        
        feature_names = feature_names or self.feature_names
        
        dot_data = export_graphviz(
            self.surrogate_tree,
            out_file=None,
            feature_names=feature_names,
            class_names=["0", "1"] if self.is_classifier else None,
            filled=filled,
            rounded=rounded,
            special_characters=True,
            precision=precision
        )
        
        if output_path:
            graph = graphviz.Source(dot_data)
            graph.render(output_path, format='png', cleanup=True)
        
        return dot_data
    
    def get_text_representation(self) -> str:
        """Get text representation of the tree."""
        if self.surrogate_tree is None:
            raise ValueError("Surrogate tree not fitted yet")
        
        return export_text(
            self.surrogate_tree,
            feature_names=self.feature_names
        )
    
    def prune_tree(self, alpha: float) -> 'DecisionTreeSurrogate':
        """Prune the tree using cost-complexity pruning.
        
        Args:
            alpha: Complexity parameter for pruning
            
        Returns:
            New DecisionTreeSurrogate with pruned tree
        """
        new_config = SurrogateTreeConfig(
            **{**self.config.__dict__, "ccp_alpha": alpha}
        )
        
        new_surrogate = DecisionTreeSurrogate(self.original_model, new_config)
        return new_surrogate