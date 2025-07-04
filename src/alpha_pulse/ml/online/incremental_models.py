"""Incremental machine learning models for streaming data."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import SGDRegressor, SGDClassifier, PassiveAggressiveRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from collections import deque
import math

from alpha_pulse.ml.online.online_learner import BaseOnlineLearner

logger = logging.getLogger(__name__)


class IncrementalSGD(BaseOnlineLearner):
    """Incremental Stochastic Gradient Descent."""
    
    def __init__(self, config: Dict[str, Any], task_type: str = 'regression'):
        super().__init__(config)
        self.task_type = task_type
        self.loss = config.get('loss', 'squared_error' if task_type == 'regression' else 'log_loss')
        self.penalty = config.get('penalty', 'l2')
        self.alpha = config.get('alpha', 0.0001)
        self.l1_ratio = config.get('l1_ratio', 0.15)
        self.epsilon = config.get('epsilon', 0.1)
        self.learning_rate_type = config.get('learning_rate_type', 'invscaling')
        self.eta0 = config.get('eta0', 0.01)
        self.power_t = config.get('power_t', 0.25)
        
        # Initialize model
        if task_type == 'regression':
            self.model = SGDRegressor(
                loss=self.loss,
                penalty=self.penalty,
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                epsilon=self.epsilon,
                learning_rate=self.learning_rate_type,
                eta0=self.eta0,
                power_t=self.power_t,
                warm_start=True,
                average=True
            )
        else:
            self.model = SGDClassifier(
                loss=self.loss,
                penalty=self.penalty,
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                epsilon=self.epsilon,
                learning_rate=self.learning_rate_type,
                eta0=self.eta0,
                power_t=self.power_t,
                warm_start=True,
                average=True
            )
            
        # Feature scaling
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: Optional[np.ndarray] = None) -> None:
        """Update model with new data."""
        # Update scaler
        if not self.scaler_fitted:
            self.scaler.partial_fit(X)
            self.scaler_fitted = True
        else:
            self.scaler.partial_fit(X)
            
        # Transform features
        X_scaled = self.scaler.transform(X)
        
        # Update model
        if self.task_type == 'classification':
            # Need to specify classes for first fit
            if not self._is_initialized:
                classes = np.unique(y)
                self.model.partial_fit(X_scaled, y, classes=classes, sample_weight=sample_weight)
            else:
                self.model.partial_fit(X_scaled, y, sample_weight=sample_weight)
        else:
            self.model.partial_fit(X_scaled, y, sample_weight=sample_weight)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_initialized:
            raise ValueError("Model not fitted yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
            
        if not self._is_initialized:
            raise ValueError("Model not fitted yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class IncrementalNaiveBayes(BaseOnlineLearner):
    """Incremental Gaussian Naive Bayes."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.var_smoothing = config.get('var_smoothing', 1e-9)
        
        self.model = GaussianNB(var_smoothing=self.var_smoothing)
        self.classes_seen = set()
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: Optional[np.ndarray] = None) -> None:
        """Update model with new data."""
        # Track classes
        self.classes_seen.update(y)
        classes = np.array(sorted(self.classes_seen))
        
        # Update model
        self.model.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_initialized:
            raise ValueError("Model not fitted yet")
            
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self._is_initialized:
            raise ValueError("Model not fitted yet")
            
        return self.model.predict_proba(X)


class IncrementalPassiveAggressive(BaseOnlineLearner):
    """Incremental Passive-Aggressive algorithm."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.C = config.get('C', 1.0)
        self.loss = config.get('loss', 'epsilon_insensitive')
        self.epsilon = config.get('epsilon', 0.1)
        
        self.model = PassiveAggressiveRegressor(
            C=self.C,
            loss=self.loss,
            epsilon=self.epsilon,
            warm_start=True,
            average=True
        )
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: Optional[np.ndarray] = None) -> None:
        """Update model with new data."""
        # Update scaler
        if not self.scaler_fitted:
            self.scaler.partial_fit(X)
            self.scaler_fitted = True
        else:
            self.scaler.partial_fit(X)
            
        # Transform features
        X_scaled = self.scaler.transform(X)
        
        # Update model
        self.model.partial_fit(X_scaled, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_initialized:
            raise ValueError("Model not fitted yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """PA doesn't support probability predictions."""
        raise NotImplementedError("Passive-Aggressive doesn't support probability predictions")


class HoeffdingTree(BaseOnlineLearner):
    """Hoeffding Tree (VFDT) for streaming classification."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.grace_period = config.get('grace_period', 200)
        self.split_confidence = config.get('split_confidence', 1e-6)
        self.tie_threshold = config.get('tie_threshold', 0.05)
        self.min_samples_split = config.get('min_samples_split', 5)
        
        self.root = None
        self.n_features = None
        self.feature_names = None
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: Optional[np.ndarray] = None) -> None:
        """Update tree with new data."""
        if self.n_features is None:
            self.n_features = X.shape[1]
            self.root = _HoeffdingNode(
                n_features=self.n_features,
                grace_period=self.grace_period,
                split_confidence=self.split_confidence,
                tie_threshold=self.tie_threshold
            )
            
        # Process each sample
        for i in range(len(X)):
            weight = sample_weight[i] if sample_weight is not None else 1.0
            self.root.update(X[i], y[i], weight)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.root is None:
            raise ValueError("Model not fitted yet")
            
        predictions = []
        for x in X:
            leaf = self._find_leaf(x)
            predictions.append(leaf.predict())
            
        return np.array(predictions)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.root is None:
            raise ValueError("Model not fitted yet")
            
        probas = []
        for x in X:
            leaf = self._find_leaf(x)
            probas.append(leaf.predict_proba())
            
        return np.array(probas)
        
    def _find_leaf(self, x: np.ndarray) -> '_HoeffdingNode':
        """Find leaf node for sample."""
        node = self.root
        while not node.is_leaf:
            if x[node.split_feature] <= node.split_value:
                node = node.left_child
            else:
                node = node.right_child
        return node


class _HoeffdingNode:
    """Node in Hoeffding Tree."""
    
    def __init__(self, n_features: int, grace_period: int, 
                 split_confidence: float, tie_threshold: float):
        self.n_features = n_features
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        
        self.is_leaf = True
        self.n_samples = 0
        self.class_counts = {}
        self.sufficient_stats = {}
        
        self.split_feature = None
        self.split_value = None
        self.left_child = None
        self.right_child = None
        
    def update(self, x: np.ndarray, y: Any, weight: float = 1.0) -> None:
        """Update node with new sample."""
        self.n_samples += weight
        
        # Update class counts
        if y not in self.class_counts:
            self.class_counts[y] = 0
        self.class_counts[y] += weight
        
        # Update sufficient statistics
        for i in range(self.n_features):
            if i not in self.sufficient_stats:
                self.sufficient_stats[i] = {
                    'sum': 0, 'sum_sq': 0, 
                    'class_sums': {}
                }
                
            self.sufficient_stats[i]['sum'] += x[i] * weight
            self.sufficient_stats[i]['sum_sq'] += x[i]**2 * weight
            
            if y not in self.sufficient_stats[i]['class_sums']:
                self.sufficient_stats[i]['class_sums'][y] = 0
            self.sufficient_stats[i]['class_sums'][y] += x[i] * weight
            
        # Check if should split
        if self.n_samples >= self.grace_period and self.n_samples % self.grace_period == 0:
            self._attempt_split()
            
    def _attempt_split(self) -> None:
        """Attempt to split node."""
        if len(self.class_counts) == 1:
            return  # Pure node
            
        # Calculate information gain for each feature
        gains = []
        for feature in range(self.n_features):
            gain = self._calculate_info_gain(feature)
            gains.append((gain, feature))
            
        # Sort by gain
        gains.sort(reverse=True)
        
        if len(gains) < 2:
            return
            
        # Check Hoeffding bound
        best_gain = gains[0][0]
        second_best_gain = gains[1][0]
        
        hoeffding_bound = self._calculate_hoeffding_bound()
        
        if best_gain - second_best_gain > hoeffding_bound or hoeffding_bound < self.tie_threshold:
            # Split on best feature
            self._split(gains[0][1])
            
    def _calculate_info_gain(self, feature: int) -> float:
        """Calculate information gain for feature."""
        # Simplified calculation - would need proper implementation
        if feature not in self.sufficient_stats:
            return 0.0
            
        # Calculate split point (mean)
        mean = self.sufficient_stats[feature]['sum'] / self.n_samples
        
        # Calculate entropy reduction
        # This is simplified - proper implementation would consider all split points
        return np.random.rand() * 0.1  # Placeholder
        
    def _calculate_hoeffding_bound(self) -> float:
        """Calculate Hoeffding bound."""
        R = np.log2(len(self.class_counts))  # Range of information gain
        delta = self.split_confidence
        n = self.n_samples
        
        return np.sqrt(R**2 * np.log(1/delta) / (2 * n))
        
    def _split(self, feature: int) -> None:
        """Split node on feature."""
        self.is_leaf = False
        self.split_feature = feature
        self.split_value = self.sufficient_stats[feature]['sum'] / self.n_samples
        
        # Create children
        self.left_child = _HoeffdingNode(
            self.n_features, self.grace_period, 
            self.split_confidence, self.tie_threshold
        )
        self.right_child = _HoeffdingNode(
            self.n_features, self.grace_period, 
            self.split_confidence, self.tie_threshold
        )
        
    def predict(self) -> Any:
        """Predict most common class."""
        if not self.class_counts:
            return 0
            
        return max(self.class_counts, key=self.class_counts.get)
        
    def predict_proba(self) -> np.ndarray:
        """Predict class probabilities."""
        if not self.class_counts:
            return np.array([0.5, 0.5])  # Default for binary
            
        total = sum(self.class_counts.values())
        classes = sorted(self.class_counts.keys())
        
        proba = []
        for c in classes:
            proba.append(self.class_counts.get(c, 0) / total)
            
        return np.array(proba)


class AdaptiveRandomForest(BaseOnlineLearner):
    """Adaptive Random Forest for streaming data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_estimators = config.get('n_estimators', 10)
        self.max_features = config.get('max_features', 'sqrt')
        self.lambda_param = config.get('lambda_param', 6.0)
        self.drift_detection = config.get('drift_detection', True)
        
        self.trees = []
        self.tree_weights = []
        self.background_trees = []
        self.tree_drift_detectors = []
        
        # Initialize trees
        for i in range(self.n_estimators):
            tree_config = config.copy()
            tree = HoeffdingTree(tree_config)
            self.trees.append(tree)
            self.tree_weights.append(1.0)
            
            if self.drift_detection:
                from alpha_pulse.ml.online.concept_drift_detector import ConceptDriftDetector
                detector_config = {'method': 'adwin'}
                self.tree_drift_detectors.append(ConceptDriftDetector(detector_config))
                
    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: Optional[np.ndarray] = None) -> None:
        """Update forest with new data."""
        n_samples = len(X)
        
        for i in range(n_samples):
            # Poisson sampling for each tree
            for j, tree in enumerate(self.trees):
                k = np.random.poisson(self.lambda_param)
                
                if k > 0:
                    # Create bootstrap sample
                    weight = (sample_weight[i] if sample_weight is not None else 1.0) * k
                    
                    # Random feature subset
                    if self.max_features == 'sqrt':
                        n_features = int(np.sqrt(X.shape[1]))
                    elif self.max_features == 'log2':
                        n_features = int(np.log2(X.shape[1]))
                    else:
                        n_features = X.shape[1]
                        
                    feature_subset = np.random.choice(
                        X.shape[1], n_features, replace=False
                    )
                    
                    # Update tree
                    X_subset = X[i:i+1, feature_subset]
                    tree.partial_fit(X_subset, y[i:i+1], sample_weight=np.array([weight]))
                    
                    # Check drift
                    if self.drift_detection and tree._is_initialized:
                        prediction = tree.predict(X_subset)[0]
                        error = 1.0 if prediction != y[i] else 0.0
                        self.tree_drift_detectors[j].add_element(error)
                        
                        if self.tree_drift_detectors[j].detected_change():
                            # Reset tree
                            logger.info(f"Drift detected in tree {j}, resetting")
                            self.trees[j] = HoeffdingTree(self.config)
                            self.tree_drift_detectors[j].reset()
                            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using weighted voting."""
        predictions = []
        
        for x in X:
            tree_predictions = []
            for tree, weight in zip(self.trees, self.tree_weights):
                if tree._is_initialized:
                    pred = tree.predict(x.reshape(1, -1))[0]
                    tree_predictions.append((pred, weight))
                    
            if tree_predictions:
                # Weighted majority vote
                vote_counts = {}
                for pred, weight in tree_predictions:
                    if pred not in vote_counts:
                        vote_counts[pred] = 0
                    vote_counts[pred] += weight
                    
                prediction = max(vote_counts, key=vote_counts.get)
            else:
                prediction = 0
                
            predictions.append(prediction)
            
        return np.array(predictions)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using weighted averaging."""
        probas = []
        
        for x in X:
            tree_probas = []
            weights = []
            
            for tree, weight in zip(self.trees, self.tree_weights):
                if tree._is_initialized:
                    proba = tree.predict_proba(x.reshape(1, -1))[0]
                    tree_probas.append(proba)
                    weights.append(weight)
                    
            if tree_probas:
                # Weighted average
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                avg_proba = np.zeros_like(tree_probas[0])
                for proba, weight in zip(tree_probas, weights):
                    avg_proba += proba * weight
                    
                probas.append(avg_proba)
            else:
                probas.append(np.array([0.5, 0.5]))
                
        return np.array(probas)


class OnlineGradientBoosting(BaseOnlineLearner):
    """Online gradient boosting for streaming data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_estimators = config.get('n_estimators', 10)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.subsample = config.get('subsample', 0.8)
        self.loss = config.get('loss', 'squared_error')
        
        self.estimators = []
        self.estimator_weights = []
        self.residuals_buffer = deque(maxlen=1000)
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: Optional[np.ndarray] = None) -> None:
        """Update boosting ensemble."""
        # Get current predictions
        if self.estimators:
            current_pred = self.predict(X)
            residuals = y - current_pred
        else:
            residuals = y - np.mean(y)
            
        # Store residuals for next estimator
        for i in range(len(X)):
            self.residuals_buffer.append((X[i], residuals[i]))
            
        # Train new estimator if enough data
        if len(self.residuals_buffer) >= 100 and len(self.estimators) < self.n_estimators:
            # Create new weak learner
            estimator = IncrementalSGD({'learning_rate': 0.01}, task_type='regression')
            
            # Train on residuals
            X_buffer = np.array([x for x, _ in self.residuals_buffer])
            y_buffer = np.array([r for _, r in self.residuals_buffer])
            
            # Subsample
            if self.subsample < 1.0:
                n_samples = int(len(X_buffer) * self.subsample)
                indices = np.random.choice(len(X_buffer), n_samples, replace=False)
                X_buffer = X_buffer[indices]
                y_buffer = y_buffer[indices]
                
            estimator.partial_fit(X_buffer, y_buffer)
            
            self.estimators.append(estimator)
            self.estimator_weights.append(self.learning_rate)
            
            logger.info(f"Added estimator {len(self.estimators)} to gradient boosting")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using boosting ensemble."""
        if not self.estimators:
            return np.zeros(len(X))
            
        predictions = np.zeros(len(X))
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions += weight * estimator.predict(X)
            
        return predictions
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Gradient boosting for regression doesn't support probabilities."""
        raise NotImplementedError("Probability prediction not supported for regression")