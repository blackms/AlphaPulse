"""Boosting algorithms for sequential ensemble learning."""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
from collections import deque
import joblib
import os

from alpha_pulse.ml.ensemble.ensemble_manager import BaseEnsemble, AgentSignal, EnsembleSignal

logger = logging.getLogger(__name__)


class AdaptiveBoosting(BaseEnsemble):
    """Adaptive boosting ensemble for trading signals."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_estimators = config.get('n_estimators', 50)
        self.learning_rate = config.get('learning_rate', 1.0)
        self.base_estimator = config.get('base_estimator', 'decision_tree')
        self.max_depth = config.get('max_depth', 3)
        self.online_learning = config.get('online_learning', True)
        
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []
        self.feature_importances = None
        
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Train adaptive boosting ensemble."""
        # Create feature matrix
        X, agent_ids = self._create_feature_matrix(signals)
        y = outcomes[:len(X)]
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No data to fit adaptive boosting")
            return
            
        # Initialize sample weights
        sample_weights = np.ones(len(X)) / len(X)
        
        for i in range(self.n_estimators):
            # Train base estimator
            estimator = self._create_base_estimator()
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Make predictions
            predictions = estimator.predict(X)
            
            # Calculate error
            error = self._calculate_weighted_error(y, predictions, sample_weights)
            
            if error >= 0.5:
                # Stop if error is too high
                logger.warning(f"Stopping at iteration {i}, error: {error}")
                break
                
            # Calculate estimator weight
            alpha = self.learning_rate * np.log((1 - error) / (error + 1e-10))
            
            # Update sample weights
            sample_weights = self._update_sample_weights(
                sample_weights, y, predictions, alpha
            )
            
            # Store estimator
            self.estimators.append(estimator)
            self.estimator_weights.append(alpha)
            self.estimator_errors.append(error)
            
            logger.info(f"Iteration {i+1}: error={error:.4f}, alpha={alpha:.4f}")
            
        # Calculate feature importances
        self._calculate_feature_importances(agent_ids)
        
        self.is_fitted = True
        logger.info(f"Fitted adaptive boosting with {len(self.estimators)} estimators")
        
    def _create_base_estimator(self):
        """Create base estimator for boosting."""
        if self.base_estimator == 'decision_tree':
            return DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown base estimator: {self.base_estimator}")
            
    def _calculate_weighted_error(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 weights: np.ndarray) -> float:
        """Calculate weighted prediction error."""
        # Convert to classification for error calculation
        y_true_sign = np.sign(y_true)
        y_pred_sign = np.sign(y_pred)
        
        incorrect = y_true_sign != y_pred_sign
        error = np.sum(weights[incorrect]) / np.sum(weights)
        
        return max(min(error, 0.999), 0.001)  # Clip to avoid numerical issues
        
    def _update_sample_weights(self, weights: np.ndarray, y_true: np.ndarray,
                             y_pred: np.ndarray, alpha: float) -> np.ndarray:
        """Update sample weights based on predictions."""
        # Increase weights for misclassified samples
        y_true_sign = np.sign(y_true)
        y_pred_sign = np.sign(y_pred)
        
        incorrect = y_true_sign != y_pred_sign
        
        # Update weights
        weights = weights * np.exp(alpha * incorrect)
        
        # Normalize
        weights = weights / np.sum(weights)
        
        return weights
        
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate prediction using adaptive boosting."""
        if not self.is_fitted or not self.estimators:
            return self._create_neutral_signal(signals, "model_not_fitted")
            
        # Create feature matrix
        X, agent_ids = self._create_feature_matrix(signals)
        
        if len(X) == 0:
            return self._create_neutral_signal(signals, "no_features")
            
        # Get predictions from all estimators
        predictions = np.zeros(X.shape[0])
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            pred = estimator.predict(X)
            predictions += weight * pred
            
        # Normalize by total weight
        total_weight = sum(self.estimator_weights)
        if total_weight > 0:
            predictions = predictions / total_weight
            
        # Get final prediction
        ensemble_signal = float(predictions[0]) if len(predictions) > 0 else 0.0
        
        # Calculate confidence based on estimator agreement
        estimator_predictions = []
        for estimator in self.estimators:
            estimator_predictions.append(estimator.predict(X)[0])
            
        prediction_std = np.std(estimator_predictions)
        confidence = 1.0 / (1.0 + prediction_std)
        
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=float(np.clip(ensemble_signal, -1, 1)),
            confidence=float(confidence),
            contributing_agents=[s.agent_id for s in signals],
            weights=self.agent_weights,
            metadata={
                'method': 'adaptive_boosting',
                'n_estimators': len(self.estimators),
                'avg_error': np.mean(self.estimator_errors) if self.estimator_errors else 0,
                'prediction_std': float(prediction_std)
            }
        )
        
    def _calculate_feature_importances(self, agent_ids: List[str]) -> None:
        """Calculate feature importances from all estimators."""
        if not self.estimators:
            return
            
        # Aggregate importances from all trees
        importances = np.zeros(len(agent_ids) * 3)  # 3 features per agent
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            if hasattr(estimator, 'feature_importances_'):
                importances += estimator.feature_importances_ * weight
                
        # Normalize
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
            
        # Map to agent weights
        for i, agent_id in enumerate(agent_ids):
            # Average importance across features for this agent
            start_idx = i * 3
            end_idx = start_idx + 3
            self.agent_weights[agent_id] = float(np.mean(importances[start_idx:end_idx]))
            
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update boosting weights based on recent performance."""
        if self.online_learning and self.estimators:
            # Adjust estimator weights based on recent performance
            # This is a simplified online update
            for i, (estimator, current_weight) in enumerate(
                zip(self.estimators, self.estimator_weights)
            ):
                # Decay old weights
                self.estimator_weights[i] = current_weight * 0.95
                
    def _create_feature_matrix(self, signals: List[AgentSignal]) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix from signals."""
        if not signals:
            return np.array([]), []
            
        # Group by timestamp
        signal_groups = {}
        for signal in signals:
            timestamp = signal.timestamp.replace(microsecond=0)
            if timestamp not in signal_groups:
                signal_groups[timestamp] = {}
            signal_groups[timestamp][signal.agent_id] = signal
            
        agent_ids = sorted(set(s.agent_id for s in signals))
        
        features = []
        for timestamp, agent_signals in signal_groups.items():
            row = []
            for agent_id in agent_ids:
                if agent_id in agent_signals:
                    signal = agent_signals[agent_id]
                    row.extend([
                        signal.signal,
                        signal.confidence,
                        signal.signal * signal.confidence
                    ])
                else:
                    row.extend([0.0, 0.0, 0.0])
            features.append(row)
            
        return np.array(features), agent_ids
        
    def _create_neutral_signal(self, signals: List[AgentSignal], reason: str) -> EnsembleSignal:
        """Create neutral signal."""
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=0.0,
            confidence=0.0,
            contributing_agents=[s.agent_id for s in signals],
            weights={},
            metadata={'method': 'adaptive_boosting', 'reason': reason}
        )


class GradientBoosting(BaseEnsemble):
    """Gradient boosting ensemble for signal refinement."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_estimators = config.get('n_estimators', 100)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.max_depth = config.get('max_depth', 5)
        self.subsample = config.get('subsample', 0.8)
        self.loss_function = config.get('loss', 'squared_error')
        
        self.model = None
        
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Train gradient boosting model."""
        X, agent_ids = self._create_feature_matrix(signals)
        y = outcomes[:len(X)]
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No data to fit gradient boosting")
            return
            
        # Create gradient boosting model
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            loss=self.loss_function,
            random_state=42
        )
        
        # Fit model
        self.model.fit(X, y)
        
        # Extract feature importances
        if hasattr(self.model, 'feature_importances_'):
            self._update_agent_weights(agent_ids, self.model.feature_importances_)
            
        self.is_fitted = True
        logger.info(f"Fitted gradient boosting with {self.n_estimators} estimators")
        
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate prediction using gradient boosting."""
        if not self.is_fitted or self.model is None:
            return self._create_neutral_signal(signals, "model_not_fitted")
            
        X, _ = self._create_feature_matrix(signals)
        
        if len(X) == 0:
            return self._create_neutral_signal(signals, "no_features")
            
        # Make prediction
        prediction = self.model.predict(X)
        ensemble_signal = float(prediction[0]) if len(prediction) > 0 else 0.0
        
        # Calculate confidence using prediction intervals
        if hasattr(self.model, 'estimators_'):
            # Get predictions from individual trees
            tree_predictions = []
            for estimator in self.model.estimators_:
                tree_pred = estimator[0].predict(X)
                tree_predictions.append(tree_pred[0])
                
            prediction_std = np.std(tree_predictions)
            confidence = 1.0 / (1.0 + prediction_std)
        else:
            confidence = 0.7
            
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=float(np.clip(ensemble_signal, -1, 1)),
            confidence=float(confidence),
            contributing_agents=[s.agent_id for s in signals],
            weights=self.agent_weights,
            metadata={
                'method': 'gradient_boosting',
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate
            }
        )
        
    def _update_agent_weights(self, agent_ids: List[str], importances: np.ndarray) -> None:
        """Update agent weights from feature importances."""
        features_per_agent = 3
        
        for i, agent_id in enumerate(agent_ids):
            start_idx = i * features_per_agent
            end_idx = start_idx + features_per_agent
            
            if end_idx <= len(importances):
                agent_importance = np.mean(importances[start_idx:end_idx])
                self.agent_weights[agent_id] = float(agent_importance)
                
    def _create_feature_matrix(self, signals: List[AgentSignal]) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix from signals."""
        if not signals:
            return np.array([]), []
            
        signal_groups = {}
        for signal in signals:
            timestamp = signal.timestamp.replace(microsecond=0)
            if timestamp not in signal_groups:
                signal_groups[timestamp] = {}
            signal_groups[timestamp][signal.agent_id] = signal
            
        agent_ids = sorted(set(s.agent_id for s in signals))
        
        features = []
        for timestamp, agent_signals in signal_groups.items():
            row = []
            for agent_id in agent_ids:
                if agent_id in agent_signals:
                    signal = agent_signals[agent_id]
                    row.extend([
                        signal.signal,
                        signal.confidence,
                        signal.signal * signal.confidence
                    ])
                else:
                    row.extend([0.0, 0.0, 0.0])
            features.append(row)
            
        return np.array(features), agent_ids
        
    def _create_neutral_signal(self, signals: List[AgentSignal], reason: str) -> EnsembleSignal:
        """Create neutral signal."""
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=0.0,
            confidence=0.0,
            contributing_agents=[s.agent_id for s in signals],
            weights={},
            metadata={'method': 'gradient_boosting', 'reason': reason}
        )
        
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update weights based on performance."""
        # Gradient boosting typically doesn't update weights online
        # But we can adjust agent weights based on recent performance
        for agent_id, metric in performance_metrics.items():
            if agent_id in self.agent_weights:
                # Blend with existing weight
                old_weight = self.agent_weights[agent_id]
                new_weight = metric
                self.agent_weights[agent_id] = 0.8 * old_weight + 0.2 * new_weight


class XGBoostEnsemble(BaseEnsemble):
    """XGBoost ensemble for high-performance prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = {
            'n_estimators': config.get('n_estimators', 100),
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.1),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
            'gamma': config.get('gamma', 0),
            'reg_alpha': config.get('reg_alpha', 0),
            'reg_lambda': config.get('reg_lambda', 1),
            'objective': config.get('objective', 'reg:squarederror'),
            'random_state': 42
        }
        
        self.model = None
        self.feature_names = None
        
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Train XGBoost model."""
        X, agent_ids = self._create_feature_matrix(signals)
        y = outcomes[:len(X)]
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No data to fit XGBoost")
            return
            
        # Create feature names
        self.feature_names = []
        for agent_id in agent_ids:
            self.feature_names.extend([
                f"{agent_id}_signal",
                f"{agent_id}_confidence",
                f"{agent_id}_weighted"
            ])
            
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params['n_estimators']
        )
        
        # Get feature importances
        importance_dict = self.model.get_score(importance_type='gain')
        self._update_agent_weights_from_importance(agent_ids, importance_dict)
        
        self.is_fitted = True
        logger.info(f"Fitted XGBoost with {self.params['n_estimators']} rounds")
        
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate prediction using XGBoost."""
        if not self.is_fitted or self.model is None:
            return self._create_neutral_signal(signals, "model_not_fitted")
            
        X, _ = self._create_feature_matrix(signals)
        
        if len(X) == 0:
            return self._create_neutral_signal(signals, "no_features")
            
        # Create DMatrix
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        
        # Make prediction
        prediction = self.model.predict(dtest)
        ensemble_signal = float(prediction[0]) if len(prediction) > 0 else 0.0
        
        # Calculate confidence using prediction leaf values
        leaf_values = self.model.predict(dtest, pred_leaf=True)
        
        # Estimate uncertainty from leaf diversity
        if len(leaf_values.shape) > 1:
            leaf_diversity = len(np.unique(leaf_values[0])) / leaf_values.shape[1]
            confidence = 1.0 - leaf_diversity * 0.5
        else:
            confidence = 0.8
            
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=float(np.clip(ensemble_signal, -1, 1)),
            confidence=float(confidence),
            contributing_agents=[s.agent_id for s in signals],
            weights=self.agent_weights,
            metadata={
                'method': 'xgboost',
                'n_estimators': self.params['n_estimators'],
                'max_depth': self.params['max_depth']
            }
        )
        
    def _update_agent_weights_from_importance(self, agent_ids: List[str], 
                                            importance_dict: Dict[str, float]) -> None:
        """Update agent weights from XGBoost feature importances."""
        for agent_id in agent_ids:
            # Sum importance across all features for this agent
            agent_importance = 0.0
            for feature_type in ['signal', 'confidence', 'weighted']:
                feature_name = f"{agent_id}_{feature_type}"
                if feature_name in importance_dict:
                    agent_importance += importance_dict[feature_name]
                    
            # Normalize and store
            self.agent_weights[agent_id] = float(agent_importance)
            
        # Normalize weights
        total_weight = sum(self.agent_weights.values())
        if total_weight > 0:
            for agent_id in self.agent_weights:
                self.agent_weights[agent_id] /= total_weight
                
    def _create_feature_matrix(self, signals: List[AgentSignal]) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix from signals."""
        if not signals:
            return np.array([]), []
            
        signal_groups = {}
        for signal in signals:
            timestamp = signal.timestamp.replace(microsecond=0)
            if timestamp not in signal_groups:
                signal_groups[timestamp] = {}
            signal_groups[timestamp][signal.agent_id] = signal
            
        agent_ids = sorted(set(s.agent_id for s in signals))
        
        features = []
        for timestamp, agent_signals in signal_groups.items():
            row = []
            for agent_id in agent_ids:
                if agent_id in agent_signals:
                    signal = agent_signals[agent_id]
                    row.extend([
                        signal.signal,
                        signal.confidence,
                        signal.signal * signal.confidence
                    ])
                else:
                    row.extend([0.0, 0.0, 0.0])
            features.append(row)
            
        return np.array(features), agent_ids
        
    def _create_neutral_signal(self, signals: List[AgentSignal], reason: str) -> EnsembleSignal:
        """Create neutral signal."""
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=0.0,
            confidence=0.0,
            contributing_agents=[s.agent_id for s in signals],
            weights={},
            metadata={'method': 'xgboost', 'reason': reason}
        )
        
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update weights based on performance."""
        # XGBoost doesn't typically update online, but we can adjust agent weights
        for agent_id, metric in performance_metrics.items():
            if agent_id in self.agent_weights:
                self.agent_weights[agent_id] = (
                    0.9 * self.agent_weights[agent_id] + 0.1 * metric
                )


class OnlineBoosting(BaseEnsemble):
    """Online boosting for streaming data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_size = config.get('window_size', 1000)
        self.update_frequency = config.get('update_frequency', 100)
        self.min_samples = config.get('min_samples', 50)
        self.base_model_type = config.get('base_model', 'xgboost')
        
        self.signal_buffer = deque(maxlen=self.window_size)
        self.outcome_buffer = deque(maxlen=self.window_size)
        self.update_counter = 0
        self.base_model = None
        
    def add_sample(self, signals: List[AgentSignal], outcome: float) -> None:
        """Add new sample to buffer."""
        self.signal_buffer.append(signals)
        self.outcome_buffer.append(outcome)
        self.update_counter += 1
        
        # Check if we should update model
        if (self.update_counter % self.update_frequency == 0 and 
            len(self.signal_buffer) >= self.min_samples):
            self._update_model()
            
    def _update_model(self) -> None:
        """Update model with buffered data."""
        # Convert buffer to training data
        all_signals = []
        for signal_list in self.signal_buffer:
            all_signals.extend(signal_list)
            
        outcomes = np.array(self.outcome_buffer)
        
        # Create appropriate base model
        if self.base_model_type == 'xgboost':
            self.base_model = XGBoostEnsemble(self.config)
        elif self.base_model_type == 'gradient':
            self.base_model = GradientBoosting(self.config)
        else:
            self.base_model = AdaptiveBoosting(self.config)
            
        # Fit model
        self.base_model.fit(all_signals, outcomes)
        
        # Update weights
        self.agent_weights = self.base_model.agent_weights
        self.is_fitted = True
        
        logger.info(f"Updated online boosting model with {len(self.signal_buffer)} samples")
        
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Initial fit with historical data."""
        # Add all samples to buffer
        for i in range(len(outcomes)):
            # Get signals at time i
            time_signals = [s for s in signals if s.timestamp.second == i]
            if time_signals:
                self.add_sample(time_signals, outcomes[i])
                
        # Force initial model update
        if len(self.signal_buffer) >= self.min_samples:
            self._update_model()
            
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate prediction using online boosting."""
        if not self.is_fitted or self.base_model is None:
            return self._create_neutral_signal(signals, "model_not_fitted")
            
        # Use base model for prediction
        return self.base_model.predict(signals)
        
    def _create_neutral_signal(self, signals: List[AgentSignal], reason: str) -> EnsembleSignal:
        """Create neutral signal."""
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=0.0,
            confidence=0.0,
            contributing_agents=[s.agent_id for s in signals],
            weights={},
            metadata={'method': 'online_boosting', 'reason': reason}
        )
        
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update weights based on performance."""
        if self.base_model:
            self.base_model.update_weights(performance_metrics)
            self.agent_weights = self.base_model.agent_weights


class LightGBMEnsemble(BaseEnsemble):
    """LightGBM ensemble for efficient training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = {
            'objective': config.get('objective', 'regression'),
            'metric': config.get('metric', 'rmse'),
            'boosting_type': config.get('boosting_type', 'gbdt'),
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.1),
            'feature_fraction': config.get('feature_fraction', 0.8),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'verbose': -1,
            'random_state': 42
        }
        
        self.n_estimators = config.get('n_estimators', 100)
        self.model = None
        
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Train LightGBM model."""
        X, agent_ids = self._create_feature_matrix(signals)
        y = outcomes[:len(X)]
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No data to fit LightGBM")
            return
            
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators
        )
        
        # Get feature importances
        importance = self.model.feature_importance(importance_type='gain')
        self._update_agent_weights(agent_ids, importance)
        
        self.is_fitted = True
        logger.info(f"Fitted LightGBM with {self.n_estimators} iterations")
        
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate prediction using LightGBM."""
        if not self.is_fitted or self.model is None:
            return self._create_neutral_signal(signals, "model_not_fitted")
            
        X, _ = self._create_feature_matrix(signals)
        
        if len(X) == 0:
            return self._create_neutral_signal(signals, "no_features")
            
        # Make prediction
        prediction = self.model.predict(X, num_iteration=self.model.best_iteration)
        ensemble_signal = float(prediction[0]) if len(prediction) > 0 else 0.0
        
        # Estimate confidence
        # For LightGBM, we can use prediction variance from trees
        confidence = 0.8  # Default confidence
        
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=float(np.clip(ensemble_signal, -1, 1)),
            confidence=float(confidence),
            contributing_agents=[s.agent_id for s in signals],
            weights=self.agent_weights,
            metadata={
                'method': 'lightgbm',
                'n_estimators': self.n_estimators,
                'best_iteration': self.model.best_iteration if self.model else 0
            }
        )
        
    def _update_agent_weights(self, agent_ids: List[str], importances: np.ndarray) -> None:
        """Update agent weights from feature importances."""
        features_per_agent = 3
        
        for i, agent_id in enumerate(agent_ids):
            start_idx = i * features_per_agent
            end_idx = start_idx + features_per_agent
            
            if end_idx <= len(importances):
                agent_importance = np.sum(importances[start_idx:end_idx])
                self.agent_weights[agent_id] = float(agent_importance)
                
        # Normalize weights
        total_weight = sum(self.agent_weights.values())
        if total_weight > 0:
            for agent_id in self.agent_weights:
                self.agent_weights[agent_id] /= total_weight
                
    def _create_feature_matrix(self, signals: List[AgentSignal]) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix from signals."""
        if not signals:
            return np.array([]), []
            
        signal_groups = {}
        for signal in signals:
            timestamp = signal.timestamp.replace(microsecond=0)
            if timestamp not in signal_groups:
                signal_groups[timestamp] = {}
            signal_groups[timestamp][signal.agent_id] = signal
            
        agent_ids = sorted(set(s.agent_id for s in signals))
        
        features = []
        for timestamp, agent_signals in signal_groups.items():
            row = []
            for agent_id in agent_ids:
                if agent_id in agent_signals:
                    signal = agent_signals[agent_id]
                    row.extend([
                        signal.signal,
                        signal.confidence,
                        signal.signal * signal.confidence
                    ])
                else:
                    row.extend([0.0, 0.0, 0.0])
            features.append(row)
            
        return np.array(features), agent_ids
        
    def _create_neutral_signal(self, signals: List[AgentSignal], reason: str) -> EnsembleSignal:
        """Create neutral signal."""
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=0.0,
            confidence=0.0,
            contributing_agents=[s.agent_id for s in signals],
            weights={},
            metadata={'method': 'lightgbm', 'reason': reason}
        )
        
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update weights based on performance."""
        for agent_id, metric in performance_metrics.items():
            if agent_id in self.agent_weights:
                self.agent_weights[agent_id] = (
                    0.85 * self.agent_weights[agent_id] + 0.15 * metric
                )