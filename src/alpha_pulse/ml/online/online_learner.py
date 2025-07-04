"""Base online learning framework for real-time model adaptation."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
import json

from alpha_pulse.ml.online.memory_manager import MemoryManager
from alpha_pulse.ml.online.concept_drift_detector import ConceptDriftDetector

logger = logging.getLogger(__name__)


@dataclass
class OnlineDataPoint:
    """Single data point for online learning."""
    timestamp: datetime
    features: np.ndarray
    label: Optional[float] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningState:
    """Current state of online learning."""
    n_samples_seen: int = 0
    n_updates: int = 0
    last_update: Optional[datetime] = None
    performance_window: List[float] = field(default_factory=list)
    drift_detected: bool = False
    drift_history: List[datetime] = field(default_factory=list)
    current_accuracy: float = 0.0
    model_version: int = 1


class BaseOnlineLearner(ABC):
    """Base class for online learning algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.01)
        self.batch_size = config.get('batch_size', 1)
        self.update_frequency = config.get('update_frequency', 1)
        self.performance_window_size = config.get('performance_window_size', 100)
        
        self.state = LearningState()
        self.memory_manager = MemoryManager(config.get('memory', {}))
        self.drift_detector = ConceptDriftDetector(config.get('drift_detection', {}))
        
        self._buffer = deque(maxlen=self.batch_size)
        self._is_initialized = False
        
    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: Optional[np.ndarray] = None) -> None:
        """Update model with new data."""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data."""
        pass
        
    def learn_one(self, data_point: OnlineDataPoint) -> Optional[float]:
        """Learn from a single data point."""
        # Make prediction before learning (prequential evaluation)
        prediction = None
        if self._is_initialized and data_point.label is not None:
            prediction = self.predict(data_point.features.reshape(1, -1))[0]
            self._update_performance(prediction, data_point.label)
            
        # Add to buffer
        self._buffer.append(data_point)
        
        # Update model if buffer is full or update frequency reached
        if len(self._buffer) >= self.batch_size or \
           self.state.n_samples_seen % self.update_frequency == 0:
            self._process_buffer()
            
        self.state.n_samples_seen += 1
        
        # Check for concept drift
        if self.state.n_samples_seen % self.drift_detector.check_frequency == 0:
            self._check_drift()
            
        return prediction
        
    def _process_buffer(self) -> None:
        """Process buffered data points."""
        if not self._buffer:
            return
            
        # Extract features and labels
        X = np.vstack([dp.features for dp in self._buffer])
        y = np.array([dp.label for dp in self._buffer if dp.label is not None])
        weights = np.array([dp.weight for dp in self._buffer if dp.label is not None])
        
        if len(y) > 0:
            # Update model
            self.partial_fit(X[:len(y)], y, sample_weight=weights)
            self.state.n_updates += 1
            self.state.last_update = datetime.now()
            self._is_initialized = True
            
        # Clear buffer
        self._buffer.clear()
        
    def _update_performance(self, prediction: float, actual: float) -> None:
        """Update performance metrics."""
        # Calculate error
        error = abs(prediction - actual)
        
        # Update performance window
        self.state.performance_window.append(error)
        if len(self.state.performance_window) > self.performance_window_size:
            self.state.performance_window.pop(0)
            
        # Update current accuracy
        if len(self.state.performance_window) > 0:
            self.state.current_accuracy = 1.0 - np.mean(self.state.performance_window)
            
        # Update drift detector
        self.drift_detector.add_element(error)
        
    def _check_drift(self) -> None:
        """Check for concept drift."""
        if self.drift_detector.detected_change():
            self.state.drift_detected = True
            self.state.drift_history.append(datetime.now())
            logger.warning(f"Concept drift detected at sample {self.state.n_samples_seen}")
            
            # Handle drift
            self._handle_drift()
            
    def _handle_drift(self) -> None:
        """Handle detected concept drift."""
        strategy = self.config.get('drift_strategy', 'adaptive')
        
        if strategy == 'reset':
            # Reset model completely
            self.reset()
        elif strategy == 'adaptive':
            # Increase learning rate temporarily
            self.learning_rate *= 2.0
            logger.info(f"Increased learning rate to {self.learning_rate}")
        elif strategy == 'ensemble':
            # Add new model to ensemble (handled by ensemble manager)
            pass
            
    def reset(self) -> None:
        """Reset the learner to initial state."""
        self.state = LearningState()
        self._buffer.clear()
        self._is_initialized = False
        logger.info("Online learner reset")
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about current learning state."""
        return {
            'n_samples_seen': self.state.n_samples_seen,
            'n_updates': self.state.n_updates,
            'last_update': self.state.last_update.isoformat() if self.state.last_update else None,
            'current_accuracy': self.state.current_accuracy,
            'drift_detected': self.state.drift_detected,
            'drift_count': len(self.state.drift_history),
            'model_version': self.state.model_version,
            'memory_usage': self.memory_manager.get_memory_usage()
        }


class OnlineLearnerEnsemble:
    """Ensemble of online learners for robust adaptation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_models = config.get('max_models', 5)
        self.combination_method = config.get('combination_method', 'weighted_average')
        
        self.learners: List[BaseOnlineLearner] = []
        self.learner_weights: List[float] = []
        self.performance_history: Dict[int, List[float]] = {}
        
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
    def add_learner(self, learner: BaseOnlineLearner, weight: float = 1.0) -> None:
        """Add a new learner to the ensemble."""
        if len(self.learners) >= self.max_models:
            # Remove worst performing learner
            self._remove_worst_learner()
            
        self.learners.append(learner)
        self.learner_weights.append(weight)
        self.performance_history[len(self.learners) - 1] = []
        
        logger.info(f"Added learner to ensemble (total: {len(self.learners)})")
        
    def learn_one(self, data_point: OnlineDataPoint) -> Optional[float]:
        """Learn from a single data point across all learners."""
        # Parallel learning
        futures = []
        for i, learner in enumerate(self.learners):
            future = self.executor.submit(learner.learn_one, data_point)
            futures.append((i, future))
            
        # Collect predictions and update performance
        predictions = []
        for i, future in futures:
            try:
                prediction = future.result(timeout=1.0)
                if prediction is not None:
                    predictions.append((i, prediction))
                    
                    # Update performance history
                    if data_point.label is not None:
                        error = abs(prediction - data_point.label)
                        self.performance_history[i].append(error)
                        
            except Exception as e:
                logger.error(f"Error in learner {i}: {e}")
                
        # Combine predictions
        if predictions:
            return self._combine_predictions(predictions)
        return None
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        
        for i, learner in enumerate(self.learners):
            try:
                pred = learner.predict(X)
                predictions.append((i, pred))
            except Exception as e:
                logger.error(f"Prediction error in learner {i}: {e}")
                
        if not predictions:
            raise ValueError("No learners available for prediction")
            
        # Combine predictions
        combined = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sample_preds = [(idx, pred[i]) for idx, pred in predictions]
            combined[i] = self._combine_predictions(sample_preds)
            
        return combined
        
    def _combine_predictions(self, predictions: List[Tuple[int, float]]) -> float:
        """Combine predictions from multiple learners."""
        if self.combination_method == 'weighted_average':
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for idx, pred in predictions:
                weight = self.learner_weights[idx]
                weighted_sum += pred * weight
                weight_sum += weight
                
            return weighted_sum / weight_sum if weight_sum > 0 else 0.0
            
        elif self.combination_method == 'median':
            return np.median([pred for _, pred in predictions])
            
        elif self.combination_method == 'best':
            # Use prediction from best performing learner
            best_idx = self._get_best_learner()
            for idx, pred in predictions:
                if idx == best_idx:
                    return pred
            return predictions[0][1]  # Fallback
            
    def _get_best_learner(self) -> int:
        """Get index of best performing learner."""
        best_idx = 0
        best_performance = float('inf')
        
        for idx, history in self.performance_history.items():
            if len(history) >= 10:  # Minimum samples
                recent_performance = np.mean(history[-50:])
                if recent_performance < best_performance:
                    best_performance = recent_performance
                    best_idx = idx
                    
        return best_idx
        
    def _remove_worst_learner(self) -> None:
        """Remove the worst performing learner."""
        if len(self.learners) <= 1:
            return
            
        worst_idx = 0
        worst_performance = -float('inf')
        
        for idx, history in self.performance_history.items():
            if len(history) >= 10:
                recent_performance = np.mean(history[-50:])
                if recent_performance > worst_performance:
                    worst_performance = recent_performance
                    worst_idx = idx
                    
        # Remove learner
        self.learners.pop(worst_idx)
        self.learner_weights.pop(worst_idx)
        del self.performance_history[worst_idx]
        
        # Reindex performance history
        new_history = {}
        for old_idx, history in self.performance_history.items():
            new_idx = old_idx if old_idx < worst_idx else old_idx - 1
            new_history[new_idx] = history
        self.performance_history = new_history
        
        logger.info(f"Removed worst performing learner (remaining: {len(self.learners)})")
        
    def update_weights(self) -> None:
        """Update learner weights based on recent performance."""
        for idx, history in self.performance_history.items():
            if len(history) >= 10:
                # Calculate weight based on inverse of recent error
                recent_error = np.mean(history[-50:])
                self.learner_weights[idx] = 1.0 / (1.0 + recent_error)
                
        # Normalize weights
        total_weight = sum(self.learner_weights)
        if total_weight > 0:
            self.learner_weights = [w / total_weight for w in self.learner_weights]
            
    def detect_drift(self) -> List[int]:
        """Check which learners have detected drift."""
        drift_indices = []
        
        for i, learner in enumerate(self.learners):
            if learner.state.drift_detected:
                drift_indices.append(i)
                learner.state.drift_detected = False  # Reset flag
                
        return drift_indices
        
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble."""
        learner_infos = []
        for i, learner in enumerate(self.learners):
            info = learner.get_info()
            info['weight'] = self.learner_weights[i]
            info['recent_performance'] = np.mean(self.performance_history[i][-50:]) if self.performance_history[i] else None
            learner_infos.append(info)
            
        return {
            'n_learners': len(self.learners),
            'combination_method': self.combination_method,
            'learners': learner_infos,
            'best_learner_idx': self._get_best_learner()
        }


class AdaptiveLearningController:
    """Controller for managing adaptive learning strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exploration_rate = config.get('exploration_rate', 0.1)
        self.adaptation_interval = config.get('adaptation_interval', 100)
        self.performance_threshold = config.get('performance_threshold', 0.7)
        
        self.strategy_performance: Dict[str, List[float]] = {}
        self.current_strategy = config.get('initial_strategy', 'conservative')
        self.n_adaptations = 0
        
    def should_adapt(self, current_performance: float) -> bool:
        """Determine if adaptation is needed."""
        return current_performance < self.performance_threshold
        
    def select_strategy(self, context: Dict[str, Any]) -> str:
        """Select learning strategy based on context."""
        if np.random.random() < self.exploration_rate:
            # Exploration: try random strategy
            strategies = ['conservative', 'aggressive', 'adaptive', 'ensemble']
            return np.random.choice(strategies)
        else:
            # Exploitation: use best performing strategy
            if not self.strategy_performance:
                return self.current_strategy
                
            best_strategy = max(
                self.strategy_performance.items(),
                key=lambda x: np.mean(x[1][-10:]) if x[1] else 0
            )[0]
            
            return best_strategy
            
    def update_strategy_performance(self, strategy: str, performance: float) -> None:
        """Update performance history for a strategy."""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
            
        self.strategy_performance[strategy].append(performance)
        
        # Keep only recent history
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy].pop(0)
            
    def adapt_learning_rate(self, base_rate: float, 
                           volatility: float, 
                           performance: float) -> float:
        """Adapt learning rate based on market conditions."""
        # Increase rate with volatility
        volatility_factor = 1.0 + volatility
        
        # Decrease rate with good performance
        performance_factor = 2.0 - performance
        
        # Time decay
        time_factor = 1.0 / (1.0 + self.n_adaptations * 0.01)
        
        adapted_rate = base_rate * volatility_factor * performance_factor * time_factor
        
        # Clip to reasonable range
        return np.clip(adapted_rate, base_rate * 0.1, base_rate * 10.0)