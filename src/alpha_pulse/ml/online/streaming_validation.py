"""Streaming validation utilities for online learning."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from collections import deque
import json
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StreamingMetric:
    """Single streaming metric with history."""
    name: str
    value: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    window_size: int = 100
    timestamp: datetime = field(default_factory=datetime.now)
    
    def update(self, value: float) -> None:
        """Update metric with new value."""
        self.value = value
        self.history.append(value)
        self.timestamp = datetime.now()
        
    def get_window_average(self) -> float:
        """Get average over recent window."""
        if not self.history:
            return 0.0
        window = list(self.history)[-self.window_size:]
        return np.mean(window)
        
    def get_trend(self) -> str:
        """Get trend direction."""
        if len(self.history) < 10:
            return "stable"
            
        recent = list(self.history)[-10:]
        older = list(self.history)[-20:-10] if len(self.history) >= 20 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "degrading"
        else:
            return "stable"


class PrequentialEvaluator:
    """Prequential (test-then-train) evaluation for streaming data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get('window_size', 1000)
        self.fade_factor = config.get('fade_factor', 0.999)
        self.task_type = config.get('task_type', 'regression')
        
        self.predictions = deque(maxlen=self.window_size)
        self.actuals = deque(maxlen=self.window_size)
        self.weights = deque(maxlen=self.window_size)
        
        self.n_samples = 0
        self.metrics = {}
        self._initialize_metrics()
        
    def _initialize_metrics(self) -> None:
        """Initialize metrics based on task type."""
        if self.task_type == 'regression':
            self.metrics = {
                'mse': StreamingMetric('mse'),
                'mae': StreamingMetric('mae'),
                'rmse': StreamingMetric('rmse'),
                'r2': StreamingMetric('r2'),
                'mape': StreamingMetric('mape')
            }
        else:  # classification
            self.metrics = {
                'accuracy': StreamingMetric('accuracy'),
                'precision': StreamingMetric('precision'),
                'recall': StreamingMetric('recall'),
                'f1': StreamingMetric('f1')
            }
            
    def add_result(self, prediction: float, actual: float, weight: float = 1.0) -> Dict[str, float]:
        """Add prediction result and update metrics."""
        self.n_samples += 1
        
        # Apply fading to weights
        for i in range(len(self.weights)):
            self.weights[i] *= self.fade_factor
            
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.weights.append(weight)
        
        # Update metrics
        return self._update_metrics()
        
    def _update_metrics(self) -> Dict[str, float]:
        """Update all metrics with current window."""
        if len(self.predictions) < 2:
            return {}
            
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        weights = np.array(self.weights)
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)
            
        results = {}
        
        if self.task_type == 'regression':
            # MSE
            mse = np.average((preds - acts) ** 2, weights=weights)
            self.metrics['mse'].update(mse)
            results['mse'] = mse
            
            # MAE
            mae = np.average(np.abs(preds - acts), weights=weights)
            self.metrics['mae'].update(mae)
            results['mae'] = mae
            
            # RMSE
            rmse = np.sqrt(mse)
            self.metrics['rmse'].update(rmse)
            results['rmse'] = rmse
            
            # R2
            ss_tot = np.average((acts - np.average(acts, weights=weights)) ** 2, weights=weights)
            ss_res = np.average((acts - preds) ** 2, weights=weights)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.metrics['r2'].update(r2)
            results['r2'] = r2
            
            # MAPE
            mask = acts != 0
            if mask.sum() > 0:
                mape = np.average(np.abs((acts[mask] - preds[mask]) / acts[mask]), weights=weights[mask]) * 100
            else:
                mape = 0
            self.metrics['mape'].update(mape)
            results['mape'] = mape
            
        else:  # classification
            # Convert to binary for metrics
            preds_binary = (preds > 0.5).astype(int)
            acts_binary = acts.astype(int)
            
            # Accuracy
            acc = accuracy_score(acts_binary, preds_binary, sample_weight=weights)
            self.metrics['accuracy'].update(acc)
            results['accuracy'] = acc
            
            # Precision, Recall, F1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                prec = precision_score(acts_binary, preds_binary, sample_weight=weights, zero_division=0)
                self.metrics['precision'].update(prec)
                results['precision'] = prec
                
                rec = recall_score(acts_binary, preds_binary, sample_weight=weights, zero_division=0)
                self.metrics['recall'].update(rec)
                results['recall'] = rec
                
                f1 = f1_score(acts_binary, preds_binary, sample_weight=weights, zero_division=0)
                self.metrics['f1'].update(f1)
                results['f1'] = f1
                
        return results
        
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        performance = {
            'n_samples': self.n_samples,
            'window_size': len(self.predictions),
            'metrics': {}
        }
        
        for name, metric in self.metrics.items():
            performance['metrics'][name] = {
                'current': metric.value,
                'window_avg': metric.get_window_average(),
                'trend': metric.get_trend()
            }
            
        return performance
        
    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Get confusion matrix for classification tasks."""
        if self.task_type != 'classification' or len(self.predictions) < 2:
            return None
            
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        
        preds_binary = (preds > 0.5).astype(int)
        acts_binary = acts.astype(int)
        
        return confusion_matrix(acts_binary, preds_binary)
        
    def reset(self) -> None:
        """Reset evaluator state."""
        self.predictions.clear()
        self.actuals.clear()
        self.weights.clear()
        self.n_samples = 0
        self._initialize_metrics()


class StreamingValidator:
    """Validator for streaming machine learning models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_interval = config.get('validation_interval', 100)
        self.stability_threshold = config.get('stability_threshold', 0.05)
        self.min_samples = config.get('min_samples', 50)
        
        self.evaluators: Dict[str, PrequentialEvaluator] = {}
        self.stability_tracker = StabilityTracker(config)
        self.anomaly_detector = AnomalyDetector(config.get('anomaly_detection', {}))
        
    def validate_stream(self, model_id: str, 
                       predictions: np.ndarray, 
                       actuals: np.ndarray,
                       weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate streaming predictions."""
        if model_id not in self.evaluators:
            self.evaluators[model_id] = PrequentialEvaluator(self.config)
            
        evaluator = self.evaluators[model_id]
        
        # Process each prediction
        if weights is None:
            weights = np.ones(len(predictions))
            
        for pred, actual, weight in zip(predictions, actuals, weights):
            evaluator.add_result(pred, actual, weight)
            
        # Get validation results
        performance = evaluator.get_current_performance()
        
        # Check stability
        is_stable = self.stability_tracker.check_stability(
            performance['metrics']
        )
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(
            predictions, actuals
        )
        
        return {
            'model_id': model_id,
            'performance': performance,
            'is_stable': is_stable,
            'anomalies': anomalies,
            'timestamp': datetime.now()
        }
        
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple models."""
        if not model_ids:
            return {}
            
        comparisons = {}
        
        for metric_type in ['mse', 'mae', 'accuracy', 'f1']:
            values = {}
            for model_id in model_ids:
                if model_id in self.evaluators:
                    evaluator = self.evaluators[model_id]
                    if metric_type in evaluator.metrics:
                        values[model_id] = evaluator.metrics[metric_type].value
                        
            if values:
                best_model = min(values, key=values.get) if metric_type in ['mse', 'mae'] else max(values, key=values.get)
                comparisons[metric_type] = {
                    'values': values,
                    'best_model': best_model,
                    'best_value': values[best_model]
                }
                
        return comparisons
        
    def get_validation_report(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive validation report."""
        if model_id not in self.evaluators:
            return None
            
        evaluator = self.evaluators[model_id]
        performance = evaluator.get_current_performance()
        
        report = {
            'model_id': model_id,
            'timestamp': datetime.now(),
            'performance': performance,
            'stability': self.stability_tracker.get_stability_report(),
            'recommendations': self._generate_recommendations(performance)
        }
        
        # Add confusion matrix for classification
        if evaluator.task_type == 'classification':
            cm = evaluator.get_confusion_matrix()
            if cm is not None:
                report['confusion_matrix'] = cm.tolist()
                
        return report
        
    def _generate_recommendations(self, performance: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance."""
        recommendations = []
        
        metrics = performance.get('metrics', {})
        
        # Check for degrading performance
        for metric_name, metric_info in metrics.items():
            if metric_info['trend'] == 'degrading':
                recommendations.append(
                    f"{metric_name} is degrading - consider model retraining or adaptation"
                )
                
        # Check for poor absolute performance
        if 'accuracy' in metrics and metrics['accuracy']['current'] < 0.7:
            recommendations.append(
                "Accuracy below 70% - review feature engineering or model architecture"
            )
            
        if 'mape' in metrics and metrics['mape']['current'] > 20:
            recommendations.append(
                "MAPE above 20% - model predictions may be unreliable"
            )
            
        return recommendations


class StabilityTracker:
    """Track model stability over time."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get('stability_window', 50)
        self.threshold = config.get('stability_threshold', 0.05)
        
        self.metric_history: Dict[str, deque] = {}
        self.stability_scores: Dict[str, float] = {}
        
    def check_stability(self, metrics: Dict[str, Any]) -> bool:
        """Check if metrics are stable."""
        stable_count = 0
        total_count = 0
        
        for metric_name, metric_info in metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = deque(maxlen=self.window_size)
                
            # Add current value
            current_value = metric_info['current']
            self.metric_history[metric_name].append(current_value)
            
            # Check stability
            if len(self.metric_history[metric_name]) >= 10:
                values = list(self.metric_history[metric_name])
                std = np.std(values)
                mean = np.mean(values)
                
                # Coefficient of variation
                cv = std / mean if mean != 0 else 0
                self.stability_scores[metric_name] = cv
                
                total_count += 1
                if cv < self.threshold:
                    stable_count += 1
                    
        # Consider stable if most metrics are stable
        return stable_count / total_count > 0.7 if total_count > 0 else True
        
    def get_stability_report(self) -> Dict[str, Any]:
        """Get stability analysis report."""
        return {
            'stability_scores': self.stability_scores.copy(),
            'overall_stable': all(
                score < self.threshold 
                for score in self.stability_scores.values()
            ),
            'threshold': self.threshold
        }


class AnomalyDetector:
    """Detect anomalies in streaming predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get('method', 'zscore')
        self.threshold = config.get('threshold', 3.0)
        self.window_size = config.get('window_size', 100)
        
        self.error_history = deque(maxlen=self.window_size)
        
    def detect_anomalies(self, predictions: np.ndarray, 
                        actuals: np.ndarray) -> List[Dict[str, Any]]:
        """Detect anomalous predictions."""
        anomalies = []
        
        errors = np.abs(predictions - actuals)
        
        for i, error in enumerate(errors):
            self.error_history.append(error)
            
            if len(self.error_history) >= 10:
                if self.method == 'zscore':
                    is_anomaly = self._detect_zscore(error)
                elif self.method == 'iqr':
                    is_anomaly = self._detect_iqr(error)
                elif self.method == 'isolation':
                    is_anomaly = self._detect_isolation(error)
                else:
                    is_anomaly = False
                    
                if is_anomaly:
                    anomalies.append({
                        'index': i,
                        'prediction': float(predictions[i]),
                        'actual': float(actuals[i]),
                        'error': float(error),
                        'method': self.method
                    })
                    
        return anomalies
        
    def _detect_zscore(self, error: float) -> bool:
        """Z-score based anomaly detection."""
        errors = np.array(self.error_history)
        mean = np.mean(errors)
        std = np.std(errors)
        
        if std == 0:
            return False
            
        z_score = abs((error - mean) / std)
        return z_score > self.threshold
        
    def _detect_iqr(self, error: float) -> bool:
        """IQR based anomaly detection."""
        errors = np.array(self.error_history)
        q1 = np.percentile(errors, 25)
        q3 = np.percentile(errors, 75)
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        return error < lower or error > upper
        
    def _detect_isolation(self, error: float) -> bool:
        """Isolation forest inspired detection."""
        # Simplified version
        errors = np.array(self.error_history)
        percentile = np.percentile(errors, 95)
        return error > percentile


class StreamingCrossValidator:
    """Cross-validation for streaming data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get('n_splits', 5)
        self.test_size = config.get('test_size', 0.2)
        
    def time_series_split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series cross-validation splits."""
        splits = []
        
        min_train_size = int(n_samples * 0.2)
        test_size = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            train_end = min_train_size + i * test_size
            test_end = min(train_end + test_size, n_samples)
            
            if test_end > n_samples:
                break
                
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            
            splits.append((train_idx, test_idx))
            
        return splits
        
    def prequential_blocks(self, n_samples: int, 
                          block_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate prequential block splits."""
        splits = []
        
        for i in range(0, n_samples - block_size, block_size):
            if i == 0:
                continue
                
            train_idx = np.arange(0, i)
            test_idx = np.arange(i, min(i + block_size, n_samples))
            
            splits.append((train_idx, test_idx))
            
        return splits