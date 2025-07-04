"""Validation utilities for ensemble methods."""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from alpha_pulse.ml.ensemble.ensemble_manager import AgentSignal, EnsembleSignal
from alpha_pulse.utils.metrics import PerformanceMetrics, SignalMetrics

logger = logging.getLogger(__name__)


class EnsembleValidator:
    """Validates ensemble methods and performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get('n_splits', 5)
        self.test_size = config.get('test_size', 0.2)
        self.walk_forward_window = config.get('walk_forward_window', 252)
        self.min_train_size = config.get('min_train_size', 100)
        
    def cross_validate_ensemble(self, ensemble, signals: List[AgentSignal], 
                              outcomes: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation on ensemble method."""
        # Time series split for financial data
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        cv_results = {
            'train_scores': [],
            'test_scores': [],
            'predictions': [],
            'actuals': [],
            'feature_importances': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(outcomes)):
            # Split signals by index
            train_signals = [s for i, s in enumerate(signals) if i in train_idx]
            test_signals = [s for i, s in enumerate(signals) if i in test_idx]
            
            train_outcomes = outcomes[train_idx]
            test_outcomes = outcomes[test_idx]
            
            # Train ensemble
            ensemble.fit(train_signals, train_outcomes)
            
            # Evaluate on train set
            train_predictions = []
            for i in range(len(train_outcomes)):
                time_signals = [s for s in train_signals if s.timestamp.second == i]
                if time_signals:
                    pred = ensemble.predict(time_signals)
                    train_predictions.append(pred.signal)
                    
            train_score = self._calculate_score(
                np.array(train_predictions), 
                train_outcomes[:len(train_predictions)]
            )
            
            # Evaluate on test set
            test_predictions = []
            for i in range(len(test_outcomes)):
                time_signals = [s for s in test_signals if s.timestamp.second == i]
                if time_signals:
                    pred = ensemble.predict(time_signals)
                    test_predictions.append(pred.signal)
                    
            test_score = self._calculate_score(
                np.array(test_predictions),
                test_outcomes[:len(test_predictions)]
            )
            
            cv_results['train_scores'].append(train_score)
            cv_results['test_scores'].append(test_score)
            cv_results['predictions'].extend(test_predictions)
            cv_results['actuals'].extend(test_outcomes[:len(test_predictions)])
            
            # Store feature importances if available
            if hasattr(ensemble, 'agent_weights'):
                cv_results['feature_importances'].append(ensemble.agent_weights.copy())
                
            logger.info(f"Fold {fold+1}: Train={train_score:.4f}, Test={test_score:.4f}")
            
        # Calculate summary statistics
        cv_results['mean_train_score'] = np.mean(cv_results['train_scores'])
        cv_results['std_train_score'] = np.std(cv_results['train_scores'])
        cv_results['mean_test_score'] = np.mean(cv_results['test_scores'])
        cv_results['std_test_score'] = np.std(cv_results['test_scores'])
        
        return cv_results
        
    def walk_forward_validation(self, ensemble, signals: List[AgentSignal],
                              outcomes: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform walk-forward validation."""
        results = {
            'predictions': [],
            'actuals': [],
            'dates': [],
            'ensemble_weights': [],
            'performance_metrics': []
        }
        
        window_size = self.walk_forward_window
        step_size = window_size // 4  # Retrain every quarter
        
        for start_idx in range(self.min_train_size, len(outcomes) - step_size, step_size):
            # Training window
            train_start = max(0, start_idx - window_size)
            train_end = start_idx
            
            # Test window
            test_start = train_end
            test_end = min(len(outcomes), test_start + step_size)
            
            # Get training data
            train_signals = [
                s for i, s in enumerate(signals) 
                if train_start <= i < train_end
            ]
            train_outcomes = outcomes[train_start:train_end]
            
            # Train ensemble
            ensemble.fit(train_signals, train_outcomes)
            
            # Make predictions on test window
            for i in range(test_start, test_end):
                time_signals = [s for s in signals if s.timestamp.second == i]
                if time_signals:
                    pred = ensemble.predict(time_signals)
                    results['predictions'].append(pred.signal)
                    results['actuals'].append(outcomes[i])
                    results['dates'].append(market_data.index[i] if i < len(market_data) else datetime.now())
                    results['ensemble_weights'].append(ensemble.agent_weights.copy())
                    
            # Calculate performance for this window
            if results['predictions']:
                window_metrics = PerformanceMetrics.calculate_comprehensive_metrics(
                    np.array(results['predictions'][-step_size:]),
                    np.array(results['actuals'][-step_size:])
                )
                results['performance_metrics'].append(window_metrics)
                
            logger.info(f"Walk-forward: trained on {train_end-train_start} samples, "
                       f"tested on {test_end-test_start} samples")
                       
        return results
        
    def validate_signal_quality(self, ensemble_signals: List[EnsembleSignal],
                              outcomes: np.ndarray) -> Dict[str, Any]:
        """Validate quality of ensemble signals."""
        if not ensemble_signals:
            return {}
            
        # Extract signal values and confidences
        signals = np.array([s.signal for s in ensemble_signals])
        confidences = np.array([s.confidence for s in ensemble_signals])
        
        # Calculate signal metrics
        signal_metrics = SignalMetrics.calculate_signal_metrics(
            signals[:len(outcomes)],
            outcomes[:len(signals)],
            confidences[:len(outcomes)]
        )
        
        # Additional quality checks
        quality_metrics = {
            **signal_metrics,
            'signal_range': float(np.ptp(signals)),
            'confidence_range': float(np.ptp(confidences)),
            'signal_skew': float(stats.skew(signals)),
            'signal_kurtosis': float(stats.kurtosis(signals)),
            'confidence_consistency': self._check_confidence_consistency(signals, confidences),
            'signal_autocorrelation': self._calculate_autocorrelation(signals)
        }
        
        # Check for common issues
        quality_metrics['warnings'] = self._check_signal_issues(signals, confidences)
        
        return quality_metrics
        
    def compare_ensemble_methods(self, ensembles: Dict[str, Any], 
                               signals: List[AgentSignal],
                               outcomes: np.ndarray) -> pd.DataFrame:
        """Compare multiple ensemble methods."""
        comparison_results = []
        
        for name, ensemble in ensembles.items():
            logger.info(f"Evaluating ensemble: {name}")
            
            # Cross-validation
            cv_results = self.cross_validate_ensemble(ensemble, signals, outcomes)
            
            # Calculate additional metrics
            predictions = np.array(cv_results['predictions'])
            actuals = np.array(cv_results['actuals'])
            
            if len(predictions) > 0 and len(actuals) > 0:
                metrics = {
                    'ensemble': name,
                    'cv_mean_score': cv_results['mean_test_score'],
                    'cv_std_score': cv_results['std_test_score'],
                    'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                    'mae': mean_absolute_error(actuals, predictions),
                    'correlation': np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0,
                    'directional_accuracy': np.mean(np.sign(predictions) == np.sign(actuals)),
                    'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(predictions * actuals)
                }
                
                comparison_results.append(metrics)
                
        return pd.DataFrame(comparison_results).sort_values('cv_mean_score', ascending=False)
        
    def validate_agent_contributions(self, ensemble, signals: List[AgentSignal],
                                   outcomes: np.ndarray) -> Dict[str, Any]:
        """Validate individual agent contributions to ensemble."""
        # Get unique agents
        agent_ids = list(set(s.agent_id for s in signals))
        
        agent_metrics = {}
        
        for agent_id in agent_ids:
            # Get signals for this agent
            agent_signals = [s for s in signals if s.agent_id == agent_id]
            
            if agent_signals:
                # Calculate individual agent performance
                agent_predictions = np.array([s.signal for s in agent_signals])
                agent_confidences = np.array([s.confidence for s in agent_signals])
                
                # Match with outcomes
                agent_outcomes = outcomes[:len(agent_predictions)]
                
                # Calculate metrics
                agent_metrics[agent_id] = {
                    'signal_count': len(agent_signals),
                    'avg_confidence': float(np.mean(agent_confidences)),
                    'accuracy': SignalMetrics.calculate_signal_accuracy(
                        agent_predictions, agent_outcomes
                    ),
                    'correlation': SignalMetrics.calculate_signal_correlation(
                        agent_predictions, agent_outcomes
                    ),
                    'weight': ensemble.agent_weights.get(agent_id, 0.0) if hasattr(ensemble, 'agent_weights') else 0.0,
                    'contribution': self._calculate_contribution(
                        agent_predictions, agent_outcomes, 
                        ensemble.agent_weights.get(agent_id, 1.0) if hasattr(ensemble, 'agent_weights') else 1.0
                    )
                }
                
        return agent_metrics
        
    def _calculate_score(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate validation score."""
        if len(predictions) == 0 or len(actuals) == 0:
            return 0.0
            
        # Use correlation as primary score
        if len(predictions) > 1:
            return float(np.corrcoef(predictions, actuals)[0, 1])
        else:
            return 0.0
            
    def _check_confidence_consistency(self, signals: np.ndarray, 
                                    confidences: np.ndarray) -> float:
        """Check if confidence levels are consistent with signal strength."""
        if len(signals) == 0:
            return 1.0
            
        # Correlation between absolute signal and confidence
        return float(np.corrcoef(np.abs(signals), confidences)[0, 1])
        
    def _calculate_autocorrelation(self, signals: np.ndarray, lag: int = 1) -> float:
        """Calculate signal autocorrelation."""
        if len(signals) <= lag:
            return 0.0
            
        return float(np.corrcoef(signals[:-lag], signals[lag:])[0, 1])
        
    def _check_signal_issues(self, signals: np.ndarray, 
                           confidences: np.ndarray) -> List[str]:
        """Check for common signal quality issues."""
        warnings = []
        
        # Check for constant signals
        if np.std(signals) < 0.01:
            warnings.append("Low signal variance - possible constant predictions")
            
        # Check for extreme clustering
        unique_signals = len(np.unique(np.round(signals, 2)))
        if unique_signals < 5:
            warnings.append(f"Low signal diversity - only {unique_signals} unique values")
            
        # Check confidence distribution
        if np.mean(confidences) < 0.3:
            warnings.append("Low average confidence - model uncertainty high")
        elif np.mean(confidences) > 0.9:
            warnings.append("Very high confidence - possible overconfidence")
            
        # Check for signal saturation
        saturated = np.sum(np.abs(signals) > 0.95) / len(signals)
        if saturated > 0.1:
            warnings.append(f"Signal saturation - {saturated*100:.1f}% signals at extremes")
            
        return warnings
        
    def _calculate_contribution(self, predictions: np.ndarray, 
                              outcomes: np.ndarray, weight: float) -> float:
        """Calculate weighted contribution of predictions to outcomes."""
        if len(predictions) == 0 or weight == 0:
            return 0.0
            
        # Weighted correlation * weight
        correlation = np.corrcoef(predictions, outcomes)[0, 1] if len(predictions) > 1 else 0
        return float(correlation * weight)


class EnsembleMonitor:
    """Monitor ensemble performance in real-time."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_window = config.get('performance_window', 100)
        self.alert_thresholds = config.get('alert_thresholds', {
            'accuracy': 0.45,
            'confidence': 0.3,
            'diversity': 0.1
        })
        
        self.signal_history = []
        self.outcome_history = []
        self.performance_history = []
        
    def update(self, ensemble_signal: EnsembleSignal, outcome: Optional[float] = None) -> Dict[str, Any]:
        """Update monitor with new signal and outcome."""
        self.signal_history.append(ensemble_signal)
        
        if outcome is not None:
            self.outcome_history.append(outcome)
            
        # Calculate recent performance if we have enough data
        if len(self.signal_history) >= self.performance_window and len(self.outcome_history) >= self.performance_window:
            recent_metrics = self._calculate_recent_performance()
            self.performance_history.append(recent_metrics)
            
            # Check for alerts
            alerts = self._check_alerts(recent_metrics)
            
            return {
                'metrics': recent_metrics,
                'alerts': alerts,
                'trend': self._calculate_performance_trend()
            }
            
        return {'metrics': {}, 'alerts': [], 'trend': {}}
        
    def _calculate_recent_performance(self) -> Dict[str, float]:
        """Calculate performance over recent window."""
        recent_signals = self.signal_history[-self.performance_window:]
        recent_outcomes = self.outcome_history[-self.performance_window:]
        
        signals = np.array([s.signal for s in recent_signals])
        confidences = np.array([s.confidence for s in recent_signals])
        outcomes = np.array(recent_outcomes)
        
        # Calculate diversity
        diversities = []
        for signal in recent_signals:
            if hasattr(signal, 'metadata') and 'diversity' in signal.metadata:
                diversities.append(signal.metadata['diversity'])
                
        return {
            'accuracy': SignalMetrics.calculate_signal_accuracy(signals, outcomes),
            'avg_confidence': float(np.mean(confidences)),
            'avg_diversity': float(np.mean(diversities)) if diversities else 0.0,
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(signals * outcomes),
            'signal_stability': SignalMetrics.calculate_signal_stability(signals)
        }
        
    def _check_alerts(self, metrics: Dict[str, float]) -> List[str]:
        """Check if any metrics breach alert thresholds."""
        alerts = []
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                alerts.append(f"{metric} below threshold: {metrics[metric]:.3f} < {threshold}")
                
        return alerts
        
    def _calculate_performance_trend(self) -> Dict[str, float]:
        """Calculate trend in performance metrics."""
        if len(self.performance_history) < 2:
            return {}
            
        recent = self.performance_history[-1]
        previous = self.performance_history[-2]
        
        trends = {}
        for metric in recent:
            if metric in previous:
                trends[f"{metric}_change"] = recent[metric] - previous[metric]
                
        return trends
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.signal_history:
            return {}
            
        all_signals = np.array([s.signal for s in self.signal_history])
        all_confidences = np.array([s.confidence for s in self.signal_history])
        
        summary = {
            'total_signals': len(self.signal_history),
            'signal_stats': {
                'mean': float(np.mean(all_signals)),
                'std': float(np.std(all_signals)),
                'min': float(np.min(all_signals)),
                'max': float(np.max(all_signals))
            },
            'confidence_stats': {
                'mean': float(np.mean(all_confidences)),
                'std': float(np.std(all_confidences)),
                'min': float(np.min(all_confidences)),
                'max': float(np.max(all_confidences))
            }
        }
        
        if self.performance_history:
            # Average performance over all windows
            avg_metrics = {}
            for metric in self.performance_history[0]:
                values = [p[metric] for p in self.performance_history if metric in p]
                avg_metrics[f"avg_{metric}"] = float(np.mean(values))
                
            summary['average_performance'] = avg_metrics
            
        return summary