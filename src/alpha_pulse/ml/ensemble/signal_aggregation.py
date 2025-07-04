"""Signal aggregation methods for ensemble combination."""

import logging
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class SignalAggregator:
    """Aggregates signals using various methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get('method', 'weighted_average')
        self.outlier_detection = config.get('outlier_detection', True)
        self.outlier_threshold = config.get('outlier_threshold', 3.0)
        self.confidence_weighting = config.get('confidence_weighting', True)
        self.diversity_bonus = config.get('diversity_bonus', 0.1)
        
        # Method mappings
        self.aggregation_methods = {
            'weighted_average': self.weighted_average,
            'median': self.robust_median,
            'trimmed_mean': self.trimmed_mean,
            'quantile': self.quantile_aggregation,
            'entropy_weighted': self.entropy_weighted,
            'rank_based': self.rank_based,
            'harmonic_mean': self.harmonic_mean,
            'geometric_mean': self.geometric_mean,
            'winsorized_mean': self.winsorized_mean,
            'adaptive': self.adaptive_aggregation
        }
        
    def aggregate(self, signals: List[float], confidences: List[float], 
                 weights: Optional[List[float]] = None) -> Dict[str, float]:
        """Aggregate signals using configured method."""
        if not signals:
            return {'signal': 0.0, 'confidence': 0.0, 'diversity': 0.0}
            
        signals = np.array(signals)
        confidences = np.array(confidences)
        
        if weights is None:
            weights = np.ones(len(signals))
        else:
            weights = np.array(weights)
            
        # Apply confidence weighting if enabled
        if self.confidence_weighting:
            weights = weights * confidences
            
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(signals)) / len(signals)
            
        # Detect and handle outliers
        if self.outlier_detection:
            signals, weights = self._handle_outliers(signals, weights)
            
        # Apply aggregation method
        if self.method in self.aggregation_methods:
            result = self.aggregation_methods[self.method](signals, weights)
        else:
            result = self.weighted_average(signals, weights)
            
        # Calculate diversity metrics
        diversity = self._calculate_diversity(signals)
        
        # Apply diversity bonus to confidence
        confidence = result.get('confidence', 0.0)
        if diversity > 0.5:  # High diversity
            confidence *= (1 + self.diversity_bonus)
            
        return {
            'signal': result['signal'],
            'confidence': min(confidence, 1.0),
            'diversity': diversity,
            'method': self.method,
            'signal_count': len(signals)
        }
        
    def weighted_average(self, signals: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Weighted average aggregation."""
        signal = np.average(signals, weights=weights)
        
        # Confidence based on weight concentration
        weight_entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(len(weights))
        confidence = weight_entropy / max_entropy if max_entropy > 0 else 0.5
        
        return {'signal': float(signal), 'confidence': float(confidence)}
        
    def robust_median(self, signals: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Weighted median aggregation."""
        # Sort signals by value
        sorted_indices = np.argsort(signals)
        sorted_signals = signals[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Find weighted median
        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, 0.5)
        
        if median_idx < len(sorted_signals):
            signal = float(sorted_signals[median_idx])
        else:
            signal = float(sorted_signals[-1])
            
        # Confidence based on weight distribution around median
        median_weight = sorted_weights[median_idx] if median_idx < len(sorted_weights) else 0
        confidence = 0.5 + 0.5 * median_weight
        
        return {'signal': signal, 'confidence': float(confidence)}
        
    def trimmed_mean(self, signals: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Trimmed mean aggregation (remove top/bottom percentiles)."""
        trim_pct = self.config.get('trim_percentage', 0.1)
        
        # Sort signals
        sorted_indices = np.argsort(signals)
        n_trim = int(len(signals) * trim_pct)
        
        if n_trim > 0 and len(signals) > 2 * n_trim:
            # Trim top and bottom
            trimmed_indices = sorted_indices[n_trim:-n_trim]
            trimmed_signals = signals[trimmed_indices]
            trimmed_weights = weights[trimmed_indices]
            
            # Renormalize weights
            trimmed_weights = trimmed_weights / trimmed_weights.sum()
            
            signal = np.average(trimmed_signals, weights=trimmed_weights)
            confidence = 0.7  # Fixed confidence for trimmed mean
        else:
            # Fall back to weighted average
            signal = np.average(signals, weights=weights)
            confidence = 0.5
            
        return {'signal': float(signal), 'confidence': float(confidence)}
        
    def quantile_aggregation(self, signals: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Quantile-based aggregation."""
        quantile = self.config.get('quantile', 0.5)
        
        # Weighted quantile
        sorted_indices = np.argsort(signals)
        sorted_signals = signals[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cumsum = np.cumsum(sorted_weights)
        quantile_idx = np.searchsorted(cumsum, quantile)
        
        if quantile_idx < len(sorted_signals):
            signal = float(sorted_signals[quantile_idx])
        else:
            signal = float(sorted_signals[-1])
            
        # Confidence based on quantile position
        if quantile == 0.5:  # Median
            confidence = 0.8
        else:
            # Lower confidence for extreme quantiles
            confidence = 0.6 - 0.4 * abs(quantile - 0.5)
            
        return {'signal': signal, 'confidence': float(confidence)}
        
    def entropy_weighted(self, signals: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Entropy-weighted aggregation."""
        # Calculate signal entropy
        signal_probs = softmax(signals)
        entropy = -np.sum(signal_probs * np.log(signal_probs + 1e-8))
        
        # Adjust weights by entropy (lower entropy = higher confidence)
        entropy_factor = 1.0 / (1.0 + entropy)
        adjusted_weights = weights * entropy_factor
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        signal = np.average(signals, weights=adjusted_weights)
        confidence = entropy_factor
        
        return {'signal': float(signal), 'confidence': float(confidence)}
        
    def rank_based(self, signals: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Rank-based aggregation."""
        # Convert signals to ranks
        ranks = stats.rankdata(signals)
        normalized_ranks = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else ranks
        
        # Map ranks back to signal range [-1, 1]
        rank_signals = 2 * normalized_ranks - 1
        
        signal = np.average(rank_signals, weights=weights)
        
        # Confidence based on rank agreement
        rank_variance = np.var(normalized_ranks)
        confidence = 1.0 / (1.0 + rank_variance)
        
        return {'signal': float(signal), 'confidence': float(confidence)}
        
    def harmonic_mean(self, signals: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Harmonic mean aggregation (good for rates/ratios)."""
        # Shift signals to positive range for harmonic mean
        shifted_signals = signals + 1.1  # Now in range [0.1, 2.1]
        
        # Weighted harmonic mean
        if np.all(shifted_signals > 0):
            harmonic = 1.0 / np.average(1.0 / shifted_signals, weights=weights)
            signal = harmonic - 1.1  # Shift back
        else:
            # Fall back to arithmetic mean
            signal = np.average(signals, weights=weights)
            
        confidence = 0.7  # Fixed confidence for harmonic mean
        
        return {'signal': float(signal), 'confidence': float(confidence)}
        
    def geometric_mean(self, signals: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Geometric mean aggregation."""
        # Shift signals to positive range
        shifted_signals = signals + 1.1  # Now in range [0.1, 2.1]
        
        # Weighted geometric mean
        if np.all(shifted_signals > 0):
            log_signals = np.log(shifted_signals)
            log_mean = np.average(log_signals, weights=weights)
            geometric = np.exp(log_mean)
            signal = geometric - 1.1  # Shift back
        else:
            # Fall back to arithmetic mean
            signal = np.average(signals, weights=weights)
            
        confidence = 0.7  # Fixed confidence for geometric mean
        
        return {'signal': float(signal), 'confidence': float(confidence)}
        
    def winsorized_mean(self, signals: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Winsorized mean (clip extreme values)."""
        percentile = self.config.get('winsor_percentile', 5)
        
        # Calculate percentiles
        lower = np.percentile(signals, percentile)
        upper = np.percentile(signals, 100 - percentile)
        
        # Clip signals
        clipped_signals = np.clip(signals, lower, upper)
        
        signal = np.average(clipped_signals, weights=weights)
        
        # Confidence based on how many values were clipped
        n_clipped = np.sum((signals < lower) | (signals > upper))
        clip_ratio = n_clipped / len(signals)
        confidence = 1.0 - 0.5 * clip_ratio
        
        return {'signal': float(signal), 'confidence': float(confidence)}
        
    def adaptive_aggregation(self, signals: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Adaptive aggregation that chooses method based on signal characteristics."""
        # Analyze signal distribution
        skewness = stats.skew(signals)
        kurtosis = stats.kurtosis(signals)
        n_signals = len(signals)
        
        # Choose method based on characteristics
        if abs(skewness) > 1.0:
            # Skewed distribution - use median
            result = self.robust_median(signals, weights)
        elif kurtosis > 3.0:
            # Heavy tails - use trimmed mean
            result = self.trimmed_mean(signals, weights)
        elif n_signals < 5:
            # Few signals - use weighted average
            result = self.weighted_average(signals, weights)
        else:
            # Normal case - use entropy weighted
            result = self.entropy_weighted(signals, weights)
            
        return result
        
    def _handle_outliers(self, signals: np.ndarray, weights: np.ndarray) -> tuple:
        """Detect and handle outliers in signals."""
        # Calculate z-scores
        mean = np.mean(signals)
        std = np.std(signals)
        
        if std > 0:
            z_scores = np.abs((signals - mean) / std)
            
            # Identify outliers
            outlier_mask = z_scores > self.outlier_threshold
            
            if np.any(outlier_mask):
                # Reduce weights for outliers
                outlier_penalty = 0.1
                weights[outlier_mask] *= outlier_penalty
                
                # Renormalize
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    
                logger.info(f"Detected {np.sum(outlier_mask)} outliers in signals")
                
        return signals, weights
        
    def _calculate_diversity(self, signals: np.ndarray) -> float:
        """Calculate diversity of signals."""
        if len(signals) < 2:
            return 0.0
            
        # Multiple diversity measures
        
        # 1. Standard deviation
        std_diversity = np.std(signals)
        
        # 2. Disagreement ratio (signals with different signs)
        positive = np.sum(signals > 0)
        negative = np.sum(signals < 0)
        total = len(signals)
        disagreement = min(positive, negative) / total if total > 0 else 0
        
        # 3. Entropy of signal distribution
        hist, _ = np.histogram(signals, bins=5)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        entropy_normalized = entropy / np.log(5) if np.log(5) > 0 else 0
        
        # Combine diversity measures
        diversity = (std_diversity + disagreement + entropy_normalized) / 3
        
        return float(min(diversity, 1.0))


class TemporalAggregator:
    """Aggregates signals over time windows."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get('window_size', 10)
        self.decay_rate = config.get('decay_rate', 0.9)
        self.signal_history = defaultdict(list)
        
    def add_signal(self, agent_id: str, signal: float, confidence: float, 
                  timestamp: datetime) -> None:
        """Add signal to history."""
        self.signal_history[agent_id].append({
            'signal': signal,
            'confidence': confidence,
            'timestamp': timestamp
        })
        
        # Keep only recent history
        if len(self.signal_history[agent_id]) > self.window_size:
            self.signal_history[agent_id].pop(0)
            
    def get_temporal_aggregate(self, agent_id: str) -> Dict[str, float]:
        """Get time-weighted aggregate for an agent."""
        if agent_id not in self.signal_history:
            return {'signal': 0.0, 'confidence': 0.0}
            
        history = self.signal_history[agent_id]
        if not history:
            return {'signal': 0.0, 'confidence': 0.0}
            
        # Calculate time-weighted average
        now = datetime.now()
        weighted_sum = 0.0
        weight_sum = 0.0
        confidence_sum = 0.0
        
        for i, entry in enumerate(reversed(history)):
            # Exponential decay based on position
            decay_weight = self.decay_rate ** i
            
            # Time-based decay
            time_diff = (now - entry['timestamp']).total_seconds()
            time_weight = np.exp(-time_diff / 3600)  # 1-hour half-life
            
            weight = decay_weight * time_weight * entry['confidence']
            weighted_sum += entry['signal'] * weight
            weight_sum += weight
            confidence_sum += entry['confidence'] * decay_weight
            
        if weight_sum > 0:
            signal = weighted_sum / weight_sum
            confidence = confidence_sum / len(history)
        else:
            signal = history[-1]['signal']
            confidence = history[-1]['confidence']
            
        return {
            'signal': float(signal),
            'confidence': float(confidence),
            'history_length': len(history)
        }
        
    def get_signal_momentum(self, agent_id: str) -> float:
        """Calculate signal momentum (trend strength)."""
        if agent_id not in self.signal_history or len(self.signal_history[agent_id]) < 2:
            return 0.0
            
        signals = [entry['signal'] for entry in self.signal_history[agent_id]]
        
        # Calculate linear regression slope
        x = np.arange(len(signals))
        slope, _ = np.polyfit(x, signals, 1)
        
        return float(slope)
        
    def get_signal_volatility(self, agent_id: str) -> float:
        """Calculate signal volatility."""
        if agent_id not in self.signal_history or len(self.signal_history[agent_id]) < 2:
            return 0.0
            
        signals = [entry['signal'] for entry in self.signal_history[agent_id]]
        return float(np.std(signals))


class ConsensusAggregator:
    """Aggregates signals based on consensus mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consensus_threshold = config.get('consensus_threshold', 0.6)
        self.super_majority = config.get('super_majority', 0.8)
        self.require_quorum = config.get('require_quorum', True)
        self.quorum_size = config.get('quorum_size', 0.5)
        
    def calculate_consensus(self, signals: List[Dict[str, Any]], 
                          total_agents: int) -> Dict[str, Any]:
        """Calculate consensus among signals."""
        if not signals:
            return {
                'consensus': False,
                'signal': 0.0,
                'confidence': 0.0,
                'consensus_type': 'no_signals'
            }
            
        # Check quorum
        participation_rate = len(signals) / total_agents if total_agents > 0 else 0
        
        if self.require_quorum and participation_rate < self.quorum_size:
            return {
                'consensus': False,
                'signal': 0.0,
                'confidence': 0.0,
                'consensus_type': 'no_quorum',
                'participation_rate': participation_rate
            }
            
        # Categorize signals
        buy_signals = [s for s in signals if s['signal'] > 0.1]
        sell_signals = [s for s in signals if s['signal'] < -0.1]
        neutral_signals = [s for s in signals if -0.1 <= s['signal'] <= 0.1]
        
        total = len(signals)
        buy_ratio = len(buy_signals) / total
        sell_ratio = len(sell_signals) / total
        neutral_ratio = len(neutral_signals) / total
        
        # Determine consensus
        consensus_signal = 0.0
        consensus_confidence = 0.0
        consensus_type = 'none'
        
        if buy_ratio >= self.super_majority:
            consensus_signal = np.mean([s['signal'] for s in buy_signals])
            consensus_confidence = np.mean([s['confidence'] for s in buy_signals])
            consensus_type = 'super_majority_buy'
        elif sell_ratio >= self.super_majority:
            consensus_signal = np.mean([s['signal'] for s in sell_signals])
            consensus_confidence = np.mean([s['confidence'] for s in sell_signals])
            consensus_type = 'super_majority_sell'
        elif buy_ratio >= self.consensus_threshold:
            consensus_signal = np.mean([s['signal'] for s in buy_signals]) * 0.7
            consensus_confidence = np.mean([s['confidence'] for s in buy_signals]) * 0.8
            consensus_type = 'majority_buy'
        elif sell_ratio >= self.consensus_threshold:
            consensus_signal = np.mean([s['signal'] for s in sell_signals]) * 0.7
            consensus_confidence = np.mean([s['confidence'] for s in sell_signals]) * 0.8
            consensus_type = 'majority_sell'
        elif neutral_ratio >= self.consensus_threshold:
            consensus_signal = 0.0
            consensus_confidence = np.mean([s['confidence'] for s in neutral_signals]) * 0.6
            consensus_type = 'majority_neutral'
        else:
            # No clear consensus - use weighted average
            all_signals = [s['signal'] for s in signals]
            all_confidences = [s['confidence'] for s in signals]
            consensus_signal = np.average(all_signals, weights=all_confidences)
            consensus_confidence = np.mean(all_confidences) * 0.5
            consensus_type = 'no_consensus'
            
        return {
            'consensus': consensus_type not in ['none', 'no_consensus', 'no_quorum'],
            'signal': float(consensus_signal),
            'confidence': float(consensus_confidence),
            'consensus_type': consensus_type,
            'participation_rate': participation_rate,
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'neutral_ratio': neutral_ratio
        }