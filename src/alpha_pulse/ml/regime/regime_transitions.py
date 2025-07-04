"""
Regime Transition Analysis and Forecasting.

This module provides advanced analysis of regime transitions including
transition probability estimation, regime persistence analysis, and
transition timing prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

from .hmm_regime_detector import RegimeType

logger = logging.getLogger(__name__)


@dataclass
class TransitionEvent:
    """Detailed information about a regime transition event."""
    from_regime: int
    to_regime: int
    timestamp: datetime
    duration_in_from: int
    market_conditions: Dict[str, float]
    trigger_factors: List[str]
    transition_speed: float  # How quickly the transition occurred
    stability_score: float  # How stable the new regime is


class RegimeTransitionAnalyzer:
    """Analyze and forecast regime transitions."""
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.transition_events: List[TransitionEvent] = []
        self.regime_durations: Dict[int, List[int]] = defaultdict(list)
        self.transition_counts = defaultdict(int)
        self.transition_features: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
        
    def add_transition(self, event: TransitionEvent):
        """Add a new transition event for analysis."""
        self.transition_events.append(event)
        self.regime_durations[event.from_regime].append(event.duration_in_from)
        self.transition_counts[(event.from_regime, event.to_regime)] += 1
        self.transition_features[(event.from_regime, event.to_regime)].append(
            event.market_conditions
        )
    
    def estimate_transition_probabilities(self, 
                                        current_regime: int,
                                        current_duration: int,
                                        market_features: Dict[str, float]) -> Dict[int, float]:
        """
        Estimate transition probabilities given current state and conditions.
        
        Args:
            current_regime: Current regime state
            current_duration: Duration in current regime
            market_features: Current market features
            
        Returns:
            Dictionary mapping target regimes to transition probabilities
        """
        # Get base transition probabilities from historical data
        base_probs = self._get_base_transition_probs(current_regime)
        
        # Adjust for duration dependence
        duration_adjusted = self._adjust_for_duration(
            base_probs, current_regime, current_duration
        )
        
        # Adjust for market conditions
        condition_adjusted = self._adjust_for_conditions(
            duration_adjusted, current_regime, market_features
        )
        
        # Normalize to ensure probabilities sum to 1
        total_prob = sum(condition_adjusted.values())
        if total_prob > 0:
            return {k: v / total_prob for k, v in condition_adjusted.items()}
        else:
            return base_probs
    
    def _get_base_transition_probs(self, from_regime: int) -> Dict[int, float]:
        """Get base transition probabilities from historical counts."""
        total_transitions = sum(
            count for (f, t), count in self.transition_counts.items() 
            if f == from_regime
        )
        
        if total_transitions == 0:
            # No historical data, return uniform probabilities
            n_regimes = len(set(e.from_regime for e in self.transition_events) | 
                           set(e.to_regime for e in self.transition_events))
            return {i: 1.0 / n_regimes for i in range(n_regimes)}
        
        probs = {}
        for (f, t), count in self.transition_counts.items():
            if f == from_regime:
                probs[t] = count / total_transitions
        
        return probs
    
    def _adjust_for_duration(self,
                           base_probs: Dict[int, float],
                           current_regime: int,
                           current_duration: int) -> Dict[int, float]:
        """Adjust transition probabilities based on regime duration."""
        if current_regime not in self.regime_durations:
            return base_probs
        
        durations = self.regime_durations[current_regime]
        if not durations:
            return base_probs
        
        # Fit survival function (simplified Weibull)
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        # Hazard rate increases with duration relative to average
        if std_duration > 0:
            z_score = (current_duration - avg_duration) / std_duration
            hazard_multiplier = 1 + 0.2 * max(0, z_score)  # Increase transition prob if overdue
        else:
            hazard_multiplier = 1.0
        
        # Apply multiplier to transition probabilities
        adjusted_probs = {}
        for regime, prob in base_probs.items():
            if regime != current_regime:
                adjusted_probs[regime] = prob * hazard_multiplier
            else:
                adjusted_probs[regime] = prob / hazard_multiplier
        
        return adjusted_probs
    
    def _adjust_for_conditions(self,
                             base_probs: Dict[int, float],
                             from_regime: int,
                             market_features: Dict[str, float]) -> Dict[int, float]:
        """Adjust transition probabilities based on market conditions."""
        adjusted_probs = base_probs.copy()
        
        # Get historical transition features
        for (f, t), feature_list in self.transition_features.items():
            if f != from_regime or t not in base_probs:
                continue
            
            if not feature_list:
                continue
            
            # Calculate similarity to historical transition conditions
            similarities = []
            for hist_features in feature_list[-20:]:  # Use recent history
                similarity = self._calculate_feature_similarity(
                    market_features, hist_features
                )
                similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                # Increase probability if conditions are similar to past transitions
                adjusted_probs[t] *= (1 + 0.5 * avg_similarity)
        
        return adjusted_probs
    
    def _calculate_feature_similarity(self,
                                    features1: Dict[str, float],
                                    features2: Dict[str, float]) -> float:
        """Calculate similarity between two feature sets."""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        # Simple cosine similarity
        vec1 = np.array([features1[k] for k in common_keys])
        vec2 = np.array([features2[k] for k in common_keys])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def forecast_regime_duration(self, 
                               current_regime: int,
                               current_duration: int) -> Dict[str, float]:
        """
        Forecast remaining duration in current regime.
        
        Returns:
            Dictionary with expected remaining duration and confidence interval
        """
        if current_regime not in self.regime_durations:
            return {'expected': 10, 'lower_95': 5, 'upper_95': 20}
        
        durations = self.regime_durations[current_regime]
        if len(durations) < 5:
            return {'expected': 10, 'lower_95': 5, 'upper_95': 20}
        
        # Fit survival model (simplified exponential)
        mean_duration = np.mean(durations)
        
        # Expected remaining duration given current duration
        # Using memoryless property of exponential as simplification
        expected_remaining = mean_duration
        
        # Confidence interval
        std_duration = np.std(durations)
        lower_95 = max(1, expected_remaining - 1.96 * std_duration)
        upper_95 = expected_remaining + 1.96 * std_duration
        
        return {
            'expected': expected_remaining,
            'lower_95': lower_95,
            'upper_95': upper_95,
            'probability_ending_soon': 1 - np.exp(-current_duration / mean_duration)
        }
    
    def identify_transition_patterns(self) -> Dict[str, Any]:
        """Identify common patterns in regime transitions."""
        patterns = {
            'cycles': self._identify_cycles(),
            'triggers': self._identify_triggers(),
            'seasonality': self._analyze_seasonality(),
            'persistence': self._analyze_persistence()
        }
        
        return patterns
    
    def _identify_cycles(self) -> List[List[int]]:
        """Identify cyclical patterns in regime transitions."""
        # Build transition sequences
        sequences = []
        current_sequence = []
        
        for event in self.transition_events:
            if not current_sequence:
                current_sequence = [event.from_regime, event.to_regime]
            elif current_sequence[-1] == event.from_regime:
                current_sequence.append(event.to_regime)
            else:
                if len(current_sequence) > 2:
                    sequences.append(current_sequence)
                current_sequence = [event.from_regime, event.to_regime]
        
        if len(current_sequence) > 2:
            sequences.append(current_sequence)
        
        # Find repeated patterns
        cycles = []
        pattern_counts = defaultdict(int)
        
        for seq in sequences:
            # Look for cycles of length 2-5
            for cycle_len in range(2, min(6, len(seq) // 2)):
                for i in range(len(seq) - cycle_len):
                    pattern = tuple(seq[i:i + cycle_len])
                    pattern_counts[pattern] += 1
        
        # Return patterns that appear multiple times
        for pattern, count in pattern_counts.items():
            if count >= 3:
                cycles.append(list(pattern))
        
        return cycles
    
    def _identify_triggers(self) -> Dict[str, List[str]]:
        """Identify common triggers for regime transitions."""
        triggers = defaultdict(list)
        
        for event in self.transition_events:
            transition_key = f"{event.from_regime}->{event.to_regime}"
            triggers[transition_key].extend(event.trigger_factors)
        
        # Summarize most common triggers
        trigger_summary = {}
        for transition, trigger_list in triggers.items():
            if trigger_list:
                # Count occurrences
                trigger_counts = defaultdict(int)
                for trigger in trigger_list:
                    trigger_counts[trigger] += 1
                
                # Get top triggers
                sorted_triggers = sorted(
                    trigger_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                trigger_summary[transition] = [t[0] for t in sorted_triggers]
        
        return trigger_summary
    
    def _analyze_seasonality(self) -> Dict[str, float]:
        """Analyze seasonal patterns in regime transitions."""
        if len(self.transition_events) < 50:
            return {}
        
        # Extract months of transitions
        monthly_transitions = defaultdict(int)
        for event in self.transition_events:
            month = event.timestamp.month
            monthly_transitions[month] += 1
        
        # Test for uniformity (chi-square test)
        observed = [monthly_transitions.get(m, 0) for m in range(1, 13)]
        expected = [len(self.transition_events) / 12] * 12
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        # Find peak months
        peak_months = sorted(
            monthly_transitions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return {
            'seasonality_p_value': p_value,
            'is_seasonal': p_value < 0.05,
            'peak_months': [m[0] for m in peak_months],
            'monthly_distribution': dict(monthly_transitions)
        }
    
    def _analyze_persistence(self) -> Dict[int, Dict[str, float]]:
        """Analyze regime persistence characteristics."""
        persistence = {}
        
        for regime, durations in self.regime_durations.items():
            if len(durations) < 3:
                continue
            
            persistence[regime] = {
                'mean_duration': np.mean(durations),
                'median_duration': np.median(durations),
                'std_duration': np.std(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'stability_score': 1 / (1 + np.std(durations) / np.mean(durations))
            }
        
        return persistence
    
    def plot_transition_matrix(self, 
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (10, 8)):
        """Plot transition probability matrix as heatmap."""
        # Build transition matrix
        all_regimes = sorted(set(e.from_regime for e in self.transition_events) | 
                           set(e.to_regime for e in self.transition_events))
        
        n_regimes = len(all_regimes)
        trans_matrix = np.zeros((n_regimes, n_regimes))
        
        for i, from_regime in enumerate(all_regimes):
            probs = self._get_base_transition_probs(from_regime)
            for j, to_regime in enumerate(all_regimes):
                trans_matrix[i, j] = probs.get(to_regime, 0)
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            trans_matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=[f"Regime {r}" for r in all_regimes],
            yticklabels=[f"Regime {r}" for r in all_regimes],
            cbar_kws={'label': 'Transition Probability'}
        )
        
        plt.title('Regime Transition Probability Matrix')
        plt.xlabel('To Regime')
        plt.ylabel('From Regime')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_duration_distributions(self,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (12, 8)):
        """Plot regime duration distributions."""
        n_regimes = len(self.regime_durations)
        if n_regimes == 0:
            return
        
        fig, axes = plt.subplots(
            (n_regimes + 1) // 2, 2, 
            figsize=figsize,
            tight_layout=True
        )
        
        if n_regimes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (regime, durations) in enumerate(self.regime_durations.items()):
            ax = axes[idx]
            
            # Plot histogram
            ax.hist(durations, bins=20, alpha=0.7, color='blue', edgecolor='black')
            
            # Add statistics
            mean_dur = np.mean(durations)
            median_dur = np.median(durations)
            
            ax.axvline(mean_dur, color='red', linestyle='--', 
                      label=f'Mean: {mean_dur:.1f}')
            ax.axvline(median_dur, color='green', linestyle='--', 
                      label=f'Median: {median_dur:.1f}')
            
            ax.set_title(f'Regime {regime} Duration Distribution')
            ax.set_xlabel('Duration (periods)')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # Hide unused subplots
        for idx in range(n_regimes, len(axes)):
            axes[idx].set_visible(False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_transition_report(self) -> str:
        """Generate comprehensive transition analysis report."""
        report = []
        report.append("=" * 50)
        report.append("REGIME TRANSITION ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 30)
        report.append(f"Total transitions: {len(self.transition_events)}")
        report.append(f"Unique regime pairs: {len(self.transition_counts)}")
        report.append("")
        
        # Most common transitions
        report.append("MOST COMMON TRANSITIONS")
        report.append("-" * 30)
        sorted_transitions = sorted(
            self.transition_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        for (from_r, to_r), count in sorted_transitions:
            prob = count / sum(c for (f, t), c in self.transition_counts.items() 
                             if f == from_r)
            report.append(f"Regime {from_r} -> {to_r}: {count} times ({prob:.1%})")
        report.append("")
        
        # Regime persistence
        report.append("REGIME PERSISTENCE")
        report.append("-" * 30)
        persistence = self._analyze_persistence()
        
        for regime, stats in persistence.items():
            report.append(f"Regime {regime}:")
            report.append(f"  Mean duration: {stats['mean_duration']:.1f} periods")
            report.append(f"  Stability score: {stats['stability_score']:.2f}")
        report.append("")
        
        # Patterns
        report.append("IDENTIFIED PATTERNS")
        report.append("-" * 30)
        patterns = self.identify_transition_patterns()
        
        if patterns['cycles']:
            report.append("Cyclical patterns found:")
            for cycle in patterns['cycles'][:3]:
                report.append(f"  {' -> '.join(map(str, cycle))}")
        else:
            report.append("No clear cyclical patterns identified")
        
        if patterns['seasonality']['is_seasonal']:
            report.append(f"Seasonal effects detected (p-value: {patterns['seasonality']['seasonality_p_value']:.3f})")
            report.append(f"Peak transition months: {patterns['seasonality']['peak_months']}")
        report.append("")
        
        return "\n".join(report)