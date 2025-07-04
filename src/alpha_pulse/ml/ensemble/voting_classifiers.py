"""Voting-based ensemble methods for signal combination."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

from alpha_pulse.ml.ensemble.ensemble_manager import BaseEnsemble, AgentSignal, EnsembleSignal

logger = logging.getLogger(__name__)


class HardVotingEnsemble(BaseEnsemble):
    """Hard voting ensemble with majority voting."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vote_threshold = config.get('vote_threshold', 0.0)
        self.tie_breaker = config.get('tie_breaker', 'confidence')  # confidence, random, neutral
        self.min_votes = config.get('min_votes', 3)
        self.signal_bins = config.get('signal_bins', 3)  # buy, neutral, sell
        
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Train voting weights based on historical performance."""
        # Calculate accuracy for each agent
        agent_accuracies = {}
        
        for agent_id in set(s.agent_id for s in signals):
            agent_signals = [s for s in signals if s.agent_id == agent_id]
            if len(agent_signals) < 10:
                continue
                
            # Calculate directional accuracy
            correct = 0
            for i, signal in enumerate(agent_signals):
                if i < len(outcomes):
                    predicted_direction = np.sign(signal.signal)
                    actual_direction = np.sign(outcomes[i])
                    if predicted_direction == actual_direction:
                        correct += 1
                        
            accuracy = correct / len(agent_signals)
            agent_accuracies[agent_id] = accuracy
            
        # Set weights based on accuracy
        if agent_accuracies:
            min_acc = min(agent_accuracies.values())
            max_acc = max(agent_accuracies.values())
            
            for agent_id, accuracy in agent_accuracies.items():
                # Normalize to [0.5, 2.0] range
                if max_acc > min_acc:
                    normalized = (accuracy - min_acc) / (max_acc - min_acc)
                    self.agent_weights[agent_id] = 0.5 + 1.5 * normalized
                else:
                    self.agent_weights[agent_id] = 1.0
                    
        self.is_fitted = True
        logger.info(f"Fitted hard voting ensemble with {len(self.agent_weights)} agents")
        
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate ensemble prediction using hard voting."""
        if len(signals) < self.min_votes:
            return self._create_neutral_signal(signals, "insufficient_votes")
            
        # Discretize signals into bins
        votes = []
        weights = []
        confidences = []
        
        for signal in signals:
            vote = self._discretize_signal(signal.signal)
            weight = self.agent_weights.get(signal.agent_id, 1.0)
            
            votes.append(vote)
            weights.append(weight)
            confidences.append(signal.confidence)
            
        # Weighted voting
        vote_counts = Counter()
        for vote, weight in zip(votes, weights):
            vote_counts[vote] += weight
            
        # Get majority vote
        if vote_counts:
            majority_vote = max(vote_counts, key=vote_counts.get)
            total_votes = sum(vote_counts.values())
            vote_confidence = vote_counts[majority_vote] / total_votes
            
            # Check for ties
            max_count = vote_counts[majority_vote]
            tied_votes = [v for v, c in vote_counts.items() if c == max_count]
            
            if len(tied_votes) > 1:
                majority_vote = self._break_tie(tied_votes, signals)
                
            # Convert discrete vote back to continuous signal
            ensemble_signal = self._vote_to_signal(majority_vote)
            
            # Calculate ensemble confidence
            ensemble_confidence = vote_confidence * np.mean(confidences)
            
            return EnsembleSignal(
                timestamp=datetime.now(),
                signal=ensemble_signal,
                confidence=ensemble_confidence,
                contributing_agents=[s.agent_id for s in signals],
                weights={s.agent_id: self.agent_weights.get(s.agent_id, 1.0) for s in signals},
                metadata={
                    'method': 'hard_voting',
                    'vote_distribution': dict(vote_counts),
                    'majority_vote': majority_vote,
                    'vote_confidence': vote_confidence
                }
            )
        else:
            return self._create_neutral_signal(signals, "no_votes")
            
    def _discretize_signal(self, signal: float) -> int:
        """Convert continuous signal to discrete vote."""
        if self.signal_bins == 3:
            if signal < -self.vote_threshold:
                return -1  # Sell
            elif signal > self.vote_threshold:
                return 1   # Buy
            else:
                return 0   # Neutral
        else:
            # More granular binning
            bins = np.linspace(-1, 1, self.signal_bins + 1)
            return int(np.digitize(signal, bins) - self.signal_bins // 2)
            
    def _vote_to_signal(self, vote: int) -> float:
        """Convert discrete vote back to continuous signal."""
        if self.signal_bins == 3:
            return float(vote) * 0.5  # -0.5, 0, 0.5
        else:
            return float(vote) / (self.signal_bins // 2)
            
    def _break_tie(self, tied_votes: List[int], signals: List[AgentSignal]) -> int:
        """Break voting ties using specified method."""
        if self.tie_breaker == 'confidence':
            # Use highest confidence signal among tied votes
            best_confidence = -1
            best_vote = 0
            
            for signal in signals:
                vote = self._discretize_signal(signal.signal)
                if vote in tied_votes and signal.confidence > best_confidence:
                    best_confidence = signal.confidence
                    best_vote = vote
                    
            return best_vote
        elif self.tie_breaker == 'random':
            return np.random.choice(tied_votes)
        else:  # neutral
            return 0
            
    def _create_neutral_signal(self, signals: List[AgentSignal], reason: str) -> EnsembleSignal:
        """Create neutral signal when voting cannot proceed."""
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=0.0,
            confidence=0.0,
            contributing_agents=[s.agent_id for s in signals],
            weights={},
            metadata={'method': 'hard_voting', 'reason': reason}
        )
        
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update agent weights based on performance."""
        for agent_id, metric in performance_metrics.items():
            if agent_id in self.agent_weights:
                # Exponential moving average update
                alpha = 0.1
                self.agent_weights[agent_id] = (
                    (1 - alpha) * self.agent_weights[agent_id] + 
                    alpha * metric
                )


class SoftVotingEnsemble(BaseEnsemble):
    """Soft voting ensemble with probability-weighted averaging."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.temperature = config.get('temperature', 1.0)
        self.use_bayesian = config.get('use_bayesian', True)
        self.entropy_weighting = config.get('entropy_weighting', False)
        self.min_confidence = config.get('min_confidence', 0.3)
        
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Train soft voting parameters."""
        # Calculate calibration scores for each agent
        agent_calibration = {}
        
        for agent_id in set(s.agent_id for s in signals):
            agent_signals = [s for s in signals if s.agent_id == agent_id]
            if len(agent_signals) < 20:
                continue
                
            # Calculate calibration score (how well confidence matches accuracy)
            confidences = [s.confidence for s in agent_signals]
            predictions = [s.signal for s in agent_signals]
            
            # Group by confidence bins and calculate actual accuracy
            confidence_bins = np.linspace(0, 1, 11)
            calibration_score = 0.0
            
            for i in range(len(confidence_bins) - 1):
                bin_mask = [
                    confidence_bins[i] <= c < confidence_bins[i+1] 
                    for c in confidences
                ]
                
                if sum(bin_mask) > 0:
                    bin_predictions = [p for p, m in zip(predictions, bin_mask) if m]
                    bin_outcomes = [o for o, m in zip(outcomes[:len(predictions)], bin_mask) if m]
                    
                    if bin_outcomes:
                        # Calculate accuracy in this confidence bin
                        accuracy = sum(
                            1 for p, o in zip(bin_predictions, bin_outcomes)
                            if np.sign(p) == np.sign(o)
                        ) / len(bin_outcomes)
                        
                        expected_confidence = (confidence_bins[i] + confidence_bins[i+1]) / 2
                        calibration_error = abs(accuracy - expected_confidence)
                        calibration_score += calibration_error * len(bin_outcomes)
                        
            if len(agent_signals) > 0:
                agent_calibration[agent_id] = 1.0 / (1.0 + calibration_score / len(agent_signals))
                
        # Set weights based on calibration
        for agent_id, calibration in agent_calibration.items():
            self.agent_weights[agent_id] = calibration
            
        self.is_fitted = True
        logger.info(f"Fitted soft voting ensemble with {len(self.agent_weights)} agents")
        
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate ensemble prediction using soft voting."""
        if not signals:
            return self._create_neutral_signal([])
            
        # Filter signals by minimum confidence
        valid_signals = [s for s in signals if s.confidence >= self.min_confidence]
        
        if not valid_signals:
            return self._create_neutral_signal(signals)
            
        # Calculate probabilities and weights
        probabilities = []
        weights = []
        
        for signal in valid_signals:
            # Convert signal to probability using sigmoid
            prob = self._signal_to_probability(signal.signal, signal.confidence)
            
            # Get agent weight
            agent_weight = self.agent_weights.get(signal.agent_id, 1.0)
            
            # Apply entropy weighting if enabled
            if self.entropy_weighting:
                entropy = self._calculate_entropy(prob)
                entropy_weight = 1.0 - entropy / np.log(2)  # Max entropy = log(2) for binary
                agent_weight *= entropy_weight
                
            probabilities.append(prob)
            weights.append(agent_weight * signal.confidence)
            
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Calculate weighted average probability
        if self.use_bayesian:
            ensemble_prob = self._bayesian_averaging(probabilities, weights)
        else:
            ensemble_prob = np.average(probabilities, weights=weights)
            
        # Convert back to signal
        ensemble_signal = self._probability_to_signal(ensemble_prob)
        
        # Calculate confidence as weighted standard deviation
        signal_variance = np.average(
            [(p - ensemble_prob)**2 for p in probabilities],
            weights=weights
        )
        ensemble_confidence = 1.0 / (1.0 + np.sqrt(signal_variance))
        
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=ensemble_signal,
            confidence=ensemble_confidence,
            contributing_agents=[s.agent_id for s in valid_signals],
            weights={s.agent_id: w for s, w in zip(valid_signals, weights)},
            metadata={
                'method': 'soft_voting',
                'ensemble_probability': float(ensemble_prob),
                'signal_variance': float(signal_variance),
                'temperature': self.temperature
            }
        )
        
    def _signal_to_probability(self, signal: float, confidence: float) -> float:
        """Convert signal to probability using temperature-scaled sigmoid."""
        # Apply temperature scaling
        scaled_signal = signal / self.temperature
        
        # Sigmoid transformation
        prob = 1.0 / (1.0 + np.exp(-scaled_signal * 2))
        
        # Adjust by confidence (pull towards 0.5 for low confidence)
        return confidence * prob + (1 - confidence) * 0.5
        
    def _probability_to_signal(self, prob: float) -> float:
        """Convert probability back to signal."""
        # Inverse sigmoid
        prob = np.clip(prob, 0.001, 0.999)
        signal = np.log(prob / (1 - prob)) / 2
        
        # Apply temperature scaling
        return signal * self.temperature
        
    def _calculate_entropy(self, prob: float) -> float:
        """Calculate entropy of binary probability."""
        if prob <= 0 or prob >= 1:
            return 0.0
        return -prob * np.log(prob) - (1 - prob) * np.log(1 - prob)
        
    def _bayesian_averaging(self, probabilities: List[float], weights: List[float]) -> float:
        """Bayesian model averaging of probabilities."""
        # Use log-odds for numerical stability
        log_odds = []
        
        for prob in probabilities:
            prob = np.clip(prob, 0.001, 0.999)
            log_odds.append(np.log(prob / (1 - prob)))
            
        # Weighted average in log-odds space
        avg_log_odds = np.average(log_odds, weights=weights)
        
        # Convert back to probability
        return 1.0 / (1.0 + np.exp(-avg_log_odds))
        
    def _create_neutral_signal(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Create neutral signal when voting cannot proceed."""
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=0.0,
            confidence=0.0,
            contributing_agents=[s.agent_id for s in signals],
            weights={},
            metadata={'method': 'soft_voting', 'reason': 'insufficient_confidence'}
        )
        
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update agent weights based on performance."""
        for agent_id, metric in performance_metrics.items():
            if agent_id in self.agent_weights:
                # Multiplicative weight update
                self.agent_weights[agent_id] *= (1 + 0.1 * (metric - 0.5))
                
        # Normalize weights
        total_weight = sum(self.agent_weights.values())
        if total_weight > 0:
            for agent_id in self.agent_weights:
                self.agent_weights[agent_id] /= total_weight


class WeightedMajorityVoting(BaseEnsemble):
    """Weighted majority voting with consensus thresholds."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.consensus_threshold = config.get('consensus_threshold', 0.6)
        self.super_majority = config.get('super_majority', 0.75)
        self.veto_threshold = config.get('veto_threshold', 0.9)
        self.use_rank_weights = config.get('use_rank_weights', True)
        
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Train weighted majority voting."""
        # Calculate precision for each agent
        agent_precision = {}
        
        for agent_id in set(s.agent_id for s in signals):
            agent_signals = [s for s in signals if s.agent_id == agent_id]
            
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            for i, signal in enumerate(agent_signals):
                if i < len(outcomes):
                    predicted = signal.signal > 0
                    actual = outcomes[i] > 0
                    
                    if predicted and actual:
                        true_positives += 1
                    elif predicted and not actual:
                        false_positives += 1
                    elif not predicted and actual:
                        false_negatives += 1
                    else:
                        true_negatives += 1
                        
            # Calculate F1 score as weight
            if true_positives > 0:
                precision = true_positives / (true_positives + false_positives + 1e-8)
                recall = true_positives / (true_positives + false_negatives + 1e-8)
                f1_score = 2 * precision * recall / (precision + recall + 1e-8)
                agent_precision[agent_id] = f1_score
            else:
                agent_precision[agent_id] = 0.5
                
        # Set weights based on F1 scores
        if self.use_rank_weights:
            # Convert to rank-based weights
            sorted_agents = sorted(agent_precision.items(), key=lambda x: x[1], reverse=True)
            rank_weights = np.linspace(2.0, 0.5, len(sorted_agents))
            
            for i, (agent_id, _) in enumerate(sorted_agents):
                self.agent_weights[agent_id] = rank_weights[i]
        else:
            for agent_id, f1_score in agent_precision.items():
                self.agent_weights[agent_id] = f1_score
                
        self.is_fitted = True
        
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate prediction using weighted majority voting."""
        if not signals:
            return self._create_neutral_signal([])
            
        # Calculate weighted votes
        buy_weight = 0.0
        sell_weight = 0.0
        neutral_weight = 0.0
        total_weight = 0.0
        
        agent_votes = {}
        
        for signal in signals:
            weight = self.agent_weights.get(signal.agent_id, 1.0) * signal.confidence
            
            if signal.signal > 0.1:
                buy_weight += weight
                agent_votes[signal.agent_id] = 'buy'
            elif signal.signal < -0.1:
                sell_weight += weight
                agent_votes[signal.agent_id] = 'sell'
            else:
                neutral_weight += weight
                agent_votes[signal.agent_id] = 'neutral'
                
            total_weight += weight
            
        if total_weight == 0:
            return self._create_neutral_signal(signals)
            
        # Normalize weights
        buy_ratio = buy_weight / total_weight
        sell_ratio = sell_weight / total_weight
        neutral_ratio = neutral_weight / total_weight
        
        # Check for super majority or veto
        if buy_ratio >= self.super_majority:
            ensemble_signal = 0.8
            decision = 'strong_buy'
        elif sell_ratio >= self.super_majority:
            ensemble_signal = -0.8
            decision = 'strong_sell'
        elif buy_ratio >= self.consensus_threshold:
            ensemble_signal = 0.5
            decision = 'buy'
        elif sell_ratio >= self.consensus_threshold:
            ensemble_signal = -0.5
            decision = 'sell'
        else:
            ensemble_signal = 0.0
            decision = 'neutral'
            
        # Check for veto conditions
        if any(s.confidence >= self.veto_threshold and abs(s.signal) > 0.8 for s in signals):
            # High confidence contrarian signal can veto
            contrarian_signals = [
                s for s in signals 
                if s.confidence >= self.veto_threshold and np.sign(s.signal) != np.sign(ensemble_signal)
            ]
            if contrarian_signals:
                ensemble_signal *= 0.5  # Reduce strength
                decision = f"{decision}_vetoed"
                
        # Calculate consensus strength
        consensus_strength = max(buy_ratio, sell_ratio, neutral_ratio)
        
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=ensemble_signal,
            confidence=consensus_strength,
            contributing_agents=[s.agent_id for s in signals],
            weights={s.agent_id: self.agent_weights.get(s.agent_id, 1.0) for s in signals},
            metadata={
                'method': 'weighted_majority',
                'decision': decision,
                'buy_ratio': float(buy_ratio),
                'sell_ratio': float(sell_ratio),
                'neutral_ratio': float(neutral_ratio),
                'consensus_strength': float(consensus_strength),
                'agent_votes': agent_votes
            }
        )
        
    def _create_neutral_signal(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Create neutral signal."""
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=0.0,
            confidence=0.0,
            contributing_agents=[s.agent_id for s in signals],
            weights={},
            metadata={'method': 'weighted_majority', 'reason': 'no_consensus'}
        )
        
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update agent weights based on performance."""
        # Rank-based update if enabled
        if self.use_rank_weights:
            sorted_agents = sorted(performance_metrics.items(), key=lambda x: x[1], reverse=True)
            rank_weights = np.linspace(2.0, 0.5, len(sorted_agents))
            
            for i, (agent_id, _) in enumerate(sorted_agents):
                if agent_id in self.agent_weights:
                    self.agent_weights[agent_id] = rank_weights[i]
        else:
            for agent_id, metric in performance_metrics.items():
                if agent_id in self.agent_weights:
                    self.agent_weights[agent_id] = max(0.1, metric)