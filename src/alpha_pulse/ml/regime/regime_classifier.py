"""
Real-time Market Regime Classification and Analysis.

This module provides real-time regime classification capabilities with
confidence estimation and regime characteristic analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import deque
import logging
from datetime import datetime, timedelta

from .hmm_regime_detector import GaussianHMM, RegimeType, HMMState
from .regime_features import RegimeFeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class RegimeInfo:
    """Information about current and historical regimes."""
    current_regime: int
    regime_type: RegimeType
    confidence: float
    regime_probabilities: np.ndarray
    entered_at: datetime
    duration: int
    features: Dict[str, float]
    transition_probability: float = 0.0
    expected_remaining_duration: float = 0.0


@dataclass
class RegimeTransition:
    """Information about regime transitions."""
    from_regime: int
    to_regime: int
    timestamp: datetime
    probability: float
    features_at_transition: Dict[str, float]
    confirmed: bool = False
    false_signal: bool = False


class RegimeClassifier:
    """Real-time regime classification with confidence estimation."""
    
    def __init__(self, 
                 hmm_model: GaussianHMM,
                 feature_engineer: RegimeFeatureEngineer,
                 window_size: int = 100,
                 min_confidence: float = 0.6):
        self.hmm_model = hmm_model
        self.feature_engineer = feature_engineer
        self.window_size = window_size
        self.min_confidence = min_confidence
        
        # State tracking
        self.current_regime_info: Optional[RegimeInfo] = None
        self.regime_history: List[RegimeInfo] = []
        self.transition_history: List[RegimeTransition] = []
        
        # Feature and probability buffers
        self.feature_buffer = deque(maxlen=window_size)
        self.probability_buffer = deque(maxlen=window_size)
        self.regime_buffer = deque(maxlen=window_size)
        
        # Transition tracking
        self.pending_transition: Optional[RegimeTransition] = None
        self.transition_confirmation_window = 5
        
    def classify(self, 
                 market_data: pd.DataFrame,
                 additional_data: Optional[Dict[str, pd.DataFrame]] = None,
                 timestamp: Optional[datetime] = None) -> RegimeInfo:
        """
        Classify current market regime.
        
        Args:
            market_data: Latest market data
            additional_data: Optional additional data (VIX, sentiment, etc.)
            timestamp: Current timestamp
            
        Returns:
            RegimeInfo with current regime classification
        """
        timestamp = timestamp or datetime.now()
        
        # Extract features
        features = self.feature_engineer.extract_features(market_data, additional_data)
        if features.empty:
            logger.warning("No features extracted")
            return self.current_regime_info
        
        # Get latest features
        latest_features = features.iloc[-1:].values
        feature_dict = dict(zip(features.columns, latest_features[0]))
        
        # Add to buffer
        self.feature_buffer.append(latest_features)
        
        # Predict regime probabilities
        if len(self.feature_buffer) >= 1:
            # Use recent history for more stable predictions
            recent_features = np.vstack(list(self.feature_buffer)[-20:])
            
            # Get state probabilities
            state_probs = self.hmm_model.predict_proba(recent_features)
            latest_probs = state_probs[-1]
            
            # Get most likely state
            current_regime = np.argmax(latest_probs)
            confidence = latest_probs[current_regime]
            
            # Add to buffers
            self.probability_buffer.append(latest_probs)
            self.regime_buffer.append(current_regime)
            
            # Check for regime change
            if self.current_regime_info is None or current_regime != self.current_regime_info.current_regime:
                if confidence >= self.min_confidence:
                    self._handle_regime_change(
                        current_regime, latest_probs, feature_dict, timestamp
                    )
            else:
                # Update current regime info
                self._update_current_regime(latest_probs, feature_dict)
            
            # Check pending transitions
            self._check_pending_transitions(current_regime, confidence)
        
        return self.current_regime_info
    
    def _handle_regime_change(self,
                            new_regime: int,
                            probabilities: np.ndarray,
                            features: Dict[str, float],
                            timestamp: datetime):
        """Handle regime transition."""
        # Create transition record
        if self.current_regime_info is not None:
            transition = RegimeTransition(
                from_regime=self.current_regime_info.current_regime,
                to_regime=new_regime,
                timestamp=timestamp,
                probability=probabilities[new_regime],
                features_at_transition=features
            )
            
            # Set as pending for confirmation
            self.pending_transition = transition
            
            logger.info(
                f"Potential regime change detected: "
                f"{self.current_regime_info.regime_type.value} -> "
                f"{self.hmm_model.states[new_regime].regime_type.value} "
                f"(confidence: {probabilities[new_regime]:.2f})"
            )
        else:
            # First regime detection
            self._set_new_regime(new_regime, probabilities, features, timestamp)
    
    def _check_pending_transitions(self, current_regime: int, confidence: float):
        """Check and confirm pending transitions."""
        if self.pending_transition is None:
            return
        
        # Count consecutive confirmations
        recent_regimes = list(self.regime_buffer)[-self.transition_confirmation_window:]
        confirmation_count = sum(r == self.pending_transition.to_regime for r in recent_regimes)
        
        if confirmation_count >= self.transition_confirmation_window * 0.8:
            # Confirm transition
            self.pending_transition.confirmed = True
            self.transition_history.append(self.pending_transition)
            
            # Set new regime
            self._set_new_regime(
                self.pending_transition.to_regime,
                list(self.probability_buffer)[-1],
                self.pending_transition.features_at_transition,
                self.pending_transition.timestamp
            )
            
            logger.info(f"Regime transition confirmed: {self.pending_transition.from_regime} -> {self.pending_transition.to_regime}")
            self.pending_transition = None
            
        elif confirmation_count < self.transition_confirmation_window * 0.2:
            # False signal
            self.pending_transition.false_signal = True
            self.transition_history.append(self.pending_transition)
            logger.info("Regime transition cancelled - false signal")
            self.pending_transition = None
    
    def _set_new_regime(self,
                       regime: int,
                       probabilities: np.ndarray,
                       features: Dict[str, float],
                       timestamp: datetime):
        """Set new regime as current."""
        # Archive current regime
        if self.current_regime_info is not None:
            self.regime_history.append(self.current_regime_info)
        
        # Create new regime info
        regime_state = self.hmm_model.states[regime]
        
        self.current_regime_info = RegimeInfo(
            current_regime=regime,
            regime_type=regime_state.regime_type,
            confidence=probabilities[regime],
            regime_probabilities=probabilities,
            entered_at=timestamp,
            duration=1,
            features=features,
            expected_remaining_duration=regime_state.typical_duration
        )
    
    def _update_current_regime(self,
                             probabilities: np.ndarray,
                             features: Dict[str, float]):
        """Update current regime information."""
        if self.current_regime_info is None:
            return
        
        self.current_regime_info.duration += 1
        self.current_regime_info.confidence = probabilities[self.current_regime_info.current_regime]
        self.current_regime_info.regime_probabilities = probabilities
        self.current_regime_info.features = features
        
        # Update expected remaining duration
        regime_state = self.hmm_model.states[self.current_regime_info.current_regime]
        self.current_regime_info.expected_remaining_duration = max(
            1, regime_state.typical_duration - self.current_regime_info.duration
        )
        
        # Calculate transition probability
        trans_probs = regime_state.transition_probs.copy()
        trans_probs[self.current_regime_info.current_regime] = 0  # Exclude self-transition
        self.current_regime_info.transition_probability = trans_probs.max()
    
    def get_regime_forecast(self, horizon: int = 10) -> pd.DataFrame:
        """
        Forecast regime probabilities over given horizon.
        
        Args:
            horizon: Forecast horizon in periods
            
        Returns:
            DataFrame with forecasted regime probabilities
        """
        if self.current_regime_info is None:
            return pd.DataFrame()
        
        # Start with current state
        current_probs = self.current_regime_info.regime_probabilities
        
        # Forecast using transition matrix powers
        trans_matrix = self.hmm_model.trans_prob
        forecasts = [current_probs]
        
        for h in range(1, horizon + 1):
            # Multiply by transition matrix
            next_probs = current_probs @ np.linalg.matrix_power(trans_matrix, h)
            forecasts.append(next_probs)
        
        # Create DataFrame
        forecast_df = pd.DataFrame(
            forecasts,
            columns=[f"regime_{i}" for i in range(len(current_probs))],
            index=range(horizon + 1)
        )
        
        # Add most likely regime
        forecast_df['most_likely_regime'] = forecast_df.values.argmax(axis=1)
        forecast_df['max_probability'] = forecast_df.iloc[:, :-1].max(axis=1)
        
        return forecast_df
    
    def get_transition_statistics(self) -> pd.DataFrame:
        """Get statistics about regime transitions."""
        if not self.transition_history:
            return pd.DataFrame()
        
        stats = []
        
        # Group by transition type
        transition_types = {}
        for trans in self.transition_history:
            if trans.false_signal:
                continue
                
            key = (trans.from_regime, trans.to_regime)
            if key not in transition_types:
                transition_types[key] = []
            transition_types[key].append(trans)
        
        # Calculate statistics
        for (from_regime, to_regime), transitions in transition_types.items():
            from_type = self.hmm_model.states[from_regime].regime_type.value
            to_type = self.hmm_model.states[to_regime].regime_type.value
            
            stats.append({
                'from_regime': from_regime,
                'to_regime': to_regime,
                'from_type': from_type,
                'to_type': to_type,
                'count': len(transitions),
                'avg_probability': np.mean([t.probability for t in transitions]),
                'false_signal_rate': sum(t.false_signal for t in self.transition_history 
                                        if t.from_regime == from_regime and t.to_regime == to_regime) / 
                                    max(1, len([t for t in self.transition_history 
                                              if t.from_regime == from_regime and t.to_regime == to_regime]))
            })
        
        return pd.DataFrame(stats)
    
    def get_regime_characteristics(self, regime: Optional[int] = None) -> Dict[str, Any]:
        """Get characteristics of a specific regime or current regime."""
        if regime is None:
            if self.current_regime_info is None:
                return {}
            regime = self.current_regime_info.current_regime
        
        if regime >= len(self.hmm_model.states):
            return {}
        
        state = self.hmm_model.states[regime]
        
        # Calculate empirical statistics from history
        regime_features = []
        for i, r in enumerate(self.regime_buffer):
            if r == regime and i < len(self.feature_buffer):
                regime_features.append(list(self.feature_buffer)[i])
        
        characteristics = {
            'regime_type': state.regime_type.value,
            'model_mean': state.mean.tolist(),
            'model_covariance': state.covariance.tolist(),
            'typical_duration': state.typical_duration,
            'transition_probabilities': state.transition_probs.tolist(),
            'self_transition_prob': state.transition_probs[regime]
        }
        
        if regime_features:
            regime_features = np.vstack(regime_features)
            characteristics.update({
                'empirical_mean': regime_features.mean(axis=0).tolist(),
                'empirical_std': regime_features.std(axis=0).tolist(),
                'n_observations': len(regime_features)
            })
        
        return characteristics