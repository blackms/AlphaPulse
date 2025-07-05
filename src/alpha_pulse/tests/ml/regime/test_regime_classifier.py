"""
Tests for Real-time Market Regime Classification.

Tests cover regime classification, transition detection, and forecasting.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from collections import deque

from alpha_pulse.ml.regime.regime_classifier import (
    RegimeClassifier, RegimeInfo, RegimeTransition
)
from alpha_pulse.ml.regime.hmm_regime_detector import (
    GaussianHMM, HMMConfig, RegimeType, HMMState
)
from alpha_pulse.ml.regime.regime_features import RegimeFeatureEngineer


class TestRegimeInfo:
    """Test RegimeInfo dataclass."""
    
    def test_regime_info_creation(self):
        """Test creating RegimeInfo."""
        info = RegimeInfo(
            current_regime=1,
            regime_type=RegimeType.BULL,
            confidence=0.85,
            regime_probabilities=np.array([0.1, 0.85, 0.05]),
            entered_at=datetime.now(),
            duration=10,
            features={'volatility': 0.15, 'return': 0.001},
            transition_probability=0.15,
            expected_remaining_duration=50.0
        )
        
        assert info.current_regime == 1
        assert info.regime_type == RegimeType.BULL
        assert info.confidence == 0.85
        assert len(info.regime_probabilities) == 3
        assert info.duration == 10
        assert 'volatility' in info.features
        assert info.transition_probability == 0.15
        assert info.expected_remaining_duration == 50.0


class TestRegimeTransition:
    """Test RegimeTransition dataclass."""
    
    def test_regime_transition_creation(self):
        """Test creating RegimeTransition."""
        transition = RegimeTransition(
            from_regime=0,
            to_regime=1,
            timestamp=datetime.now(),
            probability=0.7,
            features_at_transition={'volatility': 0.2},
            confirmed=True,
            false_signal=False
        )
        
        assert transition.from_regime == 0
        assert transition.to_regime == 1
        assert transition.probability == 0.7
        assert transition.confirmed is True
        assert transition.false_signal is False


class TestRegimeClassifier:
    """Test RegimeClassifier functionality."""
    
    @pytest.fixture
    def mock_hmm_model(self):
        """Create mock HMM model."""
        model = Mock(spec=GaussianHMM)
        model.n_states = 3
        
        # Create mock states
        states = []
        for i in range(3):
            state = Mock(spec=HMMState)
            state.regime_type = [RegimeType.BULL, RegimeType.SIDEWAYS, RegimeType.BEAR][i]
            state.typical_duration = [100, 50, 80][i]
            state.transition_probs = np.array([0.8, 0.15, 0.05])
            states.append(state)
        
        model.states = states
        
        # Mock prediction methods
        model.predict_proba = Mock(return_value=np.array([
            [0.8, 0.15, 0.05],
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.5, 0.4, 0.1]
        ]))
        
        return model
    
    @pytest.fixture
    def mock_feature_engineer(self):
        """Create mock feature engineer."""
        engineer = Mock(spec=RegimeFeatureEngineer)
        
        # Mock feature extraction
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        features = pd.DataFrame(
            np.random.randn(100, 10),
            index=dates,
            columns=[f'feature_{i}' for i in range(10)]
        )
        engineer.extract_features = Mock(return_value=features)
        
        return engineer
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 101 + np.random.randn(100).cumsum(),
            'low': 99 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': 1e6 + 1e5 * np.random.randn(100)
        }, index=dates)
        return data
    
    def test_classifier_initialization(self, mock_hmm_model, mock_feature_engineer):
        """Test classifier initialization."""
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer,
            window_size=50,
            min_confidence=0.7
        )
        
        assert classifier.hmm_model == mock_hmm_model
        assert classifier.feature_engineer == mock_feature_engineer
        assert classifier.window_size == 50
        assert classifier.min_confidence == 0.7
        assert classifier.current_regime_info is None
        assert len(classifier.regime_history) == 0
        assert len(classifier.transition_history) == 0
    
    def test_classify_regime(self, mock_hmm_model, mock_feature_engineer, sample_market_data):
        """Test regime classification."""
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer,
            window_size=50,
            min_confidence=0.6
        )
        
        # Classify regime
        regime_info = classifier.classify(sample_market_data)
        
        assert regime_info is not None
        assert isinstance(regime_info, RegimeInfo)
        assert regime_info.current_regime == 0  # Highest probability
        assert regime_info.regime_type == RegimeType.BULL
        assert regime_info.confidence == 0.8
        assert len(regime_info.regime_probabilities) == 3
        
        # Check that features were extracted
        mock_feature_engineer.extract_features.assert_called_once()
    
    def test_regime_change_detection(self, mock_hmm_model, mock_feature_engineer, sample_market_data):
        """Test regime change detection."""
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer,
            window_size=50,
            min_confidence=0.6
        )
        
        # First classification
        regime_info1 = classifier.classify(sample_market_data)
        assert classifier.current_regime_info is not None
        
        # Mock different probabilities for regime change
        mock_hmm_model.predict_proba.return_value = np.array([
            [0.1, 0.2, 0.7],  # Changed to regime 2
            [0.1, 0.2, 0.7],
            [0.1, 0.2, 0.7],
            [0.1, 0.2, 0.7]
        ])
        
        # Second classification with regime change
        regime_info2 = classifier.classify(sample_market_data)
        
        # Should detect pending transition
        assert classifier.pending_transition is not None
        assert classifier.pending_transition.from_regime == 0
        assert classifier.pending_transition.to_regime == 2
        assert classifier.pending_transition.probability == 0.7
    
    def test_transition_confirmation(self, mock_hmm_model, mock_feature_engineer, sample_market_data):
        """Test regime transition confirmation."""
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer,
            window_size=50,
            min_confidence=0.6,
            transition_confirmation_window=3
        )
        
        # Initial classification
        classifier.classify(sample_market_data)
        
        # Change to new regime
        mock_hmm_model.predict_proba.return_value = np.array([
            [0.1, 0.1, 0.8],  # New regime
            [0.1, 0.1, 0.8],
            [0.1, 0.1, 0.8],
            [0.1, 0.1, 0.8]
        ])
        
        # Need multiple confirmations
        for i in range(4):
            classifier.classify(sample_market_data)
        
        # Transition should be confirmed
        assert len(classifier.transition_history) == 1
        assert classifier.transition_history[0].confirmed is True
        assert classifier.current_regime_info.current_regime == 2
    
    def test_false_signal_detection(self, mock_hmm_model, mock_feature_engineer, sample_market_data):
        """Test false signal detection."""
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer,
            window_size=50,
            min_confidence=0.6,
            transition_confirmation_window=5
        )
        
        # Initial classification
        classifier.classify(sample_market_data)
        
        # Brief change
        mock_hmm_model.predict_proba.return_value = np.array([
            [0.1, 0.1, 0.8],  # New regime briefly
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],  # Back to original
            [0.8, 0.1, 0.1]
        ])
        
        # Detect potential transition
        classifier.classify(sample_market_data)
        assert classifier.pending_transition is not None
        
        # Return to original regime
        mock_hmm_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1]
        ])
        
        for i in range(5):
            classifier.classify(sample_market_data)
        
        # Should mark as false signal
        assert any(t.false_signal for t in classifier.transition_history)
    
    def test_regime_forecast(self, mock_hmm_model, mock_feature_engineer, sample_market_data):
        """Test regime probability forecasting."""
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer
        )
        
        # Need current regime first
        classifier.classify(sample_market_data)
        
        # Mock transition matrix
        mock_hmm_model.trans_prob = np.array([
            [0.9, 0.08, 0.02],
            [0.05, 0.9, 0.05],
            [0.02, 0.08, 0.9]
        ])
        
        # Get forecast
        forecast = classifier.get_regime_forecast(horizon=5)
        
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 6  # Current + 5 horizon
        assert 'most_likely_regime' in forecast.columns
        assert 'max_probability' in forecast.columns
        
        # Check probability evolution
        for i in range(3):
            assert f'regime_{i}' in forecast.columns
            # Probabilities should sum to 1
            row_sum = sum(forecast.iloc[0][[f'regime_{j}' for j in range(3)]])
            assert abs(row_sum - 1.0) < 0.01
    
    def test_transition_statistics(self, mock_hmm_model, mock_feature_engineer, sample_market_data):
        """Test transition statistics calculation."""
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer
        )
        
        # Add some transitions to history
        for i in range(5):
            transition = RegimeTransition(
                from_regime=0,
                to_regime=1,
                timestamp=datetime.now() - timedelta(days=i),
                probability=0.7 + 0.05 * i,
                features_at_transition={'vol': 0.2},
                confirmed=True,
                false_signal=False
            )
            classifier.transition_history.append(transition)
        
        # Add a different transition
        classifier.transition_history.append(
            RegimeTransition(
                from_regime=1,
                to_regime=2,
                timestamp=datetime.now(),
                probability=0.8,
                features_at_transition={'vol': 0.3},
                confirmed=True,
                false_signal=False
            )
        )
        
        # Get statistics
        stats = classifier.get_transition_statistics()
        
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 2  # Two unique transition types
        assert 'from_regime' in stats.columns
        assert 'to_regime' in stats.columns
        assert 'count' in stats.columns
        assert 'avg_probability' in stats.columns
        
        # Check counts
        transition_0_1 = stats[(stats['from_regime'] == 0) & (stats['to_regime'] == 1)]
        assert len(transition_0_1) == 1
        assert transition_0_1['count'].values[0] == 5
    
    def test_regime_characteristics(self, mock_hmm_model, mock_feature_engineer):
        """Test regime characteristics extraction."""
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer
        )
        
        # Add some data to buffers
        for i in range(10):
            classifier.regime_buffer.append(0)
            classifier.feature_buffer.append(np.random.randn(5))
        
        for i in range(5):
            classifier.regime_buffer.append(1)
            classifier.feature_buffer.append(np.random.randn(5))
        
        # Get characteristics for regime 0
        chars = classifier.get_regime_characteristics(0)
        
        assert isinstance(chars, dict)
        assert 'regime_type' in chars
        assert 'model_mean' in chars
        assert 'model_covariance' in chars
        assert 'typical_duration' in chars
        assert 'transition_probabilities' in chars
        assert 'self_transition_prob' in chars
        
        # Should have empirical stats
        assert 'empirical_mean' in chars
        assert 'empirical_std' in chars
        assert chars['n_observations'] == 10
    
    def test_update_current_regime(self, mock_hmm_model, mock_feature_engineer, sample_market_data):
        """Test updating current regime information."""
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer
        )
        
        # Initial classification
        classifier.classify(sample_market_data)
        initial_duration = classifier.current_regime_info.duration
        
        # Update without regime change
        classifier.classify(sample_market_data)
        
        # Duration should increase
        assert classifier.current_regime_info.duration == initial_duration + 1
        
        # Transition probability should be updated
        assert classifier.current_regime_info.transition_probability > 0
        
        # Expected remaining duration should decrease
        assert classifier.current_regime_info.expected_remaining_duration < \
               mock_hmm_model.states[0].typical_duration
    
    def test_empty_features_handling(self, mock_hmm_model, mock_feature_engineer, sample_market_data):
        """Test handling of empty features."""
        # Mock empty features
        mock_feature_engineer.extract_features.return_value = pd.DataFrame()
        
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer
        )
        
        # Should handle gracefully
        regime_info = classifier.classify(sample_market_data)
        
        # Should return existing regime info (None in this case)
        assert regime_info is None
    
    def test_regime_history_tracking(self, mock_hmm_model, mock_feature_engineer, sample_market_data):
        """Test regime history tracking."""
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer
        )
        
        # Multiple classifications
        for i in range(3):
            # Change regime each time
            mock_hmm_model.predict_proba.return_value = np.array([
                [0.1, 0.1, 0.8] if i == 0 else [0.8, 0.1, 0.1] if i == 1 else [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8] if i == 0 else [0.8, 0.1, 0.1] if i == 1 else [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8] if i == 0 else [0.8, 0.1, 0.1] if i == 1 else [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8] if i == 0 else [0.8, 0.1, 0.1] if i == 1 else [0.1, 0.8, 0.1]
            ])
            
            # Force immediate transition (skip confirmation)
            classifier.transition_confirmation_window = 1
            classifier.classify(sample_market_data)
        
        # Should have regime history
        assert len(classifier.regime_history) >= 2
        
        # Check history contains RegimeInfo objects
        for regime_info in classifier.regime_history:
            assert isinstance(regime_info, RegimeInfo)
    
    def test_buffer_management(self, mock_hmm_model, mock_feature_engineer, sample_market_data):
        """Test buffer size management."""
        window_size = 10
        classifier = RegimeClassifier(
            hmm_model=mock_hmm_model,
            feature_engineer=mock_feature_engineer,
            window_size=window_size
        )
        
        # Add more than window size
        for i in range(window_size + 5):
            classifier.classify(sample_market_data)
        
        # Buffers should not exceed window size
        assert len(classifier.feature_buffer) <= window_size
        assert len(classifier.probability_buffer) <= window_size
        assert len(classifier.regime_buffer) <= window_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])