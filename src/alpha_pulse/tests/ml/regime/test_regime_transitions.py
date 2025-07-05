"""
Tests for Regime Transition Analysis and Forecasting.

Tests cover transition probability estimation, pattern identification, and forecasting.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

from alpha_pulse.ml.regime.regime_transitions import (
    RegimeTransitionAnalyzer, TransitionEvent
)
from alpha_pulse.ml.regime.hmm_regime_detector import RegimeType


class TestTransitionEvent:
    """Test TransitionEvent dataclass."""
    
    def test_transition_event_creation(self):
        """Test creating TransitionEvent."""
        event = TransitionEvent(
            from_regime=0,
            to_regime=1,
            timestamp=datetime.now(),
            duration_in_from=50,
            market_conditions={'volatility': 0.15, 'return': 0.001},
            trigger_factors=['volatility_spike', 'extreme_return'],
            transition_speed=0.8,
            stability_score=0.9
        )
        
        assert event.from_regime == 0
        assert event.to_regime == 1
        assert event.duration_in_from == 50
        assert 'volatility' in event.market_conditions
        assert 'volatility_spike' in event.trigger_factors
        assert event.transition_speed == 0.8
        assert event.stability_score == 0.9


class TestRegimeTransitionAnalyzer:
    """Test RegimeTransitionAnalyzer functionality."""
    
    @pytest.fixture
    def sample_transition_events(self):
        """Create sample transition events."""
        events = []
        base_time = datetime.now() - timedelta(days=365)
        
        # Create a pattern: 0 -> 1 -> 2 -> 0 (cycle)
        transitions = [
            (0, 1, 100), (1, 2, 50), (2, 0, 80),
            (0, 1, 120), (1, 2, 45), (2, 0, 75),
            (0, 1, 110), (1, 0, 60),  # Different path
            (0, 2, 90),  # Direct transition
        ]
        
        for i, (from_r, to_r, duration) in enumerate(transitions):
            event = TransitionEvent(
                from_regime=from_r,
                to_regime=to_r,
                timestamp=base_time + timedelta(days=i*30),
                duration_in_from=duration,
                market_conditions={
                    'volatility': 0.1 + 0.05 * to_r,
                    'return': 0.001 - 0.0005 * to_r,
                    'vix': 15 + 5 * to_r
                },
                trigger_factors=['volatility_spike'] if to_r > from_r else ['low_volatility'],
                transition_speed=0.5 + 0.1 * i,
                stability_score=0.8 - 0.05 * to_r
            )
            events.append(event)
        
        return events
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = RegimeTransitionAnalyzer(lookback_window=252)
        
        assert analyzer.lookback_window == 252
        assert len(analyzer.transition_events) == 0
        assert len(analyzer.regime_durations) == 0
        assert analyzer.transition_counts == {}
        assert analyzer.transition_features == {}
    
    def test_add_transition(self, sample_transition_events):
        """Test adding transition events."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        assert len(analyzer.transition_events) == len(sample_transition_events)
        
        # Check duration tracking
        assert 0 in analyzer.regime_durations
        assert 1 in analyzer.regime_durations
        assert len(analyzer.regime_durations[0]) == 4  # 4 transitions from regime 0
        
        # Check transition counts
        assert (0, 1) in analyzer.transition_counts
        assert analyzer.transition_counts[(0, 1)] == 3
        assert analyzer.transition_counts[(1, 2)] == 2
        
        # Check feature tracking
        assert (0, 1) in analyzer.transition_features
        assert len(analyzer.transition_features[(0, 1)]) == 3
    
    def test_base_transition_probabilities(self, sample_transition_events):
        """Test base transition probability calculation."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        # Get base probabilities from regime 0
        probs = analyzer._get_base_transition_probs(0)
        
        assert isinstance(probs, dict)
        assert sum(probs.values()) == pytest.approx(1.0)
        
        # Check specific probabilities
        assert probs[1] == 3/5  # 3 out of 5 transitions from 0 go to 1
        assert probs[2] == 1/5  # 1 out of 5 transitions from 0 go to 2
        assert probs[0] == 1/5  # 1 self-transition (from 2->0->1 counts)
    
    def test_duration_adjustment(self, sample_transition_events):
        """Test duration-based probability adjustment."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        base_probs = {0: 0.2, 1: 0.6, 2: 0.2}
        
        # Test with short duration (should decrease transition probability)
        adjusted_short = analyzer._adjust_for_duration(base_probs, 0, 50)
        
        # Test with long duration (should increase transition probability)
        adjusted_long = analyzer._adjust_for_duration(base_probs, 0, 150)
        
        # Long duration should have higher transition probabilities
        assert adjusted_long[1] > adjusted_short[1]
        assert adjusted_long[2] > adjusted_short[2]
        
        # Self-transition should be opposite
        assert adjusted_long[0] < adjusted_short[0]
    
    def test_condition_adjustment(self, sample_transition_events):
        """Test market condition-based probability adjustment."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        base_probs = {0: 0.2, 1: 0.6, 2: 0.2}
        
        # Test with similar conditions to historical transitions
        similar_conditions = {
            'volatility': 0.15,  # Similar to 0->1 transitions
            'return': 0.0005,
            'vix': 20
        }
        
        adjusted = analyzer._adjust_for_conditions(base_probs, 0, similar_conditions)
        
        # Should increase probability for transitions with similar conditions
        assert adjusted[1] > base_probs[1]
    
    def test_transition_probability_estimation(self, sample_transition_events):
        """Test complete transition probability estimation."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        # Estimate probabilities
        probs = analyzer.estimate_transition_probabilities(
            current_regime=0,
            current_duration=110,  # Slightly above average
            market_features={'volatility': 0.15, 'return': 0.0005, 'vix': 20}
        )
        
        assert isinstance(probs, dict)
        assert sum(probs.values()) == pytest.approx(1.0)
        assert all(0 <= p <= 1 for p in probs.values())
    
    def test_regime_duration_forecast(self, sample_transition_events):
        """Test regime duration forecasting."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        # Forecast duration for regime 0
        forecast = analyzer.forecast_regime_duration(
            current_regime=0,
            current_duration=50
        )
        
        assert isinstance(forecast, dict)
        assert 'expected' in forecast
        assert 'lower_95' in forecast
        assert 'upper_95' in forecast
        assert 'probability_ending_soon' in forecast
        
        # Check reasonable values
        assert forecast['expected'] > 0
        assert forecast['lower_95'] < forecast['expected']
        assert forecast['upper_95'] > forecast['expected']
        assert 0 <= forecast['probability_ending_soon'] <= 1
    
    def test_cycle_identification(self, sample_transition_events):
        """Test cyclical pattern identification."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        patterns = analyzer.identify_transition_patterns()
        
        assert 'cycles' in patterns
        cycles = patterns['cycles']
        
        # Should identify the 0->1->2 pattern
        expected_cycle = [0, 1, 2]
        assert any(cycle[:3] == expected_cycle for cycle in cycles)
    
    def test_trigger_identification(self, sample_transition_events):
        """Test trigger factor identification."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        patterns = analyzer.identify_transition_patterns()
        
        assert 'triggers' in patterns
        triggers = patterns['triggers']
        
        # Check trigger identification
        assert '0->1' in triggers or '1->2' in triggers
        
        # Volatility spike should be common trigger for upward transitions
        for key, trigger_list in triggers.items():
            from_r, to_r = map(int, key.split('->'))
            if to_r > from_r:
                assert 'volatility_spike' in trigger_list
    
    def test_seasonality_analysis(self):
        """Test seasonality analysis."""
        analyzer = RegimeTransitionAnalyzer()
        
        # Create seasonal pattern
        base_time = datetime.now() - timedelta(days=730)
        
        # More transitions in certain months
        for year in range(2):
            for month in range(12):
                # More transitions in March, September
                n_transitions = 3 if month in [2, 8] else 1
                
                for _ in range(n_transitions):
                    event = TransitionEvent(
                        from_regime=0,
                        to_regime=1,
                        timestamp=base_time + timedelta(days=year*365 + month*30),
                        duration_in_from=50,
                        market_conditions={'vol': 0.2},
                        trigger_factors=['seasonal'],
                        transition_speed=0.5,
                        stability_score=0.8
                    )
                    analyzer.add_transition(event)
        
        patterns = analyzer.identify_transition_patterns()
        seasonality = patterns['seasonality']
        
        assert 'is_seasonal' in seasonality
        assert 'peak_months' in seasonality
        assert 'monthly_distribution' in seasonality
        
        # March (3) and September (9) should be peak months
        assert 3 in seasonality['peak_months'] or 9 in seasonality['peak_months']
    
    def test_persistence_analysis(self, sample_transition_events):
        """Test regime persistence analysis."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        patterns = analyzer.identify_transition_patterns()
        persistence = patterns['persistence']
        
        assert isinstance(persistence, dict)
        
        # Check regime 0 persistence
        assert 0 in persistence
        regime_0_stats = persistence[0]
        
        assert 'mean_duration' in regime_0_stats
        assert 'median_duration' in regime_0_stats
        assert 'std_duration' in regime_0_stats
        assert 'stability_score' in regime_0_stats
        
        # Stability score should be between 0 and 1
        assert 0 <= regime_0_stats['stability_score'] <= 1
    
    def test_most_common_transitions(self, sample_transition_events):
        """Test getting most common transitions."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        common_transitions = analyzer.get_most_common_transitions(n=3)
        
        assert len(common_transitions) <= 3
        
        # Most common should be (0, 1) with count 3
        assert common_transitions[0][0] == (0, 1)
        assert common_transitions[0][1] == 3
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_transition_matrix(self, mock_close, mock_savefig, sample_transition_events):
        """Test transition matrix plotting."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        # Test plotting
        analyzer.plot_transition_matrix(save_path='test_matrix.png')
        
        # Check that plot was saved
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_duration_distributions(self, mock_close, mock_savefig, sample_transition_events):
        """Test duration distribution plotting."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        # Test plotting
        analyzer.plot_duration_distributions(save_path='test_durations.png')
        
        # Check that plot was saved
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_transition_report(self, sample_transition_events):
        """Test transition analysis report generation."""
        analyzer = RegimeTransitionAnalyzer()
        
        for event in sample_transition_events:
            analyzer.add_transition(event)
        
        report = analyzer.generate_transition_report()
        
        assert isinstance(report, str)
        assert 'REGIME TRANSITION ANALYSIS REPORT' in report
        assert 'SUMMARY STATISTICS' in report
        assert 'MOST COMMON TRANSITIONS' in report
        assert 'REGIME PERSISTENCE' in report
        assert 'IDENTIFIED PATTERNS' in report
        
        # Check statistics are included
        assert 'Total transitions:' in report
        assert 'Mean duration:' in report
    
    def test_feature_similarity_calculation(self):
        """Test feature similarity calculation."""
        analyzer = RegimeTransitionAnalyzer()
        
        features1 = {'volatility': 0.15, 'return': 0.001, 'vix': 20}
        features2 = {'volatility': 0.16, 'return': 0.0008, 'vix': 21}
        features3 = {'volatility': 0.30, 'return': -0.005, 'vix': 40}
        
        # Similar features should have high similarity
        sim_12 = analyzer._calculate_feature_similarity(features1, features2)
        sim_13 = analyzer._calculate_feature_similarity(features1, features3)
        
        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1
        assert sim_12 > sim_13  # features1 and features2 are more similar
    
    def test_empty_analyzer(self):
        """Test analyzer with no data."""
        analyzer = RegimeTransitionAnalyzer()
        
        # Should handle empty state gracefully
        probs = analyzer.estimate_transition_probabilities(0, 50, {})
        assert isinstance(probs, dict)
        
        forecast = analyzer.forecast_regime_duration(0, 50)
        assert isinstance(forecast, dict)
        
        patterns = analyzer.identify_transition_patterns()
        assert isinstance(patterns, dict)
        
        report = analyzer.generate_transition_report()
        assert isinstance(report, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])