"""
Tests for market regime detection system.

Tests regime classification accuracy, indicator calculations, and
transition probability estimation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from alpha_pulse.models.market_regime import (
    RegimeType, RegimeIndicatorType, MarketRegime,
    RegimeIndicator, RegimeDetectionResult
)
from alpha_pulse.risk.regime_detector import MarketRegimeDetector
from alpha_pulse.utils.regime_indicators import RegimeIndicatorCalculator
from alpha_pulse.config.regime_parameters import (
    REGIME_DETECTION_PARAMS, VOLATILITY_THRESHOLDS,
    MOMENTUM_THRESHOLDS, REGIME_CLASSIFICATION_RULES
)


class TestRegimeIndicatorCalculator:
    """Test regime indicator calculations."""
    
    @pytest.fixture
    def calculator(self):
        """Create indicator calculator instance."""
        return RegimeIndicatorCalculator()
    
    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        # Create realistic market data
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        
        # Add volatility clustering
        vol_regime = np.random.choice([0.01, 0.02, 0.03], size=len(dates), p=[0.6, 0.3, 0.1])
        returns_vol_adjusted = returns * vol_regime / 0.02
        
        data = pd.DataFrame({
            'SPY': 100 * np.cumprod(1 + returns_vol_adjusted),
            'VIX': np.random.lognormal(2.9, 0.5, len(dates)),  # Mean ~20, realistic distribution
            'volume': np.random.lognormal(18, 0.5, len(dates)),
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices
        }, index=dates)
        
        return data
    
    def test_volatility_indicators(self, calculator, market_data):
        """Test volatility indicator calculations."""
        indicators = calculator.calculate_volatility_indicators(market_data)
        
        # Check VIX indicator
        assert 'vix_level' in indicators
        vix_ind = indicators['vix_level']
        assert vix_ind.indicator_type == RegimeIndicatorType.VOLATILITY
        assert 0 <= vix_ind.normalized_value <= 1
        assert vix_ind.weight == 1.5  # Higher weight for VIX
        
        # Check realized volatility
        assert 'realized_vol' in indicators
        realized_vol = indicators['realized_vol']
        assert realized_vol.value > 0
        assert realized_vol.signal < 0  # Higher vol = bearish signal
        
        # Check vol of vol
        assert 'vol_of_vol' in indicators
        vol_of_vol = indicators['vol_of_vol']
        assert vol_of_vol.value >= 0
    
    def test_momentum_indicators(self, calculator, market_data):
        """Test momentum indicator calculations."""
        indicators = calculator.calculate_momentum_indicators(market_data)
        
        # Check momentum at different horizons
        for period in ['1m', '3m', '6m']:
            key = f'momentum_{period}'
            assert key in indicators
            mom_ind = indicators[key]
            assert mom_ind.indicator_type == RegimeIndicatorType.MOMENTUM
            assert -1 <= mom_ind.normalized_value <= 2  # Allow for strong momentum
        
        # Check trend strength
        assert 'trend_strength' in indicators
        trend = indicators['trend_strength']
        assert 0 <= trend.value <= 100
        
        # Check MA positioning
        if 'ma_positioning' in indicators:
            ma_pos = indicators['ma_positioning']
            assert ma_pos.indicator_type == RegimeIndicatorType.MOMENTUM
    
    def test_liquidity_indicators(self, calculator, market_data):
        """Test liquidity indicator calculations."""
        indicators = calculator.calculate_liquidity_indicators(market_data)
        
        # Check volume ratio
        assert 'volume_ratio' in indicators
        vol_ratio = indicators['volume_ratio']
        assert vol_ratio.value > 0
        assert vol_ratio.indicator_type == RegimeIndicatorType.LIQUIDITY
        
        # Check price range
        assert 'price_range' in indicators
        price_range = indicators['price_range']
        assert price_range.value >= 0
        assert price_range.signal <= 0  # High range = low liquidity
        
        # Check composite liquidity
        assert 'liquidity_composite' in indicators
    
    def test_sentiment_indicators(self, calculator, market_data):
        """Test sentiment indicator calculations."""
        additional_data = {
            'put_call_ratio': 1.2,
            'news_sentiment': -0.3,
            'social_sentiment': 0.1
        }
        
        indicators = calculator.calculate_sentiment_indicators(
            market_data, additional_data
        )
        
        # Check put/call ratio
        assert 'put_call_ratio' in indicators
        pc_ind = indicators['put_call_ratio']
        assert pc_ind.value == 1.2
        assert pc_ind.signal < 0  # High P/C = bearish
        
        # Check news sentiment
        assert 'news_sentiment' in indicators
        news_ind = indicators['news_sentiment']
        assert news_ind.value == -0.3
        
        # Check Fear & Greed index
        assert 'fear_greed' in indicators
        fg_ind = indicators['fear_greed']
        assert 0 <= fg_ind.value <= 100
    
    def test_technical_indicators(self, calculator, market_data):
        """Test technical indicator calculations."""
        indicators = calculator.calculate_technical_indicators(market_data)
        
        # Check RSI
        assert 'rsi' in indicators
        rsi_ind = indicators['rsi']
        assert 0 <= rsi_ind.value <= 100
        assert rsi_ind.indicator_type == RegimeIndicatorType.TECHNICAL
        
        # Check Bollinger Bands position
        assert 'bb_position' in indicators
        bb_ind = indicators['bb_position']
        assert 0 <= bb_ind.value <= 1  # Position within bands
        
        # Check MACD if available
        if 'macd_signal' in indicators:
            macd_ind = indicators['macd_signal']
            assert -1 <= macd_ind.normalized_value <= 1


class TestMarketRegimeDetector:
    """Test market regime detection."""
    
    @pytest.fixture
    def detector(self):
        """Create regime detector instance."""
        return MarketRegimeDetector()
    
    @pytest.fixture
    def sample_indicators(self):
        """Create sample regime indicators."""
        return {
            'volatility': {
                'vix_level': RegimeIndicator(
                    name='vix_level',
                    indicator_type=RegimeIndicatorType.VOLATILITY,
                    value=25.0,
                    normalized_value=0.5,
                    signal=-0.3,
                    weight=1.5
                ),
                'realized_vol': RegimeIndicator(
                    name='realized_vol',
                    indicator_type=RegimeIndicatorType.VOLATILITY,
                    value=0.18,
                    normalized_value=0.45,
                    signal=-0.45,
                    weight=1.2
                )
            },
            'momentum': {
                'momentum_3m': RegimeIndicator(
                    name='momentum_3m',
                    indicator_type=RegimeIndicatorType.MOMENTUM,
                    value=0.08,
                    normalized_value=0.7,
                    signal=0.4,
                    weight=1.0
                )
            }
        }
    
    def test_detect_regime_basic(self, detector, market_data):
        """Test basic regime detection."""
        result = detector.detect_regime(market_data)
        
        assert isinstance(result, RegimeDetectionResult)
        assert result.current_regime is not None
        assert isinstance(result.current_regime.regime_type, RegimeType)
        assert 0 <= result.current_regime.confidence <= 1
        assert len(result.regime_probabilities) == 5  # All regime types
    
    def test_regime_classification_rules(self, detector):
        """Test regime classification against rules."""
        # Test BULL regime conditions
        bull_indicators = {
            'vix_level': 18,  # < 20
            'momentum_3m': 0.06,  # > 0.05
            'sentiment': 0.4,  # > 0.3
            'trend_strength': 30  # > 25
        }
        
        regime = detector._classify_regime_rule_based(bull_indicators)
        assert regime == RegimeType.BULL
        
        # Test CRISIS regime conditions
        crisis_indicators = {
            'vix_level': 45,  # > 40
            'momentum_1m': -0.12,  # < -0.10
            'liquidity_crisis': True,
            'correlation_breakdown': True
        }
        
        regime = detector._classify_regime_rule_based(crisis_indicators)
        assert regime == RegimeType.CRISIS
    
    def test_regime_transitions(self, detector, market_data):
        """Test regime transition detection."""
        # First detection
        result1 = detector.detect_regime(market_data)
        
        # Modify data to trigger regime change
        market_data_crisis = market_data.copy()
        market_data_crisis['VIX'] *= 2.5  # Spike VIX
        market_data_crisis['SPY'] *= 0.85  # Drop prices
        
        # Second detection
        result2 = detector.detect_regime(market_data_crisis)
        
        # Check transition detected
        assert result2.transition_probability is not None
        if result1.current_regime.regime_type != result2.current_regime.regime_type:
            assert result2.transition_probability > 0
    
    def test_ensemble_regime_detection(self, detector, market_data):
        """Test ensemble model regime detection."""
        # Ensure models are fitted
        detector._ensure_models_fitted(market_data)
        
        # Get ensemble predictions
        result = detector.detect_regime(market_data)
        
        # Check model agreement
        model_agreement = detector._calculate_model_agreement(
            result.regime_probabilities
        )
        assert 0 <= model_agreement <= 1
        
        # Higher agreement should increase confidence
        if model_agreement > 0.8:
            assert result.current_regime.confidence > 0.7
    
    def test_confidence_calculation(self, detector):
        """Test confidence score calculation."""
        # High probability, high agreement
        regime_probs = {
            RegimeType.BULL: 0.8,
            RegimeType.BEAR: 0.1,
            RegimeType.SIDEWAYS: 0.05,
            RegimeType.CRISIS: 0.03,
            RegimeType.RECOVERY: 0.02
        }
        
        confidence = detector._calculate_confidence(
            RegimeType.BULL,
            regime_probs,
            model_agreement=0.9,
            indicator_strength=0.8
        )
        
        assert confidence > 0.8  # High confidence expected
        
        # Low probability, low agreement
        regime_probs_uncertain = {
            RegimeType.BULL: 0.3,
            RegimeType.BEAR: 0.25,
            RegimeType.SIDEWAYS: 0.25,
            RegimeType.CRISIS: 0.1,
            RegimeType.RECOVERY: 0.1
        }
        
        confidence_low = detector._calculate_confidence(
            RegimeType.BULL,
            regime_probs_uncertain,
            model_agreement=0.4,
            indicator_strength=0.5
        )
        
        assert confidence_low < 0.5  # Low confidence expected
    
    def test_regime_stability_analysis(self, detector, market_data):
        """Test regime stability metrics."""
        # Detect regime multiple times
        for i in range(5):
            # Slightly modify data
            data_slice = market_data.iloc[i*10:].copy()
            result = detector.detect_regime(data_slice)
        
        # Get stability metrics
        stability_metrics = detector.get_regime_stability_metrics()
        
        assert 'transition_matrix' in stability_metrics
        assert 'avg_regime_duration' in stability_metrics
        assert 'regime_persistence' in stability_metrics
        
        # Transition matrix should be stochastic
        trans_matrix = stability_metrics['transition_matrix']
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)
    
    def test_volatility_regime_detection(self, detector, market_data):
        """Test volatility-specific regime detection."""
        # Create high volatility scenario
        high_vol_data = market_data.copy()
        high_vol_data['VIX'] = 35  # Elevated VIX
        
        # Add volatility indicators
        vol_indicators = {
            'vix_level': 35,
            'realized_vol': 0.30,
            'vol_of_vol': 0.15
        }
        
        result = detector.detect_regime(high_vol_data, vol_indicators)
        
        # Should not be BULL regime with high volatility
        assert result.current_regime.regime_type != RegimeType.BULL
        assert result.current_regime.volatility_level in ['high', 'extreme']
    
    @pytest.mark.parametrize("regime_type,expected_characteristics", [
        (RegimeType.BULL, {
            'volatility_level': 'low',
            'trend_direction': 'up',
            'suggested_leverage': 1.2
        }),
        (RegimeType.CRISIS, {
            'volatility_level': 'extreme',
            'trend_direction': 'down',
            'suggested_leverage': 0.4
        })
    ])
    def test_regime_characteristics(self, detector, regime_type, expected_characteristics):
        """Test regime characteristic assignments."""
        regime = MarketRegime(
            regime_type=regime_type,
            confidence=0.8,
            start_date=datetime.now()
        )
        
        # Apply characteristics
        detector._set_regime_characteristics(regime)
        
        for key, expected_value in expected_characteristics.items():
            if isinstance(expected_value, str):
                assert getattr(regime, key) == expected_value
            else:
                assert abs(getattr(regime, key) - expected_value) < 0.1


class TestRegimeTransitionProbabilities:
    """Test regime transition probability estimation."""
    
    @pytest.fixture
    def detector(self):
        """Create detector with history."""
        detector = MarketRegimeDetector()
        
        # Add synthetic regime history
        regimes = [
            (RegimeType.BULL, 100),
            (RegimeType.SIDEWAYS, 30),
            (RegimeType.BEAR, 60),
            (RegimeType.CRISIS, 20),
            (RegimeType.RECOVERY, 40),
            (RegimeType.BULL, 80)
        ]
        
        current_date = datetime.now()
        for regime_type, duration in regimes:
            regime = MarketRegime(
                regime_type=regime_type,
                confidence=0.8,
                start_date=current_date,
                duration_days=duration
            )
            detector.regime_history.append(regime)
            current_date += timedelta(days=duration)
        
        return detector
    
    def test_transition_matrix_estimation(self, detector):
        """Test transition matrix estimation from history."""
        trans_matrix = detector._estimate_transition_matrix()
        
        # Check matrix properties
        assert trans_matrix.shape == (5, 5)
        assert np.all(trans_matrix >= 0)
        assert np.all(trans_matrix <= 1)
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)
        
        # Check specific transitions from history
        # BULL -> SIDEWAYS should have probability > 0
        bull_idx = list(RegimeType).index(RegimeType.BULL)
        sideways_idx = list(RegimeType).index(RegimeType.SIDEWAYS)
        assert trans_matrix[bull_idx, sideways_idx] > 0
    
    def test_regime_duration_prediction(self, detector):
        """Test expected regime duration predictions."""
        current_regime = MarketRegime(
            regime_type=RegimeType.BULL,
            confidence=0.85,
            start_date=datetime.now()
        )
        
        expected_duration = detector._predict_regime_duration(current_regime)
        
        # Should return reasonable duration
        assert 1 <= expected_duration <= 365  # Between 1 day and 1 year
        
        # Crisis regimes should have shorter expected duration
        crisis_regime = MarketRegime(
            regime_type=RegimeType.CRISIS,
            confidence=0.9,
            start_date=datetime.now()
        )
        
        crisis_duration = detector._predict_regime_duration(crisis_regime)
        assert crisis_duration < expected_duration
    
    def test_transition_probability_calculation(self, detector):
        """Test transition probability calculations."""
        current_regime = RegimeType.BULL
        
        # Get transition probabilities
        trans_probs = detector._get_transition_probabilities(current_regime)
        
        assert len(trans_probs) == 5
        assert sum(trans_probs.values()) == pytest.approx(1.0)
        
        # Staying in same regime should often have highest probability
        assert trans_probs[current_regime] > 0.5


class TestRegimeDetectorIntegration:
    """Integration tests for regime detector."""
    
    @pytest.fixture
    def full_market_data(self):
        """Create comprehensive market data."""
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        
        # Simulate different market regimes
        regime_periods = [
            (0, 200, 'bull'),      # Bull market
            (200, 250, 'sideways'), # Consolidation
            (250, 350, 'bear'),     # Bear market
            (350, 380, 'crisis'),   # Crisis
            (380, 500, 'recovery')  # Recovery
        ]
        
        data = pd.DataFrame(index=dates)
        
        # Generate regime-specific data
        spy_prices = []
        vix_values = []
        volumes = []
        
        for start, end, regime in regime_periods:
            n_days = end - start
            
            if regime == 'bull':
                returns = np.random.normal(0.001, 0.01, n_days)
                vix = np.random.normal(15, 2, n_days)
                vol = np.random.normal(1e9, 1e8, n_days)
            elif regime == 'bear':
                returns = np.random.normal(-0.001, 0.015, n_days)
                vix = np.random.normal(25, 3, n_days)
                vol = np.random.normal(1.5e9, 2e8, n_days)
            elif regime == 'crisis':
                returns = np.random.normal(-0.003, 0.03, n_days)
                vix = np.random.normal(45, 5, n_days)
                vol = np.random.normal(2e9, 3e8, n_days)
            elif regime == 'recovery':
                returns = np.random.normal(0.0015, 0.012, n_days)
                vix = np.random.normal(22, 3, n_days)
                vol = np.random.normal(1.2e9, 1.5e8, n_days)
            else:  # sideways
                returns = np.random.normal(0, 0.008, n_days)
                vix = np.random.normal(20, 2, n_days)
                vol = np.random.normal(1e9, 1e8, n_days)
            
            if start == 0:
                base_price = 100
            else:
                base_price = spy_prices[-1]
            
            prices = base_price * np.cumprod(1 + returns)
            spy_prices.extend(prices)
            vix_values.extend(np.maximum(vix, 10))  # VIX floor
            volumes.extend(np.maximum(vol, 0))
        
        data['SPY'] = spy_prices
        data['VIX'] = vix_values
        data['volume'] = volumes
        data['high'] = data['SPY'] * 1.005
        data['low'] = data['SPY'] * 0.995
        data['close'] = data['SPY']
        
        return data
    
    def test_regime_detection_accuracy(self, full_market_data):
        """Test regime detection accuracy over different periods."""
        detector = MarketRegimeDetector()
        
        # Expected regimes for each period
        expected_regimes = {
            100: RegimeType.BULL,      # Day 100
            225: RegimeType.SIDEWAYS,  # Day 225
            300: RegimeType.BEAR,      # Day 300
            365: RegimeType.CRISIS,    # Day 365
            450: RegimeType.RECOVERY   # Day 450
        }
        
        correct_detections = 0
        total_detections = 0
        
        for day, expected_regime in expected_regimes.items():
            # Use data up to specific day
            data_slice = full_market_data.iloc[:day]
            result = detector.detect_regime(data_slice)
            
            detected_regime = result.current_regime.regime_type
            
            # Allow some flexibility (e.g., SIDEWAYS vs BULL)
            if detected_regime == expected_regime:
                correct_detections += 1
            elif (expected_regime == RegimeType.SIDEWAYS and 
                  detected_regime in [RegimeType.BULL, RegimeType.BEAR]):
                correct_detections += 0.5  # Partial credit
            
            total_detections += 1
        
        accuracy = correct_detections / total_detections
        assert accuracy >= 0.6  # At least 60% accuracy
    
    def test_regime_persistence_and_stability(self, full_market_data):
        """Test that regimes show appropriate persistence."""
        detector = MarketRegimeDetector()
        
        # Detect regimes over sliding window
        window_size = 252  # 1 year
        step_size = 5  # 5 days
        
        regime_sequence = []
        
        for i in range(window_size, len(full_market_data), step_size):
            data_window = full_market_data.iloc[i-window_size:i]
            result = detector.detect_regime(data_window)
            regime_sequence.append(result.current_regime.regime_type)
        
        # Calculate regime persistence
        regime_changes = 0
        for i in range(1, len(regime_sequence)):
            if regime_sequence[i] != regime_sequence[i-1]:
                regime_changes += 1
        
        # Regimes should show some persistence (not changing every step)
        change_frequency = regime_changes / len(regime_sequence)
        assert change_frequency < 0.3  # Less than 30% change frequency
    
    def test_real_time_regime_updates(self, full_market_data):
        """Test real-time regime detection updates."""
        detector = MarketRegimeDetector()
        
        # Initialize with historical data
        initial_data = full_market_data.iloc[:250]
        initial_result = detector.detect_regime(initial_data)
        
        # Simulate real-time updates
        for i in range(250, 260):  # 10 days of updates
            new_data = full_market_data.iloc[:i+1]
            
            # Add some intraday indicators
            additional_indicators = {
                'intraday_volatility': np.random.uniform(0.001, 0.003),
                'order_flow_imbalance': np.random.uniform(-0.5, 0.5)
            }
            
            result = detector.detect_regime(new_data, additional_indicators)
            
            # Check that confidence adjusts with new data
            assert result.current_regime.confidence > 0
            assert result.indicators_used is not None
            
            # Regime shouldn't change too frequently in stable periods
            if i < 255:  # First 5 days
                assert result.current_regime.regime_type == initial_result.current_regime.regime_type