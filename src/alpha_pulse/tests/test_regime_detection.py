"""
Comprehensive tests for the market regime detection system.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from unittest.mock import Mock, MagicMock, patch

from alpha_pulse.ml.regime.regime_features import RegimeFeatureEngineer, RegimeFeatureConfig
from alpha_pulse.ml.regime.hmm_regime_detector import (
    GaussianHMM, HMMConfig, RegimeType, RegimeSwitchingGARCH, HierarchicalHMM
)
from alpha_pulse.ml.regime.regime_classifier import RegimeClassifier, RegimeInfo
from alpha_pulse.ml.regime.regime_transitions import RegimeTransitionAnalyzer, TransitionEvent
from alpha_pulse.models.market_regime_hmm import MarketRegimeHMM, MarketRegimeConfig
from alpha_pulse.models.regime_state import (
    RegimeState, RegimeStateManager, RegimeStateFactory, RegimeCharacteristic
)
from alpha_pulse.utils.hmm_optimization import HMMOptimizer, HMMModelSelector
from alpha_pulse.services.regime_detection_service import (
    RegimeDetectionService, RegimeDetectionConfig
)


class TestRegimeFeatureEngineer:
    """Test regime feature engineering."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)
        
        # Simulate market data with regime changes
        np.random.seed(42)
        returns = np.concatenate([
            np.random.normal(0.001, 0.01, n//3),    # Bull regime
            np.random.normal(-0.001, 0.02, n//3),   # Bear regime
            np.random.normal(0.0005, 0.015, n//3)   # Recovery regime
        ])
        
        prices = 100 * np.exp(np.cumsum(returns))
        volume = np.random.uniform(1e6, 1e7, n)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
            'high': prices * (1 + np.random.uniform(0, 0.02, n)),
            'low': prices * (1 + np.random.uniform(-0.02, 0, n)),
            'close': prices,
            'volume': volume
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer."""
        config = RegimeFeatureConfig(
            volatility_windows=[5, 10, 20],
            return_windows=[1, 5, 10],
            normalize_features=True
        )
        return RegimeFeatureEngineer(config)
    
    def test_feature_extraction(self, feature_engineer, sample_data):
        """Test feature extraction."""
        features = feature_engineer.extract_features(sample_data)
        
        assert not features.empty
        assert len(features) > 0
        assert len(features.columns) > 10
        
        # Check for expected features
        expected_features = [
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'return_1d', 'return_5d', 'return_10d',
            'rsi_14', 'macd_hist'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
        
        # Check normalization
        if feature_engineer.config.normalize_features:
            # Most features should be within [-3, 3] after normalization
            assert (features.abs() <= 3).sum().sum() / features.size > 0.95
    
    def test_volatility_features(self, feature_engineer, sample_data):
        """Test volatility feature extraction."""
        features = feature_engineer._extract_volatility_features(sample_data)
        
        # Check volatility increases during bear market period
        vol_20d = features['volatility_20d'].dropna()
        bear_start = len(vol_20d) // 3
        bear_end = 2 * len(vol_20d) // 3
        
        bull_vol = vol_20d[:bear_start].mean()
        bear_vol = vol_20d[bear_start:bear_end].mean()
        
        assert bear_vol > bull_vol  # Higher volatility in bear market
    
    def test_feature_importance(self, feature_engineer, sample_data):
        """Test feature importance calculation."""
        features = feature_engineer.extract_features(sample_data)
        
        # Create mock regimes
        n = len(features)
        regimes = np.array([0] * (n//3) + [1] * (n//3) + [2] * (n//3))
        
        importance = feature_engineer.get_feature_importance(features, regimes)
        
        assert not importance.empty
        assert 'f_statistic' in importance.columns
        assert 'p_value' in importance.columns
        assert 'mutual_information' in importance.columns
        
        # Volatility features should be important for regime detection
        top_features = importance.head(10)['feature'].tolist()
        assert any('volatility' in f for f in top_features)


class TestHMMRegimeDetector:
    """Test Hidden Markov Model regime detection."""
    
    @pytest.fixture
    def hmm_config(self):
        """Create HMM configuration."""
        return HMMConfig(
            n_states=3,
            covariance_type="diag",
            n_iter=50,
            random_state=42
        )
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for HMM."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create features with 3 distinct regimes
        features = []
        for i in range(3):
            mean = [i * 0.5, i * 0.3]
            cov = [[0.1, 0.05], [0.05, 0.1]]
            regime_data = np.random.multivariate_normal(mean, cov, n_samples // 3)
            features.append(regime_data)
        
        return np.vstack(features)
    
    def test_gaussian_hmm_fit(self, hmm_config, sample_features):
        """Test Gaussian HMM fitting."""
        model = GaussianHMM(hmm_config)
        model.fit(sample_features)
        
        assert model.is_fitted
        assert model.n_states == 3
        assert model.means.shape == (3, 2)
        assert model.trans_prob.shape == (3, 3)
        assert np.allclose(model.trans_prob.sum(axis=1), 1.0)
        
        # Check convergence
        assert len(model.convergence_history) > 0
        assert model.convergence_history[-1] > model.convergence_history[0]
    
    def test_hmm_prediction(self, hmm_config, sample_features):
        """Test HMM state prediction."""
        model = GaussianHMM(hmm_config)
        model.fit(sample_features)
        
        # Predict states
        states = model.predict(sample_features)
        
        assert len(states) == len(sample_features)
        assert np.unique(states).tolist() == [0, 1, 2]
        
        # Test on new data
        new_features = np.random.randn(100, 2)
        new_states = model.predict(new_features)
        assert len(new_states) == 100
    
    def test_hmm_probability_prediction(self, hmm_config, sample_features):
        """Test HMM probability prediction."""
        model = GaussianHMM(hmm_config)
        model.fit(sample_features)
        
        # Predict probabilities
        probs = model.predict_proba(sample_features[:100])
        
        assert probs.shape == (100, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0) and np.all(probs <= 1)
    
    def test_regime_classification(self, hmm_config, sample_features):
        """Test regime type classification."""
        model = GaussianHMM(hmm_config)
        model.fit(sample_features)
        
        # Check regime classification
        assert len(model.states) == 3
        for state in model.states:
            assert isinstance(state.regime_type, RegimeType)
            assert state.typical_duration > 0
            assert state.transition_probs.sum() == pytest.approx(1.0)
    
    def test_regime_switching_garch(self, hmm_config, sample_features):
        """Test regime-switching GARCH model."""
        model = RegimeSwitchingGARCH(hmm_config)
        model.fit(sample_features)
        
        assert model.is_fitted
        assert hasattr(model, 'garch_params')
        assert len(model.garch_params) == 3
    
    def test_hierarchical_hmm(self, hmm_config, sample_features):
        """Test hierarchical HMM."""
        model = HierarchicalHMM(hmm_config, n_levels=2)
        model.fit(sample_features)
        
        assert model.is_fitted
        assert len(model.sub_models) > 0
        
        # Test hierarchical prediction
        top_states, sub_states = model.predict_hierarchical(sample_features)
        assert len(top_states) == len(sample_features)


class TestRegimeClassifier:
    """Test real-time regime classification."""
    
    @pytest.fixture
    def trained_model(self, hmm_config, sample_features):
        """Create trained HMM model."""
        model = GaussianHMM(hmm_config)
        model.fit(sample_features)
        return model
    
    @pytest.fixture
    def classifier(self, trained_model):
        """Create regime classifier."""
        feature_engineer = RegimeFeatureEngineer()
        return RegimeClassifier(trained_model, feature_engineer)
    
    def test_classification(self, classifier, sample_data):
        """Test regime classification."""
        regime_info = classifier.classify(sample_data)
        
        assert isinstance(regime_info, RegimeInfo)
        assert 0 <= regime_info.current_regime < 3
        assert 0 <= regime_info.confidence <= 1
        assert isinstance(regime_info.regime_type, RegimeType)
        assert regime_info.duration >= 1
    
    def test_regime_forecast(self, classifier, sample_data):
        """Test regime forecasting."""
        # First classify to set current regime
        classifier.classify(sample_data)
        
        # Get forecast
        forecast = classifier.get_regime_forecast(horizon=5)
        
        assert not forecast.empty
        assert len(forecast) == 6  # Current + 5 horizon
        assert 'most_likely_regime' in forecast.columns
        assert 'max_probability' in forecast.columns
    
    def test_transition_detection(self, classifier, sample_data):
        """Test regime transition detection."""
        # Simulate regime change
        regime_info1 = classifier.classify(sample_data[:100])
        
        # Modify data to force regime change
        modified_data = sample_data.copy()
        modified_data['close'] *= 0.8  # Simulate crash
        
        regime_info2 = classifier.classify(modified_data[100:200])
        
        # Check transition history
        trans_stats = classifier.get_transition_statistics()
        if not trans_stats.empty:
            assert 'from_regime' in trans_stats.columns
            assert 'to_regime' in trans_stats.columns
            assert 'count' in trans_stats.columns


class TestRegimeTransitions:
    """Test regime transition analysis."""
    
    @pytest.fixture
    def transition_analyzer(self):
        """Create transition analyzer."""
        return RegimeTransitionAnalyzer()
    
    def test_transition_probability_estimation(self, transition_analyzer):
        """Test transition probability estimation."""
        # Add some historical transitions
        for i in range(10):
            event = TransitionEvent(
                from_regime=0,
                to_regime=1,
                timestamp=datetime.now() - timedelta(days=i),
                duration_in_from=20 + i,
                market_conditions={'volatility': 0.2},
                trigger_factors=['volatility_spike'],
                transition_speed=1.0,
                stability_score=0.8
            )
            transition_analyzer.add_transition(event)
        
        # Estimate probabilities
        probs = transition_analyzer.estimate_transition_probabilities(
            current_regime=0,
            current_duration=25,
            market_features={'volatility': 0.2}
        )
        
        assert isinstance(probs, dict)
        assert 1 in probs
        assert probs[1] > 0
        assert sum(probs.values()) == pytest.approx(1.0)
    
    def test_pattern_identification(self, transition_analyzer):
        """Test transition pattern identification."""
        # Create cyclical pattern
        cycle = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        for i in range(len(cycle) - 1):
            event = TransitionEvent(
                from_regime=cycle[i],
                to_regime=cycle[i + 1],
                timestamp=datetime.now() - timedelta(days=i),
                duration_in_from=10,
                market_conditions={},
                trigger_factors=[],
                transition_speed=1.0,
                stability_score=0.8
            )
            transition_analyzer.add_transition(event)
        
        patterns = transition_analyzer.identify_transition_patterns()
        
        assert 'cycles' in patterns
        assert 'persistence' in patterns
        assert len(patterns['cycles']) > 0


class TestMarketRegimeHMM:
    """Test integrated market regime HMM system."""
    
    @pytest.fixture
    def market_regime_config(self):
        """Create market regime configuration."""
        return MarketRegimeConfig(
            hmm_config=HMMConfig(n_states=3, n_iter=20),
            classification_window=50,
            min_confidence=0.6
        )
    
    @pytest.fixture
    def market_regime_hmm(self, market_regime_config):
        """Create market regime HMM."""
        return MarketRegimeHMM(market_regime_config)
    
    def test_integrated_system(self, market_regime_hmm, sample_data):
        """Test integrated regime detection system."""
        # Fit model
        market_regime_hmm.fit(sample_data, validation_split=0.2)
        
        assert market_regime_hmm.is_fitted
        assert market_regime_hmm.classifier is not None
        
        # Predict regime
        regime_info = market_regime_hmm.predict_regime(sample_data[-100:])
        
        assert isinstance(regime_info, RegimeInfo)
        assert hasattr(regime_info, 'regime_type')
        
        # Get trading signals
        signals = market_regime_hmm.get_regime_trading_signals(
            regime_info, 
            risk_tolerance="moderate"
        )
        
        assert 'position_sizing' in signals
        assert 'recommended_strategies' in signals
        assert 'risk_adjustments' in signals
    
    def test_model_persistence(self, market_regime_hmm, sample_data, tmp_path):
        """Test model save/load functionality."""
        # Fit model
        market_regime_hmm.fit(sample_data)
        
        # Save model
        save_path = tmp_path / "test_model.pkl"
        market_regime_hmm.save(str(save_path))
        
        # Load model
        loaded_model = MarketRegimeHMM.load(str(save_path))
        
        assert loaded_model.is_fitted
        assert loaded_model.hmm_model.n_states == market_regime_hmm.hmm_model.n_states
        
        # Test prediction with loaded model
        regime_info = loaded_model.predict_regime(sample_data[-100:])
        assert isinstance(regime_info, RegimeInfo)


class TestRegimeStates:
    """Test regime state management."""
    
    def test_regime_state_creation(self):
        """Test creating regime states."""
        state = RegimeStateFactory.create_bull_market()
        
        assert state.regime_type == "bull"
        assert state.mean_returns > 0
        assert state.optimal_leverage > 1.0
        assert RegimeCharacteristic.TRENDING in state.characteristics
        assert "trend_following" in state.recommended_strategies
    
    def test_regime_state_manager(self):
        """Test regime state manager."""
        manager = RegimeStateManager()
        
        # Check default states
        assert len(manager.states) == 5
        
        # Test state transition
        manager.transition_to(0)  # Bull market
        current = manager.get_current_state()
        assert current is not None
        assert current.regime_type == "bull"
        
        # Test state update
        returns = np.random.normal(0.001, 0.01, 100)
        features = pd.DataFrame({'feature1': np.random.randn(100)})
        manager.update_state(0, returns, features)
        
        # Check statistics updated
        state = manager.get_state_by_type("bull")
        assert state.total_occurrences > 0
    
    def test_position_sizing(self):
        """Test risk-adjusted position sizing."""
        state = RegimeStateFactory.create_crisis_market()
        
        # Crisis regime should reduce position size
        position = state.get_risk_adjusted_position_size(
            base_position=1.0,
            current_volatility=0.3
        )
        
        assert position < 1.0  # Reduced in crisis
        assert position >= 0.1  # But not below minimum


class TestHMMOptimization:
    """Test HMM optimization utilities."""
    
    @pytest.fixture
    def optimizer(self):
        """Create HMM optimizer."""
        return HMMOptimizer()
    
    def test_n_states_optimization(self, optimizer, sample_features):
        """Test optimal number of states selection."""
        optimal_n, scores = optimizer.optimize_n_states(
            sample_features,
            min_states=2,
            max_states=5,
            cv_splits=2
        )
        
        assert 2 <= optimal_n <= 5
        assert len(scores) == 4
        assert all(isinstance(v, float) for v in scores.values())
    
    @pytest.mark.slow
    def test_hyperparameter_optimization(self, optimizer, sample_features):
        """Test hyperparameter optimization."""
        param_grid = {
            'n_states': [2, 3],
            'covariance_type': ['diag'],
            'init_method': ['kmeans']
        }
        
        result = optimizer.optimize_hyperparameters(
            sample_features,
            param_grid=param_grid,
            use_optuna=False
        )
        
        assert result.best_params is not None
        assert result.best_score > -np.inf
        assert len(result.all_results) > 0
    
    def test_model_stability_analysis(self, optimizer, sample_features):
        """Test model stability analysis."""
        config = HMMConfig(n_states=3, n_iter=20)
        model = GaussianHMM(config)
        model.fit(sample_features)
        
        stability = optimizer.analyze_model_stability(
            model,
            sample_features,
            n_bootstrap=5
        )
        
        assert 'stability_score' in stability
        assert 0 <= stability['stability_score'] <= 1
        assert 'mode_states' in stability


class TestRegimeDetectionService:
    """Test regime detection service."""
    
    @pytest.fixture
    def service_config(self):
        """Create service configuration."""
        return RegimeDetectionConfig(
            update_interval_minutes=1,
            enable_alerts=False,
            track_performance=True
        )
    
    @pytest.fixture
    async def detection_service(self, service_config):
        """Create detection service."""
        service = RegimeDetectionService(service_config)
        await service.initialize()
        return service
    
    @pytest.mark.asyncio
    async def test_service_lifecycle(self, detection_service):
        """Test service start/stop."""
        # Start service
        await detection_service.start()
        assert detection_service.is_running
        
        # Stop service
        await detection_service.stop()
        assert not detection_service.is_running
    
    @pytest.mark.asyncio
    async def test_regime_detection_flow(self, detection_service, sample_data):
        """Test regime detection flow."""
        # Mock data pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.fetch_historical_data.return_value = sample_data
        detection_service.data_pipeline = mock_pipeline
        
        # Initialize model
        await detection_service._initialize_model()
        
        # Perform detection
        await detection_service._detect_regime()
        
        # Check results
        current_regime = await detection_service.get_current_regime()
        assert current_regime is not None
    
    @pytest.mark.asyncio
    async def test_trading_signals(self, detection_service):
        """Test trading signal generation."""
        # Create mock regime info
        detection_service.current_regime_info = RegimeInfo(
            current_regime=0,
            regime_type=RegimeType.BULL,
            confidence=0.8,
            regime_probabilities=np.array([0.8, 0.1, 0.1]),
            entered_at=datetime.now(),
            duration=10,
            features={},
            transition_probability=0.1
        )
        
        signals = await detection_service.get_trading_signals("moderate")
        
        assert 'position_sizing' in signals
        assert 'recommended_strategies' in signals
        assert signals['position_sizing'] > 0


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, sample_data):
        """Test complete workflow from data to signals."""
        # Create system
        config = MarketRegimeConfig(
            hmm_config=HMMConfig(n_states=3, n_iter=10)
        )
        regime_system = MarketRegimeHMM(config)
        
        # Train model
        regime_system.fit(sample_data, validation_split=0.1)
        
        # Create service
        service_config = RegimeDetectionConfig(
            model_config=config,
            enable_alerts=False
        )
        service = RegimeDetectionService(service_config)
        service.regime_model = regime_system
        
        # Detect regime
        regime_info = regime_system.predict_regime(sample_data[-50:])
        
        # Get signals
        signals = regime_system.get_regime_trading_signals(
            regime_info,
            risk_tolerance="moderate"
        )
        
        # Validate complete flow
        assert regime_info is not None
        assert signals['position_sizing'] > 0
        assert len(signals['recommended_strategies']) > 0
        
        # Get analysis
        analysis = regime_system.get_transition_analysis()
        assert 'current_regime' in analysis
        assert 'regime_characteristics' in analysis