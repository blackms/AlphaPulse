"""
Tests for Market Regime Feature Engineering.

Tests cover feature extraction, normalization, and feature importance calculation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import talib

from alpha_pulse.ml.regime.regime_features import (
    RegimeFeatureConfig, RegimeFeatureEngineer
)


class TestRegimeFeatureConfig:
    """Test regime feature configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RegimeFeatureConfig()
        
        assert config.volatility_windows == [5, 10, 20, 60]
        assert config.return_windows == [1, 5, 20, 60]
        assert config.volume_windows == [5, 10, 20]
        assert config.use_vix is True
        assert config.use_sentiment is True
        assert config.use_liquidity is True
        assert config.normalize_features is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RegimeFeatureConfig(
            volatility_windows=[10, 20],
            return_windows=[5, 10],
            use_vix=False,
            normalize_features=False
        )
        
        assert config.volatility_windows == [10, 20]
        assert config.return_windows == [5, 10]
        assert config.use_vix is False
        assert config.normalize_features is False


class TestRegimeFeatureEngineer:
    """Test regime feature engineering."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        
        # Generate price data with trend
        price = 100
        prices = []
        for i in range(500):
            price *= (1 + np.random.normal(0.0002, 0.01))
            prices.append(price)
        
        # Add some volatility clustering
        volatility_regime = np.where(
            (np.arange(500) > 200) & (np.arange(500) < 300),
            2.0, 1.0
        )
        
        data = pd.DataFrame({
            'open': prices * (1 - np.random.uniform(0, 0.005, 500)),
            'high': prices * (1 + np.random.uniform(0, 0.01, 500) * volatility_regime),
            'low': prices * (1 - np.random.uniform(0, 0.01, 500) * volatility_regime),
            'close': prices,
            'volume': np.random.uniform(1e6, 1e7, 500) * (1 + np.random.normal(0, 0.2, 500))
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def additional_data(self):
        """Generate additional market data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        
        # VIX data
        vix_data = pd.DataFrame({
            'close': 15 + 5 * np.sin(np.arange(500) / 50) + np.random.normal(0, 2, 500),
            'vix9d': 14 + 4 * np.sin(np.arange(500) / 40) + np.random.normal(0, 1.5, 500),
            'vix30d': 16 + 5 * np.sin(np.arange(500) / 60) + np.random.normal(0, 2, 500)
        }, index=dates)
        
        # Sentiment data
        sentiment_data = pd.DataFrame({
            'put_call_ratio': 0.8 + 0.4 * np.sin(np.arange(500) / 30) + np.random.normal(0, 0.1, 500),
            'sentiment_score': 50 + 20 * np.sin(np.arange(500) / 40) + np.random.normal(0, 5, 500),
            'fear_greed_index': 50 + 30 * np.sin(np.arange(500) / 35) + np.random.normal(0, 10, 500)
        }, index=dates)
        
        # Credit spread data
        credit_data = pd.DataFrame({
            'ig_spread': 100 + 20 * np.sin(np.arange(500) / 60) + np.random.normal(0, 5, 500),
            'hy_spread': 400 + 100 * np.sin(np.arange(500) / 60) + np.random.normal(0, 20, 500),
            'ted_spread': 30 + 10 * np.sin(np.arange(500) / 45) + np.random.normal(0, 3, 500)
        }, index=dates)
        
        # Yield curve data
        yield_data = pd.DataFrame({
            '2y_yield': 2.0 + 0.5 * np.sin(np.arange(500) / 80) + np.random.normal(0, 0.1, 500),
            '5y_yield': 2.5 + 0.6 * np.sin(np.arange(500) / 85) + np.random.normal(0, 0.1, 500),
            '10y_yield': 3.0 + 0.7 * np.sin(np.arange(500) / 90) + np.random.normal(0, 0.1, 500),
            '30y_yield': 3.5 + 0.8 * np.sin(np.arange(500) / 95) + np.random.normal(0, 0.1, 500),
            '10y_real_yield': 0.5 + 0.3 * np.sin(np.arange(500) / 70) + np.random.normal(0, 0.05, 500)
        }, index=dates)
        
        return {
            'vix': vix_data,
            'sentiment': sentiment_data,
            'credit_spreads': credit_data,
            'yield_curve': yield_data
        }
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization."""
        config = RegimeFeatureConfig()
        engineer = RegimeFeatureEngineer(config)
        
        assert engineer.config == config
        assert engineer.feature_names == []
        assert engineer.scaler_params == {}
    
    def test_extract_volatility_features(self, sample_market_data):
        """Test volatility feature extraction."""
        config = RegimeFeatureConfig(volatility_windows=[5, 20])
        engineer = RegimeFeatureEngineer(config)
        
        vol_features = engineer._extract_volatility_features(sample_market_data)
        
        # Check feature columns
        expected_features = [
            'volatility_5d', 'volatility_20d',
            'garch_vol_5d', 'garch_vol_20d',
            'ewma_vol_5d', 'ewma_vol_20d',
            'vol_of_vol_5d', 'vol_of_vol_20d',
            'parkinson_vol_5d', 'parkinson_vol_20d',
            'vol_term_structure'
        ]
        
        for feature in expected_features:
            assert feature in vol_features.columns
        
        # Check feature values
        assert not vol_features['volatility_5d'].isna().all()
        assert np.all(vol_features['volatility_5d'].dropna() >= 0)
        
        # Check term structure
        assert not vol_features['vol_term_structure'].isna().all()
    
    def test_extract_return_features(self, sample_market_data):
        """Test return feature extraction."""
        config = RegimeFeatureConfig(return_windows=[1, 5, 20])
        engineer = RegimeFeatureEngineer(config)
        
        return_features = engineer._extract_return_features(sample_market_data)
        
        # Check feature columns
        expected_features = [
            'return_1d', 'return_5d', 'return_20d',
            'log_return_1d', 'log_return_5d', 'log_return_20d',
            'momentum_5d', 'momentum_20d',
            'skewness_1d', 'skewness_5d', 'skewness_20d',
            'kurtosis_1d', 'kurtosis_5d', 'kurtosis_20d',
            'max_drawdown_1d', 'max_drawdown_5d', 'max_drawdown_20d',
            'adx_14'
        ]
        
        for feature in expected_features:
            assert feature in return_features.columns
        
        # Check ADX calculation
        assert not return_features['adx_14'].isna().all()
        assert np.all(return_features['adx_14'].dropna() >= 0)
        assert np.all(return_features['adx_14'].dropna() <= 100)
    
    def test_extract_liquidity_features(self, sample_market_data):
        """Test liquidity feature extraction."""
        config = RegimeFeatureConfig(volume_windows=[5, 10])
        engineer = RegimeFeatureEngineer(config)
        
        liquidity_features = engineer._extract_liquidity_features(sample_market_data)
        
        # Check feature columns
        expected_features = [
            'volume_ma_5d', 'volume_ma_10d',
            'volume_std_5d', 'volume_std_10d',
            'relative_volume_5d', 'relative_volume_10d',
            'dollar_volume_5d', 'dollar_volume_10d',
            'amihud_illiquidity',
            'volume_price_corr'
        ]
        
        for feature in expected_features:
            assert feature in liquidity_features.columns
        
        # Check relative volume
        assert not liquidity_features['relative_volume_5d'].isna().all()
        
        # Check Amihud illiquidity
        assert not liquidity_features['amihud_illiquidity'].isna().all()
        assert np.all(liquidity_features['amihud_illiquidity'].dropna() >= 0)
    
    def test_extract_vix_features(self, additional_data):
        """Test VIX feature extraction."""
        engineer = RegimeFeatureEngineer()
        
        vix_features = engineer._extract_vix_features(additional_data['vix'])
        
        # Check feature columns
        expected_features = [
            'vix_level',
            'vix_change_1d',
            'vix_change_5d',
            'vix_term_structure',
            'vix_percentile_252d',
            'vix_high_regime',
            'vix_extreme_regime'
        ]
        
        for feature in expected_features:
            assert feature in vix_features.columns
        
        # Check VIX regimes
        assert vix_features['vix_high_regime'].dtype == int
        assert vix_features['vix_extreme_regime'].dtype == int
        assert np.all(vix_features['vix_high_regime'].isin([0, 1]))
    
    def test_extract_sentiment_features(self, sample_market_data, additional_data):
        """Test sentiment feature extraction."""
        engineer = RegimeFeatureEngineer()
        
        sentiment_features = engineer._extract_sentiment_features(
            sample_market_data,
            additional_data['sentiment']
        )
        
        # Check feature columns
        expected_features = [
            'put_call_ratio',
            'put_call_ma_20',
            'sentiment_score',
            'sentiment_ma_5',
            'sentiment_std_20',
            'fear_greed_index',
            'fear_greed_change'
        ]
        
        for feature in expected_features:
            assert feature in sentiment_features.columns
        
        # Check put/call ratio
        assert not sentiment_features['put_call_ratio'].isna().all()
        assert np.all(sentiment_features['put_call_ratio'].dropna() > 0)
    
    def test_extract_credit_spread_features(self, additional_data):
        """Test credit spread feature extraction."""
        engineer = RegimeFeatureEngineer()
        
        credit_features = engineer._extract_credit_spread_features(
            additional_data['credit_spreads']
        )
        
        # Check feature columns
        expected_features = [
            'ig_spread_level',
            'ig_spread_ma_20',
            'ig_spread_z_score',
            'hy_spread_level',
            'hy_spread_ma_20',
            'hy_spread_percentile',
            'credit_spread_diff',
            'spread_widening_5d',
            'spread_widening_20d',
            'ted_spread',
            'ted_spread_extreme'
        ]
        
        for feature in expected_features:
            assert feature in credit_features.columns
        
        # Check spread levels
        assert np.all(credit_features['ig_spread_level'].dropna() > 0)
        assert np.all(credit_features['hy_spread_level'].dropna() > 
                     credit_features['ig_spread_level'].dropna())
    
    def test_extract_yield_curve_features(self, additional_data):
        """Test yield curve feature extraction."""
        engineer = RegimeFeatureEngineer()
        
        yield_features = engineer._extract_yield_curve_features(
            additional_data['yield_curve']
        )
        
        # Check feature columns
        expected_features = [
            'term_spread_10y2y',
            'curve_inversion',
            'term_spread_ma_20',
            'term_spread_30y5y',
            '10y_level',
            '10y_percentile_252d',
            '10y_change_20d',
            'curve_butterfly',
            'real_yield_10y',
            'real_yield_negative'
        ]
        
        for feature in expected_features:
            assert feature in yield_features.columns
        
        # Check curve inversion indicator
        assert yield_features['curve_inversion'].dtype == int
        assert np.all(yield_features['curve_inversion'].isin([0, 1]))
    
    def test_extract_technical_features(self, sample_market_data):
        """Test technical indicator feature extraction."""
        engineer = RegimeFeatureEngineer()
        
        tech_features = engineer._extract_technical_features(sample_market_data)
        
        # Check feature columns
        expected_features = [
            'rsi_14',
            'rsi_oversold',
            'rsi_overbought',
            'macd_hist',
            'macd_signal',
            'bb_position',
            'bb_width',
            'atr_14',
            'atr_ratio'
        ]
        
        for feature in expected_features:
            assert feature in tech_features.columns
        
        # Check RSI bounds
        rsi_values = tech_features['rsi_14'].dropna()
        assert np.all(rsi_values >= 0)
        assert np.all(rsi_values <= 100)
        
        # Check binary indicators
        assert tech_features['rsi_oversold'].dtype == int
        assert tech_features['rsi_overbought'].dtype == int
    
    def test_complete_feature_extraction(self, sample_market_data, additional_data):
        """Test complete feature extraction pipeline."""
        config = RegimeFeatureConfig(
            volatility_windows=[5, 20],
            return_windows=[1, 5, 20],
            volume_windows=[5, 10]
        )
        engineer = RegimeFeatureEngineer(config)
        
        features = engineer.extract_features(sample_market_data, additional_data)
        
        # Check that we have features
        assert not features.empty
        assert len(features.columns) > 50  # Should have many features
        
        # Check that feature names are stored
        assert len(engineer.feature_names) == len(features.columns)
        assert engineer.feature_names == features.columns.tolist()
        
        # Check normalization
        if config.normalize_features:
            # Should be roughly normalized
            for col in features.columns:
                col_data = features[col].dropna()
                if len(col_data) > 100:
                    assert abs(col_data.mean()) < 1.0  # Roughly centered
                    assert col_data.std() < 3.0  # Roughly scaled
    
    def test_feature_normalization(self, sample_market_data):
        """Test feature normalization."""
        config = RegimeFeatureConfig(
            normalize_features=True,
            volatility_windows=[20],
            return_windows=[20]
        )
        engineer = RegimeFeatureEngineer(config)
        
        features = engineer.extract_features(sample_market_data)
        
        # Check normalization parameters are stored
        assert len(engineer.scaler_params) > 0
        
        for col in features.columns:
            assert col in engineer.scaler_params
            assert 'mean' in engineer.scaler_params[col]
            assert 'std' in engineer.scaler_params[col]
            
            # Check clipping
            assert features[col].min() >= -3
            assert features[col].max() <= 3
    
    def test_transform_features(self, sample_market_data):
        """Test feature transformation with stored parameters."""
        config = RegimeFeatureConfig(normalize_features=True)
        engineer = RegimeFeatureEngineer(config)
        
        # Extract features to get normalization parameters
        train_features = engineer.extract_features(sample_market_data.iloc[:400])
        
        # Transform new features
        test_features_raw = engineer._extract_volatility_features(sample_market_data.iloc[400:])
        test_features = engineer.transform_features(test_features_raw)
        
        # Check transformation
        assert not test_features.empty
        for col in test_features.columns:
            if col in engineer.scaler_params:
                # Should be normalized
                assert test_features[col].min() >= -3
                assert test_features[col].max() <= 3
    
    def test_feature_importance(self, sample_market_data):
        """Test feature importance calculation."""
        config = RegimeFeatureConfig(
            volatility_windows=[5, 20],
            return_windows=[5, 20]
        )
        engineer = RegimeFeatureEngineer(config)
        
        features = engineer.extract_features(sample_market_data)
        
        # Create mock regimes
        np.random.seed(42)
        regimes = np.random.randint(0, 3, size=len(features))
        
        importance = engineer.get_feature_importance(features, regimes)
        
        # Check output format
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'f_statistic' in importance.columns
        assert 'p_value' in importance.columns
        assert 'mutual_information' in importance.columns
        
        # Check values
        assert len(importance) == len(features.columns)
        assert np.all(importance['f_statistic'] >= 0)
        assert np.all(importance['p_value'] >= 0)
        assert np.all(importance['p_value'] <= 1)
        assert np.all(importance['mutual_information'] >= 0)
    
    def test_no_additional_data(self, sample_market_data):
        """Test feature extraction without additional data."""
        config = RegimeFeatureConfig(
            use_vix=True,
            use_sentiment=True,
            use_liquidity=True
        )
        engineer = RegimeFeatureEngineer(config)
        
        # Should work without additional data
        features = engineer.extract_features(sample_market_data)
        
        assert not features.empty
        # Should still have basic features
        assert any('volatility' in col for col in features.columns)
        assert any('return' in col for col in features.columns)
        assert any('volume' in col for col in features.columns)
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Create data with gaps
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [100] * 100,
            'volume': [1e6] * 100
        }, index=dates)
        
        # Add some NaN values
        data.loc[dates[20:30], 'close'] = np.nan
        data.loc[dates[50:55], 'volume'] = np.nan
        
        engineer = RegimeFeatureEngineer()
        features = engineer.extract_features(data)
        
        # Should handle NaN values gracefully
        assert not features.empty
        # Forward fill should have been applied
        assert features.shape[0] < 100  # Some rows dropped after forward fill


if __name__ == "__main__":
    pytest.main([__file__, "-v"])