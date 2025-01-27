"""
Unit tests for the feature engineering module.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from features.feature_engineering import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_rolling_stats,
    FeatureStore
)


class TestTechnicalIndicators(unittest.TestCase):
    """Test suite for technical indicator calculations."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data with enough periods for MACD calculation
        np.random.seed(42)  # For reproducibility
        n_periods = 50  # Enough for MACD (26) + signal (9) periods
        base_price = 100
        random_walk = np.random.randn(n_periods) * 2  # Random price movements
        prices = base_price + np.cumsum(random_walk)  # Random walk price series
        
        self.prices = pd.Series(
            prices,
            index=pd.date_range(start='2024-01-01', periods=n_periods, freq='D')
        )

    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        window = 3
        sma = calculate_sma(self.prices, window=window)
        
        # Test length and NaN values
        self.assertEqual(len(sma), len(self.prices))
        self.assertTrue(np.isnan(sma.iloc[0]))  # First two values should be NaN
        self.assertTrue(np.isnan(sma.iloc[1]))
        
        # Test calculation manually
        first_valid_value = self.prices.iloc[0:window].mean()
        self.assertAlmostEqual(sma.iloc[window-1], first_valid_value, places=5)
        
        # Test random spot check
        mid_point = len(self.prices) // 2
        manual_sma = self.prices.iloc[mid_point-window+1:mid_point+1].mean()
        self.assertAlmostEqual(sma.iloc[mid_point], manual_sma, places=5)

    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        ema = calculate_ema(self.prices, window=3)
        self.assertEqual(len(ema), len(self.prices))
        self.assertTrue(np.isnan(ema.iloc[0]))
        self.assertFalse(np.isnan(ema.iloc[-1]))

    def test_rsi_calculation(self):
        """Test Relative Strength Index calculation."""
        rsi = calculate_rsi(self.prices, window=3)
        self.assertEqual(len(rsi), len(self.prices))
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))  # RSI should be between 0 and 100

    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = calculate_macd(self.prices)
        self.assertEqual(len(macd_line), len(self.prices))
        self.assertEqual(len(signal_line), len(self.prices))
        self.assertEqual(len(histogram), len(self.prices))
        
        # MACD line should be the difference between fast and slow EMAs
        self.assertAlmostEqual(
            macd_line.iloc[-1],
            calculate_ema(self.prices, 12).iloc[-1] - calculate_ema(self.prices, 26).iloc[-1],
            places=10
        )

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = calculate_bollinger_bands(self.prices, window=3)
        self.assertEqual(len(upper), len(self.prices))
        self.assertEqual(len(middle), len(self.prices))
        self.assertEqual(len(lower), len(self.prices))
        
        # Upper band should be higher than middle band
        self.assertTrue(all(u >= m for u, m in zip(upper.dropna(), middle.dropna())))
        # Lower band should be lower than middle band
        self.assertTrue(all(l <= m for l, m in zip(lower.dropna(), middle.dropna())))

    def test_rolling_stats_calculation(self):
        """Test rolling statistics calculation."""
        stats = calculate_rolling_stats(self.prices, window=3)
        self.assertIsInstance(stats, dict)
        self.assertTrue(all(len(v) == len(self.prices) for v in stats.values()))
        
        # Test specific statistics
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)


class TestFeatureStore(unittest.TestCase):
    """Test suite for FeatureStore class."""
    
    def setUp(self):
        """Set up test data and FeatureStore instance."""
        self.temp_dir = tempfile.mkdtemp()
        self.feature_store = FeatureStore(cache_dir=self.temp_dir)
        
        # Create sample price data
        self.prices = pd.Series(
            [100, 102, 99, 101, 103, 98, 96, 99, 102, 104],
            index=pd.date_range(start='2024-01-01', periods=10, freq='D')
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_compute_technical_indicators(self):
        """Test computing multiple technical indicators."""
        features = self.feature_store.compute_technical_indicators(
            self.prices,
            windows=[2, 3]
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertTrue(len(features.columns) > 0)
        
        # Check for specific feature columns
        expected_columns = [
            'sma_2', 'ema_2', 'bb_upper_2', 'bb_middle_2', 'bb_lower_2',
            'sma_3', 'ema_3', 'bb_upper_3', 'bb_middle_3', 'bb_lower_3',
            'rsi', 'macd', 'macd_signal', 'macd_hist'
        ]
        for col in expected_columns:
            self.assertIn(col, features.columns)

    def test_feature_store_caching(self):
        """Test FeatureStore caching functionality."""
        # Compute and store features
        features = self.feature_store.compute_technical_indicators(self.prices)
        self.feature_store.add_features('test_features', features)
        
        # Retrieve features
        cached_features = self.feature_store.get_features('test_features')
        self.assertIsNotNone(cached_features)
        pd.testing.assert_frame_equal(features, cached_features)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test with empty series
        empty_series = pd.Series([])
        features = self.feature_store.compute_technical_indicators(empty_series)
        self.assertTrue(features.empty)
        
        # Test with series containing NaN
        nan_series = pd.Series([100, np.nan, 102])
        features = self.feature_store.compute_technical_indicators(nan_series)
        self.assertTrue(features.isna().any().any())  # Should contain NaN values

    def test_feature_persistence(self):
        """Test feature persistence to disk."""
        features = self.feature_store.compute_technical_indicators(self.prices)
        self.feature_store.add_features('persistent_features', features)
        
        # Create new FeatureStore instance with same cache directory
        new_store = FeatureStore(cache_dir=self.temp_dir)
        loaded_features = new_store.get_features('persistent_features')
        
        self.assertIsNotNone(loaded_features)
        pd.testing.assert_frame_equal(features, loaded_features)


if __name__ == '__main__':
    unittest.main()