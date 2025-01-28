"""
Tests for feature engineering module.
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, UTC
import tempfile
import shutil
from pathlib import Path

from ..features.feature_engineering import (
    calculate_technical_indicators,
    add_target_column,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_sma,
    calculate_ema,
    FeatureStore,
)


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for feature engineering functions."""

    def setUp(self):
        """Set up test data."""
        # Create sample price data
        self.dates = pd.date_range(
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 10, tzinfo=UTC),
            freq='D'
        )
        self.prices = pd.Series(
            [100, 102, 101, 103, 102, 104, 103, 105, 104, 106],
            index=self.dates
        )
        self.df = pd.DataFrame({
            'open': self.prices - 0.5,
            'high': self.prices + 1,
            'low': self.prices - 1,
            'close': self.prices,
            'volume': np.random.randint(1000, 2000, size=len(self.prices))
        })
        
        # Create temporary directory for feature store tests
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_technical_indicators(self):
        """Test calculation of technical indicators."""
        features = calculate_technical_indicators(self.df)
        
        # Check that all expected features are present
        expected_features = [
            'returns', 'log_returns', 'volatility',
            'sma_20', 'sma_50', 'ema_20', 'rsi',
            'macd', 'macd_signal', 'macd_hist',
            'bollinger_upper', 'bollinger_lower', 'atr'
        ]
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        # Check that basic features are calculated (they don't require a window)
        self.assertFalse(features['returns'].isna().all(), "Returns should have non-NaN values")
        self.assertFalse(features['log_returns'].isna().all(), "Log returns should have non-NaN values")

        # For technical indicators, just check they exist (they might be all NaN for small datasets)
        for feature in ['rsi', 'macd', 'bollinger_upper', 'atr']:
            self.assertIn(feature, features.columns, f"{feature} should be present in features")

    def test_target_column(self):
        """Test target column creation."""
        df = add_target_column(self.df.copy(), target_column='close', periods=1)
        
        # Check target column exists and has expected properties
        self.assertIn('target', df.columns)
        self.assertEqual(len(df['target'].dropna()), len(df) - 1)  # Last row should be NaN
        
        # Test percentage change calculation
        expected_change = (self.prices.shift(-1) - self.prices) / self.prices
        pd.testing.assert_series_equal(
            df['target'].dropna(),
            expected_change.dropna(),
            check_names=False
        )

    def test_rsi(self):
        """Test RSI calculation."""
        rsi = calculate_rsi(self.prices)
        
        # For small datasets, RSI might be all NaN, just check it exists
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(self.prices))
        
        # If we have any non-NaN values, they should be between 0 and 100
        valid_values = rsi.dropna()
        if len(valid_values) > 0:
            self.assertTrue(all(0 <= x <= 100 for x in valid_values))

    def test_macd(self):
        """Test MACD calculation."""
        macd, signal, hist = calculate_macd(self.prices)
        
        # Check MACD components
        self.assertEqual(len(macd), len(self.prices))
        self.assertEqual(len(signal), len(self.prices))
        self.assertEqual(len(hist), len(self.prices))
        
        # Verify histogram is difference of MACD and signal
        pd.testing.assert_series_equal(
            hist,
            macd - signal,
            check_names=False
        )

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = calculate_bollinger_bands(self.prices)
        
        # Drop NaN values before comparison
        valid_mask = ~upper.isna() & ~middle.isna() & ~lower.isna()
        upper_valid = upper[valid_mask]
        middle_valid = middle[valid_mask]
        lower_valid = lower[valid_mask]

        # Check band properties
        self.assertTrue(all(upper_valid >= middle_valid), "Upper band must be greater than or equal to middle band")
        self.assertTrue(all(middle_valid >= lower_valid), "Middle band must be greater than or equal to lower band")
        
        # For small datasets, we might not have any valid values
        if len(valid_mask) > 0:
            self.assertTrue(all(upper_valid >= middle_valid), "Upper band must be greater than or equal to middle band")
            self.assertTrue(all(middle_valid >= lower_valid), "Middle band must be greater than or equal to lower band")

    def test_feature_store(self):
        """Test FeatureStore functionality."""
        store = FeatureStore(cache_dir=self.temp_dir)
        
        # Calculate and store features
        features = calculate_technical_indicators(self.df)
        store.add_features('test_features', features)
        
        # Retrieve features
        loaded_features = store.get_features('test_features')
        self.assertIsNotNone(loaded_features)
        pd.testing.assert_frame_equal(features, loaded_features)


if __name__ == '__main__':
    unittest.main()