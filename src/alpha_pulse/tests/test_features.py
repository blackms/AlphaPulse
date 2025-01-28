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
        
        # Check that there are no NaN values in key features
        self.assertFalse(features['returns'].isna().all())
        self.assertFalse(features['rsi'].isna().all())
        self.assertFalse(features['macd'].isna().all())

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
        
        # Check RSI properties
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))
        # For small datasets, we expect all values except the first one
        self.assertEqual(len(rsi.dropna()), len(self.prices) - 1)

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
        
        # Check band properties
        self.assertTrue(all(upper >= middle))
        self.assertTrue(all(middle >= lower))
        self.assertEqual(len(upper.dropna()), len(self.prices) - 19)  # Default window is 20

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