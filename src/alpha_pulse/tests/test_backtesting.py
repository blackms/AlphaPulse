"""
Unit tests for the AlphaPulse backtesting framework.
"""
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from ..backtesting import (
    Backtester,
    DefaultStrategy,
    TrendFollowingStrategy,
    MeanReversionStrategy,
    Position
)


class TestBacktester(unittest.TestCase):
    """Test cases for the Backtester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.dates = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 10),
            freq='D'
        )
        self.backtester = Backtester(
            commission=0.001,
            initial_capital=100000
        )

    def test_basic_functionality(self):
        """Test basic backtesting functionality with simple up trend."""
        # Create upward trending prices and matching signals
        prices = pd.Series(
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            index=self.dates
        )
        signals = pd.Series([1.0] * len(prices), index=self.dates)
        
        results = self.backtester.backtest(
            prices=prices,
            signals=signals
        )
        
        self.assertGreater(results.total_return, 0)
        self.assertEqual(results.total_trades, 1)  # Single trade held throughout
        self.assertEqual(results.winning_trades, 1)
        self.assertEqual(results.losing_trades, 0)
        self.assertEqual(results.win_rate, 1.0)

    def test_no_trades(self):
        """Test behavior when no trades are executed."""
        prices = pd.Series(
            [100] * len(self.dates),
            index=self.dates
        )
        signals = pd.Series([0.0] * len(self.dates), index=self.dates)
        
        results = self.backtester.backtest(
            prices=prices,
            signals=signals
        )
        
        self.assertEqual(results.total_return, 0.0)
        self.assertEqual(results.total_trades, 0)
        self.assertEqual(results.winning_trades, 0)
        self.assertEqual(results.losing_trades, 0)
        self.assertEqual(results.sharpe_ratio, 0.0)

    def test_commission_impact(self):
        """Test the impact of different commission rates."""
        prices = pd.Series(
            [100, 110, 100, 110, 100],
            index=self.dates[:5]
        )
        signals = pd.Series([1, 1, -1, 1, -1], index=self.dates[:5])
        
        # Test with no commission
        results_no_comm = self.backtester.backtest(
            prices=prices,
            signals=signals
        )
        
        # Test with high commission
        backtester_high_comm = Backtester(
            commission=0.01,  # 1% commission
            initial_capital=100000,
            position_size=1.0
        )
        results_high_comm = backtester_high_comm.backtest(
            prices=prices,
            signals=signals
        )
        
        self.assertGreater(
            results_no_comm.total_return,
            results_high_comm.total_return
        )

    def test_different_strategies(self):
        """Test different trading strategies on the same data."""
        prices = pd.Series(
            [100, 95, 90, 85, 80, 85, 90, 95, 100, 105],
            index=self.dates
        )
        signals = pd.Series(
            [-0.5, -1.0, -1.5, -2.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5],
            index=self.dates
        )
        
        # Test trend following strategy
        trend_results = self.backtester.backtest(
            prices=prices,
            signals=signals
        )
        
        # Test mean reversion strategy
        mean_rev_results = self.backtester.backtest(
            prices=prices,
            signals=signals
        )
        
        # Different strategies should produce different results
        self.assertNotEqual(
            trend_results.total_return,
            mean_rev_results.total_return
        )

    def test_position_sizing(self):
        """Test different position sizing impacts."""
        prices = pd.Series(
            [100, 110, 120, 130, 140],
            index=self.dates[:5]
        )
        signals = pd.Series([1.0] * 5, index=self.dates[:5])
        
        # Test with full position size
        full_size_backtester = Backtester(
            commission=0.001,
            initial_capital=100000
        )
        full_results = full_size_backtester.backtest(
            prices=prices,
            signals=signals
        )
        
        # Test with half position size
        half_size_backtester = Backtester(
            commission=0.001,
            initial_capital=100000
        )
        half_results = half_size_backtester.backtest(
            prices=prices,
            signals=signals
        )
        
        # Full position size should have approximately double the returns
        self.assertAlmostEqual(
            full_results.total_return / half_results.total_return,
            2.0,
            places=1
        )

    def test_edge_cases(self):
        """Test various edge cases and error conditions."""
        prices = pd.Series([100, 101, 102], index=self.dates[:3])
        signals = pd.Series([1, 1, 1], index=self.dates[:3])
        
        # Test mismatched data lengths
        wrong_signals = pd.Series([1, 1], index=self.dates[:2])
        with self.assertRaises(ValueError):
            self.backtester.backtest(
                prices=prices,
                signals=wrong_signals
            )
        
        # Test invalid position size
        invalid_backtester = Backtester(
            commission=0.001,
            initial_capital=100000,
            position_size=0.0  # Should not allow zero position size
        )
        results = invalid_backtester.backtest(
            prices=prices,
            signals=signals
        )
        self.assertEqual(results.total_trades, 0)
        
        # Test negative prices
        negative_prices = pd.Series([-100, -101, -102], index=self.dates[:3])
        with self.assertRaises(ValueError):
            self.backtester.backtest(
                prices=negative_prices,
                signals=signals
            )


if __name__ == '__main__':
    unittest.main()