"""
Unit tests for risk management components.
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from alpha_pulse.risk_management import (
    RiskManager,
    RiskConfig,
    RiskAnalyzer,
    RiskMetrics,
    AdaptivePositionSizer,
    AdaptivePortfolioOptimizer,
    PortfolioConstraints,
)


class TestRiskAnalyzer(unittest.TestCase):
    """Test risk analysis calculations."""

    def setUp(self):
        """Set up test environment."""
        self.analyzer = RiskAnalyzer()
        
        # Generate sample return data
        np.random.seed(42)
        self.returns = pd.Series(
            np.random.normal(0.001, 0.02, 252),
            index=pd.date_range(
                start='2024-01-01',
                periods=252,
                freq='D'
            )
        )

    def test_risk_metrics_calculation(self):
        """Test calculation of risk metrics."""
        metrics = self.analyzer.calculate_metrics(self.returns)
        
        self.assertIsInstance(metrics, RiskMetrics)
        self.assertTrue(0 < metrics.volatility < 1)
        self.assertTrue(metrics.var_95 > 0)
        self.assertTrue(metrics.max_drawdown < 0)
        self.assertIsInstance(metrics.sharpe_ratio, float)

    def test_var_calculation_methods(self):
        """Test different VaR calculation methods."""
        # Historical VaR
        hist_var = self.analyzer.calculate_var(
            self.returns,
            confidence_level=0.95,
            method="historical"
        )
        
        # Parametric VaR
        param_var = self.analyzer.calculate_var(
            self.returns,
            confidence_level=0.95,
            method="parametric"
        )
        
        # Monte Carlo VaR
        mc_var = self.analyzer.calculate_var(
            self.returns,
            confidence_level=0.95,
            method="monte_carlo"
        )
        
        # All methods should give similar results
        self.assertAlmostEqual(hist_var, param_var, places=2)
        self.assertAlmostEqual(hist_var, mc_var, places=2)

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        prices = (1 + self.returns).cumprod()
        drawdowns = self.analyzer.calculate_drawdown(prices)
        
        self.assertIsInstance(drawdowns, pd.Series)
        self.assertTrue(all(drawdowns <= 0))  # Drawdowns should be negative
        self.assertEqual(len(drawdowns), len(prices))


class TestPositionSizer(unittest.TestCase):
    """Test position sizing calculations."""

    def setUp(self):
        """Set up test environment."""
        self.position_sizer = AdaptivePositionSizer()
        self.portfolio_value = 100000.0
        self.symbol = "BTC/USD"
        self.current_price = 50000.0

    def test_position_size_calculation(self):
        """Test basic position size calculation."""
        result = self.position_sizer.calculate_position_size(
            symbol=self.symbol,
            current_price=self.current_price,
            portfolio_value=self.portfolio_value,
            volatility=0.02,
            signal_strength=0.8,
        )
        
        self.assertTrue(0 <= result.size <= self.portfolio_value)
        self.assertTrue(0 <= result.confidence <= 1)
        self.assertIsInstance(result.metrics, dict)

    def test_position_size_limits(self):
        """Test position size respects limits."""
        # Test with high volatility (should reduce size)
        high_vol_result = self.position_sizer.calculate_position_size(
            symbol=self.symbol,
            current_price=self.current_price,
            portfolio_value=self.portfolio_value,
            volatility=0.5,  # High volatility
            signal_strength=0.8,
        )
        
        # Test with low volatility (should allow larger size)
        low_vol_result = self.position_sizer.calculate_position_size(
            symbol=self.symbol,
            current_price=self.current_price,
            portfolio_value=self.portfolio_value,
            volatility=0.01,  # Low volatility
            signal_strength=0.8,
        )
        
        self.assertGreater(low_vol_result.size, high_vol_result.size)


class TestPortfolioOptimizer(unittest.TestCase):
    """Test portfolio optimization."""

    def setUp(self):
        """Set up test environment."""
        self.optimizer = AdaptivePortfolioOptimizer()
        
        # Generate sample return data for multiple assets
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        self.returns = pd.DataFrame({
            'BTC': np.random.normal(0.001, 0.02, 252),
            'ETH': np.random.normal(0.002, 0.03, 252),
            'BNB': np.random.normal(0.001, 0.025, 252),
        }, index=dates)

    def test_portfolio_optimization(self):
        """Test portfolio weight optimization."""
        weights = self.optimizer.optimize(
            self.returns,
            risk_free_rate=0.0,
            constraints=PortfolioConstraints(
                min_weight=0.0,
                max_weight=1.0,
                max_total_weight=1.0,
            )
        )
        
        self.assertIsInstance(weights, dict)
        self.assertEqual(set(weights.keys()), set(self.returns.columns))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertTrue(all(0 <= w <= 1 for w in weights.values()))

    def test_portfolio_constraints(self):
        """Test portfolio optimization respects constraints."""
        # Test with leverage constraint
        leveraged_weights = self.optimizer.optimize(
            self.returns,
            constraints=PortfolioConstraints(
                min_weight=0.0,
                max_weight=1.0,
                max_total_weight=1.5,  # Allow 150% leverage
            )
        )
        
        self.assertLessEqual(sum(leveraged_weights.values()), 1.5)


class TestRiskManager(unittest.TestCase):
    """Test comprehensive risk management system."""

    def setUp(self):
        """Set up test environment."""
        self.risk_manager = RiskManager(
            config=RiskConfig(
                max_position_size=0.2,
                max_portfolio_leverage=1.5,
                max_drawdown=0.25,
                stop_loss=0.1,
            )
        )
        
        self.portfolio_value = 100000.0
        self.symbol = "BTC/USD"
        self.current_price = 50000.0

    def test_trade_evaluation(self):
        """Test trade evaluation logic."""
        # Test valid trade
        valid_trade = self.risk_manager.evaluate_trade(
            symbol=self.symbol,
            side="buy",
            quantity=0.1,  # Small position
            current_price=self.current_price,
            portfolio_value=self.portfolio_value,
            current_positions={},
        )
        self.assertTrue(valid_trade)
        
        # Test oversized trade
        invalid_trade = self.risk_manager.evaluate_trade(
            symbol=self.symbol,
            side="buy",
            quantity=10.0,  # Very large position
            current_price=self.current_price,
            portfolio_value=self.portfolio_value,
            current_positions={},
        )
        self.assertFalse(invalid_trade)

    def test_stop_loss_calculation(self):
        """Test stop-loss price calculation."""
        positions = {
            self.symbol: {
                'quantity': 1.0,
                'avg_entry_price': self.current_price,
                'current_price': self.current_price,
            }
        }
        
        stop_losses = self.risk_manager.get_stop_loss_prices(positions)
        
        self.assertIn(self.symbol, stop_losses)
        expected_stop = self.current_price * (1 - self.risk_manager.config.stop_loss)
        self.assertAlmostEqual(stop_losses[self.symbol], expected_stop)

    def test_risk_metrics_update(self):
        """Test risk metrics update process."""
        # Generate sample return data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        portfolio_returns = pd.Series(
            np.random.normal(0.001, 0.02, 100),
            index=dates
        )
        asset_returns = {
            'BTC/USD': pd.Series(np.random.normal(0.001, 0.02, 100), index=dates),
            'ETH/USD': pd.Series(np.random.normal(0.002, 0.03, 100), index=dates),
        }
        
        # Update metrics
        self.risk_manager.update_risk_metrics(portfolio_returns, asset_returns)
        
        # Get risk report
        report = self.risk_manager.get_risk_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('risk_metrics', report)
        self.assertIsInstance(report['risk_metrics'], dict)


if __name__ == '__main__':
    unittest.main()