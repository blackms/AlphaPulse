"""
Unit tests for portfolio management components.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.portfolio.strategies.mpt_strategy import MPTStrategy
from alpha_pulse.portfolio.strategies.hrp_strategy import HRPStrategy
from alpha_pulse.portfolio.strategies.black_litterman_strategy import BlackLittermanStrategy
from alpha_pulse.portfolio.strategies.llm_assisted_strategy import LLMAssistedStrategy
from alpha_pulse.exchanges.mock import MockExchange


class TestPortfolioStrategies(unittest.TestCase):
    """Test suite for portfolio strategies."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create sample historical data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        assets = ['BTC', 'ETH', 'BNB', 'SOL', 'USDT', 'USDC']
        
        data = {}
        for asset in assets:
            if asset in ['USDT', 'USDC']:
                data[asset] = np.ones(100)
            else:
                returns = np.random.normal(0.0002, 0.02, 100)
                data[asset] = 100 * np.exp(np.cumsum(returns))
        
        cls.historical_data = pd.DataFrame(data, index=dates)
        
        # Sample configuration
        cls.config = {
            'min_position_size': 0.05,
            'max_position_size': 0.4,
            'rebalancing_threshold': 0.1,
            'stablecoin_fraction': 0.3,
            'allowed_assets': assets,
            'risk_aversion': 2.5,
            'optimization_objective': 'sharpe'
        }

    def test_mpt_strategy(self):
        """Test Modern Portfolio Theory strategy."""
        strategy = MPTStrategy(self.config)
        
        current_allocation = {
            'BTC': 0.3,
            'ETH': 0.2,
            'BNB': 0.1,
            'SOL': 0.1,
            'USDT': 0.15,
            'USDC': 0.15
        }
        
        result = strategy.compute_target_allocation(
            current_allocation,
            self.historical_data,
            {'volatility_target': 0.15}
        )
        
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        self.assertTrue(all(w >= 0.05 and w <= 0.4 for w in result.values()))

    def test_hrp_strategy(self):
        """Test Hierarchical Risk Parity strategy."""
        strategy = HRPStrategy(self.config)
        
        current_allocation = {
            'BTC': 0.25,
            'ETH': 0.25,
            'BNB': 0.1,
            'SOL': 0.1,
            'USDT': 0.15,
            'USDC': 0.15
        }
        
        result = strategy.compute_target_allocation(
            current_allocation,
            self.historical_data,
            {'volatility_target': 0.15}
        )
        
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        self.assertTrue(all(w >= 0.05 and w <= 0.4 for w in result.values()))

    def test_black_litterman_strategy(self):
        """Test Black-Litterman strategy."""
        config = self.config.copy()
        config.update({
            'market_cap_weights': {
                'BTC': 0.4,
                'ETH': 0.2,
                'BNB': 0.1,
                'SOL': 0.1,
                'USDT': 0.1,
                'USDC': 0.1
            }
        })
        
        strategy = BlackLittermanStrategy(config)
        
        current_allocation = {
            'BTC': 0.3,
            'ETH': 0.2,
            'BNB': 0.1,
            'SOL': 0.1,
            'USDT': 0.15,
            'USDC': 0.15
        }
        
        result = strategy.compute_target_allocation(
            current_allocation,
            self.historical_data,
            {'volatility_target': 0.15}
        )
        
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        self.assertTrue(all(w >= 0.05 and w <= 0.4 for w in result.values()))

    @patch('alpha_pulse.portfolio.strategies.llm_assisted_strategy.LLMAssistedStrategy._get_llm_analysis')
    def test_llm_assisted_strategy(self, mock_llm):
        """Test LLM-assisted strategy."""
        mock_llm.return_value = "Analysis suggests maintaining current allocation"
        
        base_strategy = MPTStrategy(self.config)
        config = self.config.copy()
        config['llm'] = {
            'enabled': True,
            'model_name': 'gpt-4',
            'temperature': 0.7
        }
        
        strategy = LLMAssistedStrategy(base_strategy, config)
        
        current_allocation = {
            'BTC': 0.3,
            'ETH': 0.2,
            'BNB': 0.1,
            'SOL': 0.1,
            'USDT': 0.15,
            'USDC': 0.15
        }
        
        result = strategy.compute_target_allocation(
            current_allocation,
            self.historical_data,
            {'volatility_target': 0.15}
        )
        
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        self.assertTrue(all(w >= 0.05 and w <= 0.4 for w in result.values()))
        mock_llm.assert_called()


class TestPortfolioManager(unittest.TestCase):
    """Test suite for portfolio manager."""

    def setUp(self):
        """Set up test environment."""
        self.config_path = Path(__file__).parent.parent / "portfolio" / "portfolio_config.yaml"
        
        # Create mock exchange
        self.exchange = MockExchange(
            initial_balances={
                'BTC': 1.0,
                'ETH': 10.0,
                'BNB': 50.0,
                'SOL': 100.0,
                'USDT': 50000.0,
                'USDC': 50000.0
            },
            price_data=pd.DataFrame({
                'BTC': [40000] * 10,
                'ETH': [2000] * 10,
                'BNB': [200] * 10,
                'SOL': [100] * 10,
                'USDT': [1] * 10,
                'USDC': [1] * 10
            })
        )

    def test_portfolio_manager_initialization(self):
        """Test portfolio manager initialization."""
        manager = PortfolioManager(str(self.config_path))
        self.assertIsInstance(manager, PortfolioManager)

    def test_get_current_allocation(self):
        """Test getting current allocation."""
        manager = PortfolioManager(str(self.config_path))
        allocation = manager.get_current_allocation(self.exchange)
        
        self.assertIsInstance(allocation, dict)
        self.assertAlmostEqual(sum(allocation.values()), 1.0, places=6)

    def test_compute_rebalancing_trades(self):
        """Test computing rebalancing trades."""
        manager = PortfolioManager(str(self.config_path))
        
        current = {
            'BTC': 0.3,
            'ETH': 0.2,
            'USDT': 0.5
        }
        
        target = {
            'BTC': 0.25,
            'ETH': 0.25,
            'USDT': 0.5
        }
        
        trades = manager.compute_rebalancing_trades(current, target, 100000)
        
        self.assertIsInstance(trades, list)
        self.assertTrue(all(isinstance(t, dict) for t in trades))
        self.assertTrue(all(set(t.keys()) >= {'asset', 'value', 'type'} for t in trades))


if __name__ == '__main__':
    unittest.main()