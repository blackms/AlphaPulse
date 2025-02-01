"""
Unit tests for portfolio module.
"""
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from alpha_pulse.portfolio.allocation_strategy import AllocationStrategy
from alpha_pulse.portfolio.mpt_strategy import MPTStrategy
from alpha_pulse.portfolio.hrp_strategy import HRPStrategy
from alpha_pulse.portfolio.analyzer import PortfolioAnalyzer
from alpha_pulse.exchange_conn.interface import Balance


class TestAllocationStrategy:
    """Test base allocation strategy."""
    
    def test_calculate_rebalance_score(self):
        """Test rebalance score calculation."""
        strategy = MPTStrategy()  # Use concrete implementation for testing
        
        # Test perfect match
        current = {'BTC': Decimal('0.5'), 'ETH': Decimal('0.5')}
        target = {'BTC': Decimal('0.5'), 'ETH': Decimal('0.5')}
        score = strategy.calculate_rebalance_score(current, target)
        assert score == Decimal('1.0')
        
        # Test complete mismatch
        current = {'BTC': Decimal('1.0'), 'ETH': Decimal('0.0')}
        target = {'BTC': Decimal('0.0'), 'ETH': Decimal('1.0')}
        score = strategy.calculate_rebalance_score(current, target)
        assert score == Decimal('0.0')
        
        # Test partial mismatch
        current = {'BTC': Decimal('0.7'), 'ETH': Decimal('0.3')}
        target = {'BTC': Decimal('0.3'), 'ETH': Decimal('0.7')}
        score = strategy.calculate_rebalance_score(current, target)
        assert Decimal('0.0') < score < Decimal('1.0')


@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    data = {
        'BTC': np.random.normal(0.001, 0.02, 100),
        'ETH': np.random.normal(0.002, 0.03, 100),
        'BNB': np.random.normal(0.001, 0.025, 100)
    }
    return pd.DataFrame(data, index=dates)


class TestMPTStrategy:
    """Test Modern Portfolio Theory strategy."""
    
    def test_mpt_allocation(self, sample_returns):
        """Test MPT allocation calculation."""
        strategy = MPTStrategy()
        current_weights = {
            'BTC': Decimal('0.4'),
            'ETH': Decimal('0.3'),
            'BNB': Decimal('0.3')
        }
        
        result = strategy.calculate_allocation(
            sample_returns,
            current_weights,
            constraints={
                'min_weight': 0.1,
                'max_weight': 0.5
            }
        )
        
        # Check results
        assert isinstance(result.weights, dict)
        assert all(0.1 <= float(w) <= 0.5 for w in result.weights.values())
        assert abs(sum(float(w) for w in result.weights.values()) - 1.0) < 1e-6
        assert result.expected_return > 0
        assert result.expected_risk > 0
        assert result.sharpe_ratio > 0


class TestHRPStrategy:
    """Test Hierarchical Risk Parity strategy."""
    
    def test_hrp_allocation(self, sample_returns):
        """Test HRP allocation calculation."""
        strategy = HRPStrategy()
        current_weights = {
            'BTC': Decimal('0.4'),
            'ETH': Decimal('0.3'),
            'BNB': Decimal('0.3')
        }
        
        result = strategy.calculate_allocation(
            sample_returns,
            current_weights,
            constraints={
                'min_weight': 0.1,
                'max_weight': 0.5
            }
        )
        
        # Check results
        assert isinstance(result.weights, dict)
        assert all(0.1 <= float(w) <= 0.5 for w in result.weights.values())
        assert abs(sum(float(w) for w in result.weights.values()) - 1.0) < 1e-6
        assert result.expected_return > 0
        assert result.expected_risk > 0
        assert result.sharpe_ratio > 0


class TestPortfolioAnalyzer:
    """Test portfolio analyzer."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange."""
        exchange = AsyncMock()
        exchange.get_balances.return_value = {
            'BTC': Balance(
                free=Decimal('1.0'),
                locked=Decimal('0.0'),
                total=Decimal('1.0'),
                in_base_currency=Decimal('40000')
            ),
            'ETH': Balance(
                free=Decimal('10.0'),
                locked=Decimal('0.0'),
                total=Decimal('10.0'),
                in_base_currency=Decimal('30000')
            ),
            'USDT': Balance(
                free=Decimal('30000'),
                locked=Decimal('0.0'),
                total=Decimal('30000'),
                in_base_currency=Decimal('30000')
            )
        }
        return exchange
    
    @pytest.mark.asyncio
    async def test_get_current_allocation(self, mock_exchange):
        """Test getting current allocation."""
        analyzer = PortfolioAnalyzer(mock_exchange)
        weights = await analyzer.get_current_allocation()
        
        total = sum(float(w) for w in weights.values())
        assert abs(total - 1.0) < 1e-6
        assert all(0 <= float(w) <= 1 for w in weights.values())
    
    @pytest.mark.asyncio
    async def test_get_rebalancing_trades(self, mock_exchange):
        """Test calculating rebalancing trades."""
        analyzer = PortfolioAnalyzer(mock_exchange)
        
        current_weights = {
            'BTC': Decimal('0.4'),
            'ETH': Decimal('0.3'),
            'USDT': Decimal('0.3')
        }
        
        target_weights = {
            'BTC': Decimal('0.3'),
            'ETH': Decimal('0.4'),
            'USDT': Decimal('0.3')
        }
        
        total_value = Decimal('100000')
        trades = analyzer.get_rebalancing_trades(
            current_weights,
            target_weights,
            total_value,
            min_trade_value=Decimal('100')
        )
        
        assert isinstance(trades, list)
        assert all(isinstance(t, dict) for t in trades)
        assert all(set(t.keys()) == {'asset', 'side', 'value'} for t in trades)