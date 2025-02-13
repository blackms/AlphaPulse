"""
Tests for portfolio management components.
"""
import pytest
import pytest_asyncio
import pandas as pd
import numpy as np
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.portfolio.strategies.mpt_strategy import MPTStrategy
from alpha_pulse.portfolio.strategies.hrp_strategy import HRPStrategy
from alpha_pulse.portfolio.strategies.black_litterman_strategy import BlackLittermanStrategy
from alpha_pulse.portfolio.strategies.llm_assisted_strategy import LLMAssistedStrategy
from alpha_pulse.exchanges.implementations.binance import BinanceExchange


@pytest.fixture
def test_data():
    """Create test market data."""
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
    
    return {
        'historical_data': pd.DataFrame(data, index=dates),
        'assets': assets
    }


@pytest.fixture
def strategy_config():
    """Create test strategy configuration."""
    return {
        'min_position_size': 0.05,
        'max_position_size': 0.4,
        'rebalancing_threshold': 0.1,
        'stablecoin_fraction': 0.3,
        'allowed_assets': ['BTC', 'ETH', 'BNB', 'SOL', 'USDT', 'USDC'],
        'risk_aversion': 2.5,
        'optimization_objective': 'sharpe'
    }


class MockBalance:
    def __init__(self, total, available=None, locked=None):
        self.total = Decimal(str(total))
        self.available = Decimal(str(available or total))
        self.locked = Decimal(str(locked or 0))

@pytest_asyncio.fixture
async def exchange():
    """Create mock exchange."""
    exchange = BinanceExchange(testnet=True)
    await exchange.initialize()
    
    # Mock exchange methods
    async def mock_get_balances():
        return {
            'BTC': MockBalance(1.0),
            'ETH': MockBalance(10.0),
            'BNB': MockBalance(50.0),
            'SOL': MockBalance(100.0),
            'USDT': MockBalance(50000.0),
            'USDC': MockBalance(50000.0)
        }
    
    async def mock_get_ticker_price(symbol):
        prices = {
            'BTC/USDT': Decimal('40000'),
            'ETH/USDT': Decimal('2000'),
            'BNB/USDT': Decimal('200'),
            'SOL/USDT': Decimal('100'),
            'USDC/USDT': Decimal('1')
        }
        return prices.get(symbol, Decimal('1'))
    
    async def mock_get_portfolio_value():
        return Decimal('100000')
    
    # Apply mocks
    exchange.get_balances = mock_get_balances
    exchange.get_ticker_price = mock_get_ticker_price
    exchange.get_portfolio_value = mock_get_portfolio_value
    
    yield exchange
    await exchange.close()


@pytest.mark.asyncio
async def test_mpt_strategy(test_data, strategy_config):
    """Test Modern Portfolio Theory strategy."""
    strategy = MPTStrategy(strategy_config)
    
    current_allocation = {
        'BTC': Decimal('0.3'),
        'ETH': Decimal('0.2'),
        'BNB': Decimal('0.1'),
        'SOL': Decimal('0.1'),
        'USDT': Decimal('0.15'),
        'USDC': Decimal('0.15')
    }
    
    result = strategy.compute_target_allocation(
        current_allocation,
        test_data['historical_data'],
        {'volatility_target': 0.15}
    )
    
    assert isinstance(result, dict)
    assert abs(sum(Decimal(str(v)) for v in result.values()) - Decimal('1.0')) < Decimal('0.000001')
    assert all(Decimal('0.05') <= Decimal(str(w)) <= Decimal('0.4') for w in result.values())


@pytest.mark.asyncio
async def test_hrp_strategy(test_data, strategy_config):
    """Test Hierarchical Risk Parity strategy."""
    strategy = HRPStrategy(strategy_config)
    
    current_allocation = {
        'BTC': Decimal('0.25'),
        'ETH': Decimal('0.25'),
        'BNB': Decimal('0.1'),
        'SOL': Decimal('0.1'),
        'USDT': Decimal('0.15'),
        'USDC': Decimal('0.15')
    }
    
    result = strategy.compute_target_allocation(
        current_allocation,
        test_data['historical_data'],
        {'volatility_target': 0.15}
    )
    
    assert isinstance(result, dict)
    assert abs(sum(Decimal(str(v)) for v in result.values()) - Decimal('1.0')) < Decimal('0.000001')
    assert all(Decimal('0.05') <= Decimal(str(w)) <= Decimal('0.4') for w in result.values())


@pytest.mark.asyncio
async def test_black_litterman_strategy(test_data, strategy_config):
    """Test Black-Litterman strategy."""
    config = strategy_config.copy()
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
        'BTC': Decimal('0.3'),
        'ETH': Decimal('0.2'),
        'BNB': Decimal('0.1'),
        'SOL': Decimal('0.1'),
        'USDT': Decimal('0.15'),
        'USDC': Decimal('0.15')
    }
    
    result = strategy.compute_target_allocation(
        current_allocation,
        test_data['historical_data'],
        {'volatility_target': 0.15}
    )
    
    assert isinstance(result, dict)
    assert abs(sum(Decimal(str(v)) for v in result.values()) - Decimal('1.0')) < Decimal('0.000001')
    assert all(Decimal('0.05') <= Decimal(str(w)) <= Decimal('0.4') for w in result.values())


@pytest.mark.asyncio
async def test_llm_assisted_strategy(test_data, strategy_config):
    """Test LLM-assisted strategy."""
    with patch('alpha_pulse.portfolio.strategies.llm_assisted_strategy.LLMAssistedStrategy._get_llm_analysis') as mock_llm:
        mock_llm.return_value = "Analysis suggests maintaining current allocation"
        
        base_strategy = MPTStrategy(strategy_config)
        config = strategy_config.copy()
        config['llm'] = {
            'enabled': True,
            'model_name': 'gpt-4',
            'temperature': 0.7
        }
        
        strategy = LLMAssistedStrategy(base_strategy, config)
        
        current_allocation = {
            'BTC': Decimal('0.3'),
            'ETH': Decimal('0.2'),
            'BNB': Decimal('0.1'),
            'SOL': Decimal('0.1'),
            'USDT': Decimal('0.15'),
            'USDC': Decimal('0.15')
        }
        
        result = strategy.compute_target_allocation(
            current_allocation,
            test_data['historical_data'],
            {'volatility_target': 0.15}
        )
        
        assert isinstance(result, dict)
        assert abs(sum(Decimal(str(v)) for v in result.values()) - Decimal('1.0')) < Decimal('0.000001')
        assert all(Decimal('0.05') <= Decimal(str(w)) <= Decimal('0.4') for w in result.values())
        mock_llm.assert_called()
@pytest.fixture
def portfolio_config():
    """Create test portfolio configuration."""
    return {
        'strategy': {
            'name': 'mpt',
            'lookback_period': 100,
            'rebalancing_threshold': 0.1
        },
        'rebalancing_frequency': 'daily',
        'trading': {
            'base_currency': 'USDT',
            'min_trade_value': 10.0
        },
        'volatility_target': 0.15,
        'max_drawdown_limit': 0.25,
        'correlation_threshold': 0.7
    }


@pytest.fixture
def config_path(tmp_path, portfolio_config):
    """Create temporary config file."""
    config_file = tmp_path / "portfolio_config.yaml"
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(portfolio_config, f)
    return str(config_file)


@pytest.mark.asyncio
async def test_portfolio_manager_initialization(config_path):
    """Test portfolio manager initialization."""
    manager = PortfolioManager(config_path)
    assert isinstance(manager, PortfolioManager)


@pytest.mark.asyncio
async def test_get_current_allocation(config_path, exchange):
    """Test getting current allocation."""
    manager = PortfolioManager(config_path)
    allocation = await manager.get_current_allocation(exchange)
    
    assert isinstance(allocation, dict)
    total = sum(Decimal(str(v)) for v in allocation.values())
    assert abs(total - Decimal('1.0')) < Decimal('0.000001')


@pytest.mark.asyncio
async def test_compute_rebalancing_trades(config_path):
    """Test computing rebalancing trades."""
    manager = PortfolioManager(config_path)
    
    current = {
        'BTC': Decimal('0.3'),
        'ETH': Decimal('0.2'),
        'USDT': Decimal('0.5')
    }
    
    target = {
        'BTC': Decimal('0.25'),
        'ETH': Decimal('0.25'),
        'USDT': Decimal('0.5')
    }
    
    trades = manager.compute_rebalancing_trades(
        current,
        target,
        Decimal('100000')
    )
    
    assert isinstance(trades, list)
    assert all(isinstance(t, dict) for t in trades)
    assert all(set(t.keys()) >= {'asset', 'value', 'type'} for t in trades)


@pytest.mark.asyncio
async def test_rebalance_portfolio(config_path, exchange, test_data):
    """Test full portfolio rebalancing."""
    manager = PortfolioManager(config_path)
    
    # Mock fetch_historical_data to return test data
    async def mock_fetch_historical_data(exchange, assets):
        return test_data['historical_data']
    
    manager._fetch_historical_data = mock_fetch_historical_data
    
    result = await manager.rebalance_portfolio(exchange)
    
    assert isinstance(result, dict)
    assert 'status' in result
    assert result['status'] in ['completed', 'skipped', 'failed']
    
    if result['status'] == 'completed':
        assert 'trades' in result
        assert isinstance(result['trades'], list)
        assert all(isinstance(t, dict) for t in result['trades'])