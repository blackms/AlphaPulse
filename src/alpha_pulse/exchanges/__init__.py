"""
Exchange module initialization.

This module provides a flexible and extensible system for interacting with
cryptocurrency exchanges. It uses modern design patterns and best practices
to ensure maintainability and ease of extension.

Example usage:
    ```python
    from alpha_pulse.exchanges import ExchangeFactory, ExchangeType
    
    # Create exchange instance
    exchange = ExchangeFactory.create_exchange(
        exchange_type=ExchangeType.BINANCE,
        api_key="your-api-key",
        api_secret="your-api-secret",
        testnet=True
    )
    
    # Use exchange
    async with exchange:
        # Get account balances
        balances = await exchange.get_balances()
        
        # Execute trade
        order = await exchange.execute_trade(
            symbol="BTC/USDT",
            side="buy",
            amount=0.1
        )
    ```
"""
from typing import Dict, Any

# Core interfaces and types
from .interfaces import (
    ExchangeAdapter,
    MarketDataProvider,
    TradingOperations,
    AccountOperations,
    ExchangeConnection,
    ExchangeConfiguration,
    ExchangeError,
    ConnectionError,
    OrderError,
    MarketDataError,
    ConfigurationError
)
from .base import Balance, OHLCV
from .types import ExchangeType

# Factory for creating exchanges
from .factories import ExchangeFactory, ExchangeRegistry

# Exchange implementations
from .implementations.binance import BinanceExchange
from .implementations.bybit import BybitExchange

# Register implementations with factory
ExchangeRegistry.register(
    ExchangeType.BINANCE,
    BinanceExchange,
    'binance',
    defaultType='spot',
    adjustForTimeDifference=True,
    recvWindow=60000
)

ExchangeRegistry.register(
    ExchangeType.BYBIT,
    BybitExchange,
    'bybit',
    defaultType='spot',
    adjustForTimeDifference=True,
    recvWindow=60000,
    createMarketBuyOrderRequiresPrice=True
)

# Expose public API
__all__ = [
    # Core interfaces
    "ExchangeAdapter",
    "MarketDataProvider",
    "TradingOperations",
    "AccountOperations",
    "ExchangeConnection",
    "ExchangeConfiguration",
    
    # Data classes
    "Balance",
    "OHLCV",
    
    # Factory and registry
    "ExchangeFactory",
    "ExchangeRegistry",
    
    # Exchange implementations
    "BinanceExchange",
    "BybitExchange",
    
    # Types and enums
    "ExchangeType",
    
    # Exceptions
    "ExchangeError",
    "ConnectionError",
    "OrderError",
    "MarketDataError",
    "ConfigurationError"
]