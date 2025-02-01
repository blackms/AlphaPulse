"""
Exchange module initialization.
"""
from enum import Enum
from typing import Optional, Type

from loguru import logger

from .base import BaseExchange, Balance, OHLCV
from .binance import BinanceExchange
from .bybit import BybitExchange
from .credentials.manager import credentials_manager


class ExchangeType(Enum):
    """Supported exchange types."""
    BINANCE = 'binance'
    BYBIT = 'bybit'


class ExchangeFactory:
    """Factory for creating exchange instances."""
    
    _exchange_map = {
        ExchangeType.BINANCE: BinanceExchange,
        ExchangeType.BYBIT: BybitExchange
    }
    
    @classmethod
    async def create_exchange(
        cls,
        exchange_type: ExchangeType,
        testnet: bool = False
    ) -> BaseExchange:
        """Create and initialize an exchange instance.
        
        Args:
            exchange_type: Type of exchange to create
            testnet: Whether to use testnet
            
        Returns:
            Initialized exchange instance
            
        Raises:
            ValueError: If exchange type is not supported
        """
        exchange_class = cls._exchange_map.get(exchange_type)
        if not exchange_class:
            raise ValueError(f"Unsupported exchange type: {exchange_type}")
        
        try:
            exchange = exchange_class(testnet=testnet)
            await exchange.initialize()
            return exchange
            
        except Exception as e:
            logger.error(f"Error creating {exchange_type.value} exchange: {e}")
            raise


# Export public API
__all__ = [
    'ExchangeType',
    'ExchangeFactory',
    'BaseExchange',
    'Balance',
    'OHLCV',
    'credentials_manager'
]