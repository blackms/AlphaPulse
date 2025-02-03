"""
Exchange data provider for real-time market data.
"""
import asyncio
from decimal import Decimal
from typing import Optional, Dict
from loguru import logger

from alpha_pulse.exchanges import ExchangeFactory, ExchangeType, BaseExchange


class ExchangeDataProvider:
    """Provides real-time market data from exchanges."""
    
    def __init__(self, exchange_type: ExchangeType = ExchangeType.BINANCE, testnet: bool = True):
        """
        Initialize exchange data provider.
        
        Args:
            exchange_type: Type of exchange to use
            testnet: Whether to use testnet
        """
        self.exchange_type = exchange_type
        self.testnet = testnet
        self.exchange: Optional[BaseExchange] = None
        self._prices: Dict[str, Decimal] = {}
        self._running = False
        
    async def initialize(self) -> None:
        """Initialize exchange connection."""
        try:
            self.exchange = await ExchangeFactory.create_exchange(
                self.exchange_type,
                testnet=self.testnet
            )
            logger.info(f"Initialized {self.exchange_type.value} data provider")
        except Exception as e:
            logger.error(f"Error initializing exchange data provider: {e}")
            raise
            
    async def start(self, symbols: list[str]) -> None:
        """
        Start market data updates.
        
        Args:
            symbols: List of symbols to track
        """
        if not self.exchange:
            await self.initialize()
            
        self._running = True
        
        while self._running:
            try:
                for symbol in symbols:
                    price = await self.exchange.get_ticker_price(symbol)
                    if price:
                        self._prices[symbol] = price
                        
                # Update every second
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error updating market data: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    def stop(self) -> None:
        """Stop market data updates."""
        self._running = False
        
    def get_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            
        Returns:
            Current price or None if not available
        """
        return self._prices.get(symbol)
        
    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current price directly from exchange.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            
        Returns:
            Current price or None if not available
        """
        if not self.exchange:
            await self.initialize()
            
        try:
            return await self.exchange.get_ticker_price(symbol)
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
            
    async def close(self) -> None:
        """Close exchange connection."""
        self.stop()
        if self.exchange:
            await self.exchange.close()