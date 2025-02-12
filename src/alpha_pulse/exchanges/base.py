"""
Base exchange interface and common functionality.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from decimal import Decimal
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass
from loguru import logger

from ..monitoring.metrics import track_latency, API_LATENCY


@dataclass
class Balance:
    """Exchange balance information."""
    total: Decimal
    available: Decimal
    locked: Decimal = Decimal('0')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Balance':
        """Create Balance from dictionary."""
        return cls(
            total=Decimal(str(data.get('total', '0'))),
            available=Decimal(str(data.get('available', '0'))),
            locked=Decimal(str(data.get('locked', '0')))
        )


@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    @classmethod
    def from_list(cls, data: List[Any]) -> 'OHLCV':
        """Create OHLCV from list."""
        return cls(
            timestamp=datetime.fromtimestamp(data[0] / 1000, tz=timezone.utc),
            open=Decimal(str(data[1])),
            high=Decimal(str(data[2])),
            low=Decimal(str(data[3])),
            close=Decimal(str(data[4])),
            volume=Decimal(str(data[5]))
        )


class BaseExchange(ABC):
    """Base class for exchange implementations."""

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize exchange.

        Args:
            api_key: API key
            api_secret: API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self._session = None

    @track_latency("get_balances")
    @abstractmethod
    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances."""
        pass

    @track_latency("get_ticker_price")
    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol."""
        pass

    @track_latency("get_portfolio_value")
    @abstractmethod
    async def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in base currency."""
        pass

    @track_latency("fetch_ohlcv")
    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """
        Fetch OHLCV candles.

        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            since: Start time in milliseconds
            limit: Maximum number of candles

        Returns:
            List of OHLCV candles
        """
        pass

    @track_latency("execute_trade")
    @abstractmethod
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """
        Execute trade.

        Args:
            symbol: Trading symbol
            side: Trade side (buy/sell)
            amount: Trade amount
            price: Limit price (optional)
            order_type: Order type (market/limit)

        Returns:
            Order execution details
        """
        pass

    @track_latency("get_positions")
    @abstractmethod
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions."""
        pass

    @track_latency("get_order_history")
    @abstractmethod
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get order history.

        Args:
            symbol: Trading symbol (optional)
            since: Start time in milliseconds (optional)
            limit: Maximum number of orders (optional)

        Returns:
            List of historical orders
        """
        pass

    @track_latency("get_average_entry_price")
    async def get_average_entry_price(self, symbol: str) -> Optional[Decimal]:
        """
        Calculate average entry price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Average entry price or None if no trades found
        """
        try:
            orders = await self.get_order_history(symbol=symbol)
            if not orders:
                return None

            total_quantity = Decimal('0')
            total_value = Decimal('0')

            for order in orders:
                if order['side'] == 'buy' and order['status'] == 'filled':
                    quantity = Decimal(str(order['amount']))
                    price = Decimal(str(order['price']))
                    total_quantity += quantity
                    total_value += quantity * price

            if total_quantity == 0:
                return None

            return total_value / total_quantity

        except Exception as e:
            logger.error(f"Error calculating average entry price for {symbol}: {str(e)}")
            return None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session and not self._session.closed:
            await self._session.close()