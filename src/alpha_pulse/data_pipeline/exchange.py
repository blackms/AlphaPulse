"""
Exchange connectivity module for AlphaPulse.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
import ccxt

from loguru import logger

from .interfaces import IExchange, IExchangeFactory


class CCXTExchange(IExchange):
    """CCXT-based exchange implementation."""

    def __init__(self, api_key: str = "", api_secret: str = "", exchange_id: str = ""):
        """Initialize exchange connection."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange_id = exchange_id
        
        # Initialize CCXT exchange
        exchange_class = getattr(ccxt, exchange_id.lower())
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        logger.info(f"Initialized CCXTExchange for {exchange_id}")

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict:
        """
        Fetch historical market data from the exchange.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Candle timeframe (e.g., "1m", "1h", "1d")
            start_time: Start time for data fetch
            end_time: End time for data fetch

        Returns:
            Dict containing OHLCV data
        """
        ohlcv = self.fetch_ohlcv(
            symbol,
            timeframe,
            since=int(start_time.timestamp() * 1000) if start_time else None
        )
        return {
            'open': [candle[1] for candle in ohlcv],
            'high': [candle[2] for candle in ohlcv],
            'low': [candle[3] for candle in ohlcv],
            'close': [candle[4] for candle in ohlcv],
            'volume': [candle[5] for candle in ohlcv]
        }

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")

        Returns:
            Current price
        """
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']

    def get_available_pairs(self) -> List[str]:
        """
        Get list of available trading pairs.

        Returns:
            List of trading pair symbols
        """
        markets = self.exchange.load_markets()
        return list(markets.keys())

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[List[float]]:
        """
        Fetch OHLCV data from exchange.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch

        Returns:
            List of OHLCV candles [timestamp, open, high, low, close, volume]
        """
        try:
            # Ensure the exchange supports OHLCV data
            if not self.exchange.has['fetchOHLCV']:
                raise NotImplementedError(f"{self.exchange_id} does not support OHLCV data")

            # Fetch the OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since,
                limit=limit
            )
            
            return ohlcv

        except ccxt.NetworkError as e:
            logger.error(f"Network error while fetching OHLCV data: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error while fetching OHLCV data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching OHLCV data: {e}")
            raise

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dict containing ticker data
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except ccxt.NetworkError as e:
            logger.error(f"Network error while fetching ticker: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error while fetching ticker: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching ticker: {e}")
            raise


class CCXTExchangeFactory(IExchangeFactory):
    """Factory for creating CCXT exchange instances."""

    def create_exchange(self, exchange_id: str) -> IExchange:
        """
        Create an exchange instance.

        Args:
            exchange_id: ID of the exchange to create

        Returns:
            Exchange instance
        """
        # Import settings at method level to avoid circular imports
        from ..config.settings import settings
        return CCXTExchange(
            api_key=settings.exchange.api_key,
            api_secret=settings.exchange.api_secret,
            exchange_id=exchange_id
        )


class ExchangeManager:
    """Manages exchange connections."""

    def __init__(self):
        """Initialize exchange manager."""
        self.factory = CCXTExchangeFactory()
        self._exchanges: Dict[str, IExchange] = {}

    def get_exchange(self, exchange_id: str) -> IExchange:
        """
        Get or create an exchange instance.

        Args:
            exchange_id: ID of the exchange

        Returns:
            Exchange instance
        """
        if exchange_id not in self._exchanges:
            self._exchanges[exchange_id] = self.factory.create_exchange(exchange_id)
        return self._exchanges[exchange_id]

    def get_ticker(self, exchange_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get ticker data for a symbol.

        Args:
            exchange_id: ID of the exchange
            symbol: Trading pair symbol

        Returns:
            Dict containing ticker data
        """
        exchange = self.get_exchange(exchange_id)
        return exchange.fetch_ticker(symbol)


# Make CCXTExchange the default implementation
Exchange = CCXTExchange