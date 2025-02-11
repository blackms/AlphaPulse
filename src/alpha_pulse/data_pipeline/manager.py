"""
Main data manager coordinating all data providers.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os

from .interfaces import (
    IDataManager,
    MarketData,
    FundamentalData,
    SentimentData,
    TechnicalIndicators
)
from .providers.market.binance_provider import BinanceMarketDataProvider
from .providers.fundamental.alpha_vantage_provider import AlphaVantageProvider
from .providers.sentiment.finnhub_provider import FinnhubProvider
from .providers.technical.talib_provider import TALibProvider

logger = logging.getLogger(__name__)


def _interpolate_env_vars(value: str) -> str:
    """Interpolate environment variables in string."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, value)
    return value


def _process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process configuration dictionary and interpolate environment variables."""
    processed = {}
    for key, value in config.items():
        if isinstance(value, dict):
            processed[key] = _process_config(value)
        elif isinstance(value, list):
            processed[key] = [
                _process_config(item) if isinstance(item, dict)
                else _interpolate_env_vars(item)
                for item in value
            ]
        else:
            processed[key] = _interpolate_env_vars(value)
    return processed


class DataManager(IDataManager):
    """
    Main data manager implementing the Facade pattern.
    
    Coordinates all data providers and provides a unified interface
    for accessing market, fundamental, sentiment, and technical data.
    
    Features:
    - Unified data access interface
    - Provider coordination
    - Error handling and recovery
    - Data validation and cleaning
    - Concurrent data fetching
    """

    def __init__(
        self,
        config: Dict[str, Any],
        max_workers: int = 4
    ):
        """
        Initialize data manager.

        Args:
            config: Configuration dictionary containing API keys and settings
            max_workers: Maximum number of worker threads
        """
        self._config = _process_config(config)
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        
        # Initialize providers
        self._market_provider = None
        self._fundamental_provider = None
        self._sentiment_provider = None
        self._technical_provider = None

    async def initialize(self) -> None:
        """Initialize all data providers."""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:  # Double-check pattern
                return

            try:
                # Initialize market data provider (Binance for crypto)
                if self._config.get("market_data", {}).get("binance"):
                    self._market_provider = BinanceMarketDataProvider(
                        api_key=self._config["market_data"]["binance"].get("api_key"),
                        api_secret=self._config["market_data"]["binance"].get("api_secret"),
                        testnet=self._config["market_data"]["binance"].get("testnet", True)
                    )

                # Initialize fundamental data provider (Alpha Vantage)
                if self._config.get("fundamental_data", {}).get("alpha_vantage"):
                    self._fundamental_provider = AlphaVantageProvider(
                        api_key=self._config["fundamental_data"]["alpha_vantage"].get("api_key")
                    )

                # Initialize sentiment data provider (Finnhub)
                if self._config.get("sentiment_data", {}).get("finnhub"):
                    finnhub_config = self._config["sentiment_data"]["finnhub"]
                    self._sentiment_provider = FinnhubProvider(
                        api_key=finnhub_config.get("api_key"),
                        cache_ttl=finnhub_config.get("cache_ttl", 300)
                    )

                # Initialize technical analysis provider
                self._technical_provider = TALibProvider(
                    max_workers=self._max_workers
                )

                self._initialized = True
                logger.info("Data manager initialized successfully")

            except Exception as e:
                logger.error(f"Error initializing data manager: {str(e)}")
                raise

    async def _ensure_initialized(self):
        """Ensure data manager is initialized."""
        if not self._initialized:
            await self.initialize()

    async def get_market_data(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d"
    ) -> Dict[str, List[MarketData]]:
        """
        Get market data for multiple symbols.

        Args:
            symbols: List of trading symbols
            start_time: Start time for historical data
            end_time: End time for historical data
            interval: Data interval

        Returns:
            Dictionary mapping symbols to their market data
        """
        await self._ensure_initialized()
        if not self._market_provider:
            raise RuntimeError("Market data provider not initialized")

        async def fetch_symbol_data(symbol: str) -> tuple[str, List[MarketData]]:
            try:
                data = await self._market_provider.get_historical_data(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    interval=interval
                )
                return symbol, data
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {str(e)}")
                return symbol, []

        # Fetch data concurrently
        tasks = [fetch_symbol_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return {symbol: data for symbol, data in results if data}

    async def get_fundamental_data(
        self,
        symbols: List[str]
    ) -> Dict[str, FundamentalData]:
        """
        Get fundamental data for multiple symbols.

        Args:
            symbols: List of trading symbols

        Returns:
            Dictionary mapping symbols to their fundamental data
        """
        await self._ensure_initialized()
        if not self._fundamental_provider:
            raise RuntimeError("Fundamental data provider not initialized")

        async def fetch_symbol_fundamentals(symbol: str) -> tuple[str, Optional[FundamentalData]]:
            try:
                data = await self._fundamental_provider.get_financial_statements(symbol)
                return symbol, data
            except Exception as e:
                logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
                return symbol, None

        # Fetch data concurrently
        tasks = [fetch_symbol_fundamentals(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return {
            symbol: data for symbol, data in results
            if data is not None
        }

    async def get_sentiment_data(
        self,
        symbols: List[str]
    ) -> Dict[str, SentimentData]:
        """
        Get sentiment data for multiple symbols.

        Args:
            symbols: List of trading symbols

        Returns:
            Dictionary mapping symbols to their sentiment data
        """
        await self._ensure_initialized()
        if not self._sentiment_provider:
            raise RuntimeError("Sentiment data provider not initialized")

        # Get configuration
        finnhub_config = self._config["sentiment_data"]["finnhub"]
        lookback_days = finnhub_config.get("news_lookback_days", 7)
        lookback_hours = finnhub_config.get("social_lookback_hours", 24)

        async def fetch_symbol_sentiment(symbol: str) -> tuple[str, Optional[SentimentData]]:
            try:
                data = await self._sentiment_provider.get_sentiment_data(
                    symbol=symbol,
                    lookback_days=lookback_days,
                    lookback_hours=lookback_hours
                )
                return symbol, data
            except Exception as e:
                logger.error(f"Error fetching sentiment data for {symbol}: {str(e)}")
                return symbol, None

        # Fetch data concurrently
        tasks = [fetch_symbol_sentiment(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return {
            symbol: data for symbol, data in results
            if data is not None
        }

    def get_technical_indicators(
        self,
        market_data: Dict[str, List[MarketData]]
    ) -> Dict[str, TechnicalIndicators]:
        """
        Calculate technical indicators for market data.

        Args:
            market_data: Dictionary of market data by symbol

        Returns:
            Dictionary mapping symbols to their technical indicators
        """
        if not self._technical_provider:
            raise RuntimeError("Technical analysis provider not initialized")

        results = {}
        
        for symbol, data in market_data.items():
            try:
                # Convert MarketData list to DataFrame
                df = pd.DataFrame([
                    {
                        'open': float(d.open),
                        'high': float(d.high),
                        'low': float(d.low),
                        'close': float(d.close),
                        'volume': float(d.volume)
                    }
                    for d in data
                ])
                
                indicators = self._technical_provider.get_technical_indicators(
                    df=df,  # Changed from data=df to df=df
                    symbol=symbol
                )
                results[symbol] = indicators
                
            except Exception as e:
                logger.error(
                    f"Error calculating technical indicators for {symbol}: {str(e)}"
                )
                continue
                
        return results

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._executor.shutdown(wait=True)