"""
Main data manager coordinating all data providers.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
from loguru import logger
import sys
import inspect

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


def _filter_init_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter parameters to only include those accepted by the class's __init__."""
    sig = inspect.signature(cls.__init__)
    valid_params = {
        name: params[name]
        for name in sig.parameters
        if name in params and name != 'self'
    }
    return valid_params


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
        
        # Track providers for cleanup
        self._providers = []

    async def initialize(self) -> None:
        """Initialize all data providers."""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:  # Double-check pattern
                return

            try:
                # Get connection settings
                conn_config = self._config.get("connection", {})
                common_params = {
                    "request_timeout": conn_config.get("request_timeout", 30),
                    "tcp_connector_limit": conn_config.get("tcp_connector_limit", 100)
                }

                # Initialize market data provider (Binance for crypto)
                if self._config.get("market_data", {}).get("binance"):
                    logger.debug("Initializing Binance market data provider")
                    binance_params = {
                        "api_key": self._config["market_data"]["binance"].get("api_key"),
                        "api_secret": self._config["market_data"]["binance"].get("api_secret"),
                        "testnet": self._config["market_data"]["binance"].get("testnet", True),
                        **common_params
                    }
                    filtered_params = _filter_init_params(BinanceMarketDataProvider, binance_params)
                    self._market_provider = BinanceMarketDataProvider(**filtered_params)
                    self._providers.append(self._market_provider)

                # Initialize fundamental data provider (Alpha Vantage)
                if self._config.get("fundamental_data", {}).get("alpha_vantage"):
                    logger.debug("Initializing Alpha Vantage fundamental data provider")
                    av_params = {
                        "api_key": self._config["fundamental_data"]["alpha_vantage"].get("api_key"),
                        **common_params
                    }
                    filtered_params = _filter_init_params(AlphaVantageProvider, av_params)
                    self._fundamental_provider = AlphaVantageProvider(**filtered_params)
                    self._providers.append(self._fundamental_provider)

                # Initialize sentiment data provider (Finnhub)
                if self._config.get("sentiment_data", {}).get("finnhub"):
                    logger.debug("Initializing Finnhub sentiment data provider")
                    finnhub_config = self._config["sentiment_data"]["finnhub"]
                    finnhub_params = {
                        "api_key": finnhub_config.get("api_key"),
                        "cache_ttl": finnhub_config.get("cache_ttl", 300),
                        **common_params
                    }
                    filtered_params = _filter_init_params(FinnhubProvider, finnhub_params)
                    self._sentiment_provider = FinnhubProvider(**filtered_params)
                    self._providers.append(self._sentiment_provider)

                # Initialize technical analysis provider
                logger.debug("Initializing TA-Lib technical analysis provider")
                talib_params = {
                    "max_workers": self._max_workers,
                    **common_params
                }
                filtered_params = _filter_init_params(TALibProvider, talib_params)
                self._technical_provider = TALibProvider(**filtered_params)
                self._providers.append(self._technical_provider)

                self._initialized = True
                logger.info("Data manager initialized successfully")

            except Exception as e:
                logger.exception(f"Error initializing data manager: {str(e)}")
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
                logger.debug(f"Fetching market data for {symbol}")
                data = await self._market_provider.get_historical_data(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    interval=interval
                )
                logger.debug(f"Received {len(data)} data points for {symbol}")
                return symbol, data
            except Exception as e:
                logger.exception(f"Error fetching market data for {symbol}: {str(e)}")
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
                logger.debug(f"Fetching fundamental data for {symbol}")
                data = await self._fundamental_provider.get_financial_statements(symbol)
                return symbol, data
            except Exception as e:
                logger.exception(f"Error fetching fundamental data for {symbol}: {str(e)}")
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
                logger.debug(f"Fetching sentiment data for {symbol}")
                data = await self._sentiment_provider.get_sentiment_data(
                    symbol=symbol,
                    lookback_days=lookback_days,
                    lookback_hours=lookback_hours
                )
                return symbol, data
            except Exception as e:
                logger.exception(f"Error fetching sentiment data for {symbol}: {str(e)}")
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
                logger.debug(f"Converting market data to DataFrame for {symbol}")
                # Convert MarketData list to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': d.timestamp,
                        'open': float(d.open),
                        'high': float(d.high),
                        'low': float(d.low),
                        'close': float(d.close),
                        'volume': float(d.volume)
                    }
                    for d in data
                ])
                
                # Sort by timestamp and set as index
                df = df.sort_values('timestamp')
                df.set_index('timestamp', inplace=True)
                
                logger.debug(f"Calculating technical indicators for {symbol} with {len(df)} data points")
                indicators = self._technical_provider.get_technical_indicators(
                    df=df,
                    symbol=symbol
                )
                results[symbol] = indicators
                
            except Exception as e:
                logger.exception(
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
        # Clean up providers
        for provider in self._providers:
            try:
                if hasattr(provider, '__aexit__'):
                    await provider.__aexit__(exc_type, exc_val, exc_tb)
                    logger.debug(f"Cleaned up provider: {provider.provider_name}")
            except Exception as e:
                logger.exception(f"Error cleaning up provider: {str(e)}")
        self._providers.clear()
        
        # Clean up executor
        try:
            self._executor.shutdown(wait=True)
            logger.debug("Shut down executor")
        except Exception as e:
            logger.exception(f"Error shutting down executor: {str(e)}")