"""
Binance market data provider implementation.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from binance.client import AsyncClient
from binance.exceptions import BinanceAPIException

from ...interfaces import MarketData
from ..base import BaseDataProvider, retry_on_error, CacheMixin


class BinanceMarketDataProvider(BaseDataProvider, CacheMixin):
    """
    Market data provider implementation for Binance exchange.
    
    Features:
    - Real-time and historical data fetching
    - Rate limiting and error handling
    - Data validation and cleaning
    - Response caching
    - Automatic reconnection
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        cache_ttl: int = 300
    ):
        """
        Initialize Binance provider.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet environment
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__("binance", "market", api_key)
        CacheMixin.__init__(self, cache_ttl)
        self._api_secret = api_secret
        self._testnet = testnet
        self._client: Optional[AsyncClient] = None
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """Ensure client is initialized."""
        if not self._initialized:
            async with self._initialization_lock:
                if not self._initialized:  # Double-check pattern
                    self._client = await AsyncClient.create(
                        api_key=self._api_key,
                        api_secret=self._api_secret,
                        testnet=self._testnet
                    )
                    self._initialized = True

    async def _execute_request(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        data: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute Binance API request."""
        await self._ensure_initialized()
        try:
            if method == "GET":
                if endpoint == "klines":
                    return await self._client.get_klines(**params)
                elif endpoint == "ticker":
                    return await self._client.get_ticker(**params)
            raise ValueError(f"Unsupported method or endpoint: {method} {endpoint}")
        except BinanceAPIException as e:
            raise DataFetchError(f"Binance API error: {str(e)}")

    async def _process_response(self, response: Any) -> Any:
        """Process Binance API response."""
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], list):  # Klines data
                return self._process_klines(response)
            return response
        return response

    def _process_klines(self, klines: List[List]) -> List[MarketData]:
        """Process klines data into MarketData objects."""
        processed_data = []
        for kline in klines:
            try:
                market_data = MarketData(
                    symbol=kline[0],  # Symbol is added later
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5]),
                    additional_data={
                        "quote_volume": float(kline[7]),
                        "trades": int(kline[8]),
                        "taker_buy_volume": float(kline[9]),
                        "taker_buy_quote_volume": float(kline[10])
                    }
                )
                processed_data.append(market_data)
            except (IndexError, ValueError) as e:
                logger.warning(f"Error processing kline data: {str(e)}")
                continue
        return processed_data

    @retry_on_error(max_retries=3)
    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d"
    ) -> List[MarketData]:
        """
        Get historical market data from Binance.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            start_time: Start time for historical data
            end_time: End time for historical data
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)

        Returns:
            List of MarketData objects
        """
        cache_key = f"historical_{symbol}_{start_time}_{end_time}_{interval}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": 1000
        }

        all_klines = []
        current_start = start_time

        while current_start < end_time:
            params["startTime"] = int(current_start.timestamp() * 1000)
            klines = await self._make_request("klines", params=params)
            if not klines:
                break

            all_klines.extend(klines)
            
            # Update start time for next batch
            last_kline_time = datetime.fromtimestamp(klines[-1][0] / 1000)
            if last_kline_time <= current_start:
                break
            current_start = last_kline_time + timedelta(milliseconds=1)

        processed_data = self._process_klines(all_klines)
        for data in processed_data:
            data.symbol = symbol

        # Cache the results
        self._store_in_cache(cache_key, processed_data)
        return processed_data

    @retry_on_error(max_retries=3)
    async def get_real_time_data(self, symbol: str) -> MarketData:
        """
        Get real-time market data from Binance.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            MarketData object with current market data
        """
        params = {"symbol": symbol}
        ticker = await self._make_request("ticker", params=params)

        market_data = MarketData(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(ticker["closeTime"] / 1000),
            open=float(ticker["openPrice"]),
            high=float(ticker["highPrice"]),
            low=float(ticker["lowPrice"]),
            close=float(ticker["lastPrice"]),
            volume=float(ticker["volume"]),
            additional_data={
                "price_change": float(ticker["priceChange"]),
                "price_change_percent": float(ticker["priceChangePercent"]),
                "weighted_avg_price": float(ticker["weightedAvgPrice"]),
                "quote_volume": float(ticker["quoteVolume"]),
                "trades": int(ticker["count"])
            }
        )

        return await self.validate_and_clean_data(market_data)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        if self._client:
            await self._client.close_connection()
        self._initialized = False