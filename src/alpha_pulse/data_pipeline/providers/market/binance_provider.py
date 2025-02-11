"""
Binance market data provider implementation.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
from decimal import Decimal
import logging

from ...interfaces import MarketData
from ..base import BaseDataProvider, retry_on_error, CacheMixin

logger = logging.getLogger(__name__)


class BinanceMarketDataProvider(BaseDataProvider, CacheMixin):
    """
    Market data provider implementation using Binance API.
    
    Features:
    - Historical price data
    - Real-time price data
    - Order book data
    - Trade data
    """

    BASE_URL = "https://api.binance.com/api/v3/"
    BASE_URL_TESTNET = "https://testnet.binance.vision/api/v3/"

    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M"
    }

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        cache_ttl: int = 300,  # 5 minutes cache
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize Binance provider.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet
            cache_ttl: Cache time-to-live in seconds
            session: Optional aiohttp session
        """
        super().__init__("binance", "market", api_key)
        CacheMixin.__init__(self, cache_ttl)
        self._api_secret = api_secret
        self._testnet = testnet
        self._session = session
        self._base_url = self.BASE_URL_TESTNET if testnet else self.BASE_URL
        self._headers = {
            "X-MBX-APIKEY": api_key
        }

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self._session:
            self._session = aiohttp.ClientSession()

    async def _execute_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Any:
        """Execute Binance API request."""
        await self._ensure_session()
        url = self._base_url + endpoint
        
        request_headers = {**self._headers}
        if headers:
            request_headers.update(headers)

        try:
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                headers=request_headers,
                json=data
            ) as response:
                if response.status == 429:  # Rate limit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._execute_request(
                        endpoint, method, params, headers, data, signed
                    )
                
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    async def _process_response(self, response: Any) -> Any:
        """Process Binance API response."""
        if not response:
            return None
            
        # Check for API error messages
        if isinstance(response, dict) and "code" in response:
            raise ValueError(f"API Error: {response.get('msg', 'Unknown error')}")
            
        return response

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
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_time: Start time for historical data
            end_time: End time for historical data
            interval: Data interval (e.g., "1d" for daily)

        Returns:
            List of MarketData objects
        """
        cache_key = f"hist_{symbol}_{interval}_{start_time}_{end_time}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        # Convert interval to Binance format
        binance_interval = self.INTERVAL_MAP.get(interval)
        if not binance_interval:
            raise ValueError(f"Invalid interval: {interval}")

        # Prepare request parameters
        params = {
            "symbol": symbol.upper(),
            "interval": binance_interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": 1000  # Maximum allowed
        }

        try:
            # Fetch klines data
            klines = await self._make_request("klines", params=params)
            
            if not klines:
                return []

            # Process klines data
            market_data = []
            for kline in klines:
                timestamp = datetime.fromtimestamp(kline[0] / 1000)
                market_data.append(
                    MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=Decimal(str(kline[1])),
                        high=Decimal(str(kline[2])),
                        low=Decimal(str(kline[3])),
                        close=Decimal(str(kline[4])),
                        volume=Decimal(str(kline[5])),
                        trades=int(kline[8]),
                        vwap=Decimal(str(kline[7])) if kline[7] else None,
                        source="binance"
                    )
                )

            # Cache the processed data
            self._store_in_cache(cache_key, market_data)
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise

    @retry_on_error(max_retries=3)
    async def get_latest_price(self, symbol: str) -> MarketData:
        """
        Get latest price data from Binance.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")

        Returns:
            MarketData object with latest price
        """
        cache_key = f"price_{symbol}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            # Get 24hr ticker
            ticker = await self._make_request(
                "ticker/24hr",
                params={"symbol": symbol.upper()}
            )
            
            if not ticker:
                raise ValueError(f"No data available for {symbol}")

            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=Decimal(str(ticker["openPrice"])),
                high=Decimal(str(ticker["highPrice"])),
                low=Decimal(str(ticker["lowPrice"])),
                close=Decimal(str(ticker["lastPrice"])),
                volume=Decimal(str(ticker["volume"])),
                trades=int(ticker["count"]),
                vwap=Decimal(str(ticker["weightedAvgPrice"])),
                source="binance"
            )

            # Cache the data
            self._store_in_cache(cache_key, market_data)
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching latest price: {str(e)}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        if self._session:
            await self._session.close()