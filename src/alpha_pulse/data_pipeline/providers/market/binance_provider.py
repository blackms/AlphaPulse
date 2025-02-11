"""
Binance market data provider implementation.
"""
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from loguru import logger
import aiohttp

from ...interfaces import MarketData
from ..base import BaseDataProvider, retry_on_error, CacheMixin


class BinanceMarketDataProvider(BaseDataProvider, CacheMixin):
    """
    Binance market data provider implementation.
    
    Features:
    - Historical OHLCV data
    - Real-time market data
    - Order book data
    - Trade data
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        cache_ttl: int = 300,
        request_timeout: int = 30
    ):
        """
        Initialize Binance provider.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True
            cache_ttl: Cache time-to-live in seconds
            request_timeout: Request timeout in seconds
        """
        base_url = "https://testnet.binance.vision/api" if testnet else "https://api.binance.com/api"
        default_headers = {
            "X-MBX-APIKEY": api_key,
            "Accept": "application/json",
            "User-Agent": "AlphaPulse/1.0"
        }
        
        BaseDataProvider.__init__(
            self,
            provider_name="binance",
            provider_type="market",
            base_url=base_url,
            default_headers=default_headers
        )
        CacheMixin.__init__(self, cache_ttl=cache_ttl)
        
        self._api_secret = api_secret
        self._testnet = testnet
        self._request_timeout = request_timeout

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature."""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self._api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _add_signature(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add timestamp and signature to parameters."""
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._generate_signature(params)
        return params

    def _parse_kline_data(self, kline: List[Any]) -> MarketData:
        """Parse Binance kline data into MarketData object."""
        return MarketData(
            symbol=self.current_symbol,
            timestamp=datetime.fromtimestamp(kline[0] / 1000),
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5])
        )

    async def _execute_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> aiohttp.ClientResponse:
        """Execute request with proper error handling."""
        try:
            return await super()._execute_request(
                endpoint=endpoint,
                method=method,
                params=params,
                headers=headers,
                data=data,
                timeout=self._request_timeout
            )
        except aiohttp.ClientError as e:
            logger.error(f"Binance API request failed: {str(e)}")
            if "429" in str(e):
                logger.warning("Rate limit exceeded, backing off...")
                await asyncio.sleep(10)  # Back off for 10 seconds
            raise

    @retry_on_error(retries=3, delay=2.0)
    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d"
    ) -> List[MarketData]:
        """
        Get historical market data.

        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time
            interval: Data interval

        Returns:
            List of MarketData objects
        """
        self.current_symbol = symbol
        cache_key = f"historical_{symbol}_{interval}_{start_time}_{end_time}"
        
        # Try to get from cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        # Prepare request parameters
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 1000
        }

        try:
            logger.debug(f"Fetching historical data for {symbol}")
            response = await self._execute_request(
                endpoint="v3/klines",
                params=params
            )
            
            # Read response content before processing
            content = await response.read()
            if not content:
                raise ValueError("Empty response from Binance API")
                
            # Parse JSON response
            data = await response.json()
            if not isinstance(data, list):
                raise ValueError(f"Unexpected response format: {data}")
            
            # Parse response into MarketData objects
            market_data = [self._parse_kline_data(kline) for kline in data]
            
            # Cache the results
            self._set_in_cache(cache_key, market_data)
            
            return market_data
            
        except Exception as e:
            logger.exception(f"Error fetching historical data for {symbol}: {str(e)}")
            raise

    @retry_on_error(retries=3, delay=2.0)
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest price
        """
        cache_key = f"price_{symbol}"
        
        # Try to get from cache first
        cached_price = self._get_from_cache(cache_key)
        if cached_price is not None:
            return cached_price

        try:
            logger.debug(f"Fetching latest price for {symbol}")
            response = await self._execute_request(
                endpoint="v3/ticker/price",
                params={'symbol': symbol}
            )
            
            # Read response content before processing
            content = await response.read()
            if not content:
                raise ValueError("Empty response from Binance API")
                
            # Parse JSON response
            data = await response.json()
            if not isinstance(data, dict) or 'price' not in data:
                raise ValueError(f"Unexpected response format: {data}")
            
            price = float(data['price'])
            
            # Cache the result
            self._set_in_cache(cache_key, price)
            
            return price
            
        except Exception as e:
            logger.exception(f"Error fetching price for {symbol}: {str(e)}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await super().__aexit__(exc_type, exc_val, exc_tb)