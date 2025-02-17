"""
Binance market data provider implementation.
"""
import hmac
import hashlib
import time
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import pandas as pd
from loguru import logger
import aiohttp

from ...interfaces import MarketData, DataFetchError
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

    # Binance interval mappings
    INTERVALS = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "8h": 28800,
        "12h": 43200,
        "1d": 86400,
        "3d": 259200,
        "1w": 604800,
        "1M": 2592000,
    }

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        cache_ttl: int = 300,
        request_timeout: int = 30,
        tcp_connector_limit: int = 100
    ):
        """
        Initialize Binance provider.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True
            cache_ttl: Cache time-to-live in seconds
            request_timeout: Request timeout in seconds
            tcp_connector_limit: Maximum number of concurrent connections
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
            default_headers=default_headers,
            request_timeout=request_timeout,
            tcp_connector_limit=tcp_connector_limit
        )
        CacheMixin.__init__(self, cache_ttl=cache_ttl)
        
        self._api_secret = api_secret
        self._testnet = testnet

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
            timestamp=datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5])
        )

    async def _fetch_historical_chunk(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        interval: str,
        limit: int = 1000
    ) -> List[MarketData]:
        """
        Fetch a chunk of historical data.

        Args:
            symbol: Trading symbol
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            interval: Data interval
            limit: Maximum number of candles to fetch

        Returns:
            List of MarketData objects
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': limit
        }
        
        response = await self._execute_request(
            endpoint="v3/klines",
            params=params
        )
        
        data = await self._process_response(response)
        if not isinstance(data, list):
            raise DataFetchError(f"Unexpected response format: {data}")
        
        return [self._parse_kline_data(kline) for kline in data]

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

        # Convert timestamps to UTC milliseconds
        start_ms = int(start_time.astimezone(timezone.utc).timestamp() * 1000)
        end_ms = int(end_time.astimezone(timezone.utc).timestamp() * 1000)

        try:
            logger.debug(f"Fetching historical data for {symbol}")
            
            # Calculate number of intervals needed
            interval_seconds = self.INTERVALS[interval]
            total_seconds = (end_ms - start_ms) / 1000
            total_intervals = int(total_seconds / interval_seconds)
            
            logger.debug(f"Need {total_intervals} intervals for {symbol}")
            
            # Fetch data in chunks of 1000 candles
            all_data = []
            chunk_start_ms = start_ms
            
            while chunk_start_ms < end_ms:
                chunk_data = await self._fetch_historical_chunk(
                    symbol=symbol,
                    start_ms=chunk_start_ms,
                    end_ms=end_ms,
                    interval=interval
                )
                
                if not chunk_data:
                    break
                    
                all_data.extend(chunk_data)
                
                # Update start time for next chunk
                chunk_start_ms = int(chunk_data[-1].timestamp.timestamp() * 1000) + 1
                
                # Add small delay to avoid rate limits
                await asyncio.sleep(0.1)
            
            # Sort by timestamp and remove duplicates
            all_data.sort(key=lambda x: x.timestamp)
            unique_data = []
            seen_timestamps = set()
            
            for item in all_data:
                ts = item.timestamp
                if ts not in seen_timestamps:
                    seen_timestamps.add(ts)
                    unique_data.append(item)
            
            logger.debug(f"Fetched {len(unique_data)} data points for {symbol}")
            
            # Cache the results
            self._set_in_cache(cache_key, unique_data)
            
            return unique_data
            
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
            
            # Parse JSON response
            data = await self._process_response(response)
            if not isinstance(data, dict) or 'price' not in data:
                raise DataFetchError(f"Unexpected response format: {data}")
            
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