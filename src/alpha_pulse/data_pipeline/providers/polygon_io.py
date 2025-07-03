"""
Polygon.io market data provider implementation.

Provides comprehensive market data including stocks, options, forex, and crypto.
Features:
- Real-time and historical stock data
- Options chains and Greeks
- Cryptocurrency and forex data
- Technical indicators
- Rate limiting compliance (5 req/sec free tier)
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from .base_provider import BaseRealDataProvider, RateLimitInfo
from ..interfaces import MarketData, DataFetchError
from loguru import logger


class PolygonIOProvider(BaseRealDataProvider):
    """Polygon.io API provider for comprehensive market data."""

    def __init__(
        self,
        api_key: str,
        tier: str = "free",  # free, basic, starter, advanced
        cache_ttl: int = 60,  # 60 seconds for real-time data
        redis_url: str = "redis://localhost:6379"
    ):
        """
        Initialize Polygon.io provider.

        Args:
            api_key: Polygon.io API key
            tier: Subscription tier (affects rate limits)
            cache_ttl: Cache TTL for real-time data in seconds
            redis_url: Redis connection URL
        """
        base_url = "https://api.polygon.io"
        
        # Rate limits by tier
        tier_limits = {
            "free": RateLimitInfo(5, 300, 18000, 432000),
            "basic": RateLimitInfo(10, 600, 36000, 864000),
            "starter": RateLimitInfo(50, 3000, 180000, 4320000),
            "advanced": RateLimitInfo(100, 6000, 360000, 8640000)
        }
        
        rate_limits = tier_limits.get(tier, tier_limits["free"])
        
        super().__init__(
            provider_name="Polygon.io",
            provider_type="market_data",
            base_url=base_url,
            api_key=api_key,
            rate_limits=rate_limits,
            default_headers={
                "Accept": "application/json",
                "User-Agent": "AlphaPulse/1.0"
            },
            request_timeout=15,
            cache_ttl=cache_ttl,
            redis_url=redis_url
        )
        
        self._tier = tier

    async def _add_auth_headers(self, headers: Dict[str, str]) -> None:
        """Add Polygon.io authentication via query parameter."""
        # Polygon.io uses API key in query params
        pass

    def _build_url_with_auth(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build URL with API key parameter."""
        query_params = params or {}
        query_params['apikey'] = self.api_key
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if query_params:
            url += f"?{urlencode(query_params)}"
        
        return url

    async def _test_connectivity(self) -> None:
        """Test Polygon.io connectivity."""
        try:
            # Test with market status endpoint
            url = self._build_url_with_auth("v1/marketstatus/now")
            session = await self._get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    raise DataFetchError(f"Polygon.io connectivity test failed: {response.status}")
                    
                data = await response.json()
                if data.get('status') != 'OK':
                    raise DataFetchError("Invalid response from Polygon.io")
                    
        except Exception as e:
            raise DataFetchError(f"Polygon.io connectivity test failed: {e}")

    async def get_real_time_quote(self, symbol: str) -> MarketData:
        """
        Get real-time quote from Polygon.io.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            MarketData object with current quote
        """
        symbol = self._normalize_symbol(symbol)
        cache_key = f"polygon_quote_{symbol}"
        
        # Check cache first
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return MarketData(**cached_data)
        
        try:
            # Get previous day's close for current quote
            endpoint = f"v2/aggs/ticker/{symbol}/prev"
            params = {'adjusted': 'true'}
            url = self._build_url_with_auth(endpoint, params)
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 404:
                    raise DataFetchError(f"Symbol {symbol} not found on Polygon.io")
                elif response.status != 200:
                    raise DataFetchError(f"Polygon.io API error: {response.status}")
                
                data = await response.json()
                
                if data.get('status') != 'OK' or not data.get('results'):
                    raise DataFetchError(f"No data available for {symbol}")
                
                # Normalize data
                result = data['results'][0]
                normalized = self._normalize_polygon_agg(result, symbol)
                
                # Cache the result
                await self._set_cached_data(cache_key, normalized.__dict__, ttl=60)
                
                return normalized
                
        except Exception as e:
            logger.error(f"Failed to get real-time quote for {symbol} from Polygon.io: {e}")
            raise DataFetchError(f"Polygon.io quote request failed: {e}")

    def _normalize_polygon_agg(self, data: Dict[str, Any], symbol: str) -> MarketData:
        """Normalize Polygon.io aggregated data to MarketData format."""
        try:
            # Parse timestamp (Polygon provides timestamp in milliseconds)
            timestamp = datetime.fromtimestamp(data['t'] / 1000)
            
            return MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=self._parse_decimal(data.get('o')),
                high=self._parse_decimal(data.get('h')),
                low=self._parse_decimal(data.get('l')),
                close=self._parse_decimal(data.get('c')),
                volume=self._parse_decimal(data.get('v', 0)),
                vwap=self._parse_decimal(data.get('vw')),
                trades=data.get('n'),
                source=self.provider_name
            )
            
        except Exception as e:
            raise DataFetchError(f"Failed to normalize Polygon.io aggregated data: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[MarketData]:
        """
        Get historical data from Polygon.io.

        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1d, 1h, 5m, 1m)

        Returns:
            List of MarketData objects
        """
        symbol = self._normalize_symbol(symbol)
        
        # Polygon.io timespan mapping
        timespan_map = {
            "1m": ("minute", 1),
            "5m": ("minute", 5),
            "15m": ("minute", 15),
            "30m": ("minute", 30),
            "1h": ("hour", 1),
            "4h": ("hour", 4),
            "1d": ("day", 1),
            "1w": ("week", 1),
            "1M": ("month", 1)
        }
        
        if interval not in timespan_map:
            raise ValueError(f"Unsupported interval: {interval}")
        
        timespan, multiplier = timespan_map[interval]
        
        cache_key = f"polygon_historical_{symbol}_{start_date.date()}_{end_date.date()}_{interval}"
        
        # Check cache for historical data
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return [MarketData(**item) for item in cached_data]
        
        try:
            # Format dates for Polygon API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Build request
            endpoint = f"v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000  # Maximum allowed by Polygon
            }
            
            url = self._build_url_with_auth(endpoint, params)
            
            # Execute request
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 404:
                    raise DataFetchError(f"Symbol {symbol} not found on Polygon.io")
                elif response.status != 200:
                    raise DataFetchError(f"Polygon.io API error: {response.status}")
                
                data = await response.json()
                
                if data.get('status') != 'OK':
                    raise DataFetchError(f"Polygon.io API returned error: {data.get('error', 'Unknown error')}")
                
                results = data.get('results', [])
                if not results:
                    logger.warning(f"No historical data found for {symbol}")
                    return []
                
                # Normalize data
                market_data = []
                for item in results:
                    try:
                        normalized = self._normalize_polygon_agg(item, symbol)
                        market_data.append(normalized)
                    except Exception as e:
                        logger.warning(f"Failed to normalize historical data point: {e}")
                        continue
                
                # Sort by timestamp
                market_data.sort(key=lambda x: x.timestamp)
                
                # Cache result (24 hour TTL for historical data)
                cache_data = [item.__dict__ for item in market_data]
                await self._set_cached_data(cache_key, cache_data, ttl=86400)
                
                return market_data
                
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol} from Polygon.io: {e}")
            raise DataFetchError(f"Polygon.io historical data request failed: {e}")

    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information from Polygon.io.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with company information
        """
        symbol = self._normalize_symbol(symbol)
        cache_key = f"polygon_company_{symbol}"
        
        # Check cache (7 day TTL for company data)
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Get ticker details
            endpoint = f"v3/reference/tickers/{symbol}"
            url = self._build_url_with_auth(endpoint)
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 404:
                    raise DataFetchError(f"Symbol {symbol} not found on Polygon.io")
                elif response.status != 200:
                    raise DataFetchError(f"Polygon.io API error: {response.status}")
                
                data = await response.json()
                
                if data.get('status') != 'OK':
                    raise DataFetchError(f"Polygon.io API error: {data.get('error', 'Unknown error')}")
                
                company_info = data.get('results', {})
                company_info['source'] = self.provider_name
                company_info['last_updated'] = datetime.utcnow().isoformat()
                
                # Cache result
                await self._set_cached_data(cache_key, company_info, ttl=604800)  # 7 days
                
                return company_info
                
        except Exception as e:
            logger.error(f"Failed to get company info for {symbol} from Polygon.io: {e}")
            raise DataFetchError(f"Polygon.io company info request failed: {e}")

    async def get_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[datetime] = None,
        option_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get options chain data from Polygon.io.

        Args:
            symbol: Underlying stock symbol
            expiration_date: Specific expiration date to filter
            option_type: 'call' or 'put' to filter option type

        Returns:
            List of option contracts
        """
        symbol = self._normalize_symbol(symbol)
        
        try:
            endpoint = "v3/reference/options/contracts"
            params = {
                'underlying_ticker': symbol,
                'limit': 1000
            }
            
            if expiration_date:
                params['expiration_date'] = expiration_date.strftime('%Y-%m-%d')
            
            if option_type:
                params['contract_type'] = option_type.lower()
            
            url = self._build_url_with_auth(endpoint, params)
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status != 200:
                    raise DataFetchError(f"Polygon.io options API error: {response.status}")
                
                data = await response.json()
                
                if data.get('status') != 'OK':
                    raise DataFetchError(f"Polygon.io options API error: {data.get('error', 'Unknown error')}")
                
                options = data.get('results', [])
                
                # Add source information
                for option in options:
                    option['source'] = self.provider_name
                    option['underlying_symbol'] = symbol
                
                return options
                
        except Exception as e:
            logger.error(f"Failed to get options chain for {symbol} from Polygon.io: {e}")
            raise DataFetchError(f"Polygon.io options chain request failed: {e}")

    async def get_crypto_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[MarketData]:
        """
        Get cryptocurrency data from Polygon.io.

        Args:
            symbol: Crypto symbol (e.g., 'X:BTCUSD')
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval

        Returns:
            List of MarketData objects
        """
        # Ensure crypto symbol format
        if not symbol.startswith('X:'):
            symbol = f"X:{symbol.upper()}"
        
        # Use the same aggregates endpoint as stocks
        return await self.get_historical_data(symbol, start_date, end_date, interval)

    async def get_forex_data(
        self,
        from_currency: str,
        to_currency: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[MarketData]:
        """
        Get forex data from Polygon.io.

        Args:
            from_currency: From currency (e.g., 'EUR')
            to_currency: To currency (e.g., 'USD')
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval

        Returns:
            List of MarketData objects
        """
        # Format forex symbol
        symbol = f"C:{from_currency}{to_currency}"
        
        # Use the same aggregates endpoint
        return await self.get_historical_data(symbol, start_date, end_date, interval)

    async def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status from Polygon.io.

        Returns:
            Dictionary with market status information
        """
        try:
            endpoint = "v1/marketstatus/now"
            url = self._build_url_with_auth(endpoint)
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status != 200:
                    raise DataFetchError(f"Polygon.io market status API error: {response.status}")
                
                data = await response.json()
                
                if data.get('status') != 'OK':
                    raise DataFetchError(f"Polygon.io market status API error: {data.get('error', 'Unknown error')}")
                
                return data
                
        except Exception as e:
            logger.error(f"Failed to get market status from Polygon.io: {e}")
            raise DataFetchError(f"Polygon.io market status request failed: {e}")

    async def get_technical_indicators(
        self,
        symbol: str,
        indicator: str,
        timestamp: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get technical indicators from Polygon.io.

        Args:
            symbol: Stock symbol
            indicator: Indicator type (sma, ema, macd, rsi, etc.)
            timestamp: Specific timestamp for indicator
            **kwargs: Additional parameters for indicator

        Returns:
            Dictionary with indicator data
        """
        symbol = self._normalize_symbol(symbol)
        
        try:
            endpoint = f"v1/indicators/{indicator}/{symbol}"
            params = kwargs.copy()
            
            if timestamp:
                params['timestamp'] = timestamp
            
            url = self._build_url_with_auth(endpoint, params)
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status != 200:
                    raise DataFetchError(f"Polygon.io indicators API error: {response.status}")
                
                data = await response.json()
                
                if data.get('status') != 'OK':
                    raise DataFetchError(f"Polygon.io indicators API error: {data.get('error', 'Unknown error')}")
                
                return data
                
        except Exception as e:
            logger.error(f"Failed to get technical indicators for {symbol} from Polygon.io: {e}")
            raise DataFetchError(f"Polygon.io technical indicators request failed: {e}")

    async def search_tickers(self, query: str, market: str = "stocks") -> List[Dict[str, Any]]:
        """
        Search for tickers using Polygon.io.

        Args:
            query: Search query
            market: Market type (stocks, options, fx, crypto)

        Returns:
            List of matching tickers
        """
        try:
            endpoint = "v3/reference/tickers"
            params = {
                'search': query,
                'market': market,
                'active': 'true',
                'limit': 100
            }
            
            url = self._build_url_with_auth(endpoint, params)
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status != 200:
                    raise DataFetchError(f"Polygon.io search API error: {response.status}")
                
                data = await response.json()
                
                if data.get('status') != 'OK':
                    raise DataFetchError(f"Polygon.io search API error: {data.get('error', 'Unknown error')}")
                
                return data.get('results', [])
                
        except Exception as e:
            logger.error(f"Failed to search tickers with query '{query}' from Polygon.io: {e}")
            raise DataFetchError(f"Polygon.io ticker search request failed: {e}")

    def get_cost_estimate(self, request_type: str, symbols_count: int = 1) -> Dict[str, Any]:
        """
        Estimate API costs for Polygon.io requests.

        Args:
            request_type: Type of request
            symbols_count: Number of symbols

        Returns:
            Cost estimation information
        """
        # Polygon.io is subscription-based, costs are based on tier
        tier_costs = {
            "free": 0,
            "basic": 99,
            "starter": 199,
            "advanced": 399
        }
        
        monthly_cost = tier_costs.get(self._tier, 0)
        
        # Request limits by tier
        tier_limits = {
            "free": 432000,  # requests per day
            "basic": 864000,
            "starter": 4320000,
            "advanced": 8640000
        }
        
        daily_limit = tier_limits.get(self._tier, 432000)
        cost_per_request = monthly_cost / (daily_limit * 30) if daily_limit > 0 else 0
        
        return {
            'request_type': request_type,
            'symbols_count': symbols_count,
            'tier': self._tier,
            'monthly_cost_usd': monthly_cost,
            'daily_limit': daily_limit,
            'cost_per_request': cost_per_request,
            'estimated_cost_usd': cost_per_request * symbols_count
        }