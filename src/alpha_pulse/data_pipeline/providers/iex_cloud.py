"""
IEX Cloud market data provider implementation.

Provides real-time and historical market data from IEX Cloud API.
Features:
- Real-time quotes with sub-second latency
- Historical price data with corporate actions
- Company fundamentals and statistics
- Dividend and split information
- Rate limiting compliance (100 req/sec)
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from .base_provider import BaseRealDataProvider, RateLimitInfo, DataNormalizationResult
from ..interfaces import MarketData, DataFetchError
from loguru import logger


class IEXCloudProvider(BaseRealDataProvider):
    """IEX Cloud API provider for market data."""

    def __init__(
        self,
        api_token: str,
        is_sandbox: bool = False,
        cache_ttl: int = 30,  # 30 seconds for real-time data
        redis_url: str = "redis://localhost:6379"
    ):
        """
        Initialize IEX Cloud provider.

        Args:
            api_token: IEX Cloud API token
            is_sandbox: Whether to use sandbox environment
            cache_ttl: Cache TTL for real-time data in seconds
            redis_url: Redis connection URL
        """
        base_url = (
            "https://sandbox.iexapis.com/stable" if is_sandbox 
            else "https://cloud.iexapis.com/stable"
        )
        
        # IEX Cloud rate limits
        rate_limits = RateLimitInfo(
            requests_per_second=100,
            requests_per_minute=6000,
            requests_per_hour=360000,
            requests_per_day=8640000
        )
        
        super().__init__(
            provider_name="IEX Cloud",
            provider_type="market_data",
            base_url=base_url,
            api_key=api_token,
            rate_limits=rate_limits,
            default_headers={
                "Accept": "application/json",
                "User-Agent": "AlphaPulse/1.0"
            },
            request_timeout=10,  # Fast timeout for real-time data
            cache_ttl=cache_ttl,
            redis_url=redis_url
        )
        
        self._is_sandbox = is_sandbox

    async def _add_auth_headers(self, headers: Dict[str, str]) -> None:
        """Add IEX Cloud authentication via query parameter."""
        # IEX Cloud uses token in query params, not headers
        pass

    def _build_url_with_token(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build URL with API token parameter."""
        query_params = params or {}
        query_params['token'] = self.api_key
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if query_params:
            url += f"?{urlencode(query_params)}"
        
        return url

    async def _test_connectivity(self) -> None:
        """Test IEX Cloud connectivity."""
        try:
            # Test with a simple quote request
            url = self._build_url_with_token("stock/AAPL/quote")
            session = await self._get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    raise DataFetchError(f"IEX Cloud connectivity test failed: {response.status}")
                    
                data = await response.json()
                if not data.get('symbol'):
                    raise DataFetchError("Invalid response from IEX Cloud")
                    
        except Exception as e:
            raise DataFetchError(f"IEX Cloud connectivity test failed: {e}")

    async def get_real_time_quote(self, symbol: str) -> MarketData:
        """
        Get real-time quote from IEX Cloud.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            MarketData object with current quote
        """
        symbol = self._normalize_symbol(symbol)
        cache_key = f"iex_quote_{symbol}"
        
        # Check cache first
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return MarketData(**cached_data)
        
        try:
            # Build request URL
            endpoint = f"stock/{symbol}/quote"
            params = {
                'displayPercent': 'true',
                'filter': 'symbol,latestPrice,latestTime,previousClose,open,high,low,volume,avgTotalVolume'
            }
            url = self._build_url_with_token(endpoint, params)
            
            # Execute request
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 404:
                    raise DataFetchError(f"Symbol {symbol} not found on IEX Cloud")
                elif response.status != 200:
                    raise DataFetchError(f"IEX Cloud API error: {response.status}")
                
                data = await response.json()
                
                # Normalize data
                normalized = self._normalize_iex_quote(data, symbol)
                
                # Cache the result
                await self._set_cached_data(cache_key, normalized.__dict__, ttl=30)
                
                return normalized
                
        except Exception as e:
            logger.error(f"Failed to get real-time quote for {symbol} from IEX Cloud: {e}")
            raise DataFetchError(f"IEX Cloud quote request failed: {e}")

    def _normalize_iex_quote(self, data: Dict[str, Any], symbol: str) -> MarketData:
        """Normalize IEX Cloud quote data to MarketData format."""
        try:
            # Parse timestamp
            timestamp = None
            if data.get('latestTime'):
                # IEX provides timestamp in milliseconds
                timestamp = datetime.fromtimestamp(data['latestTime'] / 1000)
            else:
                timestamp = datetime.utcnow()
            
            # Extract prices
            latest_price = self._parse_decimal(data.get('latestPrice'))
            previous_close = self._parse_decimal(data.get('previousClose'))
            
            return MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=self._parse_decimal(data.get('open', previous_close)),
                high=self._parse_decimal(data.get('high', latest_price)),
                low=self._parse_decimal(data.get('low', latest_price)),
                close=latest_price,
                volume=self._parse_decimal(data.get('volume', 0)),
                vwap=None,  # IEX doesn't provide VWAP in quote
                trades=None,
                source=self.provider_name
            )
            
        except Exception as e:
            raise DataFetchError(f"Failed to normalize IEX quote data: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[MarketData]:
        """
        Get historical data from IEX Cloud.

        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1d, 1m, 5m, etc.)

        Returns:
            List of MarketData objects
        """
        symbol = self._normalize_symbol(symbol)
        
        # IEX Cloud interval mapping
        interval_map = {
            "1d": "1d",
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h"
        }
        
        iex_interval = interval_map.get(interval, "1d")
        
        # Determine the range parameter
        days_diff = (end_date - start_date).days
        
        if days_diff <= 5:
            range_param = "5d"
        elif days_diff <= 30:
            range_param = "1m"
        elif days_diff <= 90:
            range_param = "3m"
        elif days_diff <= 180:
            range_param = "6m"
        elif days_diff <= 365:
            range_param = "1y"
        elif days_diff <= 730:
            range_param = "2y"
        else:
            range_param = "5y"
        
        cache_key = f"iex_historical_{symbol}_{start_date.date()}_{end_date.date()}_{interval}"
        
        # Check cache for historical data (longer TTL)
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return [MarketData(**item) for item in cached_data]
        
        try:
            # Build request
            if iex_interval == "1d":
                endpoint = f"stock/{symbol}/chart/{range_param}"
            else:
                endpoint = f"stock/{symbol}/intraday-prices"
            
            params = {
                'chartCloseOnly': 'false',
                'includeToday': 'true'
            }
            
            if iex_interval != "1d":
                params['chartInterval'] = iex_interval
            
            url = self._build_url_with_token(endpoint, params)
            
            # Execute request
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 404:
                    raise DataFetchError(f"Symbol {symbol} not found on IEX Cloud")
                elif response.status != 200:
                    raise DataFetchError(f"IEX Cloud API error: {response.status}")
                
                data = await response.json()
                
                if not isinstance(data, list):
                    raise DataFetchError("Invalid historical data format from IEX Cloud")
                
                # Normalize and filter data
                market_data = []
                for item in data:
                    try:
                        normalized = self._normalize_iex_historical(item, symbol)
                        if (start_date <= normalized.timestamp <= end_date):
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
            logger.error(f"Failed to get historical data for {symbol} from IEX Cloud: {e}")
            raise DataFetchError(f"IEX Cloud historical data request failed: {e}")

    def _normalize_iex_historical(self, data: Dict[str, Any], symbol: str) -> MarketData:
        """Normalize IEX Cloud historical data to MarketData format."""
        try:
            # Parse timestamp
            if 'date' in data:
                # Daily data
                timestamp = datetime.strptime(data['date'], '%Y-%m-%d')
            elif 'minute' in data and 'date' in data:
                # Intraday data
                timestamp = datetime.strptime(f"{data['date']} {data['minute']}", '%Y-%m-%d %H:%M')
            else:
                raise ValueError("Missing timestamp information")
            
            return MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=self._parse_decimal(data.get('open')),
                high=self._parse_decimal(data.get('high')),
                low=self._parse_decimal(data.get('low')),
                close=self._parse_decimal(data.get('close')),
                volume=self._parse_decimal(data.get('volume', 0)),
                vwap=self._parse_decimal(data.get('vwap')),
                trades=data.get('numberOfTrades'),
                source=self.provider_name
            )
            
        except Exception as e:
            raise DataFetchError(f"Failed to normalize IEX historical data: {e}")

    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information from IEX Cloud.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with company information
        """
        symbol = self._normalize_symbol(symbol)
        cache_key = f"iex_company_{symbol}"
        
        # Check cache (7 day TTL for company data)
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Get company info
            endpoint = f"stock/{symbol}/company"
            url = self._build_url_with_token(endpoint)
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 404:
                    raise DataFetchError(f"Symbol {symbol} not found on IEX Cloud")
                elif response.status != 200:
                    raise DataFetchError(f"IEX Cloud API error: {response.status}")
                
                company_data = await response.json()
            
            # Get key stats
            stats_endpoint = f"stock/{symbol}/stats"
            stats_url = self._build_url_with_token(stats_endpoint)
            
            async with session.get(stats_url) as response:
                if response.status == 200:
                    stats_data = await response.json()
                else:
                    stats_data = {}
            
            # Combine data
            combined_data = {
                **company_data,
                'stats': stats_data,
                'source': self.provider_name,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Cache result
            await self._set_cached_data(cache_key, combined_data, ttl=604800)  # 7 days
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Failed to get company info for {symbol} from IEX Cloud: {e}")
            raise DataFetchError(f"IEX Cloud company info request failed: {e}")

    async def get_dividends(self, symbol: str, range_param: str = "1y") -> List[Dict[str, Any]]:
        """
        Get dividend information from IEX Cloud.

        Args:
            symbol: Stock symbol
            range_param: Range for dividend data (1y, 2y, 5y)

        Returns:
            List of dividend records
        """
        symbol = self._normalize_symbol(symbol)
        
        try:
            endpoint = f"stock/{symbol}/dividends/{range_param}"
            url = self._build_url_with_token(endpoint)
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 404:
                    return []  # No dividends found
                elif response.status != 200:
                    raise DataFetchError(f"IEX Cloud API error: {response.status}")
                
                dividends = await response.json()
                
                # Add source information
                for dividend in dividends:
                    dividend['source'] = self.provider_name
                
                return dividends
                
        except Exception as e:
            logger.error(f"Failed to get dividends for {symbol} from IEX Cloud: {e}")
            raise DataFetchError(f"IEX Cloud dividends request failed: {e}")

    async def get_splits(self, symbol: str, range_param: str = "5y") -> List[Dict[str, Any]]:
        """
        Get stock split information from IEX Cloud.

        Args:
            symbol: Stock symbol
            range_param: Range for split data (1y, 2y, 5y)

        Returns:
            List of split records
        """
        symbol = self._normalize_symbol(symbol)
        
        try:
            endpoint = f"stock/{symbol}/splits/{range_param}"
            url = self._build_url_with_token(endpoint)
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 404:
                    return []  # No splits found
                elif response.status != 200:
                    raise DataFetchError(f"IEX Cloud API error: {response.status}")
                
                splits = await response.json()
                
                # Add source information
                for split in splits:
                    split['source'] = self.provider_name
                
                return splits
                
        except Exception as e:
            logger.error(f"Failed to get splits for {symbol} from IEX Cloud: {e}")
            raise DataFetchError(f"IEX Cloud splits request failed: {e}")

    async def batch_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Get batch quotes for multiple symbols efficiently.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to MarketData
        """
        if not symbols:
            return {}
        
        # Normalize symbols
        normalized_symbols = [self._normalize_symbol(s) for s in symbols]
        symbols_param = ",".join(normalized_symbols)
        
        try:
            endpoint = f"stock/market/batch"
            params = {
                'symbols': symbols_param,
                'types': 'quote',
                'filter': 'symbol,latestPrice,latestTime,previousClose,open,high,low,volume'
            }
            url = self._build_url_with_token(endpoint, params)
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status != 200:
                    raise DataFetchError(f"IEX Cloud batch API error: {response.status}")
                
                batch_data = await response.json()
                
                # Process results
                results = {}
                for symbol, data in batch_data.items():
                    try:
                        if 'quote' in data:
                            quote_data = data['quote']
                            normalized = self._normalize_iex_quote(quote_data, symbol)
                            results[symbol] = normalized
                    except Exception as e:
                        logger.warning(f"Failed to process batch quote for {symbol}: {e}")
                        continue
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get batch quotes from IEX Cloud: {e}")
            raise DataFetchError(f"IEX Cloud batch quotes request failed: {e}")

    def get_cost_estimate(self, request_type: str, symbols_count: int = 1) -> Dict[str, Any]:
        """
        Estimate API costs for IEX Cloud requests.

        Args:
            request_type: Type of request (quote, historical, etc.)
            symbols_count: Number of symbols

        Returns:
            Cost estimation information
        """
        # IEX Cloud pricing (credits per request)
        pricing = {
            'quote': 1,
            'historical_1d': 10,
            'historical_intraday': 1,
            'company': 1,
            'stats': 1,
            'dividends': 10,
            'splits': 10,
            'batch_quote': 1  # per symbol
        }
        
        credits_per_request = pricing.get(request_type, 1)
        total_credits = credits_per_request * symbols_count
        
        # Approximate USD cost (varies by plan)
        cost_per_credit = 0.0001 if self._is_sandbox else 0.0005
        estimated_cost = total_credits * cost_per_credit
        
        return {
            'request_type': request_type,
            'symbols_count': symbols_count,
            'credits_per_request': credits_per_request,
            'total_credits': total_credits,
            'estimated_cost_usd': estimated_cost,
            'is_sandbox': self._is_sandbox
        }