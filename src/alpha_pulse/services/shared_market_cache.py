"""
Shared market data cache service.

Implements shared caching for market data (OHLCV, ticker, orderbook) that is
identical across all tenants. Reduces memory usage by 90%+ by caching once
instead of per-tenant. Per ADR-004 specification.
"""

import logging
import json
from typing import Optional, Dict, Any, List
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class SharedMarketCache:
    """
    Shared cache for market data accessed by all tenants.

    Key Design:
    - shared:market:{exchange}:{symbol}:{interval}:ohlcv
    - shared:market:{exchange}:{symbol}:ticker
    - shared:market:{exchange}:{symbol}:orderbook

    Benefits:
    - 90%+ memory reduction (cache once vs per-tenant)
    - Consistent data across all tenants
    - Lower cache churn (single TTL vs multiple)
    - No cross-tenant data leakage (read-only access)
    """

    def __init__(
        self,
        redis_client: Redis,
        default_ttl_seconds: int = 60
    ):
        """
        Initialize shared market cache.

        Args:
            redis_client: Async Redis client
            default_ttl_seconds: Default TTL for cached data (default: 60s)
        """
        self.redis = redis_client
        self.default_ttl = default_ttl_seconds

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT", "btc_usdt")

        Returns:
            Normalized symbol (e.g., "BTC_USDT")
        """
        return symbol.replace("/", "_").replace("-", "_").upper()

    def _normalize_exchange(self, exchange: str) -> str:
        """
        Normalize exchange name.

        Args:
            exchange: Exchange name (e.g., "BiNaNcE")

        Returns:
            Normalized exchange (e.g., "binance")
        """
        return exchange.lower().strip()

    def get_ohlcv_key(
        self,
        exchange: str,
        symbol: str,
        interval: str
    ) -> str:
        """
        Get Redis key for OHLCV data.

        Args:
            exchange: Exchange name (e.g., "binance")
            symbol: Trading symbol (e.g., "BTC_USDT")
            interval: Time interval (e.g., "1m", "5m", "1h")

        Returns:
            Redis key string (shared:market:{exchange}:{symbol}:{interval}:ohlcv)
        """
        exchange = self._normalize_exchange(exchange)
        symbol = self._normalize_symbol(symbol)

        return f"shared:market:{exchange}:{symbol}:{interval}:ohlcv"

    def get_ticker_key(
        self,
        exchange: str,
        symbol: str
    ) -> str:
        """
        Get Redis key for ticker data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Redis key string (shared:market:{exchange}:{symbol}:ticker)
        """
        exchange = self._normalize_exchange(exchange)
        symbol = self._normalize_symbol(symbol)

        return f"shared:market:{exchange}:{symbol}:ticker"

    def get_orderbook_key(
        self,
        exchange: str,
        symbol: str
    ) -> str:
        """
        Get Redis key for orderbook data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Redis key string (shared:market:{exchange}:{symbol}:orderbook)
        """
        exchange = self._normalize_exchange(exchange)
        symbol = self._normalize_symbol(symbol)

        return f"shared:market:{exchange}:{symbol}:orderbook"

    async def set_ohlcv(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set OHLCV data in shared cache.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            interval: Time interval
            data: OHLCV data dictionary
            ttl_seconds: TTL in seconds (default: use instance default)

        Raises:
            redis.RedisError: If Redis operation fails
        """
        key = self.get_ohlcv_key(exchange, symbol, interval)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        try:
            # Serialize data to JSON
            value = json.dumps(data)

            # Store with TTL
            await self.redis.setex(key, ttl, value)

            logger.debug(
                f"OHLCV cached: exchange={exchange}, symbol={symbol}, "
                f"interval={interval}, ttl={ttl}s"
            )

        except Exception as e:
            logger.error(
                f"OHLCV cache set failed: exchange={exchange}, symbol={symbol}, "
                f"interval={interval}, error={e}"
            )
            raise

    async def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        interval: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get OHLCV data from shared cache.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            interval: Time interval

        Returns:
            OHLCV data dictionary if found, None if cache miss

        Raises:
            redis.RedisError: If Redis operation fails
            json.JSONDecodeError: If cached data is invalid JSON
        """
        key = self.get_ohlcv_key(exchange, symbol, interval)

        try:
            value = await self.redis.get(key)

            if value is None:
                logger.debug(
                    f"OHLCV cache miss: exchange={exchange}, symbol={symbol}, "
                    f"interval={interval}"
                )
                return None

            # Deserialize JSON
            data = json.loads(value)

            logger.debug(
                f"OHLCV cache hit: exchange={exchange}, symbol={symbol}, "
                f"interval={interval}"
            )

            return data

        except json.JSONDecodeError as e:
            logger.error(
                f"OHLCV cache data corrupt: exchange={exchange}, symbol={symbol}, "
                f"interval={interval}, error={e}"
            )
            raise

        except Exception as e:
            logger.error(
                f"OHLCV cache get failed: exchange={exchange}, symbol={symbol}, "
                f"interval={interval}, error={e}"
            )
            raise

    async def set_ticker(
        self,
        exchange: str,
        symbol: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set ticker data in shared cache.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data: Ticker data dictionary
            ttl_seconds: TTL in seconds (default: use instance default)

        Raises:
            redis.RedisError: If Redis operation fails
        """
        key = self.get_ticker_key(exchange, symbol)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        try:
            value = json.dumps(data)
            await self.redis.setex(key, ttl, value)

            logger.debug(
                f"Ticker cached: exchange={exchange}, symbol={symbol}, ttl={ttl}s"
            )

        except Exception as e:
            logger.error(
                f"Ticker cache set failed: exchange={exchange}, symbol={symbol}, "
                f"error={e}"
            )
            raise

    async def get_ticker(
        self,
        exchange: str,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get ticker data from shared cache.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Ticker data dictionary if found, None if cache miss

        Raises:
            redis.RedisError: If Redis operation fails
            json.JSONDecodeError: If cached data is invalid JSON
        """
        key = self.get_ticker_key(exchange, symbol)

        try:
            value = await self.redis.get(key)

            if value is None:
                logger.debug(
                    f"Ticker cache miss: exchange={exchange}, symbol={symbol}"
                )
                return None

            data = json.loads(value)

            logger.debug(
                f"Ticker cache hit: exchange={exchange}, symbol={symbol}"
            )

            return data

        except json.JSONDecodeError as e:
            logger.error(
                f"Ticker cache data corrupt: exchange={exchange}, symbol={symbol}, "
                f"error={e}"
            )
            raise

        except Exception as e:
            logger.error(
                f"Ticker cache get failed: exchange={exchange}, symbol={symbol}, "
                f"error={e}"
            )
            raise

    async def set_orderbook(
        self,
        exchange: str,
        symbol: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set orderbook data in shared cache.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data: Orderbook data dictionary (bids/asks)
            ttl_seconds: TTL in seconds (default: use instance default)

        Raises:
            redis.RedisError: If Redis operation fails
        """
        key = self.get_orderbook_key(exchange, symbol)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        try:
            value = json.dumps(data)
            await self.redis.setex(key, ttl, value)

            logger.debug(
                f"Orderbook cached: exchange={exchange}, symbol={symbol}, ttl={ttl}s"
            )

        except Exception as e:
            logger.error(
                f"Orderbook cache set failed: exchange={exchange}, symbol={symbol}, "
                f"error={e}"
            )
            raise

    async def get_orderbook(
        self,
        exchange: str,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get orderbook data from shared cache.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Orderbook data dictionary if found, None if cache miss

        Raises:
            redis.RedisError: If Redis operation fails
            json.JSONDecodeError: If cached data is invalid JSON
        """
        key = self.get_orderbook_key(exchange, symbol)

        try:
            value = await self.redis.get(key)

            if value is None:
                logger.debug(
                    f"Orderbook cache miss: exchange={exchange}, symbol={symbol}"
                )
                return None

            data = json.loads(value)

            logger.debug(
                f"Orderbook cache hit: exchange={exchange}, symbol={symbol}"
            )

            return data

        except json.JSONDecodeError as e:
            logger.error(
                f"Orderbook cache data corrupt: exchange={exchange}, symbol={symbol}, "
                f"error={e}"
            )
            raise

        except Exception as e:
            logger.error(
                f"Orderbook cache get failed: exchange={exchange}, symbol={symbol}, "
                f"error={e}"
            )
            raise

    async def delete_ohlcv(
        self,
        exchange: str,
        symbol: str,
        interval: str
    ) -> bool:
        """
        Delete OHLCV data from cache.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            interval: Time interval

        Returns:
            True if key was deleted, False if key didn't exist

        Raises:
            redis.RedisError: If Redis operation fails
        """
        key = self.get_ohlcv_key(exchange, symbol, interval)

        try:
            deleted = await self.redis.delete(key)

            if deleted:
                logger.debug(
                    f"OHLCV cache deleted: exchange={exchange}, symbol={symbol}, "
                    f"interval={interval}"
                )
            else:
                logger.debug(
                    f"OHLCV cache key not found: exchange={exchange}, symbol={symbol}, "
                    f"interval={interval}"
                )

            return deleted > 0

        except Exception as e:
            logger.error(
                f"OHLCV cache delete failed: exchange={exchange}, symbol={symbol}, "
                f"interval={interval}, error={e}"
            )
            raise

    async def delete_ticker(
        self,
        exchange: str,
        symbol: str
    ) -> bool:
        """
        Delete ticker data from cache.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            True if key was deleted, False if key didn't exist

        Raises:
            redis.RedisError: If Redis operation fails
        """
        key = self.get_ticker_key(exchange, symbol)

        try:
            deleted = await self.redis.delete(key)

            if deleted:
                logger.debug(
                    f"Ticker cache deleted: exchange={exchange}, symbol={symbol}"
                )

            return deleted > 0

        except Exception as e:
            logger.error(
                f"Ticker cache delete failed: exchange={exchange}, symbol={symbol}, "
                f"error={e}"
            )
            raise

    async def delete_orderbook(
        self,
        exchange: str,
        symbol: str
    ) -> bool:
        """
        Delete orderbook data from cache.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            True if key was deleted, False if key didn't exist

        Raises:
            redis.RedisError: If Redis operation fails
        """
        key = self.get_orderbook_key(exchange, symbol)

        try:
            deleted = await self.redis.delete(key)

            if deleted:
                logger.debug(
                    f"Orderbook cache deleted: exchange={exchange}, symbol={symbol}"
                )

            return deleted > 0

        except Exception as e:
            logger.error(
                f"Orderbook cache delete failed: exchange={exchange}, symbol={symbol}, "
                f"error={e}"
            )
            raise

    async def clear_symbol(
        self,
        exchange: str,
        symbol: str
    ) -> int:
        """
        Clear all cached data for a symbol (OHLCV, ticker, orderbook).

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Number of keys deleted

        Raises:
            redis.RedisError: If Redis operation fails
        """
        exchange = self._normalize_exchange(exchange)
        symbol = self._normalize_symbol(symbol)

        pattern = f"shared:market:{exchange}:{symbol}:*"

        try:
            # Find all matching keys
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if not keys:
                logger.debug(
                    f"No keys to clear: exchange={exchange}, symbol={symbol}"
                )
                return 0

            # Delete all keys
            deleted = await self.redis.delete(*keys)

            logger.info(
                f"Symbol cache cleared: exchange={exchange}, symbol={symbol}, "
                f"keys_deleted={deleted}"
            )

            return deleted

        except Exception as e:
            logger.error(
                f"Symbol cache clear failed: exchange={exchange}, symbol={symbol}, "
                f"error={e}"
            )
            raise

    async def get_cache_info(
        self,
        exchange: str,
        symbol: str,
        interval: Optional[str] = None,
        data_type: str = "ticker"
    ) -> Dict[str, Any]:
        """
        Get cache metadata for a key.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            interval: Time interval (required for OHLCV)
            data_type: Type of data ("ohlcv", "ticker", "orderbook")

        Returns:
            Dictionary with cache metadata:
            - key: Redis key
            - exists: Whether key exists
            - ttl_seconds: Remaining TTL (-2 if nonexistent, -1 if no expiry)

        Raises:
            redis.RedisError: If Redis operation fails
        """
        # Get appropriate key
        if data_type == "ohlcv":
            if interval is None:
                raise ValueError("interval required for OHLCV data type")
            key = self.get_ohlcv_key(exchange, symbol, interval)
        elif data_type == "ticker":
            key = self.get_ticker_key(exchange, symbol)
        elif data_type == "orderbook":
            key = self.get_orderbook_key(exchange, symbol)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        try:
            # Check existence and TTL
            exists = await self.redis.exists(key)
            ttl = await self.redis.ttl(key)

            info = {
                "key": key,
                "exists": exists > 0,
                "ttl_seconds": ttl
            }

            logger.debug(
                f"Cache info: exchange={exchange}, symbol={symbol}, "
                f"data_type={data_type}, exists={info['exists']}, ttl={ttl}s"
            )

            return info

        except Exception as e:
            logger.error(
                f"Cache info failed: exchange={exchange}, symbol={symbol}, "
                f"data_type={data_type}, error={e}"
            )
            raise
