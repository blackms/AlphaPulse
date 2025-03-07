"""Cache module for the AlphaPulse API."""
from typing import Any, Optional
from enum import Enum

from ..config import config
from .memory import MemoryCache
from .redis import RedisCache


class CacheType(str, Enum):
    """Cache type enum."""
    MEMORY = "memory"
    REDIS = "redis"


async def get_cache():
    """Get cache instance based on configuration."""
    cache_type = config.cache.type.lower()
    
    if cache_type == CacheType.MEMORY:
        return MemoryCache()
    elif cache_type == CacheType.REDIS:
        return RedisCache(config.cache.redis_url)
    else:
        # Default to memory cache
        return MemoryCache()