"""Redis cache implementation."""
import json
from typing import Any, Optional
import aioredis


class RedisCache:
    """Redis cache implementation."""
    
    def __init__(self, redis_url: str):
        """Initialize Redis cache."""
        self.redis_url = redis_url
        self.redis = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self.redis:
            self.redis = await aioredis.create_redis_pool(self.redis_url)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        await self.connect()
        
        value = await self.redis.get(key)
        if value is None:
            return None
            
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.decode("utf-8")
    
    async def set(self, key: str, value: Any, expiry: int = 300) -> None:
        """Set value in cache with expiry in seconds."""
        await self.connect()
        
        # Serialize value to JSON if it's not a string
        if not isinstance(value, str):
            value = json.dumps(value)
            
        await self.redis.setex(key, expiry, value)
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        await self.connect()
        await self.redis.delete(key)
        
    async def clear(self) -> None:
        """Clear the entire cache (use with caution)."""
        await self.connect()
        await self.redis.flushdb()