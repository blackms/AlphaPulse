"""In-memory cache implementation."""
import time
from typing import Any, Dict, Optional


class MemoryCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self):
        """Initialize cache."""
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
            
        # Check if expired
        if self.cache[key]["expiry"] < time.time():
            del self.cache[key]
            return None
            
        return self.cache[key]["value"]
    
    async def set(self, key: str, value: Any, expiry: int = 300) -> None:
        """Set value in cache with expiry in seconds."""
        self.cache[key] = {
            "value": value,
            "expiry": time.time() + expiry
        }
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
            
    async def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()