"""Decorators for easy cache integration."""

import asyncio
import functools
import hashlib
import inspect
import json
import logging
from typing import Any, Callable, Dict, Optional, Union

from .redis_manager import RedisManager, CacheTier
from .cache_strategies import CacheStrategy, CacheStrategyFactory
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _generate_cache_key(
    func_name: str,
    args: tuple,
    kwargs: dict,
    namespace: Optional[str] = None,
    include_args: bool = True
) -> str:
    """Generate a cache key from function name and arguments."""
    parts = []
    
    if namespace:
        parts.append(namespace)
    
    parts.append(func_name)
    
    if include_args:
        # Create a hashable representation of args and kwargs
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        
        # Convert to JSON and hash
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()[:8]
        parts.append(key_hash)
    
    return ":".join(parts)


def cache(
    ttl: Optional[int] = None,
    tier: CacheTier = CacheTier.L2_LOCAL_REDIS,
    namespace: Optional[str] = None,
    strategy: str = "cache_aside",
    include_args: bool = True,
    key_prefix: Optional[str] = None,
    condition: Optional[Callable[..., bool]] = None
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        tier: Cache tier to use
        namespace: Optional namespace for cache keys
        strategy: Cache strategy to use
        include_args: Whether to include function arguments in cache key
        key_prefix: Optional prefix for cache key
        condition: Optional function to determine if result should be cached
    """
    def decorator(func: Callable) -> Callable:
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get cache manager from first argument if it has one
                cache_manager = None
                if args and hasattr(args[0], '_cache_manager'):
                    cache_manager = args[0]._cache_manager
                elif args and hasattr(args[0], 'cache_manager'):
                    cache_manager = args[0].cache_manager
                
                if not cache_manager:
                    # No cache manager available, call function directly
                    return await func(*args, **kwargs)
                
                # Generate cache key
                cache_key = _generate_cache_key(
                    func.__name__,
                    args[1:] if args else (),  # Skip self/cls
                    kwargs,
                    namespace,
                    include_args
                )
                
                if key_prefix:
                    cache_key = f"{key_prefix}:{cache_key}"
                
                # Create strategy
                cache_strategy = CacheStrategyFactory.create_strategy(
                    strategy,
                    cache_manager,
                    getattr(cache_manager, 'metrics', None)
                )
                
                # Define fetch function
                async def fetch_fn():
                    result = await func(*args, **kwargs)
                    
                    # Check condition if provided
                    if condition and not condition(result):
                        return result
                    
                    return result
                
                # Get from cache
                return await cache_strategy.get(cache_key, fetch_fn, ttl, tier)
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get cache manager from first argument if it has one
                cache_manager = None
                if args and hasattr(args[0], '_cache_manager'):
                    cache_manager = args[0]._cache_manager
                elif args and hasattr(args[0], 'cache_manager'):
                    cache_manager = args[0].cache_manager
                
                if not cache_manager:
                    # No cache manager available, call function directly
                    return func(*args, **kwargs)
                
                # Generate cache key
                cache_key = _generate_cache_key(
                    func.__name__,
                    args[1:] if args else (),  # Skip self/cls
                    kwargs,
                    namespace,
                    include_args
                )
                
                if key_prefix:
                    cache_key = f"{key_prefix}:{cache_key}"
                
                # For sync functions, we need to run in async context
                async def async_get():
                    cache_strategy = CacheStrategyFactory.create_strategy(
                        strategy,
                        cache_manager,
                        getattr(cache_manager, 'metrics', None)
                    )
                    
                    async def fetch_fn():
                        result = func(*args, **kwargs)
                        
                        # Check condition if provided
                        if condition and not condition(result):
                            return result
                        
                        return result
                    
                    return await cache_strategy.get(cache_key, fetch_fn, ttl, tier)
                
                # Run in event loop
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_get())
            
            return sync_wrapper
    
    return decorator


def cache_invalidate(
    namespace: Optional[str] = None,
    key_prefix: Optional[str] = None,
    related_keys: Optional[List[str]] = None
):
    """
    Decorator for invalidating cache when function is called.
    
    Args:
        namespace: Optional namespace for cache keys
        key_prefix: Optional prefix for cache key
        related_keys: List of related cache keys to invalidate
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Call original function
                result = await func(*args, **kwargs)
                
                # Get cache manager
                cache_manager = None
                if args and hasattr(args[0], '_cache_manager'):
                    cache_manager = args[0]._cache_manager
                elif args and hasattr(args[0], 'cache_manager'):
                    cache_manager = args[0].cache_manager
                
                if cache_manager:
                    # Generate cache key for this function
                    cache_key = _generate_cache_key(
                        func.__name__,
                        args[1:] if args else (),
                        kwargs,
                        namespace,
                        True
                    )
                    
                    if key_prefix:
                        cache_key = f"{key_prefix}:{cache_key}"
                    
                    # Invalidate this key
                    await cache_manager.delete(cache_key)
                    
                    # Invalidate related keys
                    if related_keys:
                        for key in related_keys:
                            await cache_manager.delete(key)
                
                return result
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Call original function
                result = func(*args, **kwargs)
                
                # Get cache manager
                cache_manager = None
                if args and hasattr(args[0], '_cache_manager'):
                    cache_manager = args[0]._cache_manager
                elif args and hasattr(args[0], 'cache_manager'):
                    cache_manager = args[0].cache_manager
                
                if cache_manager:
                    # Generate cache key for this function
                    cache_key = _generate_cache_key(
                        func.__name__,
                        args[1:] if args else (),
                        kwargs,
                        namespace,
                        True
                    )
                    
                    if key_prefix:
                        cache_key = f"{key_prefix}:{cache_key}"
                    
                    # Invalidate this key
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(cache_manager.delete(cache_key))
                    
                    # Invalidate related keys
                    if related_keys:
                        for key in related_keys:
                            loop.run_until_complete(cache_manager.delete(key))
                
                return result
            
            return sync_wrapper
    
    return decorator


def cache_clear(pattern: Optional[str] = None, tier: Optional[CacheTier] = None):
    """
    Decorator for clearing cache entries matching pattern.
    
    Args:
        pattern: Optional pattern to match cache keys
        tier: Optional cache tier to clear
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get cache manager
                cache_manager = None
                if args and hasattr(args[0], '_cache_manager'):
                    cache_manager = args[0]._cache_manager
                elif args and hasattr(args[0], 'cache_manager'):
                    cache_manager = args[0].cache_manager
                
                if cache_manager:
                    # Clear cache before calling function
                    await cache_manager.clear(pattern, tier)
                
                return await func(*args, **kwargs)
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get cache manager
                cache_manager = None
                if args and hasattr(args[0], '_cache_manager'):
                    cache_manager = args[0]._cache_manager
                elif args and hasattr(args[0], 'cache_manager'):
                    cache_manager = args[0].cache_manager
                
                if cache_manager:
                    # Clear cache before calling function
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(cache_manager.clear(pattern, tier))
                
                return func(*args, **kwargs)
            
            return sync_wrapper
    
    return decorator


def batch_cache(
    ttl: Optional[int] = None,
    tier: CacheTier = CacheTier.L2_LOCAL_REDIS,
    namespace: Optional[str] = None,
    strategy: str = "cache_aside",
    batch_size: int = 100
):
    """
    Decorator for caching batch operations.
    
    Args:
        ttl: Time to live in seconds
        tier: Cache tier to use
        namespace: Optional namespace for cache keys
        strategy: Cache strategy to use
        batch_size: Maximum batch size for multi-get operations
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get cache manager
            cache_manager = None
            if args and hasattr(args[0], '_cache_manager'):
                cache_manager = args[0]._cache_manager
            elif args and hasattr(args[0], 'cache_manager'):
                cache_manager = args[0].cache_manager
            
            if not cache_manager:
                return await func(*args, **kwargs)
            
            # Extract keys from arguments (assume first arg after self is keys)
            keys = args[1] if len(args) > 1 else kwargs.get('keys', [])
            if not keys:
                return await func(*args, **kwargs)
            
            # Generate cache keys
            cache_keys = []
            for key in keys:
                cache_key = f"{namespace}:{func.__name__}:{key}" if namespace else f"{func.__name__}:{key}"
                cache_keys.append(cache_key)
            
            # Try to get from cache in batches
            all_results = {}
            missing_keys = []
            
            for i in range(0, len(cache_keys), batch_size):
                batch_keys = cache_keys[i:i + batch_size]
                batch_results = await cache_manager.mget(batch_keys, tier)
                
                for j, (cache_key, orig_key) in enumerate(zip(batch_keys, keys[i:i + batch_size])):
                    if cache_key in batch_results:
                        all_results[orig_key] = batch_results[cache_key]
                    else:
                        missing_keys.append(orig_key)
            
            # Fetch missing data
            if missing_keys:
                # Call function with missing keys
                new_args = list(args)
                new_args[1] = missing_keys
                missing_results = await func(*new_args, **kwargs)
                
                # Cache missing results
                cache_data = {}
                for key, value in missing_results.items():
                    cache_key = f"{namespace}:{func.__name__}:{key}" if namespace else f"{func.__name__}:{key}"
                    cache_data[cache_key] = value
                    all_results[key] = value
                
                if cache_data:
                    await cache_manager.mset(cache_data, ttl, tier)
            
            return all_results
        
        return async_wrapper
    
    return decorator


class CacheContextManager:
    """Context manager for temporary cache operations."""
    
    def __init__(
        self,
        cache_manager: RedisManager,
        namespace: str,
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ):
        """Initialize cache context manager."""
        self.cache_manager = cache_manager
        self.namespace = namespace
        self.ttl = ttl
        self.tier = tier
        self._cached_keys: Set[str] = set()
    
    async def __aenter__(self):
        """Enter context."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and optionally clean up."""
        if exc_type:
            # Error occurred, optionally clean up cached keys
            for key in self._cached_keys:
                await self.cache_manager.delete(key)
    
    async def get(self, key: str) -> Any:
        """Get value from cache."""
        full_key = f"{self.namespace}:{key}"
        return await self.cache_manager.get(full_key, self.tier)
    
    async def set(self, key: str, value: Any) -> bool:
        """Set value in cache."""
        full_key = f"{self.namespace}:{key}"
        self._cached_keys.add(full_key)
        return await self.cache_manager.set(full_key, value, self.ttl, self.tier)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        full_key = f"{self.namespace}:{key}"
        self._cached_keys.discard(full_key)
        return await self.cache_manager.delete(full_key)