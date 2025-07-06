"""Demo script showing Redis caching capabilities."""

import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime

from alpha_pulse.services.caching_service import CachingService
from alpha_pulse.cache.cache_decorators import cache
from alpha_pulse.cache.cache_monitoring import CacheMonitor, CacheOperation, CacheAnalytics
from alpha_pulse.cache.redis_manager import CacheTier
from alpha_pulse.config.cache_config import CacheConfig


class TradingDataService:
    """Example service using caching."""
    
    def __init__(self, cache_service: CachingService):
        self._cache_manager = cache_service.redis_manager
        self.cache_service = cache_service
    
    @cache(ttl=300, namespace="market_data")
    async def get_market_quotes(self, symbol: str) -> dict:
        """Get market quotes (cached)."""
        print(f"Fetching market quotes for {symbol} from API...")
        await asyncio.sleep(0.5)  # Simulate API call
        
        return {
            "symbol": symbol,
            "price": 100 + np.random.random() * 10,
            "volume": int(1000000 * np.random.random()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @cache(ttl=600, namespace="indicators", strategy="refresh_ahead")
    async def calculate_indicators(self, symbol: str, period: int) -> dict:
        """Calculate technical indicators (cached with refresh-ahead)."""
        print(f"Calculating indicators for {symbol} with period {period}...")
        await asyncio.sleep(1.0)  # Simulate computation
        
        # Generate fake indicator data
        prices = 100 + np.cumsum(np.random.randn(period))
        
        return {
            "symbol": symbol,
            "period": period,
            "sma": float(np.mean(prices)),
            "rsi": float(50 + np.random.randn() * 10),
            "macd": float(np.random.randn()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Get historical data using manual caching."""
        cache_key = f"historical:{symbol}:{days}"
        
        # Try to get from cache
        cached_data = await self.cache_service.get(cache_key)
        if cached_data is not None:
            print(f"Historical data for {symbol} found in cache")
            return cached_data
        
        # Simulate fetching historical data
        print(f"Fetching historical data for {symbol} from database...")
        await asyncio.sleep(1.5)
        
        # Generate fake historical data
        dates = pd.date_range(end=datetime.utcnow(), periods=days, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(days).cumsum(),
            'high': 100 + np.random.randn(days).cumsum() + 2,
            'low': 100 + np.random.randn(days).cumsum() - 2,
            'close': 100 + np.random.randn(days).cumsum(),
            'volume': np.random.randint(100000, 1000000, days)
        })
        
        # Cache for 1 hour
        await self.cache_service.set(cache_key, df, ttl=3600)
        
        return df


async def demo_basic_caching():
    """Demonstrate basic caching operations."""
    print("\n=== Basic Caching Demo ===\n")
    
    # Create caching service
    cache_service = CachingService.create_for_trading()
    await cache_service.initialize()
    
    try:
        # Create service
        trading_service = TradingDataService(cache_service)
        
        # First call - cache miss
        start = time.time()
        quotes = await trading_service.get_market_quotes("AAPL")
        print(f"First call took: {time.time() - start:.3f}s")
        print(f"Quotes: {quotes}")
        
        # Second call - cache hit
        start = time.time()
        quotes = await trading_service.get_market_quotes("AAPL")
        print(f"Second call took: {time.time() - start:.3f}s (from cache)")
        
        # Different symbol - cache miss
        start = time.time()
        quotes = await trading_service.get_market_quotes("GOOGL")
        print(f"Different symbol took: {time.time() - start:.3f}s")
        
    finally:
        await cache_service.close()


async def demo_batch_operations():
    """Demonstrate batch caching operations."""
    print("\n=== Batch Operations Demo ===\n")
    
    cache_service = CachingService.create_for_trading()
    await cache_service.initialize()
    
    try:
        # Set multiple values
        data = {
            f"symbol:{symbol}": {"price": 100 + i, "volume": 1000000 * i}
            for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])
        }
        
        await cache_service.mset(data, ttl=300)
        print(f"Set {len(data)} values in cache")
        
        # Get multiple values
        keys = list(data.keys())
        results = await cache_service.mget(keys)
        print(f"Retrieved {len(results)} values from cache")
        
        # Verify data
        for key, value in results.items():
            print(f"{key}: {value}")
        
    finally:
        await cache_service.close()


async def demo_cache_invalidation():
    """Demonstrate cache invalidation."""
    print("\n=== Cache Invalidation Demo ===\n")
    
    cache_service = CachingService.create_for_trading()
    await cache_service.initialize()
    
    try:
        # Set values with tags
        await cache_service.set("user:123:profile", {"name": "John"}, tags=["users", "profiles"])
        await cache_service.set("user:456:profile", {"name": "Jane"}, tags=["users", "profiles"])
        await cache_service.set("product:789:details", {"name": "Widget"}, tags=["products"])
        
        print("Set 3 cache entries with tags")
        
        # Invalidate by tag
        count = await cache_service.invalidate(tags=["users"])
        print(f"Invalidated {count} entries with 'users' tag")
        
        # Check if entries still exist
        user_profile = await cache_service.get("user:123:profile")
        product_details = await cache_service.get("product:789:details")
        
        print(f"User profile exists: {user_profile is not None}")
        print(f"Product details exists: {product_details is not None}")
        
    finally:
        await cache_service.close()


async def demo_cache_monitoring():
    """Demonstrate cache monitoring and analytics."""
    print("\n=== Cache Monitoring Demo ===\n")
    
    config = CacheConfig()
    config.monitoring.enabled = True
    
    cache_service = CachingService(config)
    await cache_service.initialize()
    
    # Create monitor
    monitor = CacheMonitor(cache_service.redis_manager)
    await monitor.start_monitoring()
    
    try:
        # Simulate operations
        for i in range(50):
            key = f"test:key:{i % 10}"
            
            # Record operation
            start_time = time.time()
            if i % 3 == 0:
                # Cache miss
                success = False
            else:
                # Cache hit
                await cache_service.set(key, f"value_{i}")
                success = True
            
            latency_ms = (time.time() - start_time) * 1000
            
            operation = CacheOperation(
                operation_type="get",
                key=key,
                tier=CacheTier.L2_LOCAL_REDIS,
                success=success,
                latency_ms=latency_ms,
                size_bytes=100 if success else None
            )
            monitor.record_operation(operation)
        
        # Get metrics summary
        summary = monitor.get_metrics_summary()
        print("\nCache Metrics Summary:")
        print(f"Global hit rate: {summary['global']['hit_rate']:.2%}")
        print(f"Average latency: {summary['global']['avg_latency_ms']:.2f}ms")
        print(f"Total operations: {summary['global']['total_operations']}")
        
        # Get recommendations
        recommendations = monitor.get_cache_recommendations()
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"- {rec['recommendation']} (Severity: {rec['severity']})")
        
        # Analyze patterns
        analytics = CacheAnalytics(monitor)
        patterns = analytics.analyze_key_patterns()
        if patterns:
            print(f"\nKey access patterns:")
            print(f"- Total unique keys: {patterns['total_unique_keys']}")
            print(f"- Access skewness: {patterns.get('skewness', 0):.2f}")
            print(f"- Recommendation: {patterns.get('recommendation', 'N/A')}")
        
    finally:
        await monitor.stop_monitoring()
        await cache_service.close()


async def demo_distributed_caching():
    """Demonstrate distributed caching (requires Redis cluster)."""
    print("\n=== Distributed Caching Demo ===\n")
    
    config = CacheConfig()
    config.distributed.enabled = True
    config.distributed.node_id = "demo-node-1"
    config.distributed.static_nodes = [
        {"id": "node-2", "host": "localhost", "port": 6380, "role": "replica"},
        {"id": "node-3", "host": "localhost", "port": 6381, "role": "replica"}
    ]
    
    cache_service = CachingService(config)
    
    try:
        await cache_service.initialize()
        
        # Use distributed cache
        await cache_service.set("distributed:key", "value", tier=CacheTier.L3_DISTRIBUTED_REDIS)
        value = await cache_service.get("distributed:key", tier=CacheTier.L3_DISTRIBUTED_REDIS)
        
        print(f"Distributed cache value: {value}")
        
        # Get cluster stats
        stats = await cache_service.get_stats()
        if "distributed" in stats:
            print(f"\nCluster stats:")
            print(f"- Cluster size: {stats['distributed']['cluster_size']}")
            print(f"- Healthy nodes: {stats['distributed']['healthy_nodes']}")
        
    except Exception as e:
        print(f"Distributed caching not available: {e}")
        print("Note: This requires a Redis cluster setup")
    finally:
        await cache_service.close()


async def main():
    """Run all demos."""
    print("Redis Caching Functionality Demo")
    print("================================")
    
    # Run demos
    await demo_basic_caching()
    await demo_batch_operations()
    await demo_cache_invalidation()
    await demo_cache_monitoring()
    await demo_distributed_caching()
    
    print("\n✅ All demos completed!")


if __name__ == "__main__":
    asyncio.run(main())