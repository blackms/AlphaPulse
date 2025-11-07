"""
Prometheus metrics for shared market data cache.

Tracks cache hit rates, memory savings, and pub/sub invalidation events.
"""

from prometheus_client import Counter, Gauge, Histogram

# Cache hit/miss metrics
shared_cache_requests_total = Counter(
    'shared_cache_requests_total',
    'Total number of shared cache requests',
    ['data_type', 'exchange', 'symbol']  # data_type: ohlcv, ticker, orderbook
)

shared_cache_hits_total = Counter(
    'shared_cache_hits_total',
    'Total number of shared cache hits',
    ['data_type', 'exchange', 'symbol']
)

shared_cache_misses_total = Counter(
    'shared_cache_misses_total',
    'Total number of shared cache misses',
    ['data_type', 'exchange', 'symbol']
)

# Memory metrics
shared_cache_memory_saved_bytes = Gauge(
    'shared_cache_memory_saved_bytes',
    'Estimated memory saved by shared caching vs per-tenant caching',
    ['data_type']
)

shared_cache_entries_current = Gauge(
    'shared_cache_entries_current',
    'Current number of entries in shared cache',
    ['data_type']
)

# Write metrics
shared_cache_writes_total = Counter(
    'shared_cache_writes_total',
    'Total number of shared cache writes',
    ['data_type', 'source']  # source: data_pipeline, manual, update
)

shared_cache_deletes_total = Counter(
    'shared_cache_deletes_total',
    'Total number of shared cache deletions',
    ['data_type', 'reason']  # reason: invalidation, expiry, manual
)

# Invalidation metrics
shared_cache_invalidations_total = Counter(
    'shared_cache_invalidations_total',
    'Total number of cache invalidation events',
    ['data_type', 'trigger']  # trigger: pubsub, manual, scheduled
)

# Latency metrics
shared_cache_operation_latency_ms = Histogram(
    'shared_cache_operation_latency_milliseconds',
    'Latency of shared cache operations in milliseconds',
    ['operation'],  # operation: get, set, delete
    buckets=[0.5, 1, 2, 5, 10, 25, 50, 100]
)

# Hit rate gauge (computed metric)
shared_cache_hit_rate_percent = Gauge(
    'shared_cache_hit_rate_percent',
    'Cache hit rate as percentage',
    ['data_type']
)
