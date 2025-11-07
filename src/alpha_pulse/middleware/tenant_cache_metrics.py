"""
Prometheus metrics for tenant-aware caching.

Tracks namespace isolation, tenant-specific hit rates, and cross-tenant
shared cache access patterns.
"""

from prometheus_client import Counter, Gauge, Histogram

# Tenant cache operations
tenant_cache_requests_total = Counter(
    'tenant_cache_requests_total',
    'Total number of tenant cache requests',
    ['tenant_id', 'operation']  # operation: get, set, delete, mget, mset
)

tenant_cache_hits_total = Counter(
    'tenant_cache_hits_total',
    'Total number of tenant cache hits',
    ['tenant_id']
)

tenant_cache_misses_total = Counter(
    'tenant_cache_misses_total',
    'Total number of tenant cache misses',
    ['tenant_id']
)

# Namespace isolation metrics
tenant_namespace_violations_total = Counter(
    'tenant_namespace_violations_total',
    'Total number of namespace isolation violations detected',
    ['violation_type']  # violation_type: cross_tenant_access, missing_prefix
)

tenant_cache_keys_current = Gauge(
    'tenant_cache_keys_current',
    'Current number of cache keys per tenant',
    ['tenant_id']
)

tenant_cache_memory_bytes = Gauge(
    'tenant_cache_memory_bytes',
    'Estimated memory usage per tenant in bytes',
    ['tenant_id']
)

# Shared market data access
tenant_shared_cache_requests_total = Counter(
    'tenant_shared_cache_requests_total',
    'Total number of shared cache requests by tenants',
    ['tenant_id', 'data_type']  # data_type: ohlcv, ticker, orderbook
)

shared_cache_tenant_access_count = Gauge(
    'shared_cache_tenant_access_count',
    'Number of unique tenants accessing each shared cache key',
    ['exchange', 'symbol', 'data_type']
)

# Operation latency
tenant_cache_operation_latency_ms = Histogram(
    'tenant_cache_operation_latency_milliseconds',
    'Latency of tenant cache operations in milliseconds',
    ['tenant_id', 'operation'],
    buckets=[0.5, 1, 2, 5, 10, 25, 50, 100, 250]
)

# Tenant hit rate (computed metric)
tenant_cache_hit_rate_percent = Gauge(
    'tenant_cache_hit_rate_percent',
    'Cache hit rate percentage per tenant',
    ['tenant_id']
)
