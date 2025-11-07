"""
Prometheus metrics for quota enforcement middleware.

Provides observability for quota operations, cache performance, and enforcement decisions.
"""

from prometheus_client import Counter, Histogram, Gauge

# Quota check metrics
quota_checks_total = Counter(
    'quota_checks_total',
    'Total number of quota checks performed',
    ['tenant_id', 'decision']
)

quota_rejections_total = Counter(
    'quota_rejections_total',
    'Total number of quota rejections (429 responses)',
    ['tenant_id']
)

quota_warnings_total = Counter(
    'quota_warnings_total',
    'Total number of quota warnings (over quota but within overage)',
    ['tenant_id']
)

# Latency metrics
quota_check_latency_ms = Histogram(
    'quota_check_latency_milliseconds',
    'Latency of quota check operations in milliseconds',
    ['operation'],
    buckets=[1, 3, 5, 10, 25, 50, 100, 250, 500, 1000]
)

# Cache performance metrics
quota_cache_hits_total = Counter(
    'quota_cache_hits_total',
    'Total number of Redis cache hits for quota lookups',
    ['tenant_id']
)

quota_cache_misses_total = Counter(
    'quota_cache_misses_total',
    'Total number of Redis cache misses (PostgreSQL fallback)',
    ['tenant_id']
)

# Usage tracking metrics
quota_usage_increments_total = Counter(
    'quota_usage_increments_total',
    'Total number of usage increments',
    ['tenant_id']
)

quota_usage_rollbacks_total = Counter(
    'quota_usage_rollbacks_total',
    'Total number of usage rollbacks (after rejections)',
    ['tenant_id']
)

# Current state gauges
quota_current_usage_mb = Gauge(
    'quota_current_usage_megabytes',
    'Current quota usage in megabytes',
    ['tenant_id']
)

quota_limit_mb = Gauge(
    'quota_limit_megabytes',
    'Quota limit in megabytes',
    ['tenant_id']
)

quota_usage_percent = Gauge(
    'quota_usage_percent',
    'Quota usage as percentage of limit',
    ['tenant_id']
)

# Error metrics
quota_errors_total = Counter(
    'quota_errors_total',
    'Total number of errors in quota operations',
    ['operation', 'error_type']
)

redis_errors_total = Counter(
    'redis_errors_total',
    'Total number of Redis errors (fallback to PostgreSQL)',
    ['operation']
)

# Feature flag metrics
quota_enforcement_enabled = Gauge(
    'quota_enforcement_enabled',
    'Whether quota enforcement is currently enabled (1=enabled, 0=disabled)'
)

quota_excluded_paths_total = Counter(
    'quota_excluded_paths_total',
    'Total number of requests to excluded paths (no quota check)',
    ['path']
)
