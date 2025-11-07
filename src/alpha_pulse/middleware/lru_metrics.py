"""
Prometheus metrics for LRU eviction system.

Provides observability for LRU tracking and eviction operations.
"""

from prometheus_client import Counter, Histogram, Gauge

# LRU tracking metrics
lru_track_operations_total = Counter(
    'lru_track_operations_total',
    'Total number of LRU track operations',
    ['tenant_id']
)

lru_eviction_operations_total = Counter(
    'lru_eviction_operations_total',
    'Total number of LRU eviction operations',
    ['tenant_id', 'trigger']  # trigger: quota_exceeded, manual, scheduled
)

lru_keys_evicted_total = Counter(
    'lru_keys_evicted_total',
    'Total number of keys evicted via LRU',
    ['tenant_id']
)

lru_eviction_size_bytes_total = Counter(
    'lru_eviction_size_bytes_total',
    'Total size evicted via LRU in bytes',
    ['tenant_id']
)

# LRU state gauges
lru_tracked_keys_current = Gauge(
    'lru_tracked_keys_current',
    'Current number of keys tracked in LRU',
    ['tenant_id']
)

lru_oldest_key_age_seconds = Gauge(
    'lru_oldest_key_age_seconds',
    'Age of oldest key in LRU tracking (seconds)',
    ['tenant_id']
)

# Eviction performance metrics
lru_eviction_latency_ms = Histogram(
    'lru_eviction_latency_milliseconds',
    'Latency of LRU eviction operations in milliseconds',
    ['operation'],  # operation: single_key, batch, to_target
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
)

lru_eviction_batch_size = Histogram(
    'lru_eviction_batch_size',
    'Number of keys evicted per batch operation',
    buckets=[1, 5, 10, 20, 50, 100, 200, 500]
)

# Eviction success/failure metrics
lru_eviction_target_reached_total = Counter(
    'lru_eviction_target_reached_total',
    'Number of times eviction reached target usage',
    ['tenant_id']
)

lru_eviction_target_failed_total = Counter(
    'lru_eviction_target_failed_total',
    'Number of times eviction failed to reach target (insufficient keys)',
    ['tenant_id']
)

# LRU errors
lru_errors_total = Counter(
    'lru_errors_total',
    'Total number of LRU operation errors',
    ['operation', 'error_type']
)
