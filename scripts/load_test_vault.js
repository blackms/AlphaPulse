/**
 * HashiCorp Vault Load Testing
 *
 * Validates that Vault can handle 10k+ req/sec with P99 latency <10ms.
 *
 * Prerequisites:
 *   - k6 installed: https://k6.io/docs/getting-started/installation/
 *   - Vault running and unsealed
 *   - Test secrets populated (see setup.sh)
 *
 * Usage:
 *   k6 run --vus 100 --duration 60s scripts/load_test_vault.js
 *   k6 run --vus 200 --duration 120s scripts/load_test_vault.js  # Higher load
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const vaultLatency = new Trend('vault_latency');
const readCount = new Counter('read_operations');
const writeCount = new Counter('write_operations');

// Configuration
const VAULT_URL = __ENV.VAULT_ADDR || 'http://localhost:8200';
const VAULT_TOKEN = __ENV.VAULT_TOKEN || 'root';
const NUM_TENANTS = parseInt(__ENV.NUM_TENANTS || '1000');
const NUM_EXCHANGES = 3; // binance, coinbase, kraken

// Test options
export const options = {
  stages: [
    { duration: '30s', target: 50 },   // Ramp up to 50 VUs
    { duration: '1m', target: 100 },   // Ramp up to 100 VUs
    { duration: '2m', target: 100 },   // Stay at 100 VUs (steady state)
    { duration: '1m', target: 200 },   // Spike to 200 VUs
    { duration: '1m', target: 200 },   // Stay at spike
    { duration: '30s', target: 0 },    // Ramp down
  ],
  thresholds: {
    // Success criteria
    'http_req_duration': ['p(95)<20', 'p(99)<50'],  // P95 <20ms, P99 <50ms
    'http_req_failed': ['rate<0.01'],                // Error rate <1%
    'errors': ['rate<0.01'],                         // Custom error rate <1%
    'vault_latency': ['p(99)<10'],                   // P99 latency <10ms (target)
  },
};

/**
 * Setup: Create test secrets in Vault
 *
 * Run before load test:
 *   export VAULT_ADDR=http://localhost:8200
 *   export VAULT_TOKEN=root
 *   ./scripts/setup_vault_test_data.sh
 */
export function setup() {
  console.log(`Starting Vault load test`);
  console.log(`  Vault URL: ${VAULT_URL}`);
  console.log(`  Tenants: ${NUM_TENANTS}`);
  console.log(`  Exchanges per tenant: ${NUM_EXCHANGES}`);
  console.log(`  Total secrets: ${NUM_TENANTS * NUM_EXCHANGES}`);

  // Verify Vault is accessible
  const healthCheck = http.get(`${VAULT_URL}/v1/sys/health`);
  check(healthCheck, {
    'Vault is healthy': (r) => r.status === 200,
  });

  return { vaultUrl: VAULT_URL, vaultToken: VAULT_TOKEN };
}

/**
 * Main test scenario: Mixed read/write workload (90% read, 10% write)
 */
export default function(data) {
  const tenantId = `tenant-${Math.floor(Math.random() * NUM_TENANTS) + 1}`;
  const exchanges = ['binance', 'coinbase', 'kraken'];
  const exchange = exchanges[Math.floor(Math.random() * exchanges.length)];
  const secretPath = `secret/data/tenants/${tenantId}/exchanges/${exchange}`;

  const headers = {
    'X-Vault-Token': data.vaultToken,
    'Content-Type': 'application/json',
  };

  // 90% read operations
  if (Math.random() < 0.9) {
    const startTime = Date.now();

    const res = http.get(
      `${data.vaultUrl}/v1/${secretPath}`,
      { headers }
    );

    const latency = Date.now() - startTime;
    vaultLatency.add(latency);
    readCount.add(1);

    const success = check(res, {
      'read status is 200': (r) => r.status === 200,
      'read has data': (r) => {
        if (r.status !== 200) return false;
        const body = JSON.parse(r.body);
        return body.data && body.data.data;
      },
      'read latency <50ms': () => latency < 50,
    });

    if (!success) {
      errorRate.add(1);
    }
  }
  // 10% write operations (simulate credential rotation)
  else {
    const startTime = Date.now();

    const payload = JSON.stringify({
      data: {
        api_key: `test_key_${Date.now()}`,
        secret: `test_secret_${Date.now()}`,
        permissions: 'trading',
        created_at: new Date().toISOString(),
      }
    });

    const res = http.post(
      `${data.vaultUrl}/v1/${secretPath}`,
      payload,
      { headers }
    );

    const latency = Date.now() - startTime;
    vaultLatency.add(latency);
    writeCount.add(1);

    const success = check(res, {
      'write status is 200 or 204': (r) => r.status === 200 || r.status === 204,
      'write latency <100ms': () => latency < 100,
    });

    if (!success) {
      errorRate.add(1);
    }
  }

  // Small delay to prevent overwhelming Vault
  sleep(0.01); // 10ms delay
}

/**
 * Teardown: Print summary statistics
 */
export function teardown(data) {
  console.log('\n=== Vault Load Test Complete ===');
}

/**
 * Handle summary - custom reporting
 */
export function handleSummary(data) {
  const summary = {
    'Test Duration': `${data.state.testRunDurationMs / 1000}s`,
    'Total Requests': data.metrics.http_reqs.values.count,
    'Requests/sec': data.metrics.http_reqs.values.rate.toFixed(2),
    'Read Operations': data.metrics.read_operations?.values.count || 0,
    'Write Operations': data.metrics.write_operations?.values.count || 0,
    'Error Rate': `${(data.metrics.errors.values.rate * 100).toFixed(2)}%`,
    'P50 Latency': `${data.metrics.vault_latency.values['p(50)'].toFixed(2)}ms`,
    'P95 Latency': `${data.metrics.vault_latency.values['p(95)'].toFixed(2)}ms`,
    'P99 Latency': `${data.metrics.vault_latency.values['p(99)'].toFixed(2)}ms`,
    'Max Latency': `${data.metrics.vault_latency.values.max.toFixed(2)}ms`,
  };

  console.log('\n=== Summary Statistics ===');
  Object.entries(summary).forEach(([key, value]) => {
    console.log(`${key}: ${value}`);
  });

  // Determine pass/fail
  const p99Latency = data.metrics.vault_latency.values['p(99)'];
  const requestsPerSec = data.metrics.http_reqs.values.rate;
  const errorRate = data.metrics.errors.values.rate;

  console.log('\n=== Test Results ===');

  if (requestsPerSec >= 10000) {
    console.log('✓ PASS: Throughput >10k req/sec (target met)');
  } else if (requestsPerSec >= 5000) {
    console.log('⚠ WARNING: Throughput 5k-10k req/sec (below target but acceptable)');
  } else {
    console.log('✗ FAIL: Throughput <5k req/sec (unacceptable)');
  }

  if (p99Latency < 10) {
    console.log('✓ PASS: P99 latency <10ms (target met)');
  } else if (p99Latency < 20) {
    console.log('⚠ WARNING: P99 latency 10-20ms (above target but acceptable)');
  } else {
    console.log('✗ FAIL: P99 latency >20ms (unacceptable)');
  }

  if (errorRate < 0.01) {
    console.log('✓ PASS: Error rate <1% (target met)');
  } else {
    console.log('✗ FAIL: Error rate >1% (unacceptable)');
  }

  console.log('\n=== Decision ===');
  if (requestsPerSec >= 10000 && p99Latency < 10 && errorRate < 0.01) {
    console.log('Decision: PROCEED with Vault OSS (all targets met)');
  } else if (requestsPerSec >= 5000 && p99Latency < 20 && errorRate < 0.01) {
    console.log('Decision: PROCEED with caching mitigation (extend TTL to 1 hour)');
  } else {
    console.log('Decision: UPGRADE to Vault Enterprise or consider alternatives');
  }

  return {
    'stdout': JSON.stringify(summary, null, 2),
    'summary.json': JSON.stringify(data, null, 2),
  };
}
