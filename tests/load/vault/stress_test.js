/**
 * Vault Load Test - Stress Test (Find Breaking Points)
 *
 * Progressively increases load to find Vault's breaking point and recovery behavior.
 * Measures maximum capacity and degradation patterns.
 *
 * Usage:
 *   k6 run tests/load/vault/stress_test.js
 *   k6 run --duration 20m tests/load/vault/stress_test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { randomString } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');
const requestLatency = new Trend('request_latency');
const successfulRequests = new Counter('successful_requests');
const failedRequests = new Counter('failed_requests');
const currentLoad = new Gauge('current_vus');
const breakingPoint = new Gauge('breaking_point_vus');

// Test configuration - Progressive load increase
export const options = {
  // Gradually increase load to find breaking point
  stages: [
    { duration: '2m', target: 50 },     // Baseline: 50 VUs
    { duration: '2m', target: 100 },    // Moderate: 100 VUs
    { duration: '2m', target: 200 },    // High: 200 VUs
    { duration: '2m', target: 400 },    // Very High: 400 VUs
    { duration: '2m', target: 800 },    // Extreme: 800 VUs
    { duration: '2m', target: 1000 },   // Peak: 1000 VUs
    { duration: '2m', target: 1000 },   // Sustain peak
    { duration: '2m', target: 0 },      // Recovery test
  ],

  // Relaxed thresholds (expect failures at high load)
  thresholds: {
    'errors': ['rate<0.20'],              // Allow up to 20% errors at peak
    'http_req_duration': ['p(50)<1000'],  // P50 < 1s (degraded but functional)
  },

  tags: {
    test_type: 'stress_test',
    environment: 'dev',
  },
};

// Test setup
export function setup() {
  const vaultAddr = __ENV.VAULT_ADDR || 'http://127.0.0.1:8200';
  const vaultToken = __ENV.VAULT_TOKEN || 'root';

  console.log(`Vault Address: ${vaultAddr}`);
  console.log(`Starting Vault stress test (finding breaking point)...`);
  console.log(`WARNING: This test will push Vault to its limits!`);

  // Verify Vault is accessible
  const healthCheck = http.get(`${vaultAddr}/v1/sys/health`);
  const healthy = check(healthCheck, {
    'Vault is healthy before test': (r) => r.status === 200,
  });

  if (!healthy) {
    console.error('Vault is not healthy - aborting stress test');
    return null;
  }

  return {
    vaultAddr,
    vaultToken,
    tenants: generateTenantIds(50),
    exchanges: ['binance', 'coinbase', 'kraken'],
    credentialTypes: ['trading', 'readonly', 'admin'],
    errorThreshold: 0.10,  // 10% error rate indicates breaking point
    latencyThreshold: 2000,  // 2s latency indicates degradation
  };
}

function generateTenantIds(count) {
  const tenants = [];
  for (let i = 0; i < count; i++) {
    tenants.push(`stress-tenant-${String(i).padStart(4, '0')}`);
  }
  return tenants;
}

// Track metrics for breaking point detection
let lastMinuteErrors = [];
let lastMinuteLatencies = [];

export default function (data) {
  if (!data) {
    console.error('Setup failed - skipping test');
    return;
  }

  const { vaultAddr, vaultToken, tenants, exchanges, credentialTypes } = data;

  // Record current load
  currentLoad.add(__VU);

  // Random tenant selection
  const tenant = tenants[Math.floor(Math.random() * tenants.length)];
  const exchange = exchanges[Math.floor(Math.random() * exchanges.length)];
  const credType = credentialTypes[Math.floor(Math.random() * credentialTypes.length)];

  // Mix of operations (60% read, 40% write for higher stress)
  const operation = Math.random() < 0.6 ? 'read' : 'write';

  const path = `secret/data/tenants/${tenant}/${exchange}/${credType}`;
  const url = `${vaultAddr}/v1/${path}`;

  const startTime = Date.now();
  let response;

  try {
    if (operation === 'read') {
      response = http.get(url, {
        headers: {
          'X-Vault-Token': vaultToken,
        },
        timeout: '10s',  // Longer timeout for stress test
        tags: {
          operation: 'read',
          load_level: getCurrentLoadLevel(__VU),
        },
      });
    } else {
      const credentialData = {
        data: {
          api_key: `stress_key_${randomString(32)}`,
          api_secret: `stress_secret_${randomString(64)}`,
          created_at: new Date().toISOString(),
        },
      };

      response = http.post(
        url,
        JSON.stringify(credentialData),
        {
          headers: {
            'X-Vault-Token': vaultToken,
            'Content-Type': 'application/json',
          },
          timeout: '10s',
          tags: {
            operation: 'write',
            load_level: getCurrentLoadLevel(__VU),
          },
        }
      );
    }

    const latency = Date.now() - startTime;
    requestLatency.add(latency);

    // Track recent errors and latencies
    const isError = response.status < 200 || response.status >= 500;
    lastMinuteErrors.push(isError ? 1 : 0);
    lastMinuteLatencies.push(latency);

    // Keep only last 60 data points (approximately last minute)
    if (lastMinuteErrors.length > 60) {
      lastMinuteErrors.shift();
      lastMinuteLatencies.shift();
    }

    // Check if we've hit breaking point
    if (lastMinuteErrors.length === 60) {
      const recentErrorRate = lastMinuteErrors.reduce((a, b) => a + b, 0) / 60;
      const avgLatency = lastMinuteLatencies.reduce((a, b) => a + b, 0) / 60;

      if (recentErrorRate > data.errorThreshold || avgLatency > data.latencyThreshold) {
        breakingPoint.add(__VU);
        console.warn(`Breaking point detected at ${__VU} VUs (Error rate: ${(recentErrorRate * 100).toFixed(2)}%, Avg latency: ${avgLatency.toFixed(0)}ms)`);
      }
    }

    // Validate response
    const success = check(response, {
      'status is valid': (r) => r.status === 200 || r.status === 204 || r.status === 404,
      'no timeout': (r) => r.status !== 0,
      'no server error': (r) => r.status < 500,
    });

    if (success) {
      successfulRequests.add(1);
    } else {
      failedRequests.add(1);
      errorRate.add(1);

      if (response.status >= 500) {
        console.error(`Server error at ${__VU} VUs: ${response.status}`);
      } else if (response.status === 0) {
        console.error(`Timeout at ${__VU} VUs`);
      }
    }
  } catch (e) {
    failedRequests.add(1);
    errorRate.add(1);
    console.error(`Exception at ${__VU} VUs: ${e.message}`);
  }

  // Minimal think time (stress test)
  sleep(0.1);
}

function getCurrentLoadLevel(vus) {
  if (vus < 100) return 'low';
  if (vus < 300) return 'moderate';
  if (vus < 600) return 'high';
  if (vus < 900) return 'very_high';
  return 'extreme';
}

// Teardown and analysis
export function teardown(data) {
  if (!data) return;

  console.log('\n=== Stress Test Complete ===');
  console.log(`Total successful requests: ${successfulRequests.count}`);
  console.log(`Total failed requests: ${failedRequests.count}`);

  const totalRequests = successfulRequests.count + failedRequests.count;
  const overallErrorRate = totalRequests > 0 ? (failedRequests.count / totalRequests) * 100 : 0;

  console.log(`Overall error rate: ${overallErrorRate.toFixed(2)}%`);

  // Check Vault health after stress test
  const healthCheck = http.get(`${data.vaultAddr}/v1/sys/health`);
  const recovered = check(healthCheck, {
    'Vault recovered after stress': (r) => r.status === 200,
  });

  if (recovered) {
    console.log('✓ Vault recovered successfully after stress test');
  } else {
    console.error('✗ Vault did not recover - manual intervention may be required');
  }

  console.log('============================\n');
}
