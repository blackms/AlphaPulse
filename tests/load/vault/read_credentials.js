/**
 * Vault Load Test - Credential Read Performance
 *
 * Measures throughput and latency for reading credentials from Vault.
 * Simulates AlphaPulse multi-tenant credential retrieval patterns.
 *
 * Usage:
 *   k6 run tests/load/vault/read_credentials.js
 *   k6 run --vus 100 --duration 5m tests/load/vault/read_credentials.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const readLatency = new Trend('read_latency');
const successfulReads = new Counter('successful_reads');
const failedReads = new Counter('failed_reads');

// Test configuration
export const options = {
  // Load test stages
  stages: [
    { duration: '30s', target: 10 },   // Warm-up: 10 VUs
    { duration: '1m', target: 50 },    // Ramp-up: 50 VUs
    { duration: '3m', target: 100 },   // Peak load: 100 VUs
    { duration: '1m', target: 50 },    // Ramp-down: 50 VUs
    { duration: '30s', target: 0 },    // Cool-down: 0 VUs
  ],

  // Performance thresholds
  thresholds: {
    'errors': ['rate<0.01'],              // Error rate < 1%
    'read_latency': ['p(95)<100'],        // 95% of requests < 100ms
    'http_req_duration': ['p(99)<200'],   // 99% of requests < 200ms
    'http_req_failed': ['rate<0.05'],     // HTTP failures < 5%
  },

  // Tags for filtering
  tags: {
    test_type: 'vault_read',
    environment: 'dev',
  },
};

// Test setup
export function setup() {
  const vaultAddr = __ENV.VAULT_ADDR || 'http://127.0.0.1:8200';
  const vaultToken = __ENV.VAULT_TOKEN || 'root';

  console.log(`Vault Address: ${vaultAddr}`);
  console.log(`Starting Vault read load test...`);

  // Verify Vault is accessible
  const healthCheck = http.get(`${vaultAddr}/v1/sys/health`);
  check(healthCheck, {
    'Vault is healthy': (r) => r.status === 200,
  });

  // Create test credentials if they don't exist
  const testTenantId = '00000000-0000-0000-0000-000000000001';
  const testData = {
    data: {
      api_key: 'test_key_12345',
      api_secret: 'test_secret_67890',
      testnet: false,
    },
  };

  const createResponse = http.post(
    `${vaultAddr}/v1/secret/data/tenants/${testTenantId}/binance/trading`,
    JSON.stringify(testData),
    {
      headers: {
        'X-Vault-Token': vaultToken,
        'Content-Type': 'application/json',
      },
    }
  );

  console.log(`Test credential created: ${createResponse.status}`);

  return {
    vaultAddr,
    vaultToken,
    tenants: [
      '00000000-0000-0000-0000-000000000001',
      '00000000-0000-0000-0000-000000000002',
      '00000000-0000-0000-0000-000000000003',
      '00000000-0000-0000-0000-000000000004',
      '00000000-0000-0000-0000-000000000005',
    ],
    exchanges: ['binance', 'coinbase', 'kraken'],
    credentialTypes: ['trading', 'readonly', 'admin'],
  };
}

// Main test scenario
export default function (data) {
  const { vaultAddr, vaultToken, tenants, exchanges, credentialTypes } = data;

  // Randomly select tenant, exchange, and credential type
  const tenant = tenants[Math.floor(Math.random() * tenants.length)];
  const exchange = exchanges[Math.floor(Math.random() * exchanges.length)];
  const credType = credentialTypes[Math.floor(Math.random() * credentialTypes.length)];

  const path = `secret/data/tenants/${tenant}/${exchange}/${credType}`;
  const url = `${vaultAddr}/v1/${path}`;

  // Record start time
  const startTime = Date.now();

  // Perform credential read
  const response = http.get(url, {
    headers: {
      'X-Vault-Token': vaultToken,
    },
    tags: {
      name: 'read_credential',
      tenant: tenant,
      exchange: exchange,
    },
  });

  // Record latency
  const latency = Date.now() - startTime;
  readLatency.add(latency);

  // Check response
  const success = check(response, {
    'status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    'response time < 200ms': (r) => r.timings.duration < 200,
    'valid JSON response': (r) => {
      try {
        return r.json() !== null;
      } catch (e) {
        return false;
      }
    },
  });

  if (success) {
    successfulReads.add(1);
  } else {
    failedReads.add(1);
    errorRate.add(1);
    console.error(`Read failed: ${response.status} - ${response.body}`);
  }

  // Think time: simulate realistic user behavior
  sleep(Math.random() * 2); // 0-2 seconds between requests
}

// Teardown
export function teardown(data) {
  console.log('\n=== Load Test Complete ===');
  console.log(`Total successful reads: ${successfulReads.count}`);
  console.log(`Total failed reads: ${failedReads.count}`);
  console.log('============================\n');
}
