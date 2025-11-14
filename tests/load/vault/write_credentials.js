/**
 * Vault Load Test - Credential Write/Rotation Performance
 *
 * Measures throughput and latency for writing/updating credentials in Vault.
 * Simulates AlphaPulse credential rotation operations.
 *
 * Usage:
 *   k6 run tests/load/vault/write_credentials.js
 *   k6 run --vus 50 --duration 5m tests/load/vault/write_credentials.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomString } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');
const writeLatency = new Trend('write_latency');
const successfulWrites = new Counter('successful_writes');
const failedWrites = new Counter('failed_writes');
const rotationCount = new Counter('rotation_count');

// Test configuration
export const options = {
  // Write operations are heavier, use lower VU count
  stages: [
    { duration: '30s', target: 5 },    // Warm-up: 5 VUs
    { duration: '1m', target: 20 },    // Ramp-up: 20 VUs
    { duration: '3m', target: 50 },    // Peak load: 50 VUs
    { duration: '1m', target: 20 },    // Ramp-down: 20 VUs
    { duration: '30s', target: 0 },    // Cool-down: 0 VUs
  ],

  // Performance thresholds (write operations are slower)
  thresholds: {
    'errors': ['rate<0.01'],              // Error rate < 1%
    'write_latency': ['p(95)<200'],       // 95% of writes < 200ms
    'http_req_duration': ['p(99)<500'],   // 99% of requests < 500ms
    'http_req_failed': ['rate<0.05'],     // HTTP failures < 5%
  },

  tags: {
    test_type: 'vault_write',
    environment: 'dev',
  },
};

// Test setup
export function setup() {
  const vaultAddr = __ENV.VAULT_ADDR || 'http://127.0.0.1:8200';
  const vaultToken = __ENV.VAULT_TOKEN || 'root';

  console.log(`Vault Address: ${vaultAddr}`);
  console.log(`Starting Vault write/rotation load test...`);

  // Verify Vault is accessible
  const healthCheck = http.get(`${vaultAddr}/v1/sys/health`);
  check(healthCheck, {
    'Vault is healthy': (r) => r.status === 200,
  });

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

// Generate realistic credential data
function generateCredentialData() {
  return {
    data: {
      api_key: `key_${randomString(32)}`,
      api_secret: `secret_${randomString(64)}`,
      passphrase: Math.random() > 0.5 ? `pass_${randomString(16)}` : null,
      testnet: Math.random() > 0.7,
      created_at: new Date().toISOString(),
      rotated_at: new Date().toISOString(),
    },
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

  // Generate new credential data (simulating rotation)
  const credentialData = generateCredentialData();

  // Record start time
  const startTime = Date.now();

  // Perform credential write/rotation
  const response = http.post(
    url,
    JSON.stringify(credentialData),
    {
      headers: {
        'X-Vault-Token': vaultToken,
        'Content-Type': 'application/json',
      },
      tags: {
        name: 'write_credential',
        tenant: tenant,
        exchange: exchange,
        operation: 'rotation',
      },
    }
  );

  // Record latency
  const latency = Date.now() - startTime;
  writeLatency.add(latency);

  // Check response
  const success = check(response, {
    'status is 200 or 204': (r) => r.status === 200 || r.status === 204,
    'response time < 500ms': (r) => r.timings.duration < 500,
    'valid JSON response': (r) => {
      try {
        if (r.body) {
          return r.json() !== null;
        }
        return true; // 204 No Content is valid
      } catch (e) {
        return false;
      }
    },
  });

  if (success) {
    successfulWrites.add(1);
    rotationCount.add(1);
  } else {
    failedWrites.add(1);
    errorRate.add(1);
    console.error(`Write failed: ${response.status} - ${response.body}`);
  }

  // Verify write by reading back (10% of the time)
  if (Math.random() < 0.1) {
    const verifyResponse = http.get(url, {
      headers: {
        'X-Vault-Token': vaultToken,
      },
      tags: {
        name: 'verify_write',
      },
    });

    check(verifyResponse, {
      'verification successful': (r) => r.status === 200,
      'data persisted correctly': (r) => {
        try {
          const body = r.json();
          return body.data && body.data.data && body.data.data.api_key !== undefined;
        } catch (e) {
          return false;
        }
      },
    });
  }

  // Think time: writes are less frequent than reads
  sleep(Math.random() * 5); // 0-5 seconds between writes
}

// Teardown
export function teardown(data) {
  console.log('\n=== Write/Rotation Load Test Complete ===');
  console.log(`Total successful writes: ${successfulWrites.count}`);
  console.log(`Total failed writes: ${failedWrites.count}`);
  console.log(`Total rotations: ${rotationCount.count}`);
  console.log('==========================================\n');
}
