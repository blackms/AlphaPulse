/**
 * Vault Load Test - Multi-Tenant Concurrent Access
 *
 * Simulates multiple tenants accessing Vault simultaneously to measure:
 * - Tenant isolation performance
 * - Cross-tenant impact
 * - Scalability with increasing tenant count
 *
 * Usage:
 *   k6 run tests/load/vault/multi_tenant_concurrent.js
 *   k6 run --vus 200 --duration 10m tests/load/vault/multi_tenant_concurrent.js
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomString } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics per tenant
const errorRateByTenant = new Rate('errors_by_tenant');
const latencyByTenant = new Trend('latency_by_tenant');
const requestsPerTenant = new Counter('requests_per_tenant');
const isolationViolations = new Counter('isolation_violations');

// Test configuration
export const options = {
  // Simulate 100 concurrent tenants
  stages: [
    { duration: '1m', target: 50 },    // Ramp-up to 50 tenants
    { duration: '2m', target: 100 },   // Ramp-up to 100 tenants
    { duration: '5m', target: 100 },   // Sustain 100 tenants
    { duration: '1m', target: 50 },    // Ramp-down to 50 tenants
    { duration: '1m', target: 0 },     // Cool-down
  ],

  // Strict thresholds for multi-tenant
  thresholds: {
    'errors_by_tenant': ['rate<0.005'],        // Error rate < 0.5%
    'latency_by_tenant': ['p(95)<150'],        // P95 < 150ms
    'http_req_duration': ['p(99)<300'],        // P99 < 300ms
    'isolation_violations': ['count==0'],      // ZERO isolation violations
  },

  tags: {
    test_type: 'multi_tenant',
    environment: 'dev',
  },
};

// Generate unique tenant IDs (simulate 100 tenants)
function generateTenantPool(count) {
  const tenants = [];
  for (let i = 1; i <= count; i++) {
    tenants.push(`tenant-${String(i).padStart(4, '0')}`);
  }
  return tenants;
}

// Test setup
export function setup() {
  const vaultAddr = __ENV.VAULT_ADDR || 'http://127.0.0.1:8200';
  const vaultToken = __ENV.VAULT_TOKEN || 'root';
  const tenantCount = parseInt(__ENV.TENANT_COUNT || '100');

  console.log(`Vault Address: ${vaultAddr}`);
  console.log(`Tenant Count: ${tenantCount}`);
  console.log(`Starting multi-tenant concurrent access test...`);

  // Verify Vault is accessible
  const healthCheck = http.get(`${vaultAddr}/v1/sys/health`);
  check(healthCheck, {
    'Vault is healthy': (r) => r.status === 200,
  });

  const tenants = generateTenantPool(tenantCount);
  const exchanges = ['binance', 'coinbase', 'kraken', 'bitfinex', 'okx'];
  const credentialTypes = ['trading', 'readonly', 'admin'];

  // Pre-populate some credentials for each tenant
  console.log('Pre-populating credentials...');
  let created = 0;
  for (let i = 0; i < Math.min(tenantCount, 20); i++) {
    const tenant = tenants[i];
    const exchange = exchanges[i % exchanges.length];

    const testData = {
      data: {
        api_key: `key_${randomString(32)}`,
        api_secret: `secret_${randomString(64)}`,
        testnet: false,
        tenant_id: tenant,
      },
    };

    http.post(
      `${vaultAddr}/v1/secret/data/tenants/${tenant}/${exchange}/trading`,
      JSON.stringify(testData),
      {
        headers: {
          'X-Vault-Token': vaultToken,
          'Content-Type': 'application/json',
        },
      }
    );
    created++;
  }
  console.log(`Pre-populated ${created} credentials`);

  return {
    vaultAddr,
    vaultToken,
    tenants,
    exchanges,
    credentialTypes,
  };
}

// Main test scenario
export default function (data) {
  const { vaultAddr, vaultToken, tenants, exchanges, credentialTypes } = data;

  // Each VU represents a tenant - assign tenant based on VU ID
  const vuId = __VU - 1;
  const tenant = tenants[vuId % tenants.length];
  const exchange = exchanges[Math.floor(Math.random() * exchanges.length)];
  const credType = credentialTypes[Math.floor(Math.random() * credentialTypes.length)];

  // 70% reads, 30% writes (realistic pattern)
  const operation = Math.random() < 0.7 ? 'read' : 'write';

  group(`Tenant ${tenant} - ${operation}`, function () {
    const path = `secret/data/tenants/${tenant}/${exchange}/${credType}`;
    const url = `${vaultAddr}/v1/${path}`;
    const startTime = Date.now();

    let response;

    if (operation === 'read') {
      // Read operation
      response = http.get(url, {
        headers: {
          'X-Vault-Token': vaultToken,
        },
        tags: {
          tenant: tenant,
          operation: 'read',
          exchange: exchange,
        },
      });
    } else {
      // Write operation
      const credentialData = {
        data: {
          api_key: `key_${randomString(32)}`,
          api_secret: `secret_${randomString(64)}`,
          tenant_id: tenant,
          updated_at: new Date().toISOString(),
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
          tags: {
            tenant: tenant,
            operation: 'write',
            exchange: exchange,
          },
        }
      );
    }

    // Record metrics
    const latency = Date.now() - startTime;
    latencyByTenant.add(latency, { tenant: tenant });
    requestsPerTenant.add(1, { tenant: tenant });

    // Check response
    const success = check(response, {
      'status is valid': (r) => r.status === 200 || r.status === 204 || r.status === 404,
      'response time acceptable': (r) => r.timings.duration < 300,
    });

    if (!success) {
      errorRateByTenant.add(1, { tenant: tenant });
      console.error(`[${tenant}] ${operation} failed: ${response.status}`);
    }

    // Verify tenant isolation (critical security check)
    if (response.status === 200 && operation === 'read') {
      try {
        const body = response.json();
        if (body.data && body.data.data) {
          const returnedTenantId = body.data.data.tenant_id;

          // Check if returned tenant_id matches requested tenant
          if (returnedTenantId && returnedTenantId !== tenant) {
            isolationViolations.add(1);
            console.error(`ISOLATION VIOLATION: Tenant ${tenant} received data for ${returnedTenantId}!`);
          }
        }
      } catch (e) {
        // Ignore parse errors
      }
    }
  });

  // Simulate realistic tenant activity patterns
  sleep(Math.random() * 3);
}

// Teardown and analysis
export function teardown(data) {
  console.log('\n=== Multi-Tenant Concurrent Access Test Complete ===');
  console.log(`Total tenants simulated: ${data.tenants.length}`);
  console.log(`Isolation violations: ${isolationViolations.count}`);

  if (isolationViolations.count > 0) {
    console.error('WARNING: Tenant isolation violations detected!');
  }

  console.log('=====================================================\n');
}
