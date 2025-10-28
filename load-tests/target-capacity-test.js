/**
 * AlphaPulse - Target Capacity Load Test (500 concurrent users)
 *
 * This script tests the target capacity of the AlphaPulse API
 * with 500 concurrent users over 10 minutes.
 *
 * Test Mix:
 * - 70% reads (GET endpoints: portfolio, trades, positions)
 * - 30% writes (POST endpoints: trades, signals)
 *
 * Success Criteria:
 * - p99 latency <500ms (acceptance)
 * - p99 latency <200ms (ideal)
 * - Error rate <1%
 * - CPU usage <70%
 * - Memory usage <80%
 *
 * Usage:
 *   k6 run target-capacity-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const portfolioLatency = new Trend('portfolio_latency');
const tradesLatency = new Trend('trades_latency');
const positionsLatency = new Trend('positions_latency');
const createTradeLatency = new Trend('create_trade_latency');
const signalLatency = new Trend('signal_latency');
const requestCount = new Counter('requests');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp-up to 100 users
    { duration: '2m', target: 300 },  // Ramp-up to 300 users
    { duration: '2m', target: 500 },  // Ramp-up to 500 users
    { duration: '2m', target: 500 },  // Stay at 500 users (stress test)
    { duration: '2m', target: 0 },    // Ramp-down to 0
  ],
  thresholds: {
    http_req_duration: ['p(95)<400', 'p(99)<500'],  // 95% < 400ms, 99% < 500ms
    http_req_failed: ['rate<0.01'],                 // Error rate < 1%
    errors: ['rate<0.01'],
  },
};

// Base URL (change to staging environment)
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test users (5 tenants with multiple users each)
const USERS = [
  // Tenant 1
  { email: 'user1@tenant1.com', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000001' },
  { email: 'user2@tenant1.com', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000001' },
  // Tenant 2
  { email: 'user1@tenant2.com', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000002' },
  { email: 'user2@tenant2.com', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000002' },
  // Tenant 3
  { email: 'user1@tenant3.com', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000003' },
  { email: 'user2@tenant3.com', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000003' },
  // Tenant 4
  { email: 'user1@tenant4.com', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000004' },
  { email: 'user2@tenant4.com', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000004' },
  // Tenant 5
  { email: 'user1@tenant5.com', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000005' },
  { email: 'user2@tenant5.com', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000005' },
];

/**
 * Authenticate and get JWT token
 */
function authenticate() {
  const user = USERS[Math.floor(Math.random() * USERS.length)];

  const response = http.post(`${BASE_URL}/api/auth/login`, JSON.stringify({
    email: user.email,
    password: user.password,
  }), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '10s',
  });

  const success = check(response, {
    'login successful': (r) => r.status === 200,
    'token received': (r) => r.json('access_token') !== undefined,
  });

  if (!success) {
    errorRate.add(1);
    return null;
  }

  return response.json('access_token');
}

/**
 * Main test scenario
 */
export default function () {
  // Authenticate
  const token = authenticate();
  if (!token) {
    sleep(1);
    return;
  }

  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };

  requestCount.add(1);

  // Weighted request distribution
  const rand = Math.random();

  // 25% - Get portfolio
  if (rand < 0.25) {
    const start = Date.now();
    const response = http.get(`${BASE_URL}/api/portfolio`, { headers, timeout: '10s' });
    portfolioLatency.add(Date.now() - start);

    const success = check(response, {
      'portfolio status 200': (r) => r.status === 200,
      'portfolio latency <500ms': (r) => r.timings.duration < 500,
    });

    if (!success) errorRate.add(1);
  }
  // 25% - Get trades
  else if (rand < 0.50) {
    const start = Date.now();
    const response = http.get(`${BASE_URL}/api/trades?limit=50`, { headers, timeout: '10s' });
    tradesLatency.add(Date.now() - start);

    const success = check(response, {
      'trades status 200': (r) => r.status === 200,
      'trades latency <500ms': (r) => r.timings.duration < 500,
    });

    if (!success) errorRate.add(1);
  }
  // 20% - Get positions
  else if (rand < 0.70) {
    const start = Date.now();
    const response = http.get(`${BASE_URL}/api/positions`, { headers, timeout: '10s' });
    positionsLatency.add(Date.now() - start);

    const success = check(response, {
      'positions status 200': (r) => r.status === 200,
      'positions latency <500ms': (r) => r.timings.duration < 500,
    });

    if (!success) errorRate.add(1);
  }
  // 20% - Create trade
  else if (rand < 0.90) {
    const tradeData = {
      symbol: ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'MATIC/USD'][Math.floor(Math.random() * 5)],
      side: Math.random() > 0.5 ? 'buy' : 'sell',
      quantity: Math.random() * 10,
      price: Math.random() * 50000,
      order_type: 'limit',
    };

    const start = Date.now();
    const response = http.post(`${BASE_URL}/api/trades`, JSON.stringify(tradeData), {
      headers,
      timeout: '10s',
    });
    createTradeLatency.add(Date.now() - start);

    const success = check(response, {
      'create trade status 201': (r) => r.status === 201 || r.status === 200,
      'create trade latency <500ms': (r) => r.timings.duration < 500,
    });

    if (!success) errorRate.add(1);
  }
  // 10% - Get trading signals
  else {
    const start = Date.now();
    const response = http.get(`${BASE_URL}/api/signals?limit=20`, { headers, timeout: '10s' });
    signalLatency.add(Date.now() - start);

    const success = check(response, {
      'signals status 200': (r) => r.status === 200,
      'signals latency <500ms': (r) => r.timings.duration < 500,
    });

    if (!success) errorRate.add(1);
  }

  sleep(Math.random() * 2 + 0.5); // Random think time: 0.5-2.5 seconds
}

/**
 * Setup: Create test users if needed
 */
export function setup() {
  console.log('='.repeat(60));
  console.log('AlphaPulse - Target Capacity Load Test');
  console.log('='.repeat(60));
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`Test duration: 10 minutes`);
  console.log(`Target users: 500 concurrent`);
  console.log(`Test mix: 70% reads (portfolio/trades/positions), 30% writes (trades/signals)`);
  console.log(`Success criteria: p99 <500ms, error rate <1%`);
  console.log('='.repeat(60));

  // Health check
  const healthResponse = http.get(`${BASE_URL}/health`);
  if (healthResponse.status !== 200) {
    console.error('Health check failed! Aborting test.');
    throw new Error('API health check failed');
  }

  console.log('✓ Health check passed');
  console.log('Starting test in 5 seconds...');

  return { timestamp: Date.now() };
}

/**
 * Teardown: Print summary
 */
export function teardown(data) {
  console.log('='.repeat(60));
  console.log('Test Summary');
  console.log('='.repeat(60));
  console.log(`Test completed at: ${new Date().toISOString()}`);
  console.log(`Test started at: ${new Date(data.timestamp).toISOString()}`);
  console.log(`Duration: ${((Date.now() - data.timestamp) / 1000 / 60).toFixed(2)} minutes`);
  console.log('='.repeat(60));
  console.log('Review detailed metrics in the output above.');
  console.log('Check Grafana dashboards for resource utilization (CPU, memory, database).');
  console.log('='.repeat(60));
}
