/**
 * AlphaPulse - Baseline Load Test (100 concurrent users)
 *
 * This script tests the baseline performance of the AlphaPulse API
 * with 100 concurrent users over 10 minutes.
 *
 * Test Mix:
 * - 70% reads (GET endpoints: portfolio, trades, positions)
 * - 30% writes (POST endpoints: trades, signals)
 *
 * Success Criteria:
 * - p99 latency <500ms (acceptance)
 * - p99 latency <200ms (ideal)
 * - Error rate <1%
 *
 * Usage:
 *   k6 run baseline-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const portfolioLatency = new Trend('portfolio_latency');
const tradesLatency = new Trend('trades_latency');
const createTradeLatency = new Trend('create_trade_latency');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp-up to 100 users
    { duration: '6m', target: 100 },  // Stay at 100 users
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

// Test users (rotate between them)
const USERS = [
  { email: 'user1@alphapulse.dev', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000001' },
  { email: 'user2@alphapulse.dev', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000002' },
  { email: 'user3@alphapulse.dev', password: 'test123', tenant_id: '00000000-0000-0000-0000-000000000003' },
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

  // 70% reads: Get portfolio, trades, positions
  const readWeight = Math.random();
  if (readWeight < 0.7) {
    // Get portfolio (35%)
    if (readWeight < 0.35) {
      const start = Date.now();
      const response = http.get(`${BASE_URL}/api/portfolio`, { headers });
      portfolioLatency.add(Date.now() - start);

      const success = check(response, {
        'portfolio status 200': (r) => r.status === 200,
        'portfolio has data': (r) => r.json('positions') !== undefined,
      });

      if (!success) errorRate.add(1);
    }
    // Get trades (35%)
    else {
      const start = Date.now();
      const response = http.get(`${BASE_URL}/api/trades?limit=50`, { headers });
      tradesLatency.add(Date.now() - start);

      const success = check(response, {
        'trades status 200': (r) => r.status === 200,
        'trades is array': (r) => Array.isArray(r.json()),
      });

      if (!success) errorRate.add(1);
    }
  }
  // 30% writes: Create trade signal
  else {
    const tradeData = {
      symbol: ['BTC/USD', 'ETH/USD', 'SOL/USD'][Math.floor(Math.random() * 3)],
      side: Math.random() > 0.5 ? 'buy' : 'sell',
      quantity: Math.random() * 10,
      price: Math.random() * 50000,
      order_type: 'limit',
    };

    const start = Date.now();
    const response = http.post(`${BASE_URL}/api/trades`, JSON.stringify(tradeData), { headers });
    createTradeLatency.add(Date.now() - start);

    const success = check(response, {
      'create trade status 201': (r) => r.status === 201 || r.status === 200,
      'trade has id': (r) => r.json('id') !== undefined || r.json('trade_id') !== undefined,
    });

    if (!success) errorRate.add(1);
  }

  sleep(1); // Think time: 1 second between requests
}

/**
 * Setup: Create test users if needed
 */
export function setup() {
  console.log('Setting up load test...');
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`Test duration: 10 minutes`);
  console.log(`Target users: 100`);
  console.log(`Test mix: 70% reads, 30% writes`);

  return { timestamp: Date.now() };
}

/**
 * Teardown: Print summary
 */
export function teardown(data) {
  console.log(`Test completed at: ${new Date().toISOString()}`);
  console.log(`Test started at: ${new Date(data.timestamp).toISOString()}`);
  console.log(`Duration: ${(Date.now() - data.timestamp) / 1000 / 60} minutes`);
}
