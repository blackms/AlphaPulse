# C4 Level 1: System Context Diagram

**Date**: 2025-10-21
**Sprint**: 3 (Design & Alignment)
**Author**: Tech Lead
**Related**: [HLD-MULTI-TENANT-SAAS.md](../HLD-MULTI-TENANT-SAAS.md), Issue #180

---

## Purpose

This System Context diagram shows AlphaPulse SaaS at the highest level, showing:
- The system boundary
- Users/actors interacting with the system
- External systems the platform depends on

---

## Diagram

```plantuml
@startuml C4_Level1_System_Context
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Context.puml

LAYOUT_WITH_LEGEND()

title System Context Diagram for AlphaPulse Multi-Tenant SaaS

Person(tenant_trader, "Tenant Trader", "Individual trader or fund manager using AlphaPulse for automated trading")
Person(tenant_admin, "Tenant Admin", "Tenant administrator managing credentials, billing, and team members")
Person(platform_admin, "Platform Admin", "AlphaPulse operations team managing infrastructure and tenants")

System(alphapulse, "AlphaPulse SaaS", "Multi-tenant AI-driven hedge fund platform with automated trading, risk management, and portfolio optimization")

System_Ext(exchanges, "Cryptocurrency Exchanges", "Binance, Coinbase, Kraken - Market data and trade execution")
System_Ext(market_data, "Market Data Providers", "CoinGecko, CoinMarketCap - Price feeds, historical data")
System_Ext(stripe, "Stripe", "Payment processing, subscription management, invoicing")
System_Ext(email, "Email Service", "SendGrid/AWS SES - Transactional emails, alerts, notifications")
System_Ext(monitoring, "Monitoring & Observability", "Datadog/Prometheus - Metrics, logs, alerts, dashboards")

Rel(tenant_trader, alphapulse, "Views portfolio, executes trades, monitors performance", "HTTPS/WebSocket")
Rel(tenant_admin, alphapulse, "Manages credentials, billing, team members", "HTTPS")
Rel(platform_admin, alphapulse, "Manages tenants, infrastructure, billing", "HTTPS/SSH")

Rel(alphapulse, exchanges, "Fetches market data, executes trades", "REST API/WebSocket")
Rel(alphapulse, market_data, "Fetches price data, sentiment data", "REST API")
Rel(alphapulse, stripe, "Creates subscriptions, reports usage, processes payments", "REST API/Webhooks")
Rel(alphapulse, email, "Sends alerts, trade confirmations, billing emails", "SMTP/API")
Rel(alphapulse, monitoring, "Sends metrics, logs, traces", "StatsD/OpenTelemetry")

Rel(stripe, alphapulse, "Payment status updates (success, failed, refund)", "Webhooks")
Rel(exchanges, alphapulse, "Order fill updates, balance updates", "WebSocket")

@enduml
```

---

## Actors

### Tenant Trader
**Role**: End user of the AlphaPulse platform
**Responsibilities**:
- Configure trading strategies and AI agent parameters
- Monitor portfolio performance and risk metrics
- Review trade executions and agent signals
- Receive alerts for significant events (drawdown, trade execution)

**Access Level**: Tenant-scoped (can only see own tenant's data)

---

### Tenant Admin
**Role**: Administrator within a tenant organization
**Responsibilities**:
- Manage exchange credentials (add, rotate, delete)
- Manage billing and subscription (upgrade/downgrade tier)
- Invite and manage team members (multi-user support, future)
- View usage metrics and billing history

**Access Level**: Tenant-scoped with admin privileges

---

### Platform Admin
**Role**: AlphaPulse operations and support team
**Responsibilities**:
- Provision new tenants (onboarding)
- Manage infrastructure (Kubernetes, databases, Vault)
- Troubleshoot issues (support tickets)
- Monitor platform health and performance
- Manage billing disputes and refunds

**Access Level**: Global (can view all tenants, read-only except for support actions)

---

## External Systems

### Cryptocurrency Exchanges
**Examples**: Binance, Coinbase, Kraken
**Integration Type**: REST API + WebSocket
**Purpose**:
- Fetch real-time market data (OHLCV, order books, trades)
- Execute trades (market orders, limit orders)
- Query balances and positions
- Subscribe to order fill updates

**Dependencies**:
- Exchange API availability (99.9% uptime SLA from exchanges)
- Rate limits (1200 req/min for Binance, varies by exchange)
- Credential management (API keys stored in Vault)

**Risk**: Exchange downtime impacts trading (mitigated by multi-exchange support)

---

### Market Data Providers
**Examples**: CoinGecko, CoinMarketCap, alternative data providers
**Integration Type**: REST API
**Purpose**:
- Price data (spot prices, historical OHLCV)
- Sentiment data (social media mentions, sentiment scores)
- Market metadata (coin listings, market cap rankings)

**Dependencies**:
- API availability (99% uptime)
- Rate limits (50 req/min for CoinGecko free tier, 500 req/min for Pro)

**Risk**: Data provider outage (mitigated by caching + fallback to exchange APIs)

---

### Stripe
**Integration Type**: REST API + Webhooks
**Purpose**:
- Create and manage subscriptions (Starter, Pro, Enterprise tiers)
- Process payments (credit card, ACH)
- Generate invoices
- Report usage-based charges (API calls, trades)
- Handle refunds and disputes

**Webhooks Received**:
- `invoice.payment_succeeded` → Activate subscription
- `invoice.payment_failed` → Retry payment, suspend if exhausted
- `customer.subscription.updated` → Update tenant tier
- `customer.subscription.deleted` → Cancel tenant subscription

**Dependencies**:
- Stripe API availability (99.99% uptime SLA)
- PCI compliance (handled by Stripe)

**Risk**: Stripe downtime blocks new signups (mitigated by retry logic + manual override)

---

### Email Service
**Examples**: SendGrid, AWS SES
**Integration Type**: SMTP or REST API
**Purpose**:
- Transactional emails (signup confirmation, password reset)
- Trade execution alerts (trade executed, order filled)
- Risk alerts (drawdown threshold exceeded, position limit reached)
- Billing notifications (invoice available, payment failed)

**Dependencies**:
- Email service availability (99.9% uptime)
- Deliverability (SPF, DKIM, DMARC configured)

**Risk**: Email delivery failures (non-critical, logs retained for troubleshooting)

---

### Monitoring & Observability
**Examples**: Datadog, Prometheus + Grafana
**Integration Type**: StatsD, OpenTelemetry, API
**Purpose**:
- Metrics (request rate, latency, error rate, cache hit rate)
- Logs (application logs, audit logs, error logs)
- Traces (distributed tracing for API requests)
- Alerts (PagerDuty integration for on-call)
- Dashboards (platform health, tenant usage, billing metrics)

**Dependencies**:
- Monitoring service availability (99.9% uptime)
- Data retention (30 days for logs, 1 year for metrics)

**Risk**: Monitoring outage (non-critical, platform continues running)

---

## Key Interactions

### 1. Tenant Trader → AlphaPulse
**Protocol**: HTTPS (REST API) + WebSocket (real-time updates)
**Authentication**: JWT (tenant_id embedded in claims)
**Use Cases**:
- View portfolio dashboard (GET /portfolio)
- Configure AI agents (POST /agents/config)
- Execute manual trade (POST /trades)
- Subscribe to real-time updates (WebSocket /ws)

**Security**:
- TLS 1.3 encryption
- JWT token expiration (1 hour)
- Rate limiting (100 req/min per tenant)

---

### 2. AlphaPulse → Exchanges
**Protocol**: REST API + WebSocket
**Authentication**: API key + secret (stored in Vault)
**Use Cases**:
- Fetch market data (GET /api/v3/ticker/price)
- Execute trade (POST /api/v3/order)
- Subscribe to order updates (WebSocket /ws/order)

**Security**:
- API keys rotated every 90 days
- Permissions: Trading + read-only (no withdrawals)
- IP whitelisting (if supported by exchange)

**Rate Limits**:
- Binance: 1200 req/min (20 req/sec)
- Coinbase: 10 req/sec
- Kraken: 15 req/sec

**Caching**:
- Market data cached for 1 minute (reduces API calls by 90%)

---

### 3. AlphaPulse → Stripe
**Protocol**: REST API + Webhooks (inbound)
**Authentication**: Stripe API key (secret key stored in Vault)
**Use Cases**:
- Create subscription (POST /v1/subscriptions)
- Report usage (POST /v1/subscription_items/{id}/usage_records)
- Retrieve invoice (GET /v1/invoices/{id})

**Webhooks**:
- Stripe sends webhooks to AlphaPulse (POST /webhooks/stripe)
- Signature verification (HMAC SHA256)
- Idempotent processing (webhook_id stored to prevent duplicates)

**Security**:
- Webhook signature verification
- HTTPS only
- Retry logic (Stripe retries failed webhooks 3 times)

---

### 4. AlphaPulse → Monitoring
**Protocol**: StatsD (UDP) + HTTPS (API)
**Authentication**: API key
**Use Cases**:
- Send metrics (statsd.increment('api.requests'))
- Send logs (POST /v1/logs)
- Send traces (OpenTelemetry exporter)

**Metrics Examples**:
- `api.requests.count` (total requests)
- `api.requests.latency.p99` (99th percentile latency)
- `cache.hit_rate` (Redis cache hit rate)
- `vault.requests.latency` (Vault latency)
- `tenant.usage.api_calls` (per-tenant API usage)

---

## Security Boundaries

### 1. Internet → AlphaPulse
- **HTTPS only** (TLS 1.3)
- **JWT authentication** (tenant_id in claims)
- **Rate limiting** (100 req/min per tenant, 1000 req/min global)
- **WAF** (Web Application Firewall) - Cloudflare or AWS WAF

### 2. AlphaPulse → External Systems
- **API keys stored in Vault** (not in environment variables)
- **Credential rotation** (every 90 days)
- **TLS certificate validation** (verify SSL certificates)
- **Timeout limits** (5 seconds for API calls, 30 seconds for WebSocket)

### 3. Tenant Data Isolation
- **PostgreSQL RLS** (database-level isolation)
- **Redis namespaces** (`tenant:{id}:*`)
- **Vault policies** (tenant can only access own credentials)

---

## Scalability Considerations

### Horizontal Scaling
- **API**: Stateless containers (scale to 50+ pods)
- **Agents**: Background workers (scale to 20+ workers)
- **Database**: Read replicas (scale reads)
- **Cache**: Redis Cluster (scale to 6+ nodes)

### Vertical Scaling
- **Database**: Upgrade to db.r5.2xlarge (8 vCPU, 64GB RAM)
- **Vault**: Upgrade to t3.large (2 vCPU, 8GB RAM)

### Load Balancing
- **NGINX/Traefik** in front of API containers
- **Round-robin** load balancing
- **Health checks** (liveness + readiness probes)

---

## Compliance & Audit

### SOC2 Requirements
- **Audit logs**: All API requests logged (tenant_id, user_id, endpoint, timestamp)
- **Credential access**: Vault audit logs (who accessed what credentials)
- **Data retention**: Logs retained for 1 year

### GDPR Requirements
- **Data export**: Tenants can export their data (GET /export)
- **Data deletion**: Tenants can request data deletion (DELETE /account)
- **Privacy policy**: Displayed during signup

---

## Future Enhancements (Out of Scope for Phase 1)

1. **SSO Integration** (SAML/OAuth for Enterprise)
2. **Multi-region Deployment** (EU data residency)
3. **Mobile App** (React Native for iOS/Android)
4. **White-label Branding** (custom domain, logo, colors)
5. **Marketplace** (buy/sell custom trading strategies)

---

## References

- [HLD Section 2.1: Architecture Views](../HLD-MULTI-TENANT-SAAS.md#21-architecture-views)
- [ADR-001: Data Isolation Strategy](../adr/001-multi-tenant-data-isolation-strategy.md)
- [ADR-003: Credential Management](../adr/003-credential-management-multi-tenant.md)
- [ADR-005: Billing System](../adr/005-billing-system-selection.md)
- [C4 Model Documentation](https://c4model.com/)

---

**Diagram Status**: Draft (pending review)
**Review Date**: Sprint 3, Week 1
**Reviewers**: Tech Lead, Security Lead, CTO

---

**END OF DOCUMENT**
