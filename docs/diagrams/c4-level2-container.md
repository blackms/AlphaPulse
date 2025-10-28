# C4 Level 2: Container Diagram

**Date**: 2025-10-21
**Sprint**: 3 (Design & Alignment)
**Author**: Tech Lead
**Related**: [HLD-MULTI-TENANT-SAAS.md](../HLD-MULTI-TENANT-SAAS.md), Issue #180

---

## Purpose

This Container diagram shows the high-level technology choices and how containers/services communicate within the AlphaPulse SaaS platform.

**Note**: "Container" here refers to separately runnable/deployable units (web apps, databases, file systems), not Docker containers specifically.

---

## Diagram

```plantuml
@startuml C4_Level2_Container
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml

LAYOUT_WITH_LEGEND()

title Container Diagram for AlphaPulse Multi-Tenant SaaS

Person(tenant_trader, "Tenant Trader", "Trader using AlphaPulse")
Person(platform_admin, "Platform Admin", "Operations team")

System_Boundary(alphapulse, "AlphaPulse SaaS Platform") {
    Container(web_app, "Web Application", "React/TypeScript", "Provides UI for traders and admins")
    Container(api, "API Application", "Python/FastAPI", "Handles HTTP/WebSocket requests, orchestrates trading logic")
    Container(agent_workers, "Agent Workers", "Python/Celery", "Background workers for AI agents (Technical, Fundamental, Sentiment)")
    Container(risk_workers, "Risk Workers", "Python/Celery", "Background workers for risk management and portfolio optimization")
    Container(billing_service, "Billing Service", "Python/FastAPI", "Usage metering, quota enforcement, Stripe webhooks")
    Container(provisioning_service, "Provisioning Service", "Python/FastAPI", "Tenant onboarding, credential setup")

    ContainerDb(postgres, "Database", "PostgreSQL 14", "Stores trades, positions, portfolios, agent signals (tenant_id + RLS)")
    ContainerDb(redis, "Cache", "Redis Cluster 7", "Caches market data, credentials, agent signals (namespace isolation)")
    ContainerDb(vault, "Secrets Manager", "HashiCorp Vault", "Stores exchange API keys securely (HA, 3 replicas)")

    Container(nginx, "Load Balancer", "NGINX/Traefik", "Routes requests, TLS termination, rate limiting")
}

System_Ext(exchanges, "Exchanges API", "Binance, Coinbase, Kraken")
System_Ext(stripe, "Stripe API", "Payment processing")
System_Ext(email, "Email Service", "SendGrid/SES")
System_Ext(monitoring, "Monitoring", "Datadog/Prometheus")

Rel(tenant_trader, nginx, "Views portfolio, executes trades", "HTTPS/WebSocket")
Rel(platform_admin, nginx, "Manages tenants, views metrics", "HTTPS")

Rel(nginx, web_app, "Routes requests", "HTTPS")
Rel(nginx, api, "Routes API requests", "HTTPS")

Rel(web_app, api, "Makes API calls", "HTTPS/JSON")

Rel(api, postgres, "Reads/writes trades, positions", "SQL/asyncpg")
Rel(api, redis, "Reads/writes cache", "Redis protocol")
Rel(api, vault, "Reads exchange credentials", "HTTPS/Vault API")
Rel(api, agent_workers, "Enqueues agent tasks", "Celery/Redis")
Rel(api, risk_workers, "Enqueues risk tasks", "Celery/Redis")
Rel(api, exchanges, "Fetches market data, executes trades", "HTTPS/WebSocket")
Rel(api, stripe, "Creates subscriptions, reports usage", "HTTPS")
Rel(api, email, "Sends alerts, notifications", "SMTP/API")
Rel(api, monitoring, "Sends metrics, logs, traces", "StatsD/OTLP")

Rel(agent_workers, postgres, "Writes agent signals", "SQL/asyncpg")
Rel(agent_workers, redis, "Caches intermediate results", "Redis protocol")
Rel(agent_workers, vault, "Reads exchange credentials", "HTTPS")
Rel(agent_workers, exchanges, "Fetches market data", "HTTPS")

Rel(risk_workers, postgres, "Writes risk metrics", "SQL/asyncpg")
Rel(risk_workers, redis, "Caches portfolio state", "Redis protocol")

Rel(billing_service, postgres, "Reads usage data", "SQL/asyncpg")
Rel(billing_service, stripe, "Reports usage, handles webhooks", "HTTPS")

Rel(provisioning_service, postgres, "Creates tenant records", "SQL")
Rel(provisioning_service, vault, "Sets up credential paths", "HTTPS")
Rel(provisioning_service, redis, "Initializes cache namespaces", "Redis protocol")
Rel(provisioning_service, stripe, "Creates subscriptions", "HTTPS")

Rel(stripe, billing_service, "Payment webhooks", "HTTPS")
Rel(exchanges, api, "Order updates, balance updates", "WebSocket")

@enduml
```

---

## Containers

### 1. Web Application (React/TypeScript)
**Technology**: React 18, TypeScript, React Query, WebSocket client
**Purpose**: Single-page application providing UI for traders and admins

**Key Features**:
- Portfolio dashboard (real-time positions, P&L)
- Trading interface (execute manual trades)
- Agent configuration (adjust agent parameters)
- Billing dashboard (usage metrics, invoices)
- Admin portal (tenant management, infrastructure monitoring)

**Deployment**:
- Static files served by NGINX
- Hosted on CDN (CloudFlare/AWS CloudFront) for global performance
- Separate build per environment (dev, staging, prod)

**Security**:
- JWT stored in httpOnly cookies (XSS protection)
- CSRF tokens for state-changing requests
- Content Security Policy (CSP) headers

**Scalability**:
- Static assets (infinitely scalable via CDN)
- No server-side state

---

### 2. API Application (Python/FastAPI)
**Technology**: Python 3.11, FastAPI, asyncpg, httpx, WebSocket
**Purpose**: Main API server handling HTTP/WebSocket requests

**Key Responsibilities**:
- **Authentication**: JWT validation, tenant context extraction
- **Routing**: REST API endpoints + WebSocket connections
- **Orchestration**: Coordinate agents, risk management, portfolio optimization
- **Trade Execution**: Place orders on exchanges
- **Real-time Updates**: Stream portfolio updates via WebSocket

**Endpoints**:
- `GET /portfolio` - Get current portfolio
- `POST /trades` - Execute trade
- `GET /agents/signals` - Get agent signals
- `POST /agents/config` - Update agent configuration
- `GET /risk/metrics` - Get risk metrics
- `WS /ws` - WebSocket connection for real-time updates

**Deployment**:
- Docker containers on Kubernetes
- Horizontal scaling (10-50 pods)
- Liveness/readiness probes
- Auto-scaling based on CPU/memory (HPA)

**Security**:
- JWT validation on every request
- RLS session variable set from `tenant_id` claim
- Rate limiting (100 req/min per tenant)
- Request timeouts (30 seconds)

**Performance**:
- Target: P99 <500ms for all endpoints
- Connection pooling (PostgreSQL, Redis, HTTP clients)
- Async I/O (asyncio, uvicorn)

---

### 3. Agent Workers (Python/Celery)
**Technology**: Python 3.11, Celery, Redis (broker), asyncpg
**Purpose**: Background workers executing AI agent tasks

**Agents**:
- **Technical Agent**: Technical analysis (RSI, MACD, Bollinger Bands)
- **Fundamental Agent**: On-chain analysis (transaction volume, active addresses)
- **Sentiment Agent**: Social media sentiment (Twitter, Reddit)
- **Value Agent**: Value investing (market cap analysis, holder distribution)
- **Activist Agent**: Governance activity tracking
- **Warren Buffett Agent**: Long-term value signals

**Workflow**:
1. API enqueues task: `generate_agent_signals.delay(tenant_id, symbol)`
2. Worker pulls task from Redis queue
3. Worker fetches market data from exchanges
4. Worker generates signal (Buy/Sell/Hold)
5. Worker writes signal to PostgreSQL (with RLS)
6. Worker caches signal in Redis

**Deployment**:
- Docker containers on Kubernetes
- Horizontal scaling (5-20 workers per agent type)
- Celery concurrency: 4 tasks per worker
- Task timeout: 5 minutes

**Security**:
- WorkerTenantContext mixin ensures RLS session variable set
- No direct user input (tasks from trusted API only)

**Performance**:
- Target: 100 signals/minute per agent
- Parallel execution (6 agent types × 5 workers = 30 parallel tasks)

---

### 4. Risk Workers (Python/Celery)
**Technology**: Python 3.11, Celery, Redis, asyncpg, NumPy/SciPy
**Purpose**: Background workers for risk management and portfolio optimization

**Tasks**:
- **Risk Calculation**: VaR, CVaR, Sharpe ratio, max drawdown
- **Portfolio Optimization**: Mean-variance, Risk Parity, HRP, Black-Litterman
- **Position Sizing**: Kelly Criterion, fixed fractional
- **Rebalancing**: Periodic portfolio rebalancing

**Workflow**:
1. API enqueues task: `calculate_risk_metrics.delay(tenant_id)`
2. Worker pulls task from Redis queue
3. Worker reads portfolio from PostgreSQL
4. Worker calculates risk metrics
5. Worker writes metrics to PostgreSQL
6. Worker caches results in Redis

**Deployment**:
- Docker containers on Kubernetes
- Horizontal scaling (5-10 workers)
- CPU-intensive tasks (matrix operations)
- Task timeout: 10 minutes

**Performance**:
- Target: 50 portfolios optimized/minute
- Memory-intensive (NumPy arrays for correlation matrices)

---

### 5. Billing Service (Python/FastAPI)
**Technology**: Python 3.11, FastAPI, asyncpg, Stripe SDK
**Purpose**: Usage metering, quota enforcement, Stripe webhook handling

**Key Responsibilities**:
- **Usage Metering**: Track API calls, trades, positions per tenant
- **Quota Enforcement**: Block requests when quota exceeded
- **Stripe Webhooks**: Handle payment events (success, failed, refund)
- **Subscription Management**: Update tenant status (active, suspended, cancelled)

**Endpoints**:
- `POST /webhooks/stripe` - Stripe webhook handler
- `GET /usage` - Get current usage metrics
- `GET /invoices` - List invoices

**Webhook Events**:
- `invoice.payment_succeeded` → Activate subscription
- `invoice.payment_failed` → Suspend tenant
- `customer.subscription.updated` → Update tier
- `customer.subscription.deleted` → Cancel subscription

**Deployment**:
- Docker container on Kubernetes
- 2-3 replicas for high availability
- Webhook signature verification (HMAC SHA256)
- Idempotent processing (prevent duplicate webhook handling)

**Security**:
- Stripe webhook signature verification
- HTTPS only
- Rate limiting (1000 webhooks/min)

---

### 6. Provisioning Service (Python/FastAPI)
**Technology**: Python 3.11, FastAPI, asyncpg, Vault SDK, Stripe SDK
**Purpose**: Automated tenant onboarding and setup

**Key Responsibilities**:
- **Tenant Creation**: Create tenant record in PostgreSQL
- **Credential Setup**: Initialize Vault paths for tenant
- **Cache Initialization**: Create Redis namespace for tenant
- **Stripe Subscription**: Create Stripe customer and subscription
- **Health Checks**: Verify database, cache, Vault, Stripe connectivity

**Workflow** (<2 minutes for Starter/Pro):
1. Receive signup request
2. Create tenant record in PostgreSQL
3. Initialize Vault path (`secret/tenants/{tenant_id}/`)
4. Create Redis namespace (`tenant:{tenant_id}:*`)
5. Create Stripe customer and subscription
6. Run health checks
7. Send welcome email
8. Return tenant credentials (JWT)

**Deployment**:
- Docker container on Kubernetes
- 1-2 replicas
- Async provisioning (task queue for reliability)

**Performance**:
- Target: <2 min for Starter/Pro, <10 min for Enterprise
- Durable queue (Redis Streams or RabbitMQ) for retry logic

---

### 7. PostgreSQL Database
**Technology**: PostgreSQL 14, asyncpg driver
**Purpose**: Primary data store for all tenant data

**Key Tables**:
- `tenants` (tenant metadata)
- `trades` (all trades, partitioned by tenant_id)
- `positions` (current positions)
- `portfolios` (portfolio snapshots)
- `portfolio_history` (historical performance)
- `risk_metrics` (VaR, CVaR, Sharpe)
- `agent_signals` (buy/sell/hold signals from agents)
- `trade_executions` (execution log)

**RLS Policies**:
- All tables except `tenants` have RLS enabled
- Policy: `tenant_id = current_setting('app.current_tenant_id')::uuid`

**Indexes**:
- Composite indexes: `(tenant_id, id)`, `(tenant_id, created_at DESC)`
- Performance: <10% overhead vs no RLS

**Deployment**:
- AWS RDS or self-hosted on Kubernetes
- Instance: db.r5.xlarge (4 vCPU, 32GB RAM)
- Storage: GP3 SSD (3000 IOPS)
- High Availability: Multi-AZ deployment
- Read Replicas: 1-2 replicas for read-heavy workloads

**Backup**:
- Automated backups (daily)
- PITR (Point-In-Time Recovery, 7 days retention)
- Manual snapshots before major migrations

---

### 8. Redis Cluster
**Technology**: Redis 7, Redis Cluster mode
**Purpose**: Caching layer for market data, credentials, signals

**Namespace Structure**:
- `tenant:{tenant_id}:signal:*` - Agent signals
- `tenant:{tenant_id}:position:*` - Current positions
- `tenant:{tenant_id}:portfolio` - Portfolio snapshot
- `shared:market:{symbol}:price` - Market prices (shared)
- `meta:tenant:{tenant_id}:usage_bytes` - Usage tracking (rolling counter)
- `meta:tenant:{tenant_id}:lru` - LRU eviction (sorted set)

**Deployment**:
- Redis Cluster (6 nodes: 3 masters + 3 replicas)
- Instance: cache.t3.medium (2 vCPU, 3.09GB RAM) per node
- Max memory: 2GB per master
- Eviction policy: `allkeys-lru`
- High Availability: Automatic failover

**Performance**:
- Target: P99 <1ms for GET, P99 <2ms for SET
- Cache hit rate: >80% for market data

---

### 9. HashiCorp Vault
**Technology**: Vault 1.15.x, Raft consensus
**Purpose**: Secure storage for exchange API keys

**Path Structure**:
- `secret/tenants/{tenant_id}/exchanges/binance` - Binance credentials
- `secret/tenants/{tenant_id}/exchanges/coinbase` - Coinbase credentials
- `secret/tenants/{tenant_id}/exchanges/kraken` - Kraken credentials

**Policies**:
- Tenant policy: Allow read/write only to `secret/tenants/{tenant_id}/*`
- Admin policy: Allow read all paths

**Deployment**:
- HA cluster (3 replicas)
- Raft integrated storage
- Auto-unseal with AWS KMS
- Instance: t3.medium (2 vCPU, 4GB RAM) per replica

**Backup**:
- Raft snapshots (daily)
- Audit logs (all credential access logged)

**Performance**:
- Target: >10k req/sec, P99 <10ms
- In-memory caching (5-minute TTL) reduces load by 95%

---

### 10. NGINX Load Balancer
**Technology**: NGINX or Traefik
**Purpose**: Reverse proxy, TLS termination, load balancing

**Key Features**:
- **TLS Termination**: HTTPS (TLS 1.3)
- **Load Balancing**: Round-robin to API pods
- **Rate Limiting**: 100 req/min per tenant, 1000 req/min global
- **Health Checks**: Liveness probes to API pods
- **Static File Serving**: React SPA

**Deployment**:
- Kubernetes Ingress controller
- 2-3 replicas for high availability
- Auto-scaling based on request rate

**Security**:
- WAF (Web Application Firewall) integration
- DDoS protection (Cloudflare or AWS Shield)

---

## Communication Patterns

### Synchronous (Request/Response)
- **API ↔ Database**: SQL queries (asyncpg)
- **API ↔ Redis**: Cache reads/writes (redis-py)
- **API ↔ Vault**: Credential retrieval (hvac SDK)
- **API ↔ Exchanges**: Market data fetch (httpx)
- **API ↔ Stripe**: Subscription management (stripe SDK)

**Pattern**: Direct API calls with connection pooling

---

### Asynchronous (Task Queue)
- **API → Agent Workers**: Celery tasks (Redis as broker)
- **API → Risk Workers**: Celery tasks (Redis as broker)

**Pattern**: Producer-consumer with durable queue

**Task Example**:
```python
@celery.task
@tenant_context.required
def generate_agent_signals(tenant_id: UUID, symbol: str):
    # Worker automatically sets RLS session variable
    signals = technical_agent.generate(symbol)
    db.save(signals)  # RLS ensures tenant isolation
```

---

### Event-Driven (Webhooks)
- **Stripe → Billing Service**: Payment webhooks
- **Exchanges → API**: Order fill updates (WebSocket)

**Pattern**: Webhook receiver with signature verification

**Webhook Example**:
```python
@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    sig = request.headers.get("Stripe-Signature")
    payload = await request.body()
    event = stripe.Webhook.construct_event(payload, sig, webhook_secret)
    # Process event (invoice.payment_succeeded, etc.)
```

---

### Real-Time (WebSocket)
- **API ↔ Tenant Trader**: Portfolio updates, trade notifications

**Pattern**: Persistent WebSocket connection with heartbeat

**WebSocket Example**:
```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    tenant_id = extract_tenant_from_jwt(websocket)
    while True:
        updates = await get_portfolio_updates(tenant_id)
        await websocket.send_json(updates)
        await asyncio.sleep(1)  # 1Hz updates
```

---

## Security Boundaries

### 1. External → Load Balancer
- TLS 1.3 encryption
- WAF (Web Application Firewall)
- DDoS protection
- Rate limiting

### 2. Load Balancer → API
- Internal TLS (optional, Kubernetes network policy)
- JWT validation
- Tenant context extraction

### 3. API → Database
- Connection pooling (max 20 connections)
- RLS session variable set: `SET LOCAL app.current_tenant_id = {tenant_id}`
- Prepared statements (SQL injection prevention)

### 4. API → Vault
- TLS encryption
- Vault token authentication
- Tenant-scoped policies

### 5. API → Redis
- TLS encryption (optional)
- Namespace isolation (`tenant:{id}:*`)
- ACLs (Redis 6+, optional)

---

## Scalability

### Horizontal Scaling
- **API**: 10-50 pods (Kubernetes HPA)
- **Agent Workers**: 5-20 workers per agent type
- **Risk Workers**: 5-10 workers
- **Redis**: 6-12 nodes (3-6 masters + replicas)

### Vertical Scaling
- **Database**: db.r5.xlarge → db.r5.2xlarge (8 vCPU, 64GB RAM)
- **Vault**: t3.medium → t3.large (2 vCPU, 8GB RAM)

### Caching Strategy
- **L1 Cache**: In-memory (Python `@lru_cache`)
- **L2 Cache**: Redis (distributed cache)
- **L3 Cache**: CDN (static assets)

---

## Monitoring & Observability

### Metrics
- **Request Rate**: Requests/second per endpoint
- **Latency**: P50, P95, P99 per endpoint
- **Error Rate**: Errors/second per endpoint
- **Cache Hit Rate**: Redis cache hit percentage
- **Queue Depth**: Celery queue length
- **Database Connections**: Active connections to PostgreSQL

### Logs
- **Application Logs**: FastAPI request/response logs
- **Audit Logs**: All credential access (Vault audit logs)
- **Error Logs**: Exceptions and stack traces

### Traces
- **Distributed Tracing**: OpenTelemetry traces
- **Request Flow**: API → Worker → Database (end-to-end)

### Dashboards
- **Platform Health**: Request rate, latency, error rate
- **Tenant Usage**: API calls, trades, positions per tenant
- **Infrastructure**: CPU, memory, disk, network per container

---

## Deployment Architecture

### Kubernetes Namespaces
- `alphapulse-prod` - Production environment
- `alphapulse-staging` - Staging environment
- `alphapulse-dev` - Development environment

### Pods
- **API**: 10-50 pods (HPA)
- **Agent Workers**: 30-120 pods (6 types × 5-20 workers)
- **Risk Workers**: 5-10 pods
- **Billing Service**: 2-3 pods
- **Provisioning Service**: 1-2 pods

### Services
- **Load Balancer**: Ingress controller (NGINX)
- **API**: ClusterIP service (internal)
- **Redis**: ClusterIP service (internal)
- **Vault**: ClusterIP service (internal)

### Storage
- **PostgreSQL**: External (AWS RDS or persistent volume)
- **Redis**: StatefulSet with persistent volumes
- **Vault**: StatefulSet with persistent volumes

---

## References

- [HLD Section 2.1: Architecture Views](../HLD-MULTI-TENANT-SAAS.md#21-architecture-views)
- [ADR-001: Data Isolation Strategy](../adr/001-multi-tenant-data-isolation-strategy.md)
- [ADR-002: Tenant Provisioning](../adr/002-tenant-provisioning-architecture.md)
- [ADR-003: Credential Management](../adr/003-credential-management-multi-tenant.md)
- [ADR-004: Caching Strategy](../adr/004-caching-strategy-multi-tenant.md)
- [C4 Level 1: System Context](c4-level1-system-context.md)

---

**Diagram Status**: Draft (pending review)
**Review Date**: Sprint 3, Week 1
**Reviewers**: Tech Lead, Security Lead, DevOps Lead, CTO

---

**END OF DOCUMENT**
