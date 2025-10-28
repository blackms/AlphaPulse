# C4 Level 3: Component Diagram

**Date**: 2025-10-21
**Sprint**: 3 (Design & Alignment)
**Author**: Tech Lead
**Related**: [HLD-MULTI-TENANT-SAAS.md](../HLD-MULTI-TENANT-SAAS.md), Issue #180

---

## Purpose

This Component diagram shows the internal structure of the **API Application** container, breaking it down into key components and their responsibilities.

---

## Diagram: API Application Components

```plantuml
@startuml C4_Level3_Component_API
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Component.puml

LAYOUT_WITH_LEGEND()

title Component Diagram - API Application (FastAPI)

Container(web_app, "Web Application", "React/TypeScript")
Container_Boundary(api, "API Application") {
    Component(auth_middleware, "Auth Middleware", "Python/FastAPI", "Validates JWT, extracts tenant_id, sets tenant context")
    Component(rate_limiter, "Rate Limiter", "Python/slowapi", "Enforces rate limits per tenant (100 req/min)")

    Component(api_routes, "API Routes", "FastAPI Router", "HTTP endpoints and WebSocket handlers")

    Component(tenant_service, "Tenant Service", "Python", "Tenant CRUD, tenant context management")
    Component(trading_service, "Trading Service", "Python", "Execute trades, manage orders")
    Component(portfolio_service, "Portfolio Service", "Python", "Portfolio calculations, P&L tracking")
    Component(agent_service, "Agent Orchestrator", "Python", "Coordinate AI agents, aggregate signals")
    Component(risk_service, "Risk Manager", "Python", "Risk calculations, position sizing, stop-loss")
    Component(credential_service, "Credential Service", "Python", "Retrieve credentials from Vault")
    Component(cache_service, "Caching Service", "Python", "Redis cache operations, namespace isolation")
    Component(billing_service_client, "Billing Client", "Python", "Usage tracking, quota checks")

    Component(db_models, "Database Models", "SQLAlchemy", "ORM models with tenant_id, RLS session management")
    Component(exchange_client, "Exchange Client", "Python/CCXT", "Exchange API wrapper (Binance, Coinbase, Kraken)")
    Component(websocket_manager, "WebSocket Manager", "Python", "Manage persistent connections, broadcast updates")
}

ContainerDb(postgres, "PostgreSQL", "Database")
ContainerDb(redis, "Redis", "Cache")
ContainerDb(vault, "Vault", "Secrets")
Container(agent_workers, "Agent Workers", "Celery")
System_Ext(exchanges, "Exchanges", "Binance, Coinbase")

Rel(web_app, auth_middleware, "Sends requests with JWT", "HTTPS")
Rel(auth_middleware, rate_limiter, "After auth validation", "Internal")
Rel(rate_limiter, api_routes, "If within rate limit", "Internal")

Rel(api_routes, tenant_service, "Tenant operations")
Rel(api_routes, trading_service, "Trade execution")
Rel(api_routes, portfolio_service, "Portfolio queries")
Rel(api_routes, agent_service, "Agent management")
Rel(api_routes, risk_service, "Risk calculations")
Rel(api_routes, websocket_manager, "Real-time updates")

Rel(trading_service, credential_service, "Get API keys")
Rel(trading_service, exchange_client, "Execute orders")
Rel(trading_service, db_models, "Save trades")
Rel(trading_service, cache_service, "Cache order status")
Rel(trading_service, billing_service_client, "Track usage")

Rel(portfolio_service, db_models, "Read positions")
Rel(portfolio_service, cache_service, "Cache portfolio state")

Rel(agent_service, agent_workers, "Enqueue tasks", "Celery/Redis")
Rel(agent_service, cache_service, "Cache signals")

Rel(risk_service, db_models, "Read portfolio")
Rel(risk_service, cache_service, "Cache risk metrics")

Rel(credential_service, vault, "Get credentials", "HTTPS")
Rel(credential_service, cache_service, "Cache credentials (5min)")

Rel(cache_service, redis, "Read/write", "Redis protocol")

Rel(db_models, postgres, "SQL queries", "asyncpg")

Rel(exchange_client, exchanges, "REST/WebSocket", "HTTPS")

@enduml
```

---

## Components

### 1. Auth Middleware
**Technology**: FastAPI middleware, PyJWT
**Responsibility**: Authentication and tenant context initialization

**Key Functions**:
```python
async def tenant_context_middleware(request: Request, call_next):
    # Extract JWT from Authorization header
    token = request.headers.get("Authorization", "").replace("Bearer ", "")

    # Validate JWT and extract tenant_id
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    tenant_id = payload.get("tenant_id")

    # Set tenant context (thread-local or request state)
    request.state.tenant_id = tenant_id

    # Set PostgreSQL RLS session variable
    await db.execute(f"SET LOCAL app.current_tenant_id = '{tenant_id}'")

    response = await call_next(request)
    return response
```

**Error Handling**:
- Missing token → 401 Unauthorized
- Invalid signature → 401 Unauthorized
- Expired token → 401 Unauthorized
- Missing tenant_id → 403 Forbidden

**Performance**:
- JWT validation: <1ms (in-memory operation)
- RLS session variable: <1ms (PostgreSQL local variable)

---

### 2. Rate Limiter
**Technology**: slowapi (rate limiting library for FastAPI)
**Responsibility**: Enforce per-tenant rate limits

**Configuration**:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=lambda: request.state.tenant_id,  # Rate limit by tenant
    default_limits=["100/minute"]  # Default: 100 requests per minute
)

@app.get("/portfolio")
@limiter.limit("100/minute")  # Per-tenant limit
async def get_portfolio(request: Request):
    tenant_id = request.state.tenant_id
    # ... implementation
```

**Tier-Based Limits**:
- **Starter**: 100 req/min
- **Pro**: 500 req/min
- **Enterprise**: 2000 req/min

**Storage**: Redis (stores request counts per tenant)

**Error Response**:
```json
{
  "error": "Rate limit exceeded",
  "limit": "100/minute",
  "retry_after": 42
}
```

---

### 3. API Routes
**Technology**: FastAPI Router
**Responsibility**: HTTP endpoints and WebSocket handlers

**Route Groups**:

#### Authentication & Tenants
- `POST /auth/signup` - Tenant signup
- `POST /auth/login` - JWT login
- `GET /auth/refresh` - Refresh JWT
- `GET /tenants/me` - Get current tenant info

#### Trading
- `POST /trades` - Execute trade
- `GET /trades` - List trades
- `GET /trades/{id}` - Get trade details
- `DELETE /trades/{id}` - Cancel order

#### Portfolio
- `GET /portfolio` - Get current portfolio
- `GET /portfolio/history` - Historical performance
- `GET /positions` - Current positions
- `GET /positions/{symbol}` - Position details

#### AI Agents
- `GET /agents/signals` - Get recent signals
- `POST /agents/config` - Update agent config
- `POST /agents/run` - Trigger agent execution
- `GET /agents/status` - Agent health status

#### Risk Management
- `GET /risk/metrics` - Get risk metrics (VaR, CVaR, Sharpe)
- `GET /risk/limits` - Get risk limits
- `POST /risk/limits` - Update risk limits

#### Credentials
- `GET /credentials` - List credentials
- `POST /credentials` - Add credential
- `PUT /credentials/{id}` - Update credential
- `DELETE /credentials/{id}` - Delete credential

#### Billing & Usage
- `GET /usage` - Current usage metrics
- `GET /invoices` - List invoices
- `GET /subscription` - Subscription details
- `POST /subscription/upgrade` - Upgrade tier

#### WebSocket
- `WS /ws` - Real-time portfolio updates

**OpenAPI Documentation**: Auto-generated at `/docs`

---

### 4. Tenant Service
**Responsibility**: Tenant CRUD operations and context management

**Key Methods**:
```python
class TenantService:
    async def get_tenant(self, tenant_id: UUID) -> Tenant:
        """Get tenant by ID"""

    async def create_tenant(self, data: TenantCreate) -> Tenant:
        """Create new tenant (called by provisioning service)"""

    async def update_tenant(self, tenant_id: UUID, data: TenantUpdate) -> Tenant:
        """Update tenant metadata"""

    async def suspend_tenant(self, tenant_id: UUID) -> None:
        """Suspend tenant (billing failure)"""

    async def activate_tenant(self, tenant_id: UUID) -> None:
        """Activate tenant (payment succeeded)"""
```

**Database Interactions**:
- Table: `tenants`
- No RLS (single table for all tenants)
- Indexed by `id` (UUID)

---

### 5. Trading Service
**Responsibility**: Trade execution and order management

**Key Methods**:
```python
class TradingService:
    async def execute_trade(
        self,
        tenant_id: UUID,
        symbol: str,
        side: str,  # BUY/SELL
        quantity: Decimal,
        order_type: str  # MARKET/LIMIT
    ) -> Trade:
        """Execute trade on exchange"""

        # 1. Get credentials from Vault
        credentials = await credential_service.get_credentials(tenant_id, exchange)

        # 2. Validate risk limits
        await risk_service.validate_trade(tenant_id, symbol, quantity)

        # 3. Execute on exchange
        order = await exchange_client.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            credentials=credentials
        )

        # 4. Save to database
        trade = await db_models.Trade.create(
            tenant_id=tenant_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=order.price,
            order_id=order.id,
            status="PENDING"
        )

        # 5. Track usage
        await billing_service_client.track_trade(tenant_id)

        return trade
```

**Error Handling**:
- Insufficient funds → 400 Bad Request
- Risk limit exceeded → 403 Forbidden
- Exchange API error → 502 Bad Gateway
- Credential missing → 404 Not Found

---

### 6. Portfolio Service
**Responsibility**: Portfolio calculations and P&L tracking

**Key Methods**:
```python
class PortfolioService:
    async def get_portfolio(self, tenant_id: UUID) -> Portfolio:
        """Get current portfolio state"""

        # Try cache first
        cached = await cache_service.get(f"tenant:{tenant_id}:portfolio")
        if cached:
            return cached

        # Calculate from database
        positions = await db_models.Position.get_by_tenant(tenant_id)
        total_value = sum(p.quantity * p.current_price for p in positions)

        portfolio = Portfolio(
            tenant_id=tenant_id,
            positions=positions,
            total_value=total_value,
            cash_balance=await self._get_cash_balance(tenant_id),
            unrealized_pnl=await self._calculate_unrealized_pnl(tenant_id)
        )

        # Cache for 5 seconds (hot data)
        await cache_service.set(f"tenant:{tenant_id}:portfolio", portfolio, ttl=5)

        return portfolio

    async def get_historical_performance(
        self,
        tenant_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> List[PortfolioSnapshot]:
        """Get historical portfolio snapshots"""
```

**Caching Strategy**:
- Portfolio state: 5-second TTL (hot data)
- Historical snapshots: 1-hour TTL (cold data)

---

### 7. Agent Orchestrator
**Responsibility**: Coordinate AI agents and aggregate signals

**Key Methods**:
```python
class AgentOrchestrator:
    async def generate_signals(
        self,
        tenant_id: UUID,
        symbol: str
    ) -> List[AgentSignal]:
        """Generate signals from all agents"""

        # Enqueue tasks for each agent type
        tasks = []
        for agent_type in ["technical", "fundamental", "sentiment",
                          "value", "activist", "buffett"]:
            task = agent_workers.generate_agent_signal.delay(
                tenant_id=tenant_id,
                symbol=symbol,
                agent_type=agent_type
            )
            tasks.append(task)

        # Wait for all tasks to complete (with timeout)
        results = await asyncio.gather(*[task.get(timeout=30) for task in tasks])

        return results

    async def get_consensus_signal(
        self,
        tenant_id: UUID,
        symbol: str
    ) -> str:  # BUY/SELL/HOLD
        """Aggregate signals into consensus"""

        signals = await self.get_recent_signals(tenant_id, symbol)

        # Weighted voting
        weights = {"technical": 0.3, "fundamental": 0.3, "sentiment": 0.2,
                   "value": 0.1, "activist": 0.05, "buffett": 0.05}

        buy_score = sum(w for s, w in zip(signals, weights.values()) if s.signal == "BUY")
        sell_score = sum(w for s, w in zip(signals, weights.values()) if s.signal == "SELL")

        if buy_score > 0.5:
            return "BUY"
        elif sell_score > 0.5:
            return "SELL"
        else:
            return "HOLD"
```

**Background Worker Interaction**:
- Asynchronous task execution via Celery
- Redis as message broker
- 30-second timeout for agent execution

---

### 8. Risk Manager
**Responsibility**: Risk calculations, position sizing, stop-loss enforcement

**Key Methods**:
```python
class RiskManager:
    async def calculate_risk_metrics(self, tenant_id: UUID) -> RiskMetrics:
        """Calculate VaR, CVaR, Sharpe ratio, max drawdown"""

    async def validate_trade(
        self,
        tenant_id: UUID,
        symbol: str,
        quantity: Decimal
    ) -> None:
        """Validate trade against risk limits"""

        # Check position size limit
        position_value = quantity * current_price
        if position_value > max_position_size:
            raise RiskLimitExceeded("Position size exceeds limit")

        # Check portfolio concentration
        portfolio_value = await portfolio_service.get_total_value(tenant_id)
        concentration = position_value / portfolio_value
        if concentration > 0.2:  # Max 20% per position
            raise RiskLimitExceeded("Portfolio concentration exceeds 20%")

        # Check daily loss limit
        daily_pnl = await self._get_daily_pnl(tenant_id)
        if daily_pnl < -0.05 * portfolio_value:  # Max 5% daily loss
            raise RiskLimitExceeded("Daily loss limit exceeded")

    async def apply_position_sizing(
        self,
        tenant_id: UUID,
        signal_strength: float
    ) -> Decimal:
        """Calculate position size using Kelly Criterion"""
```

**Risk Limits** (configurable per tenant):
- Max position size: 20% of portfolio
- Max leverage: 2x (Pro), 5x (Enterprise)
- Daily loss limit: 5% of portfolio
- Max drawdown: 20%

---

### 9. Credential Service
**Responsibility**: Retrieve exchange API keys from Vault

**Key Methods**:
```python
class CredentialService:
    def __init__(self):
        self.vault_client = hvac.Client(url=VAULT_URL, token=VAULT_TOKEN)
        self.cache = {}  # In-memory cache

    async def get_credentials(
        self,
        tenant_id: UUID,
        exchange: str
    ) -> ExchangeCredentials:
        """Get credentials with 5-minute cache"""

        cache_key = f"{tenant_id}:{exchange}"

        # Check in-memory cache
        if cache_key in self.cache:
            cached, timestamp = self.cache[cache_key]
            if time.time() - timestamp < 300:  # 5-minute TTL
                return cached

        # Fetch from Vault
        path = f"secret/tenants/{tenant_id}/exchanges/{exchange}"
        secret = self.vault_client.secrets.kv.v2.read_secret_version(path=path)

        credentials = ExchangeCredentials(
            api_key=secret["data"]["api_key"],
            secret=secret["data"]["secret"],
            permissions=secret["data"]["permissions"]
        )

        # Cache in memory
        self.cache[cache_key] = (credentials, time.time())

        return credentials

    async def store_credentials(
        self,
        tenant_id: UUID,
        exchange: str,
        credentials: ExchangeCredentials
    ) -> None:
        """Store credentials in Vault after validation"""

        # Validate credentials via test API call
        await self._validate_credentials(exchange, credentials)

        # Store in Vault
        path = f"secret/tenants/{tenant_id}/exchanges/{exchange}"
        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=credentials.dict()
        )

        # Invalidate cache
        cache_key = f"{tenant_id}:{exchange}"
        if cache_key in self.cache:
            del self.cache[cache_key]
```

**Caching**:
- In-memory cache (5-minute TTL)
- Reduces Vault load by 95%
- Automatic invalidation on credential update

---

### 10. Caching Service
**Responsibility**: Redis cache operations with namespace isolation

**Key Methods**:
```python
class CachingService:
    def __init__(self):
        self.redis = redis.asyncio.from_url(REDIS_URL)

    async def get(self, tenant_id: UUID, key: str) -> Optional[Any]:
        """Get value from tenant namespace"""
        namespaced_key = f"tenant:{tenant_id}:{key}"
        value = await self.redis.get(namespaced_key)
        return json.loads(value) if value else None

    async def set(
        self,
        tenant_id: UUID,
        key: str,
        value: Any,
        ttl: int = 300
    ) -> None:
        """Set value in tenant namespace with TTL"""
        namespaced_key = f"tenant:{tenant_id}:{key}"
        await self.redis.setex(namespaced_key, ttl, json.dumps(value))

        # Track usage (rolling counter)
        usage_key = f"meta:tenant:{tenant_id}:usage_bytes"
        await self.redis.incrby(usage_key, len(json.dumps(value)))

        # Track LRU (sorted set)
        lru_key = f"meta:tenant:{tenant_id}:lru"
        await self.redis.zadd(lru_key, {namespaced_key: time.time()})

    async def get_shared(self, key: str) -> Optional[Any]:
        """Get value from shared namespace (market data)"""
        shared_key = f"shared:market:{key}"
        value = await self.redis.get(shared_key)
        return json.loads(value) if value else None

    async def set_shared(self, key: str, value: Any, ttl: int = 60) -> None:
        """Set value in shared namespace"""
        shared_key = f"shared:market:{key}"
        await self.redis.setex(shared_key, ttl, json.dumps(value))
```

**Namespace Isolation**:
- Tenant-specific: `tenant:{id}:*`
- Shared market data: `shared:market:*`
- Metadata: `meta:tenant:{id}:*`

---

### 11. Billing Service Client
**Responsibility**: Track usage and check quotas

**Key Methods**:
```python
class BillingServiceClient:
    async def track_api_call(self, tenant_id: UUID) -> None:
        """Increment API call counter"""
        await redis.incr(f"usage:tenant:{tenant_id}:api_calls")

    async def track_trade(self, tenant_id: UUID) -> None:
        """Increment trade counter"""
        await redis.incr(f"usage:tenant:{tenant_id}:trades")

    async def check_quota(self, tenant_id: UUID, resource: str) -> bool:
        """Check if tenant has quota remaining"""
        usage = await redis.get(f"usage:tenant:{tenant_id}:{resource}")
        limit = await self._get_limit(tenant_id, resource)
        return int(usage or 0) < limit
```

---

### 12. Database Models
**Technology**: SQLAlchemy (async), asyncpg driver
**Responsibility**: ORM models with RLS session management

**Key Models**:
```python
class BaseModel(Base):
    __abstract__ = True

    id = Column(UUID, primary_key=True, default=uuid4)
    tenant_id = Column(UUID, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

class Trade(BaseModel):
    __tablename__ = "trades"

    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # BUY/SELL
    quantity = Column(Numeric(18, 8), nullable=False)
    price = Column(Numeric(18, 8), nullable=False)
    order_id = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)  # PENDING/FILLED/CANCELLED

    __table_args__ = (
        Index("idx_trades_tenant_id", "tenant_id", "id"),
        Index("idx_trades_tenant_created", "tenant_id", "created_at"),
    )
```

**RLS Session Management**:
```python
async def set_tenant_context(tenant_id: UUID):
    """Set PostgreSQL RLS session variable"""
    await db.execute(f"SET LOCAL app.current_tenant_id = '{tenant_id}'")
```

---

### 13. Exchange Client
**Technology**: CCXT (unified exchange API)
**Responsibility**: Abstract exchange API differences

**Key Methods**:
```python
class ExchangeClient:
    def __init__(self):
        self.exchanges = {
            "binance": ccxt.binance(),
            "coinbase": ccxt.coinbase(),
            "kraken": ccxt.kraken()
        }

    async def create_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        credentials: ExchangeCredentials
    ) -> Order:
        """Create order on exchange"""

        client = self.exchanges[exchange]
        client.apiKey = credentials.api_key
        client.secret = credentials.secret

        order = await client.create_order(
            symbol=symbol,
            type=order_type.lower(),
            side=side.lower(),
            amount=float(quantity)
        )

        return Order(
            id=order["id"],
            symbol=order["symbol"],
            side=order["side"],
            price=Decimal(order["price"]),
            quantity=Decimal(order["amount"]),
            status=order["status"]
        )

    async def fetch_balance(
        self,
        exchange: str,
        credentials: ExchangeCredentials
    ) -> Dict[str, Decimal]:
        """Fetch account balance"""
```

---

### 14. WebSocket Manager
**Responsibility**: Manage persistent WebSocket connections

**Key Methods**:
```python
class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[UUID, List[WebSocket]] = {}

    async def connect(self, tenant_id: UUID, websocket: WebSocket):
        """Register new WebSocket connection"""
        await websocket.accept()
        if tenant_id not in self.active_connections:
            self.active_connections[tenant_id] = []
        self.active_connections[tenant_id].append(websocket)

    async def disconnect(self, tenant_id: UUID, websocket: WebSocket):
        """Unregister WebSocket connection"""
        self.active_connections[tenant_id].remove(websocket)

    async def broadcast_to_tenant(self, tenant_id: UUID, message: dict):
        """Send message to all connections for tenant"""
        if tenant_id in self.active_connections:
            for connection in self.active_connections[tenant_id]:
                await connection.send_json(message)
```

**Message Types**:
- `portfolio_update` - Portfolio state changed
- `trade_executed` - Trade completed
- `risk_alert` - Risk limit approached
- `agent_signal` - New signal generated

---

## Component Interactions

### Trade Execution Flow
1. **API Routes** receives trade request
2. **Auth Middleware** validates JWT, extracts tenant_id
3. **Rate Limiter** checks tenant quota
4. **Trading Service** coordinates execution:
   - Get credentials from **Credential Service**
   - Validate risk via **Risk Manager**
   - Execute via **Exchange Client**
   - Save to DB via **Database Models**
   - Track usage via **Billing Client**
   - Cache order status via **Caching Service**
5. **WebSocket Manager** broadcasts update to tenant

### Portfolio Query Flow
1. **API Routes** receives portfolio request
2. **Auth Middleware** validates JWT
3. **Portfolio Service** retrieves data:
   - Check **Caching Service** first
   - If miss, query **Database Models**
   - Cache result for 5 seconds
4. Return portfolio to client

### Agent Signal Generation Flow
1. **API Routes** receives signal request
2. **Agent Orchestrator** coordinates:
   - Enqueue tasks to **Agent Workers** (via Celery)
   - Wait for results (30s timeout)
   - Aggregate signals
   - Cache via **Caching Service**
3. Return consensus signal

---

## Design Principles Validation

### Simplicity over Complexity ✅
- Services have single responsibility
- No unnecessary abstractions
- Standard patterns (middleware, service layer)

### Evolutionary Architecture ✅
- Services loosely coupled via interfaces
- Exchange client abstracts API differences
- Easy to replace components (e.g., swap CCXT for custom client)

### Data Sovereignty ✅
- Each service owns its data access
- No shared database sessions
- Clear API boundaries

### Observability First ✅
- All services log to structured logger
- Metrics tracked per service
- Tracing for distributed operations

---

## References

- [C4 Level 2: Container Diagram](c4-level2-container.md)
- [HLD Section 2.1: Component View](../HLD-MULTI-TENANT-SAAS.md#21-architecture-views)
- [ADR-001: Data Isolation Strategy](../adr/001-multi-tenant-data-isolation-strategy.md)
- [ADR-004: Caching Strategy](../adr/004-caching-strategy-multi-tenant.md)

---

**Diagram Status**: Draft (pending review)
**Review Date**: Sprint 3, Week 1
**Reviewers**: Tech Lead, Senior Engineers

---

**END OF DOCUMENT**
