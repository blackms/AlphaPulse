# API Documentation

AlphaPulse exposes a FastAPI application located at `src/alpha_pulse/api`.  The
service provides REST and WebSocket endpoints covering metrics, portfolio
analytics, risk tooling, ensemble routing, and supporting services.

## Running the API locally

```bash
poetry install
poetry run uvicorn src.alpha_pulse.api.main:app --reload
```

The interactive OpenAPI documentation is available at
`http://localhost:8000/docs` once the server is running.

## Authentication

The project ships with OAuth2 password/token support
(`src/alpha_pulse/api/auth.py`).  Endpoints that require authentication follow
the FastAPI dependency pattern and will return 401 responses when credentials
are missing or invalid.

Tokens can be obtained by POSTing to `/token` with form-encoded credentials.
See the `get_current_user` dependency for request-scoped user resolution.

## Router overview

| Router module                          | Prefix              | Description                                        |
|----------------------------------------|---------------------|----------------------------------------------------|
| `routers/metrics.py`                   | `/api/v1/metrics`   | System metrics and Prometheus integration          |
| `routers/portfolio.py`                 | `/api/v1/portfolio` | Portfolio snapshots and analytics                  |
| `routers/trades.py`                    | `/api/v1/trades`    | Trade history and execution helpers                |
| `routers/regime.py`                    | `/api/v1/regime`    | Market regime detection and monitoring             |
| `routers/risk_budget.py`               | `/api/v1/risk-budget` | Risk budgeting operations                        |
| `routers/hedging.py`                   | `/api/v1/hedging`   | Tail-risk hedging analysis and execution           |
| `routers/liquidity.py`                 | `/api/v1/liquidity` | Liquidity risk checks and liquidity-aware metrics  |
| `routers/ensemble.py`                  | `/api/v1/ensemble`  | Ensemble model management and inference            |
| `routers/online_learning.py`           | `/api/v1/online-learning` | Online learning controls                      |
| `routers/gpu.py`                       | `/api/v1/gpu`       | GPU diagnostics and workload dispatch (if enabled) |
| `routers/explainability.py`            | `/api/v1/explainability` | Explainability requests                       |
| `routers/data_quality.py`              | `/api/v1/data-quality` | Data quality metrics                           |
| `routers/backtesting.py`               | `/api/v1/backtesting` | Backtesting orchestration                      |
| `routers/data_lake.py`                 | `/api/v1/data-lake` | Data lake orchestration and metadata               |
| `routers/alerts.py`                    | `/api/v1/alerts`    | Alert management and acknowledgment                |
| `routers/system.py`                    | `/api/v1/system`    | System health and maintenance endpoints            |
| `routes/audit.py`                      | `/api/v1/audit`     | Audit trail queries                                |

Refer to the source file for input/output models and error handling.

## WebSockets

WebSocket subscriptions are defined under `api/websockets`.  The router
`websockets/endpoints.py` exposes channels for real-time metrics and streaming
portfolio updates.  Subscription management is handled by
`websockets/subscription.py`.

## Error handling & middleware

Middleware for audit logging, security headers, CSRF protection, rate limiting,
tenant context management, and validation can be found under
`src/alpha_pulse/api/middleware`.  Familiarise yourself with these components
before registering new routers to ensure requests flow through the appropriate
guards.

## Adding new endpoints

1. Create a router module under `src/alpha_pulse/api/routers`.
2. Define request/response models in `api/models` (or the appropriate domain
   package).
3. Register the router in `src/alpha_pulse/api/main.py`.
4. Update automated tests (`src/alpha_pulse/tests/api`) and documentation.
5. If the endpoint requires authentication, use the existing dependencies to
   enforce token validation.
