# AI Hedge Fund Platform Overview

This document summarises the major subsystems that compose AlphaPulse.  The
focus is on how the codebase is organised today and which modules drive the core
trading flow.

## Layered architecture

| Layer | Location | Description |
|-------|----------|-------------|
| Signal generation | `src/alpha_pulse/agents/` | Specialised agents (technical, sentiment, value, fundamental, activist, valuation).  Each agent implements a `generate_signals` coroutine returning domain-specific `TradeSignal` objects. |
| Risk management | `src/alpha_pulse/risk_management/` | Position sizing, budgeting, analytics, and risk dashboards.  The `RiskBudgetingService` and `TailRiskHedgingService` orchestrate continuous monitoring tasks. |
| Portfolio management | `src/alpha_pulse/portfolio/` | Optimisation strategies (mean-variance, risk parity, Black-Litterman, etc.) and portfolio state helpers. |
| Execution | `src/alpha_pulse/execution/` | Broker factory, liquidity-aware executor, smart order router, and paper/real brokers built on CCXT. |
| Monitoring & ops | `src/alpha_pulse/monitoring/` | Prometheus metrics, alert routing, and audit logging integrations. |
| Machine learning | `src/alpha_pulse/ml/` | Ensemble models, reinforcement learning utilities, online learning, GPU pipelines, and explainability helpers. |
| Data pipeline | `src/alpha_pulse/data_pipeline/` | Historical and real-time market data fetchers, adapters, and database accessors. |

## API surface

The FastAPI application in `src/alpha_pulse/api/main.py` exposes REST and
WebSocket endpoints.  Core router groups include:

- `/api/v1/portfolio` – portfolio metrics and analytics
- `/api/v1/trades` – trade information and execution helpers
- `/api/v1/regime` – market regime detection status
- `/api/v1/risk-budget` – dynamic risk budgeting actions
- `/api/v1/hedging` – tail-risk hedging operations
- `/api/v1/ensemble` – ensemble configuration and inference
- `/api/v1/data-quality`, `/api/v1/backtesting`, `/api/v1/data-lake` – supporting services

Startup wires ensemble, risk budgeting, regime detection, throttling, caching,
database optimisation, and WebSocket subscriptions.

## Workflow summary

1. **Agents** collect market data (through the data pipeline) and emit trade
   signals.
2. **Risk management** evaluates positions, budgets, and hedging requirements.
3. **Portfolio strategies** combine signals with constraints to produce target
   allocations.
4. **Execution services** route orders via brokers and apply liquidity-aware
   adjustments when enabled.
5. **Monitoring** records metrics, emits alerts, and writes audit events.

## Development notes

- Use `poetry install` and `poetry run uvicorn src.alpha_pulse.api.main:app --reload`
  to start the API locally.
- Unit tests are under `src/alpha_pulse/tests`; run them with `poetry run pytest`.
- Refer to `docs/API_DOCUMENTATION.md` for router-level details and
  `development/CLAUDE.md` for contributor guidelines.
