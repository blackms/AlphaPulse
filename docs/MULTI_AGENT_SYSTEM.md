# Multi-Agent Trading System

AlphaPulse organises its trading intelligence around a set of specialised
Python-based agents coordinated by an asynchronous manager.  This note explains
how the pieces fit together.

## Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `AgentManager` | `src/alpha_pulse/agents/manager.py` | Coordinates specialised agents, requests market data, aggregates decisions, and plugs into ensemble services. |
| Specialised agents | `src/alpha_pulse/agents/*.py` | Technical, sentiment, valuation, activist, value, and fundamental agents implement domain-specific logic via the shared interfaces in `src/alpha_pulse/agents/interfaces.py`. |
| Supervisor | `src/alpha_pulse/agents/supervisor/` | Provides higher-level orchestration, performance tracking, and optional human-in-the-loop hooks. |
| Ensemble integration | `src/alpha_pulse/services/ensemble_service.py` | Combines agent outputs using voting/stacking/boosting strategies when configured. |
| Audit wrapper | `src/alpha_pulse/agents/audit_wrapper.py` | Adds audit logging around agent operations. |

## Data flow

1. The manager prepares a `MarketDataPayload` (using data pipeline helpers) and
   dispatches it to enabled agents.
2. Each agent emits `TradeSignal` objects with direction, confidence, and
   metadata.
3. Signals can optionally be passed to the ensemble service for aggregation.
4. The manager returns consolidated signals to downstream consumers (e.g.
   portfolio/risk modules or API endpoints).

## Runtime integration

- The FastAPI service wires the agent manager via dependency injection
  (`src/alpha_pulse/api/dependencies.py`).
- Audit logging is applied through `AuditLoggingMiddleware` and the agent
  wrappers.
- Risk budgeting, regime detection, and hedging services can consume agent
  outputs to trigger portfolio adjustments.

## Extending the system

1. Derive a new agent class from `BaseAgent` (see `interfaces.py`).
2. Implement the asynchronous `generate_signals` method.
3. Register the agent in `AgentFactory` or configure it through the manager.
4. Update relevant tests in `src/alpha_pulse/tests/agents/`.

For additional usage examples, review the scripts under `examples/trading/`.
