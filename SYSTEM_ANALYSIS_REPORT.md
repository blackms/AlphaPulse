# AlphaPulse System Analysis

This document summarises the current codebase structure after the documentation
cleanup carried out in 2024.

## High-level architecture

The platform is organised around a FastAPI service (`src/alpha_pulse/api`) which
exposes REST and WebSocket endpoints used by dashboards, monitoring tools, and
automation.  Several background services are initialised during application
start-up, including:

- ensemble signal aggregation
- risk budgeting and tail-risk hedging
- market regime detection
- caching and database optimisation helpers

Core domain logic lives under `src/alpha_pulse`:

| Area                        | Location                                     | Purpose |
|-----------------------------|----------------------------------------------|---------|
| Agents & signal generation | `src/alpha_pulse/agents`                     | Specialised trading agents and manager orchestration |
| Risk management            | `src/alpha_pulse/risk_management`            | Position sizing, risk budgeting, analytics |
| Portfolio management       | `src/alpha_pulse/portfolio`                  | Optimisation strategies and allocation helpers |
| Hedging & execution        | `src/alpha_pulse/hedging`, `src/alpha_pulse/execution` | Execution abstractions and hedging utilities |
| Monitoring & alerting      | `src/alpha_pulse/monitoring`                 | Prometheus metrics, alert routing |
| Machine learning tooling   | `src/alpha_pulse/ml`                         | Reinforcement learning, ensembles, GPU helpers, explainability |
| Data pipeline              | `src/alpha_pulse/data_pipeline`              | Market data retrieval and preprocessing |

Supporting assets reside in:

- `docs/` – feature documentation, security notes, operational guides
- `examples/` – working demos that exercise agents, risk modules, and API calls
- `load-tests/` – k6 scenarios and accompanying README for stress testing
- `dashboard/` – optional React dashboard consuming the API

## Operational dependencies

Running the full stack requires:

- PostgreSQL for persistence (see `scripts/create_alphapulse_db.sh`)
- Redis for caching, rate limiting, and throttling
- Optional: HashiCorp Vault / AWS Secrets Manager for secret storage

Most services degrade gracefully when optional systems are absent, but the best
experience comes from providing real credentials in `.env` or via the secrets
manager interfaces.

## Quality and testing

- Automated tests live under `src/alpha_pulse/tests`.
- `poetry run pytest` executes the suite with coverage configured in
  `pyproject.toml`.
- Linters and static analysis: `poetry run flake8`, `poetry run mypy`.
- Formatting: `poetry run black`.

When adding new features, extend the relevant test modules and update the
documentation index (`docs/README.md`).
