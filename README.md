# AlphaPulse

AlphaPulse is an algorithmic trading research platform that bundles together
multiple AI trading agents, risk-control utilities, data pipelines, and a public
FastAPI service.  The codebase is organised so individual subsystems can be
experimented with in isolation or composed into larger trading workflows.

## Highlights

- **Multi-agent signal generation** – specialised agents (technical, sentiment,
  value, activist, etc.) coordinate through an asynchronous agent manager.
- **Risk and portfolio controls** – position sizing, budgeting, hedging, and
  liquidity tooling live under `src/alpha_pulse/risk_management` and
  `src/alpha_pulse/portfolio`.
- **FastAPI surface** – `src/alpha_pulse/api` exposes REST and WebSocket
  endpoints for metrics, portfolio analytics, backtesting hooks, and ensemble
  orchestration.
- **ML & simulation utilities** – reinforcement learning, ensemble methods,
  distributed backtesting, GPU helpers, and online learning reside in
  `src/alpha_pulse/ml` and related packages.
- **Monitoring & operations** – Prometheus metrics, alert routing, audit
  logging, and exchange synchronisation are available in dedicated modules.

## Repository layout

```
docs/                         Project and subsystem documentation
examples/                     End-to-end and feature-specific demos
src/alpha_pulse/              Python packages for API, agents, services, etc.
src/scripts/                  Helper entry points (e.g. run_api.py)
dashboard/                    React dashboard (optional)
load-tests/                   k6 load testing scripts and notes
```

See `docs/index.md` for the currently curated documentation index.

## Requirements

- Python **3.11** or **3.12**
- [Poetry](https://python-poetry.org/) 1.6 or newer
- Node.js 16+ (dashboard optional)

## Getting started

```bash
# install dependencies
poetry install

# create a shell with the virtual environment activated
poetry shell

# copy base environment configuration and edit as needed
cp .env.example .env
```

Populate secrets (exchange credentials, database URLs, etc.) via environment
variables or the secrets manager abstractions configured in
`src/alpha_pulse/config`.

## Running services

```bash
# launch the FastAPI server with auto-reload
poetry run uvicorn src.alpha_pulse.api.main:app --reload

# start the sample trading system runner
poetry run python src/scripts/run_api.py
```

Open http://localhost:8000/docs for interactive API documentation once the
server is running.

For live trading or paper-testing integrations, review the module specific
README files in `src/alpha_pulse/**/README.md`.

## Testing and quality

```bash
# execute the pytest suite with coverage configuration defined in pyproject.toml
poetry run pytest

# formatting, linting, type checking
poetry run black src/alpha_pulse
poetry run flake8 src/alpha_pulse
poetry run mypy src/alpha_pulse
```

If you modify documentation or configuration files, ensure referenced commands
match the Poetry-based workflow.

## Documentation

- `CLAUDE.md` – working notes for AI copilots and contributors
- `docs/README.md` – curated links to subsystem documentation
- `docs/audit-logging.md`, `docs/database-encryption.md`, `docs/ensemble-methods.md`,
  etc. – deep dives into specific features

Whenever you add APIs, agents, or services, update the relevant documentation
and extend the test suite before submitting changes.
