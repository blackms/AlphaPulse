# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/alpha_pulse/` (domain modules like `api/`, `backtesting/`, `ml/`, `services/`).
- API entrypoint: `src/alpha_pulse/api/main.py` (FastAPI app `app`).
- Tests: `src/alpha_pulse/tests/` (pytest). Examples: `src/alpha_pulse/examples/` and top-level `examples/`.
- Migrations: `migrations/` (alembic). Config: `alembic.ini`.
- Ops/CI: `.github/`, `docker-compose*.yml`, `Dockerfile`, `scripts/`.

## Build, Test, and Development Commands
- Install deps: `poetry install` (Python >=3.11). CI uses `pyproject.toml`.
- Run API (dev): `poetry run uvicorn src.alpha_pulse.api.main:app --reload`.
- Test suite: `poetry run pytest` (uses `pytest.ini` and coverage to XML).
- Lint/format: `poetry run flake8`, `poetry run black .`, `poetry run mypy src`.
- DB migrate: `alembic upgrade head` (set env via `.env` or Docker).
- Docker (local stack): `docker-compose up -d`.

## Coding Style & Naming Conventions
- Python: Black formatting (PEP8), 4-space indent, type hints required in new/changed code.
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Imports: absolute within `alpha_pulse` package; group stdlib/third-party/local.
- Keep modules focused; prefer `services/` for orchestration and `utils/` for pure helpers.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`, `pytest-cov`.
- Location/patterns: place tests in `src/alpha_pulse/tests/` named `test_*.py`.
- Markers: use `@pytest.mark.unit|integration|api|ml|slow`. Example: `pytest -m "not slow"`.
- Coverage: respect configured `--cov=src/alpha_pulse`; add focused tests for new logic.

## Commit & Pull Request Guidelines
- Conventional Commits: `feat: ...`, `fix: ...`, `chore(deps): ...`, `test: ...` (seen in history).
- PRs: include clear description, rationale, linked issues, and test results. For UI/dashboard changes, add screenshots.
- CI: ensure linters/tests pass locally. Avoid committing secrets; use `.env.example` and secure managers (Vault/AWS Secrets Manager).

## Security & Configuration Tips
- Never commit real credentials. Derive config from `.env` and `config/` defaults; see `AGENT_INSTRUCTIONS.yaml` for integrations.
- When running locally, set minimal `.env` (DB URL, secrets). For API, verify `/health` and `/docs` after startup.

