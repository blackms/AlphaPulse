# Development Environment Setup

This guide provides the minimal steps required to work on AlphaPulse locally.
For broader context see `README.md` and `development/CLAUDE.md`.

## Prerequisites

- Python 3.11 or 3.12
- Poetry 1.6+
- Redis (optional, for rate limiting/caching)
- PostgreSQL (optional, for persistent storage)

## Setup steps

```bash
poetry install
poetry run uvicorn src.alpha_pulse.api.main:app --reload
```

To configure environment variables, copy `.env.example` to `.env` and adjust
credentials/secrets as needed.

For database setup utilities see the scripts in `scripts/` (for example
`create_alphapulse_db.sh`).

Refer to the module-specific READMEs under `src/alpha_pulse/**/README.md` for
feature-specific instructions.
