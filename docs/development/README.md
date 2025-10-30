# Development Documentation

This directory contains supplementary guides for contributors working on
AlphaPulse.  The primary reference is [CLAUDE.md](CLAUDE.md); the notes below
link to additional resources.

## Quick access
- [Debug Tools](../DEBUG_TOOLS.md)
- [Migration Guide](../migration-guide.md)
- [API Documentation](../API_DOCUMENTATION.md)
- [System Architecture](../SYSTEM_ARCHITECTURE.md)

## Environment recap

```bash
poetry install
poetry run uvicorn src.alpha_pulse.api.main:app --reload
```

Optional services (PostgreSQL, Redis, Vault) can be enabled locally using the
scripts in `scripts/`.  See module-specific READMEs under `src/alpha_pulse` for
feature guidance.

## Quality checks

```bash
poetry run black src/alpha_pulse
poetry run flake8 src/alpha_pulse
poetry run mypy src/alpha_pulse
poetry run pytest
```

When adding new modules, extend the relevant tests in `src/alpha_pulse/tests`
and update the documentation index in `docs/index.md`.
