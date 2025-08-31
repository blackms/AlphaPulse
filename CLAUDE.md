# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaPulse is a sophisticated AI-driven hedge fund system that combines multiple specialized trading agents, advanced risk management, portfolio optimization, and real-time monitoring. The system is designed for cryptocurrency markets but supports multiple asset classes.

## Key Commands

### Build and Run
```bash
# Install dependencies
poetry install --no-interaction
poetry shell

# Run API server
python src/scripts/run_api.py

# Run main trading system
python -m alpha_pulse.main

# Run dashboard
cd dashboard && npm start
```

### Testing
```bash
# Run tests with coverage
poetry run pytest --cov=src/alpha_pulse --cov-report=xml

# Run specific test file
poetry run pytest src/alpha_pulse/tests/test_specific.py -v

# Run integration tests
poetry run pytest -m integration
```

### Code Quality
```bash
# Lint
poetry run flake8 src/alpha_pulse --count --exit-zero --max-complexity=10 --max-line-length=127

# Format
poetry run black src/alpha_pulse

# Type check
poetry run mypy src/alpha_pulse
```

### Database Setup
```bash
./scripts/create_alphapulse_db.sh
```

### Dashboard Commands
```bash
cd dashboard
npm test        # Run tests
npm run lint    # Lint code
npm run build   # Production build
```

## Architecture Overview

The system follows a 4-layer architecture:

1. **Input Layer** (`src/alpha_pulse/agents/`): 6 specialized AI agents generate trading signals
   - Technical, Fundamental, Sentiment, Value, Activist, Warren Buffett agents
   
2. **Risk Management Layer** (`src/alpha_pulse/risk_management/`): Controls and risk limits
   - Position sizing, leverage limits, drawdown protection, stop-loss
   
3. **Portfolio Management Layer** (`src/alpha_pulse/portfolio/`): Optimization strategies
   - Mean-variance, Risk Parity, Hierarchical Risk Parity, Black-Litterman
   
4. **Output Layer** (`src/alpha_pulse/execution/`): Trade execution
   - Paper trading and live trading via CCXT

### Critical Components

- **Market Regime Detection** (`src/alpha_pulse/services/regime_detection_service.py`): HMM-based market state classification
- **Correlation Analysis** (`src/alpha_pulse/analysis/correlation_analysis.py`): Advanced correlation with tail dependencies
- **Caching Service** (`src/alpha_pulse/services/caching_service.py`): Multi-tier Redis caching
- **API** (`src/alpha_pulse/api/`): FastAPI with WebSocket support
- **Dashboard** (`dashboard/`): React/TypeScript frontend

## Important Integration Notes

**Current Status**: Many sophisticated features exist but are NOT integrated into the main system flow. Before implementing new features, check if they already exist in the codebase.

Key unintegrated components that need attention:
- HMM regime detection service (implemented but not started in main API)
- Advanced correlation analysis
- Ensemble ML methods
- Online learning capabilities

## Development Tips

1. **Always check existing code first** - Many features are already implemented but not integrated
2. **Use async/await** for I/O operations to maintain performance
3. **Follow risk-first approach** - Never bypass risk management checks
4. **Test with paper trading** before enabling live trading
5. **Use Poetry** for dependency management (Python 3.11+)
6. **Configure via YAML** files in `config/` directory
7. **Set environment variables** for credentials (never hardcode)
8. **Monitor with Prometheus** metrics exposed on port 8000

## Testing Strategy

- Unit tests for all new components
- Integration tests for API endpoints  
- Paper trading validation before live deployment
- Run tests with `poetry run pytest` before committing

## Security Requirements

- Use credential management system in `src/alpha_pulse/exchanges/credentials/`
- JWT authentication for API endpoints
- Never commit API keys or secrets
- Use environment variables for sensitive configuration