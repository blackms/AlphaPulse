# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaPulse is an AI-powered algorithmic trading system designed to operate as an automated hedge fund. It combines multiple specialized AI trading agents, advanced risk management, portfolio optimization, and real-time monitoring capabilities.

## Common Development Commands

### Environment Setup
```bash
poetry install --no-interaction
poetry shell
export PYTHONPATH=./src:$PYTHONPATH
```

### Testing
```bash
# Run all tests with coverage
poetry run pytest --cov-branch --cov-report=xml

# Run specific test file
poetry run pytest src/alpha_pulse/tests/test_specific.py -v

# Run integration tests only
poetry run pytest -m integration
```

### Code Quality
```bash
# Lint with flake8
poetry run flake8 src/alpha_pulse --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format with black
poetry run black src/alpha_pulse

# Type check with mypy
poetry run mypy src/alpha_pulse
```

### Running the Application
```bash
# Run API server
python src/scripts/run_api.py

# Run main trading system
python -m alpha_pulse.main

# Run paper trading demo
python -m alpha_pulse.examples.demo_paper_trading
```

### Database Setup
```bash
./scripts/create_alphapulse_db.sh
./scripts/setup_test_database.sh
```

### Docker Operations
```bash
docker-compose up -d --build
docker-compose logs -f alphapulse
docker-compose down
```

### Dashboard Development
```bash
cd dashboard
npm install
npm start       # Development server
npm run build   # Production build
npm test        # Run tests
```

## Architecture Overview

AlphaPulse follows a 4-layer architecture:

1. **Input Layer**: Signal generation via 6 specialized trading agents
   - Located in `src/alpha_pulse/agents/`
   - Agents: Technical, Fundamental, Sentiment, Value, Activist, Warren Buffett-style

2. **Risk Management Layer**: Signal processing and risk controls
   - Located in `src/alpha_pulse/risk_management/`
   - Features: Position sizing, leverage limits, drawdown protection, stop-loss

3. **Portfolio Management Layer**: Decision making and optimization
   - Located in `src/alpha_pulse/portfolio/`
   - Strategies: MPT, HRP, Black-Litterman, LLM-assisted optimization

4. **Output Layer**: Trade execution
   - Located in `src/alpha_pulse/execution/`
   - Supports paper trading and live trading via CCXT

### Key Components

- **Data Pipeline** (`src/alpha_pulse/data_pipeline/`): Real-time and historical data fetching
- **Exchange Integration** (`src/alpha_pulse/exchanges/`): CCXT adapter for multiple exchanges
- **Monitoring** (`src/alpha_pulse/monitoring/`): Prometheus metrics and multi-channel alerting
- **API** (`src/alpha_pulse/api/`): FastAPI REST API with WebSocket support
- **Dashboard** (`dashboard/`): React/TypeScript frontend application
- **RL Trading** (`src/alpha_pulse/rl/`): Reinforcement learning trading agents
- **Backtesting** (`src/alpha_pulse/backtesting/`): Historical strategy testing

## Development Guidelines

1. **Component-Based Architecture**: Each trading agent, risk module, and portfolio optimizer is a separate component with clear interfaces.

2. **Risk-First Approach**: Always consider risk management implications when implementing trading features. Never bypass risk checks.

3. **Testing Requirements**: 
   - Write unit tests for all new components
   - Include integration tests for API endpoints
   - Test with paper trading before live trading

4. **Security Practices**:
   - Never hardcode API credentials
   - Use the credential management system in `src/alpha_pulse/exchanges/credentials/`
   - Follow JWT authentication patterns for API endpoints

5. **Performance Considerations**:
   - Optimize for real-time data processing
   - Use async/await for I/O operations
   - Leverage caching where appropriate

6. **Database Operations**:
   - Use SQLAlchemy ORM for database interactions
   - Create Alembic migrations for schema changes
   - TimescaleDB is used for time-series data

7. **Configuration Management**:
   - Use YAML configuration files in `config/`
   - Support environment variable overrides
   - Separate configs for development/testing/production

8. **Monitoring and Logging**:
   - Use structured logging with appropriate levels
   - Expose Prometheus metrics for key operations
   - Implement health checks for all services

## Important Context

- This is a solo developer project designed for AI-assisted development
- The system is classified as "Complex" (Cynefin framework) due to multiple interacting components
- Follow domain-driven design principles
- The project uses Poetry for dependency management (Python 3.11+)
- Frontend uses React with TypeScript and Material-UI
- All trading strategies must be thoroughly backtested before deployment
- The system supports multiple exchanges through CCXT
- Real-time monitoring is critical for production use