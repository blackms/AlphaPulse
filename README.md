<h1 align="center">AlphaPulse ⚡</h1>

<p align="center">
  <strong>Production-grade autonomous AI hedge fund platform</strong><br/>
  Multi-asset trading • Intelligent risk management • Real-time portfolio optimization
</p>

<p align="center">
  <a href="https://github.com/blackms/AlphaPulse/actions/workflows/ci-enhanced.yml">
    <img alt="CI Status" src="https://github.com/blackms/AlphaPulse/actions/workflows/ci-enhanced.yml/badge.svg?branch=main&label=CI">
  </a>
  <a href="https://github.com/blackms/AlphaPulse/actions/workflows/python-app.yml">
    <img alt="Test Suite" src="https://github.com/blackms/AlphaPulse/actions/workflows/python-app.yml/badge.svg?branch=main&label=Tests">
  </a>
  <a href="https://codecov.io/gh/blackms/AlphaPulse">
    <img alt="Coverage" src="https://codecov.io/gh/blackms/AlphaPulse/branch/main/graph/badge.svg">
  </a>
  <a href="https://github.com/blackms/AlphaPulse/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg">
  </a>
  <a href="https://www.python.org/">
    <img alt="Python Versions" src="https://img.shields.io/badge/python-3.11%20|%203.12-3776AB.svg">
  </a>
  <a href="https://python-poetry.org/">
    <img alt="Poetry" src="https://img.shields.io/badge/poetry-1.6%2B-60A5FA.svg">
  </a>
  <img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20|%20macOS%20|%20Docker-lightgrey">
</p>

<p align="center">
  <a href="#-quick-demo">Quick Demo</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📌 Overview

**AlphaPulse** is an enterprise-grade, AI-powered hedge fund platform that orchestrates sophisticated trading strategies across multiple asset classes. Built with Python 3.11+, FastAPI, and cutting-edge ML frameworks, it provides a complete ecosystem for quantitative research, autonomous trading, and institutional-grade risk management.

The platform combines **6 specialized AI agents** with advanced portfolio optimization, real-time market regime detection, and explainable ML to generate, validate, and execute trading signals with institutional rigor. Every component is designed for production deployment, from multi-tenant isolation and audit logging to distributed computing and comprehensive monitoring.

### Why AlphaPulse?

- **🤖 Autonomous Intelligence**: Six specialized AI agents (Technical, Fundamental, Sentiment, Value, Activist, Buffett-style) working in concert to identify alpha across market conditions
- **🛡️ Risk-First Design**: Multi-layer risk controls with position sizing, leverage limits, drawdown protection, and dynamic hedging strategies
- **📊 Portfolio Optimization**: Advanced strategies including Mean-Variance, Risk Parity, Hierarchical Risk Parity, and Black-Litterman models
- **🔍 Explainable AI**: SHAP and LIME integrations ensure every trading decision can be explained and audited
- **⚡ Production-Ready**: Multi-tenant architecture, comprehensive audit trails, Prometheus metrics, and load-tested for institutional scale
- **🌐 Multi-Asset Support**: Trade cryptocurrencies, equities, forex, and derivatives through unified CCXT integration

> 🎯 **Perfect for**: Quantitative researchers building systematic strategies, ML engineers deploying trading models, portfolio managers requiring institutional risk controls, and platform teams operating production trading infrastructure.

## 🗂️ Table of Contents

- [📌 Overview](#-overview)
- [🚀 Quick Demo](#-quick-demo)
- [✨ Features](#-features)
- [🏁 Installation](#-installation)
- [🎯 Quick Start](#-quick-start)
- [🧠 Core Capabilities](#-core-capabilities)
- [🏗️ Architecture](#-architecture)
- [🔧 Configuration](#-configuration)
- [🧪 Testing & Quality](#-testing--quality)
- [📈 Monitoring & Operations](#-monitoring--operations)
- [📚 Documentation](#-documentation)
- [🗺️ Roadmap & Releases](#-roadmap--releases)
- [🤝 Contributing](#-contributing)
- [🛡️ Security](#-security)
- [📄 License](#-license)
- [💬 Community & Support](#-community--support)

## 🚀 Quick Demo

```bash
# 1. Clone and setup
git clone https://github.com/blackms/AlphaPulse.git
cd AlphaPulse
poetry install && cp .env.example .env

# 2. Launch the platform
poetry run uvicorn src.alpha_pulse.api.main:app --reload

# 3. Open your browser
open http://localhost:8000/docs  # Interactive API documentation
```

**Try it out**: The API comes with interactive Swagger UI where you can test endpoints, view portfolio analytics, and monitor trading agents in real-time.

## ✨ Features

### 🤖 Multi-Agent Intelligence System

**Six Specialized AI Agents** working autonomously and collaboratively:

- **Technical Agent**: Price action, momentum indicators, trend analysis, pattern recognition
- **Fundamental Agent**: Financial metrics, earnings analysis, balance sheet evaluation
- **Sentiment Agent**: News analysis, social media sentiment, market psychology indicators
- **Value Agent**: Intrinsic value calculation, relative valuation, margin of safety
- **Activist Agent**: Institutional holdings, insider trading, shareholder activism signals
- **Buffett Agent**: Quality moats, competitive advantages, long-term value investing principles

Each agent generates weighted signals with confidence scores, enabling sophisticated ensemble decision-making.

### 🛡️ Institutional-Grade Risk Management

**Multi-Layer Risk Controls**:
- **Position Sizing**: Kelly Criterion, Volatility-based, Fixed Fractional, Dynamic sizing
- **Risk Budgeting**: VaR limits, Expected Shortfall, risk allocation across strategies
- **Drawdown Protection**: Maximum drawdown limits, dynamic exposure reduction
- **Leverage Management**: Portfolio-level and position-level leverage constraints
- **Dynamic Hedging**: Options hedging, tail risk protection, correlation-based hedging
- **Stop-Loss Systems**: Time-based, volatility-adjusted, trailing stops

### 📊 Advanced Portfolio Optimization

**Multiple Optimization Strategies**:
- **Mean-Variance Optimization**: Classic Markowitz portfolio construction with constraints
- **Risk Parity**: Equal risk contribution from each position
- **Hierarchical Risk Parity (HRP)**: Cluster-based diversification using machine learning
- **Black-Litterman**: Bayesian approach combining market equilibrium with agent views
- **Custom Optimization**: Extensible framework for proprietary strategies

### 🔍 Market Regime Detection

**Hidden Markov Model (HMM)** based regime classification:
- Identifies bull, bear, high-volatility, low-volatility market states
- Dynamic strategy adaptation based on regime transitions
- Real-time regime probability monitoring
- Historical regime analysis and backtesting

### 🎯 Explainable AI & Model Transparency

**Every decision is auditable**:
- SHAP (SHapley Additive exPlanations) for feature importance
- LIME (Local Interpretable Model-agnostic Explanations) for individual predictions
- Comprehensive audit logs tracking agent decisions, risk adjustments, and executions
- Detailed performance attribution by agent and strategy

### ⚡ Production Infrastructure

**Enterprise-ready deployment**:
- **Multi-Tenant Architecture**: Secure tenant isolation with Row-Level Security (RLS)
- **FastAPI Backend**: Async, high-performance REST and WebSocket APIs
- **Distributed Computing**: Ray and Dask integration for parallel backtesting and optimization
- **Caching Layer**: Redis-based multi-tier caching for market data and signals
- **Message Queue**: Asynchronous task processing with Celery
- **Monitoring**: Prometheus metrics, Grafana dashboards, custom alerting
- **Observability**: Structured logging, distributed tracing, performance profiling

### 📈 Real-Time Dashboard

**React/TypeScript operations dashboard**:
- Live portfolio monitoring with WebSocket updates
- Agent signal visualization and performance tracking
- Risk metrics and exposure analysis
- Trade execution monitoring and order management
- Alert management and notification center
- Historical performance analytics

## 🏁 Installation

### Prerequisites

- **Python**: 3.11 or 3.12
- **Poetry**: 1.6+ for dependency management
- **Docker** (optional): For running PostgreSQL, Redis, and other services
- **Node.js**: 18+ (for dashboard development)

### Option 1: Quick Start (API Only)

Perfect for exploring the API and testing agents without infrastructure dependencies:

```bash
# Clone the repository
git clone https://github.com/blackms/AlphaPulse.git
cd AlphaPulse

# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration (API keys, database URLs, etc.)

# Launch the API server
poetry run uvicorn src.alpha_pulse.api.main:app --reload
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

### Option 2: Full Stack Deployment

For production-like setup with PostgreSQL, Redis, Prometheus, and MLflow:

```bash
# Install dependencies
poetry install

# Start infrastructure services
docker-compose -f docker-compose.dev.yml up -d

# Initialize database
poetry run alembic upgrade head
poetry run python src/scripts/init_db.py

# Launch the API
poetry run uvicorn src.alpha_pulse.api.main:app --reload

# (Optional) Launch the dashboard
cd dashboard
npm install
npm start
```

**Services will be available at**:
- API: http://localhost:8000
- Dashboard: http://localhost:3000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001
- MLflow: http://localhost:5000

### Option 3: Development Environment

For active development with all quality tools:

```bash
# Install all dependencies including dev tools
poetry install --with dev

# Activate virtual environment
poetry shell

# Run quality checks
poetry run pytest                    # Run test suite
poetry run black src/alpha_pulse     # Format code
poetry run flake8 src/alpha_pulse    # Lint
poetry run mypy src/alpha_pulse      # Type check
```

## 🎯 Quick Start

### Running Your First Trading Strategy

```python
from alpha_pulse.agents.agent_manager import AgentManager
from alpha_pulse.risk_management.risk_manager import RiskManager
from alpha_pulse.portfolio.portfolio_optimizer import PortfolioOptimizer

# Initialize components
agent_manager = AgentManager()
risk_manager = RiskManager()
portfolio_optimizer = PortfolioOptimizer(strategy="hierarchical_risk_parity")

# Get signals from all agents
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
signals = await agent_manager.aggregate_signals(symbols)

# Apply risk management
risk_adjusted_signals = risk_manager.apply_risk_controls(signals)

# Optimize portfolio allocation
optimal_portfolio = portfolio_optimizer.optimize(
    signals=risk_adjusted_signals,
    constraints={"max_position_size": 0.2, "max_leverage": 2.0}
)

print(optimal_portfolio)
```

### Using the REST API

```bash
# Get portfolio summary
curl -X GET "http://localhost:8000/portfolio/summary" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Get agent signals for a symbol
curl -X POST "http://localhost:8000/agents/signals" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTC/USDT"], "timeframe": "1h"}'

# Execute a trade
curl -X POST "http://localhost:8000/execution/trade" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USDT",
    "side": "buy",
    "amount": 0.01,
    "order_type": "limit",
    "price": 45000
  }'
```

### WebSocket Real-Time Updates

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/portfolio');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Portfolio update:', data);
};
```

See [API Documentation](docs/API_DOCUMENTATION.md) for complete endpoint reference.

## 🧠 Core Capabilities

### 🤖 Multi-Agent Intelligence System

AlphaPulse orchestrates **6 specialized AI agents** through a sophisticated manager in [src/alpha_pulse/agents/](src/alpha_pulse/agents/):

**Agent Architecture**:
- Each agent specializes in a specific market analysis domain
- Agents generate weighted signals with confidence scores (0-100)
- Signal aggregation uses ensemble methods (voting, stacking, boosting)
- Adaptive agent weighting based on historical performance

**Signal Flow**:
1. Market data ingestion from multiple sources (CCXT, APIs, websockets)
2. Parallel agent execution across selected symbols
3. Signal normalization and confidence scoring
4. Ensemble aggregation with conflict resolution
5. Risk-adjusted signal output

See [Component Architecture](docs/SYSTEM_ARCHITECTURE.md#component-architecture) and [Workflow Summary](docs/AI_HEDGE_FUND_DOCUMENTATION.md#workflow-summary) for detailed signal flow diagrams.

### 🛡️ Risk Management & Portfolio Governance

**Risk Management Layer** ([src/alpha_pulse/risk_management/](src/alpha_pulse/risk_management/)):
- **Position Sizing**: Dynamic sizing based on volatility, correlation, and confidence
- **Risk Budgeting**: Allocate risk capital across strategies with VaR/CVaR constraints
- **Drawdown Protection**: Automatic exposure reduction on portfolio drawdowns
- **Leverage Control**: Portfolio-level and position-level leverage limits
- **Dynamic Hedging**: Options strategies and correlation-based hedging
- **Stop-Loss Management**: Volatility-adjusted and time-based stops

**Portfolio Management** ([src/alpha_pulse/portfolio/](src/alpha_pulse/portfolio/)):
- Mean-Variance Optimization with custom constraints
- Risk Parity for balanced risk contribution
- Hierarchical Risk Parity using machine learning clustering
- Black-Litterman model incorporating agent views
- Multi-objective optimization (Sharpe, Sortino, Calmar ratios)

All risk controls enforce limits **before** execution, with comprehensive audit trails. See [Risk Management Layer](docs/SYSTEM_ARCHITECTURE.md#component-interactions) for control flow details.

### 🎓 Ensemble Intelligence & Explainability

**Ensemble Methods** ([docs/ensemble-methods.md](docs/ensemble-methods.md)):
- **Voting Ensembles**: Majority voting, weighted voting, soft voting
- **Stacking**: Meta-learner combines base agent predictions
- **Boosting**: Sequential agent weighting based on errors
- **Adaptive Weighting**: Performance-based agent weight adjustment

**Explainable AI**:
- SHAP analysis for feature importance across agents
- LIME for individual prediction explanations
- Agent contribution tracking for performance attribution
- Decision path visualization in dashboard

### 📊 Market Regime Detection

**Hidden Markov Model (HMM)** implementation ([src/alpha_pulse/services/regime_detection_service.py](src/alpha_pulse/services/regime_detection_service.py)):

**Detected Regimes**:
- Bull Market (trending up, low volatility)
- Bear Market (trending down, high volatility)
- High Volatility (choppy, uncertain)
- Low Volatility (range-bound, quiet)

**Features**:
- Real-time regime probability calculation
- Regime transition detection and alerting
- Strategy adaptation based on current regime
- Historical regime analysis and backtesting

**Integration Status**: Service initialized on API startup, 10% integrated with dashboard. See [Market Regime Detection Overview](docs/regime-detection.md#overview) for roadmap.

### ⚡ Distributed Computing & Scalability

**Ray Integration** ([docs/distributed-computing.md](docs/distributed-computing.md)):
- Parallel backtesting across multiple strategies and parameters
- Distributed hyperparameter tuning with Ray Tune
- Auto-scaling Ray clusters (local, AWS, GCP)
- GPU-accelerated model training

**Dask Integration**:
- Large-scale data processing pipelines
- Out-of-core computation for big datasets
- Parallel feature engineering

**Use Cases**:
- Walk-forward optimization across years of data
- Monte Carlo simulation (1000+ scenarios)
- Hyperparameter grid search with early stopping
- Multi-asset portfolio optimization

### 📈 Observability & Audit System

**Audit Logging** ([docs/audit-logging.md](docs/audit-logging.md)):
- Comprehensive audit taxonomy (agent decisions, risk adjustments, trades)
- Request-scoped context propagation
- Tenant-aware audit trails with RLS
- Structured logging with correlation IDs

**Monitoring** ([docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#monitoring-and-metrics)):
- **Prometheus Metrics**: Request latency, error rates, agent performance
- **Grafana Dashboards**: Real-time system health and trading metrics
- **MLflow Integration**: Model versioning, experiment tracking, performance comparison
- **Custom Alerts**: Drawdown alerts, risk limit breaches, system health

### 🎮 Human-in-the-Loop Dashboard

**React Dashboard** ([dashboard/README.md](dashboard/README.md)):
- Real-time portfolio monitoring with WebSocket updates
- Agent signal visualization and performance comparison
- Risk metrics and exposure breakdown
- Trade execution monitoring and order management
- Alert management and notification center
- Historical performance analytics with interactive charts

### 🔬 Load Testing & Performance Validation

**k6 Load Tests** ([load-tests/README.md](load-tests/README.md)):
- **Smoke Test**: Basic functionality validation
- **Target Capacity**: 500 concurrent users, p99 < 200ms
- **Stress Test**: Breaking point identification
- **Soak Test**: 24-hour stability validation

**Performance Criteria**:
- API latency p99 < 200ms
- Error rate < 0.1%
- Throughput > 1000 req/s
- WebSocket concurrency > 500 connections

## 🏗️ Architecture

### System Overview

AlphaPulse follows a **4-layer architecture** designed for modularity, testability, and production deployment:

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER (Agents)                     │
│  Technical • Fundamental • Sentiment • Value • Activist • Buffett │
└────────────────────┬────────────────────────────────────────┘
                     │ Signals (weighted, confidence-scored)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              RISK MANAGEMENT LAYER                           │
│  Position Sizing • Risk Budgeting • Drawdown Protection      │
│  Leverage Control • Dynamic Hedging • Stop-Loss              │
└────────────────────┬────────────────────────────────────────┘
                     │ Risk-adjusted signals
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           PORTFOLIO MANAGEMENT LAYER                         │
│  Mean-Variance • Risk Parity • HRP • Black-Litterman        │
│  Multi-objective Optimization • Rebalancing                  │
└────────────────────┬────────────────────────────────────────┘
                     │ Optimal portfolio weights
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              EXECUTION LAYER                                 │
│  Order Management • CCXT Integration • Paper Trading         │
│  Slippage Modeling • Execution Analytics                     │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```text
AlphaPulse/
├── src/alpha_pulse/
│   ├── api/                        # FastAPI application
│   │   ├── main.py                # API entry point, startup/shutdown
│   │   ├── routers/               # REST endpoints (portfolio, agents, risk)
│   │   ├── websockets/            # WebSocket handlers for real-time updates
│   │   └── middleware/            # Auth, CORS, logging, error handling
│   │
│   ├── agents/                     # AI Trading Agents
│   │   ├── agent_manager.py       # Orchestrates all agents
│   │   ├── technical_agent.py     # Technical analysis
│   │   ├── fundamental_agent.py   # Fundamental analysis
│   │   ├── sentiment_agent.py     # Sentiment analysis
│   │   ├── value_agent.py         # Value investing
│   │   ├── activist_agent.py      # Activist/institutional signals
│   │   └── buffett_agent.py       # Warren Buffett-style analysis
│   │
│   ├── risk_management/            # Risk Control Systems
│   │   ├── risk_manager.py        # Central risk management
│   │   ├── position_sizer.py      # Position sizing algorithms
│   │   ├── risk_budgeting.py      # VaR/CVaR allocation
│   │   ├── hedging_service.py     # Dynamic hedging strategies
│   │   └── stop_loss.py           # Stop-loss management
│   │
│   ├── portfolio/                  # Portfolio Optimization
│   │   ├── portfolio_optimizer.py # Optimization orchestrator
│   │   ├── mean_variance.py       # Markowitz optimization
│   │   ├── risk_parity.py         # Risk parity allocation
│   │   ├── hierarchical_rp.py     # Hierarchical risk parity
│   │   └── black_litterman.py     # Bayesian portfolio construction
│   │
│   ├── execution/                  # Trade Execution
│   │   ├── order_manager.py       # Order lifecycle management
│   │   ├── executor.py            # CCXT integration
│   │   └── paper_trading.py       # Simulated execution
│   │
│   ├── ml/                         # Machine Learning
│   │   ├── models/                # Model implementations
│   │   ├── trainers/              # Training pipelines
│   │   ├── explainability/        # SHAP, LIME integrations
│   │   └── ensemble/              # Ensemble methods
│   │
│   ├── services/                   # Cross-Cutting Services
│   │   ├── regime_detection_service.py  # HMM regime detection
│   │   ├── caching_service.py     # Redis caching layer
│   │   ├── data_service.py        # Market data aggregation
│   │   ├── monitoring_service.py  # Prometheus metrics
│   │   └── alert_service.py       # Alert management
│   │
│   ├── data_pipeline/              # Data Ingestion
│   │   ├── ccxt_adapter.py        # Exchange data via CCXT
│   │   ├── websocket_client.py    # Real-time data streams
│   │   └── data_warehouse.py      # Historical data storage
│   │
│   ├── backtesting/                # Backtesting Engine
│   │   ├── backtest_engine.py     # Simulation orchestrator
│   │   ├── evaluation.py          # Performance metrics
│   │   └── walk_forward.py        # Walk-forward optimization
│   │
│   ├── config/                     # Configuration Management
│   │   ├── settings.py            # Pydantic settings models
│   │   ├── secrets/               # Secret providers (Vault, AWS)
│   │   └── environments/          # Environment-specific configs
│   │
│   └── tests/                      # Test Suite
│       ├── unit/                  # Unit tests
│       ├── integration/           # Integration tests
│       └── performance/           # Load and performance tests
│
├── dashboard/                      # React Dashboard
│   ├── src/
│   │   ├── components/            # React components
│   │   ├── services/              # API clients, WebSocket handlers
│   │   ├── hooks/                 # Custom React hooks
│   │   └── pages/                 # Page components
│   └── public/                    # Static assets
│
├── docs/                           # Documentation
│   ├── SYSTEM_ARCHITECTURE.md     # System design and diagrams
│   ├── AI_HEDGE_FUND_DOCUMENTATION.md  # Trading workflow
│   ├── API_DOCUMENTATION.md       # API reference
│   ├── ensemble-methods.md        # Ensemble strategies
│   ├── regime-detection.md        # Regime detection guide
│   ├── distributed-computing.md   # Ray/Dask usage
│   ├── audit-logging.md           # Audit system
│   ├── security.md                # Security practices
│   └── DEPLOYMENT.md              # Deployment guide
│
├── examples/                       # Example Scripts
│   ├── trading_example.py         # End-to-end trading workflow
│   ├── portfolio_optimization.py  # Portfolio optimization demo
│   └── backtesting_example.py     # Backtesting example
│
├── load-tests/                     # Performance Testing
│   ├── k6/                        # k6 load test scenarios
│   └── results/                   # Test results and reports
│
├── scripts/                        # Utility Scripts
│   ├── init_db.py                 # Database initialization
│   ├── migrate_secrets.py         # Secret migration
│   └── create_alphapulse_db.sh    # Database setup
│
├── migrations/                     # Alembic Migrations
├── config/                         # YAML Configuration Files
├── pyproject.toml                  # Poetry dependencies
├── pytest.ini                      # Pytest configuration
├── docker-compose.dev.yml          # Development stack
└── .env.example                    # Environment template
```

### Key Design Principles

1. **Separation of Concerns**: Each layer has clear responsibilities and interfaces
2. **Risk-First**: All signals pass through risk management before execution
3. **Observable**: Comprehensive logging, metrics, and audit trails
4. **Scalable**: Async I/O, distributed computing, caching strategies
5. **Testable**: >15% test coverage with unit, integration, and performance tests
6. **Multi-Tenant**: Secure tenant isolation at database and application layers

For detailed architecture diagrams, see [System Overview](docs/SYSTEM_ARCHITECTURE.md#system-overview) and [Layered Architecture](docs/AI_HEDGE_FUND_DOCUMENTATION.md#layered-architecture).

## 🔧 Configuration

### Environment Variables

AlphaPulse uses environment variables for configuration. Create a `.env` file from the template:

```bash
cp .env.example .env
```

**Key Configuration Areas**:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/alphapulse
REDIS_URL=redis://localhost:6379/0

# API Keys (Exchange)
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret

# Security
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30

# Multi-Tenant
TENANT_ID=00000000-0000-0000-0000-000000000001

# ML & Monitoring
MLFLOW_TRACKING_URI=http://localhost:5000
PROMETHEUS_PORT=9090

# Feature Flags
ENABLE_PAPER_TRADING=true
ENABLE_REGIME_DETECTION=true
ENABLE_RISK_BUDGETING=true
```

### YAML Configuration Files

Agent and strategy configurations are in `config/` directory:

- `agent_config.yaml`: Agent weights, confidence thresholds
- `risk_config.yaml`: Risk limits, position sizing parameters
- `portfolio_config.yaml`: Optimization constraints, rebalancing rules
- `execution_config.yaml`: Order types, slippage models

See [Environment Variables](docs/DEPLOYMENT.md#environment-variables) and [Secret Management](docs/security.md#secret-management) for details.

## 🧪 Testing & Quality

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test categories
poetry run pytest -m unit              # Unit tests only
poetry run pytest -m integration       # Integration tests
poetry run pytest -m "not slow"        # Skip slow tests

# Run with coverage
poetry run pytest --cov=src/alpha_pulse --cov-report=html

# Run specific test file
poetry run pytest src/alpha_pulse/tests/test_agent_manager.py -v
```

### Code Quality Checks

```bash
# Format code
poetry run black src/alpha_pulse

# Lint
poetry run flake8 src/alpha_pulse

# Type checking
poetry run mypy src/alpha_pulse

# Security scan
poetry run bandit -r src/alpha_pulse
```

### Test Coverage

Current coverage: **>15%** (target: 30% patch coverage)

Coverage is enforced via [codecov.yml](codecov.yml):
- Project coverage: 15% (informational)
- Patch coverage: 30% (non-blocking)
- CI integration via GitHub Actions

See [Quality Checks](docs/development/README.md#quality-checks) for the complete quality workflow.

## 📈 Monitoring & Operations

### Deployment Options

**Option 1: Docker Compose (Recommended for development)**
```bash
docker-compose -f docker-compose.dev.yml up -d
```

**Option 2: Kubernetes** (see [Deployment Guide](docs/DEPLOYMENT.md#building-and-deploying))
```bash
helm install alphapulse ./helm/alphapulse
```

**Option 3: Bare Metal**
```bash
poetry run uvicorn src.alpha_pulse.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Observability Stack

**Prometheus Metrics** (`:8000/metrics`):
- Request latency (p50, p95, p99)
- Error rates by endpoint
- Agent execution time
- Portfolio metrics (PnL, Sharpe, drawdown)
- System resources (CPU, memory)

**Grafana Dashboards**:
- System health overview
- Trading performance
- Agent signal quality
- Risk metrics monitoring

**MLflow** (`:5000`):
- Model versioning and registry
- Experiment tracking
- Parameter comparison
- Performance benchmarking

### Operational Playbooks

- **Deployment**: [Deployment Guide](docs/DEPLOYMENT.md#building-and-deploying) — Docker, Kubernetes, production hardening
- **Secrets Management**: [Secret Management](docs/security.md#secret-management) — Vault, AWS Secrets Manager integration
- **Exchange Debugging**: [Debug Tools](docs/DEBUG_TOOLS.md#available-debug-tools) — Connectivity testing, API diagnostics
- **Audit Trails**: [Audit Logging](docs/audit-logging.md#architecture) — Request logging, compliance
- **Distributed Computing**: [Cluster Configuration](docs/distributed-computing.md#cluster-configuration) — Ray clusters for AWS/GCP
- **Load Testing**: [Load Tests](load-tests/README.md#overview) — k6 scenarios, capacity planning

## 📚 Documentation

### Core Documentation

| Document | Description | Key Topics |
|----------|-------------|------------|
| [System Architecture](docs/SYSTEM_ARCHITECTURE.md) | Complete system design with diagrams | 4-layer architecture, component interactions, data flows |
| [AI Hedge Fund Guide](docs/AI_HEDGE_FUND_DOCUMENTATION.md) | Trading workflow and agent design | Signal generation, risk management, execution |
| [API Documentation](docs/API_DOCUMENTATION.md) | REST and WebSocket API reference | Endpoints, authentication, request/response schemas |
| [Deployment Guide](docs/DEPLOYMENT.md) | Production deployment instructions | Docker, Kubernetes, monitoring setup |
| [Security Guide](docs/security.md) | Security practices and compliance | Secret management, RBAC, audit logging |

### Feature Deep Dives

| Feature | Documentation | Coverage |
|---------|---------------|----------|
| **Ensemble Methods** | [ensemble-methods.md](docs/ensemble-methods.md) | Voting, stacking, boosting, adaptive weighting |
| **Regime Detection** | [regime-detection.md](docs/regime-detection.md) | HMM implementation, integration status, monitoring |
| **Distributed Computing** | [distributed-computing.md](docs/distributed-computing.md) | Ray/Dask orchestration, cluster management, GPU support |
| **Audit Logging** | [audit-logging.md](docs/audit-logging.md) | Taxonomy, middleware, multi-tenant isolation |
| **Load Testing** | [load-tests/README.md](load-tests/README.md) | k6 scenarios, performance SLOs, capacity planning |
| **Dashboard** | [dashboard/README.md](dashboard/README.md) | React components, WebSocket integration, features |

### Developer Resources

- [Development Guide](docs/development/README.md) — Local setup, quality checks, debugging
- [CLAUDE.md](CLAUDE.md) — AI assistant guidance and best practices
- [MIGRATION.md](MIGRATION.md) — Upgrade guides and breaking changes
- [CHANGELOG.md](CHANGELOG.md) — Release history and version changes
- [docs/README.md](docs/README.md) — Complete documentation index

### API Reference

Interactive API documentation is available when the server is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🗺️ Roadmap & Releases

### Current Version

**v1.20.0.0** — Latest stable release with multi-tenant support

See [pyproject.toml](pyproject.toml) for full version details.

### Recent Releases

| Version | Release Date | Highlights |
|---------|--------------|------------|
| **1.20.0.0** | 2025-10 | Multi-tenant agent and risk manager support, RLS enhancements |
| **1.8.0.0** | 2025-09 | Security hardening, dependency updates, Bandit integration |
| **1.5.0.0** | 2025-08 | Regime detection service, HMM implementation |
| **1.0.0.0** | 2025-07 | Initial production release, 6-agent system |

Full release history in [CHANGELOG.md](CHANGELOG.md).

### Upcoming Features (v2.0.0)

🚀 **Multi-Tenant Architecture Completion**:
- [ ] Tenant-aware API endpoints with full RLS
- [ ] Multi-tenant dashboard support
- [ ] Tenant-specific configuration management
- [ ] Enhanced audit logging per tenant

🧠 **Advanced ML Capabilities**:
- [ ] Deep learning models for signal generation
- [ ] Reinforcement learning for portfolio optimization
- [ ] AutoML for strategy parameter tuning
- [ ] Advanced explainability features

📊 **Market Regime Integration**:
- [ ] Dashboard integration for regime visualization
- [ ] Regime-adaptive strategy switching
- [ ] Historical regime backtesting
- [ ] Real-time regime alerting

⚡ **Performance & Scalability**:
- [ ] Horizontal scaling for API services
- [ ] Message queue for async processing
- [ ] Event-driven architecture for real-time updates
- [ ] Enhanced caching strategies

See [CHANGELOG.md Unreleased section](CHANGELOG.md#unreleased) for detailed roadmap.

### Migration Guides

**Upgrading to v2.0.0 (Multi-Tenant)**:
- New `tenant_id` parameter required in many APIs
- Updated logging context with tenant information
- Database schema changes (automatic via Alembic)
- Test fixture updates for multi-tenant tests

See [MIGRATION.md](MIGRATION.md#upgrading-to-v200-multi-tenant-support) for complete upgrade instructions.

## 🤝 Contributing

We welcome contributions from the community! AlphaPulse is built by quant researchers, ML engineers, and developers passionate about systematic trading.

### How to Contribute

1. **Fork and Branch**
   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

2. **Write Quality Code**
   - Use Python 3.11+ with type hints
   - Format with Black: `poetry run black src/alpha_pulse`
   - Follow PEP 8 (enforced by flake8)
   - Add docstrings for public APIs
   - Write focused unit tests for new functionality

3. **Test Thoroughly**
   ```bash
   # Run tests
   poetry run pytest -m "not slow"  # Skip GPU-intensive tests
   poetry run pytest --cov=src/alpha_pulse

   # Quality checks
   poetry run black src/alpha_pulse
   poetry run flake8 src/alpha_pulse
   poetry run mypy src/alpha_pulse
   ```

4. **Update Documentation**
   - Add docstrings to new functions/classes
   - Update relevant docs in `docs/` directory
   - Add examples to demonstrate new features
   - Update `docs/README.md` index if adding new guides
   - Include screenshots for dashboard changes

5. **Commit with Convention**
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   feat: add new momentum indicator to technical agent
   fix: resolve race condition in risk manager
   docs: update API documentation for portfolio endpoints
   test: add integration tests for regime detection
   ```

6. **Submit Pull Request**
   - Provide clear description of changes
   - Link related issues
   - Include test evidence (coverage reports, test output)
   - Add screenshots for UI changes
   - Ensure CI passes

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/AlphaPulse.git
cd AlphaPulse

# Install with dev dependencies
poetry install --with dev
poetry shell

# Set up pre-commit hooks (optional)
poetry run pre-commit install
```

### Contribution Areas

**High-Impact Contributions**:
- 🤖 New trading agents or signal generators
- 📊 Portfolio optimization strategies
- 🛡️ Risk management enhancements
- 🧪 Test coverage improvements
- 📖 Documentation and tutorials
- 🐛 Bug fixes and performance improvements

**Good First Issues**: Look for issues tagged `good-first-issue` in the GitHub issue tracker.

### Code Review Process

1. Automated CI checks must pass
2. Maintainer review (typically 1-3 days)
3. Address review feedback
4. Final approval and merge

### Contributor Resources

- [CLAUDE.md](CLAUDE.md) — AI copilot guidance and best practices
- [Development Guide](docs/development/README.md) — Local setup and debugging
- [Architecture Docs](docs/SYSTEM_ARCHITECTURE.md) — System design reference
- `scripts/` — Utility scripts for common tasks

### Code of Conduct

Be respectful, constructive, and professional. We're building a collaborative community focused on advancing systematic trading research.

## 🛡️ Security

### Security Best Practices

AlphaPulse implements multiple security layers to protect your trading operations and sensitive data:

**Authentication & Authorization**:
- JWT-based authentication for all API endpoints
- Role-based access control (RBAC)
- Multi-tenant isolation with Row-Level Security (RLS)
- API key rotation and expiration

**Secret Management**:
- Environment-based configuration (`.env` files for dev)
- HashiCorp Vault integration for production
- AWS Secrets Manager support
- Never commit secrets to version control
- Use `scripts/migrate_secrets.py` for secret rotation

**Data Protection**:
- Encrypted database connections (PostgreSQL SSL/TLS)
- Redis encryption in transit
- Sensitive data logging prevention
- PII anonymization in audit logs

**Audit & Compliance**:
- Comprehensive audit trail for all operations
- Request-scoped correlation IDs
- Tenant-aware logging
- Retention policies for compliance

**Dependency Security**:
- Automated dependency scanning with Dependabot
- Security vulnerability monitoring
- Regular dependency updates
- Bandit static analysis for Python security issues

### Reporting Security Vulnerabilities

**Please do NOT open public GitHub issues for security vulnerabilities.**

Instead, report security issues privately:
1. Email security@alphapulse.dev (or maintainer email)
2. Provide detailed description and reproduction steps
3. Include potential impact assessment
4. Wait for acknowledgment before public disclosure

We aim to respond within 48 hours and will work with you to address the issue promptly.

### Security Resources

- [Security Overview](docs/security.md#overview) — Complete security posture documentation
- [Security Update (v1.8.0.0)](docs/security/SECURITY_UPDATE.md) — Latest hardening measures
- [Secret Management](docs/security.md#secret-management) — Credential management guide
- [Audit Logging](docs/audit-logging.md#architecture) — Compliance and audit trails

### Security Checklist for Production

- [ ] Rotate all default secrets and API keys
- [ ] Enable HTTPS/TLS for all endpoints
- [ ] Configure firewall rules and network policies
- [ ] Set up Vault or AWS Secrets Manager
- [ ] Enable audit logging and monitoring
- [ ] Configure backup and disaster recovery
- [ ] Review and test incident response plan
- [ ] Perform security scanning and penetration testing

## 📄 License

AlphaPulse is released under the **[AGPL-3.0-or-later](LICENSE)** license.

### Key Points

- **Open Source**: Free to use, modify, and distribute
- **Copyleft**: Modifications must be released under AGPL-3.0
- **Network Use**: If you run AlphaPulse as a service, you must provide source code access
- **Commercial Use**: Permitted, but derivative works must remain open source

### Commercial Support

For commercial support, bespoke integrations, or partnership opportunities:
- Open a discussion in GitHub Discussions
- Contact maintainers via repository issues
- Enterprise support packages available on request

## 💬 Community & Support

### Getting Help

- **Documentation**: Start with [docs/](docs/) directory
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Stack Overflow**: Tag questions with `alphapulse`

### Community Resources

- **GitHub Repository**: https://github.com/blackms/AlphaPulse
- **Documentation**: Available in `docs/` directory
- **Examples**: See `examples/` for usage patterns
- **Issue Tracker**: Report bugs and track feature requests

### Stay Updated

- Watch the repository for new releases
- Follow [CHANGELOG.md](CHANGELOG.md) for updates
- Check [GitHub Releases](https://github.com/blackms/AlphaPulse/releases) for version notes

### Acknowledgments

AlphaPulse is built on the shoulders of giants. Special thanks to:
- FastAPI for the excellent async web framework
- CCXT for unified exchange API access
- Ray for distributed computing capabilities
- The entire open-source trading and ML community

---

<p align="center">
  <strong>Made with ❤️ by the AlphaPulse research & engineering team</strong><br/>
  <em>Dive in, build agents, and ship alpha.</em>
</p>

<p align="center">
  <a href="#-overview">Back to Top ↑</a>
</p>
