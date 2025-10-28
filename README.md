# 📈 AlphaPulse: AI-Driven Hedge Fund System

[![CI/CD](https://github.com/blackms/AlphaPulse/actions/workflows/python-app.yml/badge.svg)](https://github.com/blackms/AlphaPulse/actions/workflows/python-app.yml)
[![Codecov](https://codecov.io/gh/blackms/AlphaPulse/branch/main/graph/badge.svg)](https://codecov.io/gh/blackms/AlphaPulse)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-github_pages-blue.svg)](https://blackms.github.io/AlphaPulse/)

AlphaPulse is a sophisticated algorithmic trading system that combines multiple specialized AI trading agents, advanced risk management controls, modern portfolio optimization techniques, high-performance caching, database optimization, market regime detection, and real-time monitoring and analytics to create a comprehensive hedge fund solution.

---

## 🚀 Multi-Tenant SaaS Transformation Status

**Current Phase**: Phase 3 (Build & Validate) - Sprint 4 (IN PROGRESS)
**Current Sprint**: Sprint 4 (2025-10-28 to 2025-11-08) - 31% complete

### Phase 2 Completion Summary

We have successfully completed the comprehensive design phase for transforming AlphaPulse into a multi-tenant SaaS platform:

- ✅ **392KB Documentation**: 16 comprehensive documents across architecture, security, operations
- ✅ **100% Protocol Compliance**: All LIFECYCLE-ORCHESTRATOR-ENHANCED-PROTO.yaml requirements satisfied
- ✅ **4/4 Quality Gates Passed**: HLD approved, TCO validated, team aligned, observability designed
- ✅ **2/3 Approval Conditions Met**: Dev environment ready, operational runbook complete (load testing in Sprint 4)
- ✅ **Architecture Validated**: All design principles validated, scalable to 1,000+ tenants
- ✅ **Security Comprehensive**: STRIDE analysis, defense-in-depth, all HIGH risks mitigated
- ✅ **Highly Profitable**: $62-476 profit per customer depending on tier

### Sprint 4 Progress (Day 1 Complete)

**Completed (4 SP / 13 SP = 31%)**:
- ✅ **Load Testing Scripts** - k6 baseline (100 users) and target capacity (500 users) tests
- ✅ **Enhanced CI/CD Pipeline** - Lint, test, security (Bandit/Safety), build, quality gate
- ✅ **Helm Charts** - Complete Kubernetes deployment (API, Workers, Redis, Vault, Ingress, HPA)
- ✅ **Environment Values** - Dev, staging, production Helm values files
- ✅ **Production Dockerfile** - Multi-stage build with security hardening
- ✅ **Seed Script** - Load test user and data creation
- ✅ **Training Materials** - Kubernetes workshop (2h), Vault training (2h)

**Next Steps**:
- ⏳ Provision staging environment (AWS/GCP)
- ⏳ Execute load testing (baseline + target capacity)
- ⏳ Team training (Kubernetes, Vault)
- ⏳ EPIC-001 preparation (database multi-tenancy)

**Sprint 4 Tracking**: See [`.agile/sprint-4-tracking.md`](.agile/sprint-4-tracking.md) and [Issue #181](https://github.com/blackms/AlphaPulse/issues/181)

### Key Documents

| Document | Description | Location |
|----------|-------------|----------|
| **HLD** | High-Level Design (65KB) | [HLD-MULTI-TENANT-SAAS.md](HLD-MULTI-TENANT-SAAS.md) |
| **C4 Diagrams** | Architecture views (4 levels) | [docs/diagrams/c4-*.md](docs/diagrams/) |
| **Security Review** | STRIDE analysis, compliance | [docs/security-design-review.md](docs/security-design-review.md) |
| **Database Migration** | Zero-downtime migration plan | [docs/database-migration-plan.md](docs/database-migration-plan.md) |
| **Architecture Review** | Design validation, TCO | [docs/architecture-review.md](docs/architecture-review.md) |
| **Dev Environment** | 60-90 min setup guide | [docs/development-environment.md](docs/development-environment.md) |
| **Operational Runbook** | Incident response, DR | [docs/operational-runbook.md](docs/operational-runbook.md) |
| **Load Testing** | Scripts, README, report template | [load-tests/](load-tests/) |
| **Helm Charts** | Kubernetes deployment | [helm/alphapulse/](helm/alphapulse/) |
| **Phase 2 Summary** | Complete status report | [docs/phase-2-completion-summary.md](docs/phase-2-completion-summary.md) |

### Architecture Highlights

- **Hybrid Isolation Strategy**: PostgreSQL RLS + Redis namespaces + Vault policies
- **Scalability**: 100 → 1,000+ tenants with horizontal scaling
- **TCO**: $2,350-3,750/month for 100 tenants (highly profitable)
- **Performance**: Projected p99 latency <200ms (validating in Sprint 4)
- **Security**: SOC2, GDPR, PCI DSS readiness documented
- **Deployment**: Kubernetes (EKS/GKE) with Helm charts, CI/CD pipeline ready

For complete details, see [Phase 2 Completion Summary](docs/phase-2-completion-summary.md).

---

## Table of Contents

- [✨ Executive Summary](#-executive-summary)
- [📚 Project Documentation System](#-project-documentation-system)
- [⬇️ Installation](#️-installation)
- [⚙️ Configuration](#️-configuration)
- [🚀 Features](#-features)
- [🔌 API Reference](#-api-reference)
- [💡 Usage Examples](#-usage-examples)
- [⚡ Performance Optimization](#-performance-optimization)
- [💾 Caching Architecture](#-caching-architecture)
- [🔍 Troubleshooting](#-troubleshooting)
- [🔒 Security](#-security)
- [🤝 Contributing](#-contributing)
- [📜 Changelog](#-changelog)
- [❓ Support](#-support)
- [📊 Architecture Documentation](#-architecture-documentation)

## ✨ Executive Summary

AlphaPulse is a state-of-the-art AI Hedge Fund system that leverages multiple specialized AI agents working in concert to generate trading signals, which are then processed through sophisticated risk management controls and portfolio optimization techniques. The system is designed to operate across various asset classes with a focus on cryptocurrency markets.

### Key Components

| Component | Description |
|-----------|-------------|
| Multi-Agent System | 6 specialized agents (Technical, Fundamental, Sentiment, Value, Activist, Warren Buffett) working in concert |
| Market Regime Detection | HMM-based regime classification with 5 distinct market states (FULLY INTEGRATED v1.18.0) |
| Correlation Analysis | Advanced correlation analysis with tail dependencies and regime detection (v1.18.0) |
| Dynamic Risk Budgeting | Regime-aware position limits and leverage controls (v1.18.0) |
| Explainable AI | SHAP, LIME, and counterfactual explanations for all decisions |
| Risk Management | Dynamic position sizing, stop-loss, drawdown protection with risk budgets |
| Portfolio Optimization | Mean-variance, risk parity, Black-Litterman with correlation integration |
| High-Performance Caching | Multi-tier Redis caching with intelligent invalidation |
| Distributed Computing | Ray & Dask for parallel backtesting and optimization |
| Execution System | Paper trading and live trading capabilities |
| Dashboard | Real-time monitoring of all system aspects |
| API | RESTful API with WebSocket support and full enterprise feature coverage |

### Performance Metrics

- Backtested Sharpe Ratio: 1.8
- Maximum Drawdown: 12%
- Win Rate: 58%
- Average Profit/Loss Ratio: 1.5

## 📚 Project Documentation System

AlphaPulse includes a comprehensive machine-readable documentation system designed to serve as the "project brain" for AI-assisted development. This system ensures that all AI agents have complete context about the project state, preventing duplicate work and ensuring proper integration of features.

### Documentation Files

The following YAML files in the project root provide critical project context:

| File | Purpose | When to Read |
|------|---------|--------------|
| `PROJECT_MEMORY.yaml` | Master project state reference | **ALWAYS READ FIRST** |
| `COMPONENT_MAP.yaml` | All components and their integration status | Before implementing any feature |
| `INTEGRATION_FLOWS.yaml` | Data flow mapping and integration gaps | When working on system integration |
| `AGENT_INSTRUCTIONS.yaml` | Development guidelines for AI agents | Before starting any development task |

### Key Project Status

**Current Phase**: Integration Audit - Many sophisticated features exist but are not integrated into the main system flow.

**Critical Integration Gap**: The HMM (Hidden Markov Model) regime detection service is fully implemented but never started in the main API, meaning the system is missing crucial market context for trading decisions.

### Integration Status Categories

- **INTEGRATED**: Feature is fully wired into main system flow and used by end users
- **IMPLEMENTED_NOT_INTEGRATED**: Feature code exists but isn't connected to the main system
- **PARTIAL_INTEGRATION**: Feature partially used but missing key connections
- **NOT_INTEGRATED**: Feature not connected to main system at all

### For AI Developers

Before implementing any new feature:
1. Check `COMPONENT_MAP.yaml` to see if it already exists
2. Prioritize integrating existing unintegrated features over building new ones
3. Update the documentation files after any integration work

This documentation system is self-maintaining - all agents must update these files after making changes to ensure future agents have accurate context.

## ⬇️ Installation

### Prerequisites

- Python 3.11+ (required for latest features)
- Node.js 14+ (for dashboard)
- PostgreSQL with TimescaleDB
- Redis 6.0+ (required for caching layer)
- Docker and Docker Compose (for containerized deployment)

### Installation Steps

#### Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/blackms/AlphaPulse.git
   cd AlphaPulse
   ```

2. Install Python dependencies using Poetry:
   ```bash
   # Install Poetry if not already installed
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install
   
   # Activate the virtual environment
   poetry shell
   ```

3. Install dashboard dependencies:
   ```bash
   cd dashboard
   npm install
   cd ..
   ```

4. Set up the database:
   ```bash
   # Make the script executable
   chmod +x scripts/create_alphapulse_db.sh
   
   # Run the script
   ./scripts/create_alphapulse_db.sh
   ```

5. Set up Redis for caching:
   ```bash
   # Install Redis (Ubuntu/Debian)
   sudo apt-get install redis-server
   
   # Install Redis (macOS)
   brew install redis
   
   # Start Redis
   redis-server
   ```

6. Configure your API credentials:
   ```bash
   cp src/alpha_pulse/exchanges/credentials/example.yaml src/alpha_pulse/exchanges/credentials/credentials.yaml
   # Edit credentials.yaml with your exchange API keys
   ```

7. Run the system:
   ```bash
   # Start the API server
   python src/scripts/run_api.py
   
   # In another terminal, start the dashboard
   cd dashboard && npm start
   ```

#### Docker Installation

1. Create a `.env` file in the project root with the required environment variables:
   ```bash
   # Exchange API credentials
   EXCHANGE_API_KEY=your_api_key
   EXCHANGE_API_SECRET=your_api_secret
   
   # MLflow settings
   MLFLOW_TRACKING_URI=http://mlflow:5000
   
   # Monitoring
   PROMETHEUS_PORT=8000
   GRAFANA_ADMIN_PASSWORD=alphapulse  # Change this in production
   ```

2. Build and start all services:
   ```bash
   docker-compose up -d --build
   ```

3. Verify all services are running:
   ```bash
   docker-compose ps
   ```

## ⚙️ Configuration

AlphaPulse uses a configuration-driven approach with YAML files for different components.

### Core Configuration Files

| File | Description | Default Location |
|------|-------------|------------------|
| API Configuration | API settings and endpoints | `config/api_config.yaml` |
| Database Configuration | Database connection settings | `config/database_config.yaml` |
| Agent Configuration | Settings for trading agents | `config/agents/*.yaml` |
| Risk Management | Risk control parameters | `config/risk_management/risk_config.yaml` |
| Portfolio Management | Portfolio optimization settings | `config/portfolio/portfolio_config.yaml` |
| Cache Configuration | Redis caching settings | `config/cache_config.py` |
| Monitoring | Metrics and alerting configuration | `config/monitoring_config.yaml` |

### Environment Variables

The following environment variables can be used to override configuration settings:

```bash
# Database settings
DB_USER="testuser"
DB_PASS="testpassword"
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="alphapulse"

# Exchange API credentials
EXCHANGE_API_KEY=your_api_key
EXCHANGE_API_SECRET=your_api_secret
ALPHA_PULSE_BYBIT_TESTNET=true/false

# OpenAI API Key (for LLM-based hedging analysis)
OPENAI_API_KEY=your_openai_api_key

# Authentication
JWT_SECRET=your_jwt_secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO

# Redis settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password
```

### Agent Configuration

Each agent can be configured in its respective YAML file:

```yaml
# Example: config/agents/technical_agent.yaml
name: "Technical Agent"
weight: 0.3
enabled: true
parameters:
  lookback_period: 14
  indicators:
    - "RSI"
    - "MACD"
    - "Bollinger"
  thresholds:
    buy: 0.7
    sell: 0.3
```

### Risk Management Configuration

Configure risk controls in `config/risk_management/risk_config.yaml`:

```yaml
position_limits:
  default: 20000.0
margin_limits:
  total: 150000.0
exposure_limits:
  total: 100000.0
drawdown_limits:
  max: 25000.0
```

## 🚀 Features

AlphaPulse provides a comprehensive set of features for algorithmic trading:

### Multi-Agent System

The system uses multiple specialized AI agents to analyze different aspects of the market:

- **Technical Agent**: Chart pattern analysis and technical indicators
- **Fundamental Agent**: Economic data analysis and company fundamentals
- **Sentiment Agent**: News and social media analysis
- **Value Agent**: Long-term value assessment
- **Activist Agent**: Market-moving event detection

### Enhanced Risk Management

The system now includes comprehensive risk management features:

- **Tail Risk Hedging**: Automated detection and hedging of extreme market events
- **Liquidity Risk Management**: Pre-trade impact assessment and slippage estimation
- **Monte Carlo VaR**: Advanced risk metrics using simulation techniques
- **Dynamic Risk Budgeting**: Regime-aware position sizing and leverage limits

### Market Regime Detection

Advanced Hidden Markov Model (HMM) based regime detection:

- **Multi-Factor Analysis**: Volatility, returns, liquidity, and sentiment features
- **Real-Time Classification**: Continuous market regime monitoring
- **5 Market Regimes**: Bull, Bear, Sideways, Crisis, and Recovery
- **Transition Forecasting**: Early warning for regime changes
- **Adaptive Strategies**: Automatic strategy adjustment per regime

### Explainable AI (XAI)

Comprehensive explainability features for transparency and compliance:

- **SHAP Explanations**: Game theory-based feature contributions for all models
- **LIME Local Explanations**: Instance-level interpretable approximations
- **Feature Importance Analysis**: Multi-method importance computation
- **Decision Tree Surrogates**: Interpretable approximations of complex models
- **Counterfactual Explanations**: "What-if" analysis for alternative outcomes
- **Regulatory Compliance**: Automated documentation and audit trails

### Ensemble Methods

Advanced ensemble techniques for combining agent signals:

- **Voting Methods**: Hard/soft voting with weighted consensus
- **Stacking**: Meta-learning with XGBoost, LightGBM, Neural Networks
- **Boosting**: Adaptive, gradient, and online boosting algorithms
- **Adaptive Weighting**: Performance-based dynamic weight optimization
- **Signal Aggregation**: Robust aggregation with outlier detection

### Risk Management

Advanced risk controls to protect your portfolio:

- **Position Size Limits**: Default max 20% per position
- **Portfolio Leverage**: Default max 1.5x exposure
- **Stop Loss**: Default ATR-based with 2% max loss
- **Drawdown Protection**: Reduces exposure when approaching limits

### Portfolio Optimization

Multiple portfolio optimization strategies:

- **Mean-Variance Optimization**: Efficient frontier approach
- **Risk Parity**: Equal risk contribution across assets
- **Hierarchical Risk Parity**: Clustering-based risk allocation
- **Black-Litterman**: Combines market equilibrium with views
- **LLM-Assisted**: AI-enhanced portfolio construction

### Machine Learning Integration

Advanced ML capabilities for adaptive trading:

- **Ensemble Methods**: Voting, stacking, and boosting for signal aggregation
- **Online Learning**: Real-time model adaptation from trading outcomes
- **Drift Detection**: Automatic detection of model performance degradation
- **GPU Acceleration**: Ready infrastructure for high-performance computing (coming soon)

### Real-Time Dashboard

The dashboard provides comprehensive monitoring and control:

- **Portfolio View**: Current allocations and performance
- **Agent Insights**: Signals from each agent
- **Risk Metrics**: Current risk exposure and limits
- **Cache Metrics**: Hit rates, latency, and memory usage
- **System Health**: Component status and data flow
- **Alerts**: System notifications and important events

### Execution System

Flexible trade execution options:

- **Paper Trading**: Test strategies without real money
- **Live Trading**: Connect to supported exchanges
- **Smart Order Routing**: Optimize execution across venues
- **Transaction Cost Analysis**: Monitor and minimize costs

### Distributed Computing

High-performance distributed backtesting and optimization:

- **Ray & Dask Support**: Choose the best framework for your workload
- **Parallel Backtesting**: Test strategies across time, symbols, or parameters
- **Hyperparameter Optimization**: Distributed grid search and Bayesian optimization
- **Auto-scaling Clusters**: Dynamic resource allocation based on demand
- **Fault Tolerance**: Automatic retry and checkpointing for reliability
- **Result Aggregation**: Smart combination of distributed results

## 🔌 API Reference

AlphaPulse provides a comprehensive RESTful API for interacting with the system.

### Authentication

The API supports two authentication methods:

#### API Key Authentication
```
X-API-Key: your_api_key
```

#### OAuth2 Authentication
1. Obtain a token:
```
POST /token
Content-Type: application/x-www-form-urlencoded

username=your_username&password=your_password
```

2. Include the token in the Authorization header:
```
Authorization: Bearer your_access_token
```

### Base URL
```
http://localhost:18001
```

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API health check |
| `/api/v1/positions/spot` | GET | Get current spot positions |
| `/api/v1/positions/futures` | GET | Get current futures positions |
| `/api/v1/positions/metrics` | GET | Get detailed position metrics |
| `/api/v1/risk/exposure` | GET | Get current risk exposure |
| `/api/v1/risk/metrics` | GET | Get detailed risk metrics |
| `/api/v1/portfolio` | GET | Get current portfolio data |
| `/api/v1/metrics/{metric_type}` | GET | Get metrics data |
| `/api/v1/hedging/*` | GET/POST | Tail risk hedging analysis and recommendations |
| `/api/v1/liquidity/*` | GET/POST | Liquidity risk assessment and impact analysis |
| `/api/v1/ensemble/*` | GET/POST | Ensemble ML methods for signal aggregation |
| `/api/v1/online-learning/*` | GET/POST | Online learning model management |

### WebSocket Endpoints

Real-time updates via WebSocket connections:

| Endpoint | Description |
|----------|-------------|
| `/ws/metrics` | Real-time metrics updates |
| `/ws/alerts` | Real-time alerts |
| `/ws/portfolio` | Real-time portfolio updates |
| `/ws/trades` | Real-time trade updates |

For complete API documentation, see the interactive API docs at `http://localhost:8000/docs` when the API is running.

## 💡 Usage Examples

### Running the System

For a complete demo with all fixes applied:
```bash
./run_fixed_demo.sh
```

For individual components:
```bash
# API only
python src/scripts/run_api.py

# Dashboard only
cd dashboard && npm start

# Trading engine
python -m alpha_pulse.main
```

### Running Caching Demo

To see the caching functionality in action:
```bash
# Run the caching demo
python src/alpha_pulse/examples/demo_caching.py
```

This demonstrates:
- Basic caching operations with performance comparison
- Batch operations for efficient data handling
- Tag-based cache invalidation
- Real-time cache monitoring and analytics
- Distributed caching capabilities

### Backtesting Strategies

1. Configure your backtest in `examples/trading/demo_backtesting.py`
2. Run the backtest:
   ```bash
   python examples/trading/demo_backtesting.py
   ```
3. View results in the `reports/` directory

### Adding Custom Agents

1. Create a new agent class in `src/alpha_pulse/agents/`
2. Implement the Agent interface defined in `src/alpha_pulse/agents/interfaces.py`
3. Register your agent in `src/alpha_pulse/agents/factory.py`
4. Add configuration in `config/agents/your_agent.yaml`

### Customizing Risk Controls

1. Edit `config/risk_management/risk_config.yaml`
2. Adjust parameters like max position size, drawdown limits, etc.
3. For advanced customization, extend `RiskManager` in `src/alpha_pulse/risk_management/manager.py`

## ⚡ Performance Optimization

### Hardware Recommendations

For optimal performance, the following hardware specifications are recommended:

- **CPU**: 8+ cores for parallel signal processing
- **RAM**: 16GB+ for large datasets and model inference
- **Storage**: SSD with at least 100GB free space
- **Network**: Low-latency connection to exchanges

### Software Optimization

For large-scale deployments:

- **Redis caching is enabled by default**: Fine-tune in `config/cache_config.py`
- **Enable distributed caching**: Set `distributed.enabled = true` for multi-node setups
- **Use cache warming**: Enable predictive warming for market open
- **Enable database sharding**: Set in `config/database_config.yaml`
- **Implement GPU acceleration**: Configure in `config/compute_config.yaml`

### Benchmarks

| Configuration | Signals per Second | Latency (ms) | Max Assets |
|---------------|-------------------|--------------|------------|
| Basic (4 cores, 8GB RAM) | 50 | 200 | 20 |
| Standard (8 cores, 16GB RAM) | 120 | 80 | 50 |
| High-Performance (16+ cores, 32GB+ RAM) | 300+ | 30 | 100+ |

## 💾 Caching Architecture

AlphaPulse includes a comprehensive Redis-based caching layer that significantly improves system performance:

### Multi-Tier Cache Architecture

| Tier | Storage | TTL | Use Cases |
|------|---------|-----|-----------|
| L1 Memory | Application Memory | 1 min | Hot data, real-time quotes |
| L2 Local Redis | Local Redis Instance | 5 min | Indicators, recent trades |
| L3 Distributed | Redis Cluster | 1 hour | Historical data, backtest results |

### Cache Strategies

- **Cache-Aside**: Lazy loading for on-demand data
- **Write-Through**: Synchronous cache and database updates
- **Write-Behind**: Asynchronous batch updates for high throughput
- **Refresh-Ahead**: Proactive cache warming for predictable access patterns

### Key Features

#### Intelligent Invalidation
- Time-based expiration with TTL variance
- Event-driven invalidation for real-time updates
- Dependency tracking for cascading updates
- Tag-based bulk invalidation

#### Performance Optimization
- MessagePack serialization for compact storage
- LZ4 compression for large objects
- Consistent hashing for distributed caching
- Connection pooling for reduced latency

#### Monitoring & Analytics
- Real-time hit rate tracking
- Latency monitoring per operation
- Hot key detection and optimization
- Automatic performance recommendations

### Usage Example

```python
from alpha_pulse.services.caching_service import CachingService
from alpha_pulse.cache.cache_decorators import cache

# Initialize caching service
cache_service = CachingService.create_for_trading()
await cache_service.initialize()

# Use cache decorator for automatic caching
@cache(ttl=300, namespace="market_data")
async def get_market_data(symbol: str):
    # This will be automatically cached
    return await fetch_market_data(symbol)

# Manual cache operations
await cache_service.set("key", value, ttl=600, tags=["market"])
value = await cache_service.get("key")

# Invalidate by tags
await cache_service.invalidate(tags=["market"])
```

### Performance Impact

- **90%+ cache hit rate** for frequently accessed data
- **<1ms latency** for L1/L2 cache hits
- **50-80% reduction** in database load
- **3-5x improvement** in API response times

### Configuration

Configure caching in `src/alpha_pulse/config/cache_config.py`:

```python
# Example configuration
config = CacheConfig()
config.tiers["l2_local_redis"].ttl = 300  # 5 minutes
config.serialization.compression = CompressionType.LZ4
config.warming.enabled = True  # Enable predictive warming
```

## 🔍 Troubleshooting

### Common Issues

#### API Connection Errors
- Check your API credentials in `credentials.yaml`
- Verify exchange status and rate limits
- Check network connectivity

#### Portfolio Rebalancing Errors
- Ensure sufficient balance on exchange
- Check minimum order size requirements
- Verify portfolio constraints are not too restrictive

#### Dashboard Connection Issues
- Ensure API is running (`python src/scripts/run_api.py`)
- Check port availability (default: 8000)
- Verify WebSocket connection in browser console

#### Redis Cache Issues
- Ensure Redis is running: `redis-cli ping` (should return PONG)
- Check Redis memory usage: `redis-cli info memory`
- Clear cache if needed: `redis-cli FLUSHDB`
- Verify Redis configuration in `config/cache_config.py`

### Diagnostic Steps

1. Check the logs:
   ```bash
   tail -f logs/alphapulse.log
   ```

2. Verify database connection:
   ```bash
   python check_database.py
   ```

3. Test API endpoints:
   ```bash
   python check_api_endpoints.py
   ```

4. Monitor system metrics:
   ```bash
   # If using Docker
   docker-compose logs -f prometheus
   ```

## 🔒 Security

### Authentication and Authorization

- API access is secured via API keys or OAuth2 tokens
- Dashboard access requires user authentication
- Role-based access control for different system functions

### Data Protection

- All API communications support TLS encryption
- Sensitive data (API keys, credentials) are stored securely
- Database connections use encrypted channels

### Best Practices

- Regularly rotate API keys
- Use strong, unique passwords for all accounts
- Limit API access to necessary IP addresses
- Monitor for unusual activity
- Keep all dependencies updated

## 🤝 Contributing

We welcome contributions to AlphaPulse! Here's how to get started:

### Code Style

- Python code follows PEP 8 guidelines
- JavaScript code follows Airbnb style guide
- All code must include appropriate documentation

### Testing Requirements

- All new features must include unit tests
- Integration tests are required for API endpoints
- Maintain or improve code coverage

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Add tests for your changes
5. Ensure all tests pass
6. Submit a pull request

## 📜 Changelog

### v1.16.0.0 - Latest

#### Added
- **Database Optimization System**: Advanced connection pooling, query optimization, and intelligent routing
- **Index Management**: Automated advisor, bloat monitoring, and concurrent operations
- **Read/Write Splitting**: Load balancing across replicas with automatic failover
- **Performance Monitoring**: Real-time metrics and comprehensive health reporting

### v1.15.0.0 - Previous

#### Added
- **Comprehensive Redis Caching Layer**: Multi-tier caching architecture with L1 memory, L2 local Redis, and L3 distributed caching
- **Intelligent Cache Strategies**: Implemented cache-aside, write-through, write-behind, and refresh-ahead patterns
- **Advanced Cache Invalidation**: Time-based, event-driven, dependency-based, and tag-based invalidation
- **Cache Monitoring & Analytics**: Real-time metrics, hot key detection, and performance recommendations
- **Optimized Serialization**: MessagePack with compression support (LZ4, Snappy, GZIP)

### v1.14.0.0

#### Added
- Distributed Computing with Ray and Dask for parallel backtesting
- Enhanced scalability for large-scale simulations
- Improved resource utilization efficiency

For a complete list of changes, see the [CHANGELOG.md](docs/releases/CHANGELOG.md) file.

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[📋 Documentation Index](docs/README.md)** - Complete documentation navigation
- **[🏗️ System Architecture](docs/SYSTEM_ARCHITECTURE.md)** - Overall system design
- **[🚀 User Guide](docs/USER_GUIDE.md)** - Setup and usage instructions
- **[👨‍💻 Developer Guide](docs/development/CLAUDE.md)** - Development guidelines
- **[📊 API Documentation](docs/API_DOCUMENTATION.md)** - REST API reference
- **[🔐 Security](docs/security.md)** - Security features and protocols

### Quick Start Documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production setup
- [Database Setup](docs/DATABASE_SETUP.md) - Database configuration
- [Debug Tools](docs/DEBUG_TOOLS.md) - Troubleshooting utilities

### Release Information
- [Release Notes](docs/releases/RELEASE_NOTES.md) - Latest updates
- [Changelog](docs/releases/CHANGELOG.md) - Complete history

## ❓ Support

For issues or questions:

1. **Check Documentation** - Comprehensive guides in [`docs/`](docs/)
2. **API Reference** - Live documentation at `http://localhost:8000/docs` when running
3. **Troubleshooting** - See [Debug Tools](docs/DEBUG_TOOLS.md) and troubleshooting guides
4. **GitHub Issues** - Open an issue in the [repository](https://github.com/blackms/AlphaPulse/issues)

## 📊 Architecture Documentation

For comprehensive architecture documentation including C4 diagrams, data flow diagrams, sequence diagrams, and more, see [docs/architecture-diagrams.md](docs/architecture-diagrams.md).

This documentation includes:
- C4 Model diagrams (Context, Container, Component levels)
- Data flow and trading signal flow diagrams
- Sequence diagrams for key processes
- Deployment and infrastructure diagrams
- State machines for order lifecycle and system health
- Entity relationship diagrams
- Performance and security architecture
- Monitoring and observability architecture# CI/CD Test
# Trigger CI
