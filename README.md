# 📈 AlphaPulse: AI-Driven Hedge Fund System

[![CI/CD](https://github.com/blackms/AlphaPulse/actions/workflows/python-app.yml/badge.svg)](https://github.com/blackms/AlphaPulse/actions/workflows/python-app.yml)
[![Codecov](https://codecov.io/gh/blackms/AlphaPulse/branch/main/graph/badge.svg)](https://codecov.io/gh/blackms/AlphaPulse)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/blackms/AlphaPulse/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/alphapulse/badge/?version=latest)](https://alphapulse.readthedocs.io/en/latest/?badge=latest)

AlphaPulse is a sophisticated algorithmic trading system that combines multiple specialized AI trading agents, advanced risk management controls, modern portfolio optimization techniques, and real-time monitoring and analytics to create a comprehensive hedge fund solution.

## Table of Contents

- [✨ Executive Summary](#executive-summary)
- [⬇️ Installation](#installation)
- [⚙️ Configuration](#configuration)
- [🚀 Features](#features)
- [🔌 API Reference](#api-reference)
- [💡 Usage Examples](#usage-examples)
- [⚡ Performance Optimization](#performance-optimization)
- [🔍 Troubleshooting](#troubleshooting)
- [🔒 Security](#security)
- [🤝 Contributing](#contributing)
- [📜 Changelog](#changelog)
- [❓ Support](#support)

## ✨ Executive Summary

AlphaPulse is a state-of-the-art AI Hedge Fund system that leverages multiple specialized AI agents working in concert to generate trading signals, which are then processed through sophisticated risk management controls and portfolio optimization techniques. The system is designed to operate across various asset classes with a focus on cryptocurrency markets.

### Key Components

| Component | Description |
|-----------|-------------|
| Multi-Agent System | 5+ specialized agents (Technical, Fundamental, Sentiment, Value, Activist) working in concert |
| Risk Management | Position sizing, stop-loss, drawdown protection |
| Portfolio Optimization | Mean-variance, risk parity, and adaptive approaches |
| Execution System | Paper trading and live trading capabilities |
| Dashboard | Real-time monitoring of all system aspects |
| API | RESTful API with WebSocket support |

### Performance Metrics

- Backtested Sharpe Ratio: 1.8
- Maximum Drawdown: 12%
- Win Rate: 58%
- Average Profit/Loss Ratio: 1.5

## ⬇️ Installation

### Prerequisites

- Python 3.8+ (3.11+ recommended)
- Node.js 14+ (for dashboard)
- PostgreSQL with TimescaleDB
- Docker and Docker Compose (for containerized deployment)
- Redis (optional, for caching)

### Installation Steps

#### Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/alpha-pulse.git
   cd alpha-pulse
   ```

2. Install Python dependencies:
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows

   # Install dependencies
   pip install -r requirements.txt
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
   chmod +x create_alphapulse_db.sh
   
   # Run the script
   ./create_alphapulse_db.sh
   ```

5. Configure your API credentials:
   ```bash
   cp src/alpha_pulse/exchanges/credentials/example.yaml src/alpha_pulse/exchanges/credentials/credentials.yaml
   # Edit credentials.yaml with your exchange API keys
   ```

6. Run the setup script:
   ```bash
   ./setup.sh
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

### Real-Time Dashboard

The dashboard provides comprehensive monitoring and control:

![Dashboard Screenshot](dashboard/public/dashboard_screenshot.png)

- **Portfolio View**: Current allocations and performance
- **Agent Insights**: Signals from each agent
- **Risk Metrics**: Current risk exposure and limits
- **System Health**: Component status and data flow
- **Alerts**: System notifications and important events

### Execution System

Flexible trade execution options:

- **Paper Trading**: Test strategies without real money
- **Live Trading**: Connect to supported exchanges
- **Smart Order Routing**: Optimize execution across venues
- **Transaction Cost Analysis**: Monitor and minimize costs

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

### WebSocket Endpoints

Real-time updates via WebSocket connections:

| Endpoint | Description |
|----------|-------------|
| `/ws/metrics` | Real-time metrics updates |
| `/ws/alerts` | Real-time alerts |
| `/ws/portfolio` | Real-time portfolio updates |
| `/ws/trades` | Real-time trade updates |

For complete API documentation, refer to the [API_DOCUMENTATION.md](API_DOCUMENTATION.md) file.

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

- **Use Redis for caching**: Configure in `config/settings.py`
- **Enable database sharding**: Set in `config/database_config.yaml`
- **Implement GPU acceleration**: Configure in `config/compute_config.yaml`

### Benchmarks

| Configuration | Signals per Second | Latency (ms) | Max Assets |
|---------------|-------------------|--------------|------------|
| Basic (4 cores, 8GB RAM) | 50 | 200 | 20 |
| Standard (8 cores, 16GB RAM) | 120 | 80 | 50 |
| High-Performance (16+ cores, 32GB+ RAM) | 300+ | 30 | 100+ |

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

### v7.0.5 (Wallaby) - 2025-04-21

#### Added
- Enhanced Command Compatibility: Introduced a new system rule ensuring that CLI commands are automatically tailored for the user's specific operating system
- New Prime Meta-Development Suite: Added three specialized 'Prime' modes and their corresponding rules

For a complete list of changes, see the [CHANGELOG.md](CHANGELOG.md) file.

## ❓ Support

For issues or questions:

1. Check the documentation in `docs/`
2. Run example scripts in `examples/`
3. Consult the API documentation at `http://localhost:8000/docs` when the API is running
4. Open an issue in the repository

For additional support, contact the development team at support@alphapulse.example.com.