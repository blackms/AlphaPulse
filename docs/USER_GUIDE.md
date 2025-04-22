# AI Hedge Fund User Guide

## Getting Started

Welcome to the AI Hedge Fund system - a comprehensive algorithmic trading platform combining multiple AI agents, risk management, and portfolio optimization for cryptocurrency markets.

### Prerequisites

- Python 3.8+
- Node.js 14+ (for dashboard)
- Docker (optional, for containerized deployment)
- API keys for supported exchanges

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/alpha-pulse.git
   cd alpha-pulse
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install dashboard dependencies:
   ```bash
   cd dashboard
   npm install
   cd ..
   ```

4. Configure your API credentials:
   ```bash
   cp src/alpha_pulse/exchanges/credentials/example.yaml src/alpha_pulse/exchanges/credentials/credentials.yaml
   # Edit credentials.yaml with your exchange API keys
   ```

5. Run the setup script:
   ```bash
   ./setup.sh
   ```

## System Components

### 1. Trading Agents

The system uses five specialized AI agents:

| Agent | Purpose | Configuration |
|-------|---------|---------------|
| Technical | Chart pattern analysis | `config/agents/technical_agent.yaml` |
| Fundamental | Economic data analysis | `config/agents/fundamental_agent.yaml` |
| Sentiment | News and social media analysis | `config/agents/sentiment_agent.yaml` |
| Value | Long-term value assessment | `config/agents/value_agent.yaml` |
| Activist | Market-moving event detection | `config/agents/activist_agent.yaml` |

### 2. Risk Management

Risk controls are configured in `config/risk_management/risk_config.yaml`:

- **Position Size Limits**: Default max 20% per position
- **Portfolio Leverage**: Default max 1.5x exposure
- **Stop Loss**: Default ATR-based with 2% max loss
- **Drawdown Protection**: Reduces exposure when approaching limits

### 3. Portfolio Management

Portfolio settings are in `config/portfolio/portfolio_config.yaml`:

- **Optimization Strategy**: Choose between Mean-Variance, Risk Parity, or Adaptive
- **Rebalancing Schedule**: Set frequency and threshold
- **Target Metrics**: Set optimization goals (Sharpe, Sortino, etc.)

### 4. Dashboard

The dashboard provides real-time monitoring and control:

- **Portfolio View**: Current allocations and performance
- **Agent Insights**: Signals from each agent
- **Risk Metrics**: Current risk exposure and limits
- **System Health**: Component status and data flow
- **Alerts**: System notifications and important events

## Usage Guides

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

## Troubleshooting

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

### Support

For additional support:
- Check the documentation in `docs/`
- Run example scripts in `examples/`
- Consult the API documentation at `http://localhost:8000/docs` when the API is running

## Advanced Configuration

### Custom Data Sources

To add a new data source:
1. Implement the `DataProvider` interface in `src/alpha_pulse/data_pipeline/providers/`
2. Register your provider in `src/alpha_pulse/data_pipeline/manager.py`
3. Configure your data source in `config/data_pipeline_config.yaml`

### Performance Optimization

For large-scale deployments:
- Use Redis for caching: Configure in `config/settings.py`
- Enable database sharding: Set in `config/database_config.yaml`
- Implement GPU acceleration: Configure in `config/compute_config.yaml`

### Logging and Monitoring

- Logs are stored in `logs/`
- Configure log levels in `config/logging_config.yaml`
- Metrics are exposed via Prometheus at `http://localhost:8000/metrics`