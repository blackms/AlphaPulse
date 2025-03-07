# AlphaPulse - AI Hedge Fund

AlphaPulse is an advanced algorithmic trading system that combines multiple AI agents, sophisticated risk management, and portfolio optimization to make data-driven investment decisions in cryptocurrency markets.

## Features

- Multi-agent architecture combining technical, fundamental, sentiment, and value analysis
- Risk-first approach with multiple layers of risk controls
- Portfolio optimization using modern portfolio theory
- Extensible framework for adding new strategies and data sources
- Real-time monitoring and performance analytics

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL with TimescaleDB (or use Docker setup)
- Redis (or use Docker setup)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/alpha-pulse.git
   cd alpha-pulse
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the database infrastructure:
   ```bash
   ./scripts/setup_database.sh
   ```

4. Initialize the database:
   ```bash
   python src/scripts/init_db.py
   ```

### Running the System

1. Start the data pipeline:
   ```bash
   python -m alpha_pulse.data_pipeline
   ```

2. Launch the trading engine:
   ```bash
   python -m alpha_pulse.main
   ```

3. Monitor performance:
   ```bash
   python -m alpha_pulse.monitoring
   ```

## Database Infrastructure

AlphaPulse uses a robust database infrastructure:

- **PostgreSQL with TimescaleDB**: For relational data and time-series metrics
- **Redis**: For caching and real-time messaging

### Database Setup

The database infrastructure can be set up using Docker Compose:

```bash
docker-compose -f docker-compose.db.yml up -d
```

This will start:
- PostgreSQL with TimescaleDB extension
- Redis
- PgAdmin (web interface for PostgreSQL)

### Database Schema

The database schema includes:
- Users and authentication
- Portfolios and positions
- Trades and orders
- Time-series metrics
- Alerts and notifications

### Database Access

The database access layer provides:
- Connection management
- Repository pattern for data access
- ORM models
- Transaction support

## Examples

Check out the examples directory for sample scripts:

- `examples/database/demo_database.py`: Demonstrates database operations
- `examples/trading/demo_ai_hedge_fund.py`: Demonstrates the AI Hedge Fund
- `examples/monitoring/demo_monitoring.py`: Demonstrates the monitoring system

To run the database demo:

```bash
cd examples/database
./run_demo.sh
```

## Documentation

- [API Documentation](API_DOCUMENTATION.md)
- [System Architecture](SYSTEM_ARCHITECTURE.md)
- [Hedge Fund Requirements](HEDGE_FUND_REQUIREMENTS.md)
- [Deployment Guide](DEPLOYMENT.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.