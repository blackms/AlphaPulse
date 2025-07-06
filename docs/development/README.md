# Development Documentation

This directory contains documentation specifically for developers working on the AlphaPulse system.

## 📋 Contents

### Developer Guides
- [**CLAUDE.md**](CLAUDE.md) - Comprehensive developer guide for AI-assisted development with Claude

### Related Development Documentation
- [**Debug Tools**](../DEBUG_TOOLS.md) - Available debugging tools and utilities
- [**Migration Guide**](../migration-guide.md) - Database and system migration procedures
- [**API Documentation**](../API_DOCUMENTATION.md) - REST API reference and examples
- [**System Architecture**](../SYSTEM_ARCHITECTURE.md) - Overall system design for developers

## 🛠️ Development Environment

### Prerequisites
- Python 3.11+
- Node.js 16+ (for dashboard development)
- Docker and Docker Compose
- Poetry (Python dependency management)
- PostgreSQL 13+ with TimescaleDB extension
- Redis 6+

### Setup Instructions
```bash
# Clone repository
git clone https://github.com/blackms/AlphaPulse.git
cd AlphaPulse

# Install Python dependencies
poetry install --no-interaction
poetry shell

# Set up environment
export PYTHONPATH=./src:$PYTHONPATH

# Install dashboard dependencies
cd dashboard
npm install
cd ..

# Set up database
./scripts/create_alphapulse_db.sh
./scripts/setup_test_database.sh
```

## 🧪 Development Workflow

### Branch Strategy
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - Individual feature development
- `hotfix/*` - Critical bug fixes
- `release/*` - Release preparation

### Code Quality Standards
```bash
# Linting
poetry run flake8 src/alpha_pulse --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Formatting
poetry run black src/alpha_pulse

# Type checking
poetry run mypy src/alpha_pulse

# Testing
poetry run pytest --cov-branch --cov-report=xml
```

### Commit Message Format
```
type(scope): brief description

Detailed description if needed

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## 🏗️ Architecture Overview

### 4-Layer Architecture
1. **Input Layer** - Signal generation via 6 specialized trading agents
2. **Risk Management Layer** - Signal processing and risk controls
3. **Portfolio Management Layer** - Decision making and optimization
4. **Output Layer** - Trade execution

### Key Components
- **Data Pipeline** (`src/alpha_pulse/data_pipeline/`) - Real-time and historical data
- **Agents** (`src/alpha_pulse/agents/`) - AI trading agents
- **Risk Management** (`src/alpha_pulse/risk_management/`) - Risk controls
- **Portfolio** (`src/alpha_pulse/portfolio/`) - Portfolio optimization
- **Execution** (`src/alpha_pulse/execution/`) - Trade execution
- **API** (`src/alpha_pulse/api/`) - REST API and WebSocket endpoints
- **Dashboard** (`dashboard/`) - React frontend

## 🧪 Testing Strategy

### Test Categories
- **Unit Tests** - Individual component testing
- **Integration Tests** - Cross-component interaction testing
- **API Tests** - REST API endpoint testing
- **End-to-End Tests** - Complete workflow testing

### Test Commands
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest src/alpha_pulse/tests/test_specific.py -v

# Run integration tests only
poetry run pytest -m integration

# Run with coverage
poetry run pytest --cov-branch --cov-report=xml
```

## 🚀 Running the System

### Local Development
```bash
# Run API server
python src/scripts/run_api.py

# Run main trading system
python -m alpha_pulse.main

# Run paper trading demo
python -m alpha_pulse.examples.demo_paper_trading

# Run dashboard
cd dashboard
npm start
```

### Docker Development
```bash
# Build and run all services
docker-compose up -d --build

# View logs
docker-compose logs -f alphapulse

# Stop services
docker-compose down
```

## 📊 Monitoring and Debugging

### Available Tools
- **Prometheus Metrics** - System performance monitoring
- **Grafana Dashboards** - Visual monitoring and alerting
- **Structured Logging** - Comprehensive application logging
- **Debug Endpoints** - Development debugging utilities

### Debug Commands
```bash
# Check system health
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics

# Test API endpoints
curl http://localhost:8000/api/v1/system/status
```

## 🔧 Development Best Practices

### Code Organization
- Follow the existing directory structure
- Use clear, descriptive naming conventions
- Implement proper error handling and logging
- Write comprehensive docstrings and comments

### Performance Considerations
- Optimize for real-time data processing
- Use async/await for I/O operations
- Implement proper caching strategies
- Profile performance-critical code paths

### Security Guidelines
- Never hardcode credentials or API keys
- Use the credential management system
- Implement proper input validation
- Follow secure coding practices

## 📚 Additional Resources

### External Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Web framework
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/) - Database ORM
- [React Documentation](https://reactjs.org/docs/) - Frontend framework
- [Material-UI Documentation](https://mui.com/) - UI component library

### Internal Resources
- [Architecture Diagrams](../architecture-diagrams.md) - Visual system overview
- [API Documentation](../API_DOCUMENTATION.md) - Detailed API reference
- [Troubleshooting Guides](../DATABASE_CONNECTION_FIX.md) - Common issue resolution

---
*For development questions, consult [CLAUDE.md](CLAUDE.md) or reach out to the development team.*