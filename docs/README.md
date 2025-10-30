# AlphaPulse Documentation

Welcome to the comprehensive documentation for AlphaPulse, an AI-powered algorithmic trading system designed to operate as an automated hedge fund.

## 📋 Table of Contents

### 🏗️ Architecture & Core Systems
- [System Architecture](SYSTEM_ARCHITECTURE.md) – Overall system design and component map
- [Backend Architecture](BACKEND_ARCHITECTURE.md) – Service orchestration and data flow
- [Architecture Diagrams](architecture-diagrams.md) – C4 diagrams and reference visuals
- [Multi-Agent System](MULTI_AGENT_SYSTEM.md) – Trading agent design
- [API Documentation](API_DOCUMENTATION.md) – REST/WebSocket overview

### 🚀 Getting Started
- [User Guide](USER_GUIDE.md) – Base configuration and quick start
- [Deployment Guide](DEPLOYMENT.md) – Production deployment considerations
- [Database Setup](DATABASE_SETUP.md) – PostgreSQL and schema preparation
- [System Specification](SPEC.md) – Technical capabilities and scope

### 🔧 Development
- [CLAUDE.md](development/CLAUDE.md) – Copilot/developer onboarding guidance
- [Development Overview](development/README.md) – Directory-level notes
- [Debug Tools](DEBUG_TOOLS.md) – Diagnostics and helper scripts
- [Migration Guide](migration-guide.md) – Encryption migration checklist

### 📊 Features & Capabilities
- [AI Hedge Fund Overview](AI_HEDGE_FUND_DOCUMENTATION.md) – End-to-end trading flow
- [Ensemble Methods](ensemble-methods.md) – Signal aggregation
- [Regime Detection](regime-detection.md) – Market regime modelling
- [Online Learning](ONLINE_LEARNING.md) – Adaptive model support
- [Distributed Computing](distributed-computing.md) – Scaling backtests and simulations
- [GPU Acceleration](gpu_acceleration.md) – GPU tooling and services
- [Explainable AI](explainable-ai.md) – Model interpretability assets
- [Key Management](key-management.md) – Encryption key handling
- [Audit Logging](audit-logging.md) – Audit trail implementation details
- [Database Optimisation](database-optimization.md) – Connection and query tooling
- [Database Encryption](database-encryption.md) – Field-level encryption types

### 🛡️ Security
- [Security Overview](security.md) – Security posture and configuration
- [Security Update Summary](security/SECURITY_UPDATE_SUMMARY.md)
- [Security Updates](security/SECURITY_UPDATE.md)

### 🛠️ Troubleshooting & Fixes
- [Database Connection Fix](DATABASE_CONNECTION_FIX.md)
- [TA-Lib Installation](TA_LIB_INSTALLATION.md)
- [TA-Lib Undefined Symbol Fix](TA_LIB_UNDEFINED_SYMBOL_FIX.md)
- [Frontend ↔ Backend Connection](frontend_backend_connection_fix.md)
- [Bybit Debug Guide](BYBIT_DEBUG_README.md)
- [Loguru Dependency Fix](LOGURU_DEPENDENCY_FIX.md)

### 📚 Examples & Reference Material
- `examples/README.md` – Directory-level overview
- `examples/trading/README.md` – Trading-focused demos
- `examples/portfolio/README.md` – Portfolio utilities
- `examples/monitoring/README.md` – Monitoring samples
- `examples/alerting/README.md` – Alerting walkthrough
- `examples/analysis/README.md` – Analysis notebooks/scripts
- `examples/data/README.md` – Data ingestion examples
- `docs/user_guide/integrated_features_guide.md` – Feature cross-reference

### 🧪 Load Testing & Dashboards
- [Load Testing README](../load-tests/README.md)
- [Dashboard README](../dashboard/README.md)

### 📈 Project-level references
- [Changelog](../CHANGELOG.md)
- [System Analysis Report](../SYSTEM_ANALYSIS_REPORT.md)

## 🏷️ Documentation Categories

### By Topic
- **🏗️ Architecture**: System design and component relationships
- **📊 Trading**: Core trading functionality and strategies  
- **🛡️ Risk Management**: Risk controls and monitoring
- **🔐 Security**: Security features and protocols
- **⚡ Performance**: Optimization and scalability
- **🔧 Development**: Developer tools and guides
- **🚀 Deployment**: Production setup and operations

### By Audience
- **👨‍💼 Business Users**: User guides, specifications, requirements
- **👨‍💻 Developers**: Architecture, APIs, troubleshooting
- **🛡️ Security Teams**: Security documentation, audit logs
- **📊 Analysts**: Integration status, performance analysis

## 🔄 Documentation Maintenance

When you add or modify features:

1. Update or create documentation in the relevant section above.
2. Append a note to the [Changelog](../CHANGELOG.md) describing the change.
3. Ensure demos in `examples/` continue to run (modify their READMEs as needed).

## 📞 Support

- Start with the [User Guide](USER_GUIDE.md) for environment setup.
- Use [Debug Tools](DEBUG_TOOLS.md) for diagnostics and troubleshooting.
- Refer to the [API Documentation](API_DOCUMENTATION.md) for endpoint details.

---

*Documentation last updated: 2024-12-01*
