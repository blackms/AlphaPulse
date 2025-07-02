# Changelog

All notable changes to the AlphaPulse project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2025-01-02
### Added
- Comprehensive field-level encryption for sensitive trading and user data
- AES-256-GCM encryption with authenticated encryption (AEAD)
- SQLAlchemy encrypted field types for transparent encryption/decryption
- Searchable encryption for queryable fields using deterministic tokens
- Hierarchical key management with rotation support
- Batch encryption operations for performance optimization
- Migration tooling for encrypting existing data
- Performance test suite for encryption operations
- Extensive documentation on database encryption and key management

### Changed
- Enhanced database models to use encrypted fields for sensitive data
- Updated database configuration to support encryption transparently
- Improved security architecture to protect data at rest

### Security
- Implemented encryption at rest for all sensitive trading data
- Added field-level encryption for user PII (emails, phone numbers, etc.)
- Protected API credentials and trading account details with encryption
- Added key versioning system for rotation without data re-encryption

## [0.1.3] - 2025-01-02
### Added
- Comprehensive secret management system with multi-provider support (AWS Secrets Manager, HashiCorp Vault, Environment Variables)
- Secure authentication module with bcrypt password hashing and JWT improvements
- Migration script to help users transition from hardcoded credentials
- Kubernetes secrets configuration templates
- Secure Docker Compose configuration with proper secret handling
- Audit logging for all secret access operations
- Comprehensive security documentation

### Changed
- Replaced all hardcoded credentials with secure secret management
- Enhanced authentication to use proper password hashing instead of plaintext
- Updated dependencies to include security libraries (passlib, boto3, hvac, cryptography)

### Security
- Removed hardcoded API keys and credentials from codebase
- Implemented encryption at rest for local secret storage
- Added proper JWT secret management with rotation support
- Enhanced .gitignore to prevent accidental credential commits

## [0.1.2] - 2025-01-02
### Added
- Comprehensive unit tests for Technical, Fundamental, and Sentiment agents
- CLAUDE.md documentation file for AI-assisted development guidance
- Test fixtures and utilities for agent testing in conftest.py

### Changed
- Enhanced test coverage for core trading agents

## [0.1.1] - 2024-06-XX
### Changed
- Refactored backtester to use new `alpha_pulse/agents` module instead of deprecated `src/agents`.
- Removed the old `src/agents` directory and all legacy agent code.
- Confirmed all documentation and diagrams are up-to-date after agents module cleanup.

## [Unreleased]

### Added
- Support for additional cryptocurrency exchanges
- Enhanced risk management controls
- New technical indicators and analysis tools
- Improved documentation and examples

## [1.0.0] - 2024-03-15
### Added
- Initial release of AlphaPulse trading system
- Multi-agent trading architecture with 5 specialized agents:
  - Technical Agent for chart pattern analysis
  - Fundamental Agent for economic data analysis
  - Sentiment Agent for news and social media analysis
  - Value Agent for long-term value assessment
  - Activist Agent for market-moving event detection
- Advanced risk management system with:
  - Position size limits
  - Portfolio leverage controls
  - Stop-loss mechanisms
  - Drawdown protection
- Portfolio optimization strategies:
  - Mean-Variance Optimization
  - Risk Parity
  - Hierarchical Risk Parity
  - Black-Litterman
  - LLM-Assisted portfolio construction
- Real-time dashboard with:
  - Portfolio view
  - Agent insights
  - Risk metrics
  - System health monitoring
  - Alert system
- Comprehensive RESTful API with:
  - Authentication (API Key and OAuth2)
  - Position management endpoints
  - Risk exposure endpoints
  - Portfolio data endpoints
  - WebSocket support for real-time updates
- Docker support for containerized deployment
- Integration with major cryptocurrency exchanges
- Support for both paper trading and live trading
- Smart order routing system
- Transaction cost analysis tools

### Changed
- Optimized performance for high-frequency trading
- Improved error handling and logging
- Enhanced security measures for API access

### Fixed
- Initial bug fixes and stability improvements
- API authentication issues
- Dashboard connection problems
- Portfolio rebalancing errors

## [0.9.0] - 2024-02-01
### Added
- Beta release with core trading functionality
- Basic risk management controls
- Initial dashboard implementation
- API framework

### Changed
- Performance optimizations
- UI/UX improvements
- Documentation updates

### Fixed
- Various stability issues
- Connection handling
- Data synchronization problems

[Unreleased]: https://github.com/blackms/AlphaPulse/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/blackms/AlphaPulse/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/blackms/AlphaPulse/releases/tag/v0.9.0