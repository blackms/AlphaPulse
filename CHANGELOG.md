# Changelog

All notable changes to AlphaPulse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.14.0.0] - 2025-01-06

### Added
- **Comprehensive Redis Caching Layer**
  - Multi-tier caching architecture (L1 memory, L2 local Redis, L3 distributed)
  - Four cache strategies: cache-aside, write-through, write-behind, refresh-ahead
  - Intelligent cache invalidation (time-based, event-driven, dependency-based, tag-based)
  - Cache decorators for easy integration (@cache, @cache_invalidate)
  - Distributed caching support with consistent hashing and replication
  - Cache warming mechanisms for predictive preloading
  - Comprehensive monitoring and analytics

- **Performance Optimizations**
  - Optimized serialization with MessagePack
  - Compression support (LZ4, Snappy, GZIP)
  - Connection pooling for reduced latency
  - TTL variance to prevent thundering herd

- **Cache Monitoring**
  - Real-time metrics (hit rates, latency, memory usage)
  - Hot key detection and optimization
  - Automatic performance recommendations
  - Prometheus metrics integration

### Changed
- Updated README.md with comprehensive caching documentation
- Redis is now a required dependency (not optional)
- Enhanced performance benchmarks with caching metrics

### Fixed
- Fixed missing repository links in README.md
- Updated installation instructions to use Poetry

## [1.13.0.0] - 2024-12-25

### Added
- Ray distributed computing infrastructure for parallel backtesting
- Comprehensive distributed computing documentation
- Market regime detection with Hidden Markov Models
- Explainable AI features (SHAP, LIME, counterfactuals)
- Enhanced risk management controls

### Changed
- Improved backtesting performance by 10x with distributed computing
- Updated architecture to support horizontal scaling

## [1.12.0.0] - 2024-12-20

### Added
- Real-time data provider integrations
- WebSocket support for live market data
- Enhanced input validation framework
- Secure credential management system

### Changed
- Migrated to Poetry for dependency management
- Updated to Python 3.11+ requirement

## [1.11.0.0] - 2024-12-15

### Added
- Multi-agent trading system with 5 specialized agents
- Portfolio optimization strategies (MPT, HRP, Black-Litterman)
- Real-time monitoring dashboard
- RESTful API with WebSocket support

### Changed
- Refactored architecture to 4-layer design
- Improved test coverage to 80%+

## [1.0.0.0] - 2024-11-01

### Added
- Initial release of AlphaPulse
- Basic trading engine
- Paper trading support
- Simple technical indicators
- PostgreSQL database integration