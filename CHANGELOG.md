# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-02-13

### Added
- ✨ Enhanced logging system with detailed debug information for signal generation and aggregation processes
- ✨ New AI hedge fund configuration features:
  - Weights for trend, momentum, volatility, volume, and pattern
  - Timeframes for short, medium, and long terms
  - Pattern recognition with confidence threshold
- ✨ Added support for custom request timeout and connection limit in data providers

### Changed
- ♻️ Enhanced signal generation logic in manager
- ♻️ Improved data pipeline:
  - Replaced logging with loguru
  - Enhanced data fetching and processing performance
  - Simplified and reorganized configuration
- ♻️ Enhanced providers:
  - Updated Binance provider to use aiohttp for asynchronous requests
  - Improved error handling and logging for API interactions
  - Enhanced indicator calculations in talib provider
  - Improved Alpha Vantage provider error handling and response processing

### Fixed
- 🐛 Fixed position value calculation in hedge fund module

### Documentation
- 📝 Added comprehensive technical documentation for exchanges
- 📝 Added system architecture documentation
- 📝 Enhanced risk management documentation with diagrams
- 📝 Added AI hedge fund technical documentation
- 📝 Added README files for example directories
- 📝 Enhanced overall documentation structure
- 📝 Added real data implementation plan

### Maintenance
- 🔧 Added symbolic link for config example
- 🗑️ Removed outdated portfolio analysis reports
- 📦 Updated gitignore for new file types

[0.2.0]: https://github.com/yourusername/AlphaPulse/compare/v0.1.0...0.2.0