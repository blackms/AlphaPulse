# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Risk Services**: Wire risk budgeting and tail risk hedging services to live data providers ([#120](https://github.com/blackms/AlphaPulse/pull/120))
  - Services now receive market data from YFinance instead of using dummy data
  - Portfolio data flows from database through new `LivePortfolioProvider`
  - Added fail-fast dependency validation with `ServiceConfigurationError`
  - Fixed service initialization to properly inject required dependencies
  - Improved concurrent market data fetching with semaphore-based rate limiting

### Added
- **Models**: New `Portfolio` and `Position` data structures for service layer (src/alpha_pulse/models/portfolio.py)
- **Services**: New `LivePortfolioProvider` to bridge database accessors with service models
- **Exceptions**: New `ServiceConfigurationError` for dependency validation
- **Tests**: Comprehensive test coverage for risk budgeting and tail risk hedging services

### Changed
- **DataFetcher**: Added `fetch_historical_data` method for multi-symbol concurrent retrieval
- **RiskBudgetingService**: Constructor now requires `data_fetcher` and `portfolio_provider` dependencies
- **TailRiskHedgingService**: Constructor now accepts optional `portfolio_provider` parameter

### Technical Details
- Services validate dependencies at startup with clear error messages
- Tests updated with proper async mocks and stub implementations
- Background service loops properly handle errors and retry with backoff
- Concurrent data fetching limited to 4 requests via semaphore

---

## Previous Releases

For releases prior to this changelog, see [GitHub Releases](https://github.com/blackms/AlphaPulse/releases).
