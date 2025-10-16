# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Risk Scenario Models**: Fixed dataclass inheritance field ordering in risk scenarios ([#106](https://github.com/blackms/AlphaPulse/issues/106))
  - Added defaults to all fields in child dataclasses: `MacroeconomicScenario`, `StressScenario`, `TailRiskScenario`, `ReverseStressScenario`
  - Fixes `TypeError: non-default argument 'asset_returns' follows default argument`
  - Risk scenarios module now imports successfully
  - Unblocks Monte Carlo simulations, stress testing, and risk analytics
  - No breaking changes: all fields preserved with same types, now more flexible with defaults

## [1.21.3] - 2025-10-13

### Fixed
- **Execution Models**: Fixed dataclass field ordering in `OptimalExecutionParams` class ([#125](https://github.com/blackms/AlphaPulse/pull/125), fixes [#107](https://github.com/blackms/AlphaPulse/issues/107))
  - Reordered required market parameter fields before optional fields
  - Fixes `TypeError: non-default argument 'daily_volatility' follows default argument`
  - Slippage estimates module now imports successfully
  - Unblocks optimal execution algorithms and Almgren-Chriss optimization
  - No breaking changes: all fields preserved with same types

## [1.21.2] - 2025-10-13

### Fixed
- **Explainability Models**: Fixed dataclass field ordering in `GlobalExplanation` class ([#124](https://github.com/blackms/AlphaPulse/pull/124), fixes [#116](https://github.com/blackms/AlphaPulse/issues/116))
  - Reordered required fields to appear before optional fields
  - Fixes `TypeError: non-default argument 'performance_by_feature' follows default argument`
  - Explainability module now imports successfully
  - Unblocks explainability tests and SHAP functionality
  - No breaking changes: all fields preserved with same types
- **ML Explainability**: Added missing `shap` (^0.48.0) dependency for SHAP-based model interpretability ([#122](https://github.com/blackms/AlphaPulse/pull/122), fixes [#109](https://github.com/blackms/AlphaPulse/issues/109))
  - SHAPExplainer module dependency satisfied
  - Supports TreeExplainer, DeepExplainer, GradientExplainer, LinearExplainer, KernelExplainer
  - Full explainability functionality now available (with #124)
  - Package size: ~10MB with binary wheels for fast installation

## [1.21.1] - 2025-10-12

### Fixed
- **ML Ensemble Dependencies**: Added missing `xgboost` (^3.0.5) and `lightgbm` (^4.6.0) dependencies ([#121](https://github.com/blackms/AlphaPulse/pull/121), fixes [#117](https://github.com/blackms/AlphaPulse/issues/117))
  - `XGBoostEnsemble` class now fully functional
  - `LightGBMEnsemble` class now fully functional
  - `StackingEnsemble` xgboost/lightgbm meta-model options now available
  - `OnlineBoosting` xgboost base model option now available
  - **Note**: macOS users must install OpenMP runtime: `brew install libomp`
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
