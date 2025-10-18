# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.21.6] - 2025-10-18

### Fixed
- **Redis Dependency**: Upgraded `redis` from 3.5.3 to 5.3.1 to support `redis.asyncio` module ([#128](https://github.com/blackms/AlphaPulse/pull/128), fixes [#112](https://github.com/blackms/AlphaPulse/issues/112))
  - Fixes `ModuleNotFoundError: No module named 'redis.asyncio'` in cache and data provider modules
  - Redis 5.x includes built-in asyncio support (available since redis-py 4.2.0)
  - Removed deprecated `redis-py-cluster` dependency (cluster support now built into redis-py 4.0+)
  - Affected files now import successfully: `cache/redis_manager.py`, `cache/distributed_cache.py`, `api/cache/redis.py`, `services/data_aggregation.py`, `data_pipeline/providers/base_provider.py`
  - No breaking changes: redis 5.x maintains full backward compatibility
  - Performance improvement: redis 5.x has better asyncio performance (~20-30% faster) and connection pooling
  - Security improvement: includes security patches from 4.x and 5.x series
  - **Note**: Redis 5.x requires Python 3.8+ (project already requires 3.11+)

## [1.21.5] - 2025-10-18

### Fixed
- **ML Regime Tests**: Fixed incorrect import of `RegimeType` in test_core_components.py ([#127](https://github.com/blackms/AlphaPulse/pull/127), fixes [#110](https://github.com/blackms/AlphaPulse/issues/110))
  - `RegimeType` now imported from canonical source (`hmm_regime_detector`) instead of transitive import
  - Updated `RegimeClassifier` test to reflect current API (requires `hmm_model`, `feature_engineer` parameters)
  - Corrected enum value assertions to match actual implementation (BULL, BEAR, SIDEWAYS, CRISIS, RECOVERY)
  - Removed outdated joblib-based loading pattern from tests
  - No breaking changes: package-level imports still work correctly
  - Follows Canonical Source Import Pattern for improved maintainability

## [1.21.4] - 2025-10-16

### Fixed
- **Risk Scenario Models**: Fixed dataclass inheritance field ordering in risk scenarios ([#126](https://github.com/blackms/AlphaPulse/pull/126), fixes [#106](https://github.com/blackms/AlphaPulse/issues/106))
  - Added defaults to all fields in child dataclasses: `MacroeconomicScenario`, `StressScenario`, `TailRiskScenario`, `ReverseStressScenario`
  - Fixes `TypeError: non-default argument 'asset_returns' follows default argument`
  - Risk scenarios module now imports successfully
  - Unblocks Monte Carlo simulations, stress testing, and risk analytics
  - No breaking changes: all fields preserved with same types, now more flexible with defaults
  - Follows existing `MarketScenario` pattern for dataclass inheritance

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
