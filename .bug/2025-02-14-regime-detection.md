# Bug Report — bug(services/regime-detection): Service starts without data pipeline and never produces regimes

Summary
- Scope/Area: services/regime-detection
- Type: data — Severity: S2
- Environment: macOS 14.6 (arm64), Python 3.11, commit HEAD, `poetry run uvicorn src.alpha_pulse.api.main:app --reload`

Expected vs Actual
- Expected: `RegimeDetectionService` should ingest recent market data on startup and publish a current regime with confidence.
- Actual: No data pipeline is injected, `_fetch_recent_data` returns an empty DataFrame, and the service logs “Model not fitted”/“No recent data available,” leaving `current_regime_info` unset.

Steps to Reproduce
1. `poetry run uvicorn src.alpha_pulse.api.main:app --reload`
2. After startup, access `/api/v1/regime/status` or inspect `app.state.regime_detection_service.current_regime_info`.
3. Observe null regime data and repeated warnings about missing data.

Evidence
- Logs: “Model not fitted, skipping regime detection” / “No recent data available”.
- Traces/Metrics: `current_market_regime` Prometheus gauge stays at default because detection never runs.
- Screenshots/Attachments: None
- Recent changes considered: `src/alpha_pulse/api/main.py:338-361`, `src/alpha_pulse/services/regime_detection_service.py:286-316`

Diagnosis Timeline
- t0: While validating regime integration, noticed API endpoints returning empty payloads.
- t1: Hypothesized data pipeline never provided to the service.
- t2: Confirmed constructor is invoked without `DataPipeline`, so `_fetch_recent_data` short-circuits.
- t3: Proposed wiring the existing data pipeline or providing a fallback fetch strategy.

Root Cause Analysis
- 5 Whys:
  1. Why is the regime missing? `current_regime_info` never updates.
  2. Why no update? `_fetch_recent_data` returns empty because `self.data_pipeline` is `None`.
  3. Why is pipeline `None`? Startup never constructs/injects `DataPipeline`.
  4. Why wasn’t this caught? Mocked tests patch the service and bypass real data acquisition.
  5. Why no fallback? Service assumes injection, but the API assembly lagged integration work.
- Causal chain: Startup → instantiate `RegimeDetectionService(config, data_pipeline=None)` → `_fetch_recent_data` returns empty → detection skipped → downstream agents relying on regimes receive `None`.

Remediation
- Workaround/Mitigation: Provide a stub data pipeline via dependency injection during startup or disable regime-dependent features temporarily.
- Proposed permanent fix: Instantiate `DataPipeline` (or equivalent data source) and pass it into the service; add guard rails that error loudly when no pipeline is configured.
- Risk & rollback considerations: Make sure pipeline setup integrates with credentials/feature flags and avoid long blocking fetches on startup.

Validation & Prevention
- Test plan: Add integration test that starts the service with a fake pipeline returning deterministic data and asserts `current_regime_info` populates.
- Regression tests to add: Unit test for `_fetch_recent_data` verifying it raises/alerts when pipeline missing.
- Monitoring/alerts: Emit startup health metric if pipeline is absent or data fetch returns empty.

Ownership & Next Steps
- Owner(s): Quant Research Engineering
- Dependencies/links: `src/alpha_pulse/api/main.py:338-361`, `src/alpha_pulse/services/regime_detection_service.py:286-316`
- Checklist:
  - [ ] Reproducible steps verified
  - [ ] Evidence attached/linked
  - [ ] RCA written and reviewed
  - [ ] Fix implemented/validated
  - [ ] Regression tests merged
