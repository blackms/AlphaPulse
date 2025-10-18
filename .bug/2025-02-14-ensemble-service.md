# Bug Report — bug(services/ensemble): Ensemble service fails to start and manager calls missing API

Summary
- Scope/Area: services/ensemble
- Type: functional — Severity: S2
- Environment: macOS 14.6 (arm64), Python 3.11, commit HEAD, `poetry run uvicorn src.alpha_pulse.api.main:app --reload`

Expected vs Actual
- Expected: FastAPI startup should register a fully working `EnsembleService` and agent aggregation should await the service API that returns an `EnsemblePredictionResponse`.
- Actual: Startup logs `EnsembleService.__init__()` missing `db_session`, disabling the service; even if injected manually, `AgentManager` calls a non-existent, non-awaited `get_ensemble_prediction`, returning a coroutine and crashing when accessing `.confidence`.

Steps to Reproduce
1. `poetry run uvicorn src.alpha_pulse.api.main:app --reload`
2. Observe startup log `Error initializing ensemble service: EnsembleService.__init__() missing 1 required positional argument: 'db_session'`.
3. If you monkey-patch the startup to pass a session, trigger signal aggregation (e.g. via `AgentManager._aggregate_signals_with_ensemble`) and hit `AttributeError: 'EnsembleService' object has no attribute 'get_ensemble_prediction'` / `'coroutine' object has no attribute 'confidence'`.

Evidence
- Logs: `Error initializing ensemble service: EnsembleService.__init__() missing 1 required positional argument: 'db_session'`
- Traces/Metrics: None (service aborts before instrumentation)
- Screenshots/Attachments: None
- Recent changes considered: `src/alpha_pulse/api/main.py:392`, `src/alpha_pulse/services/ensemble_service.py:37`, `src/alpha_pulse/agents/manager.py:535`

Diagnosis Timeline
- t0: While auditing startup services, noticed ensemble initialization failure in logs.
- t1: Hypothesized constructor signature drift between `main.py` and `EnsembleService`.
- t2: Confirmed mismatch and found the manager calling `get_ensemble_prediction` without `await`.
- t3: Proposed to inject a DB session and align the manager with the async API.

Root Cause Analysis
- 5 Whys:
  1. Why is the ensemble service disabled? Because `startup_event` catches an exception.
  2. Why the exception? `EnsembleService()` is called without the required `db_session`.
  3. Why is `db_session` missing? Constructor was refactored to require a session but `main.py` wasn’t updated.
  4. Why would aggregation still fail after adding the session? The manager still calls `get_ensemble_prediction`, which no longer exists.
  5. Why the method mismatch? The service API was renamed to `generate_ensemble_prediction` (async) but the caller wasn’t updated/awaited.
- Causal chain: API startup → instantiate `EnsembleService()` without session → `TypeError` → ensemble disabled → agent manager still set to use ensemble → call into stale method name without await → coroutine bubbles up and crashes aggregation.

Remediation
- Workaround/Mitigation: Disable ensemble features via config (`use_ensemble=False`) to avoid the broken path.
- Proposed permanent fix: Inject a real SQLAlchemy session into `EnsembleService` during startup, update `AgentManager` to call and await `generate_ensemble_prediction`, and align method naming. Optionally supply a session factory to avoid tight coupling.
- Risk & rollback considerations: Ensure session lifecycle management so startup failures tear down cleanly; adding awaits in the trading loop requires careful error handling.

Validation & Prevention
- Test plan: Extend `test_api_startup_services` to assert `app.state.ensemble_service` is populated and that aggregation returns an `EnsemblePredictionResponse`.
- Regression tests to add: Unit test on `AgentManager._aggregate_signals_with_ensemble` using a mocked `EnsembleService` verifying awaited call.
- Monitoring/alerts: Add startup health logging/metrics to confirm ensemble registration and surface async aggregation errors.

Ownership & Next Steps
- Owner(s): Trading Platform team
- Dependencies/links: `src/alpha_pulse/api/main.py:392`, `src/alpha_pulse/services/ensemble_service.py:37`, `src/alpha_pulse/agents/manager.py:535`
- Checklist:
  - [ ] Reproducible steps verified
  - [ ] Evidence attached/linked
  - [ ] RCA written and reviewed
  - [ ] Fix implemented/validated
  - [ ] Regression tests merged
