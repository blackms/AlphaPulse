# Bug Report — bug(services/online-learning): Startup uses next(get_db_session()) on Session object

Summary
- Scope/Area: services/online-learning
- Type: functional — Severity: S2
- Environment: macOS 14.6 (arm64), Python 3.11, commit HEAD, `poetry run uvicorn src.alpha_pulse.api.main:app --reload`

Expected vs Actual
- Expected: Online learning service should start with a valid SQLAlchemy session obtained from `get_db_session`.
- Actual: Startup raises `TypeError: 'Session' object is not an iterator` because `next(get_db_session())` is invoked even though the helper already returns a Session.

Steps to Reproduce
1. `poetry run uvicorn src.alpha_pulse.api.main:app --reload`
2. Watch console during startup as services initialize.
3. Observe log `Error initializing online learning service: 'Session' object is not an iterator`.

Evidence
- Logs: `Error initializing online learning service: 'Session' object is not an iterator`
- Traces/Metrics: None (service aborts before telemetry)
- Screenshots/Attachments: None
- Recent changes considered: `src/alpha_pulse/api/main.py:399-416`, `src/alpha_pulse/config/database.py:376`

Diagnosis Timeline
- t0: Startup log highlighted online learning service failure.
- t1: Hypothesized misuse of `next()` on session factory.
- t2: Verified `get_db_session` returns a Session, not a generator, causing `TypeError`.
- t3: Proposed removing `next()` (or returning an iterator) and re-testing startup.

Root Cause Analysis
- 5 Whys:
  1. Why does the service fail? The constructor receives `next(get_db_session())`.
  2. Why call `next()`? Presumably assuming `get_db_session` yielded sessions.
  3. Why is that assumption wrong? `get_db_session` returns a Session directly.
  4. Why wasn’t this caught? No integration test covers the real startup path.
  5. Why no test? Mocked tests patch out the database dependency and return iterables.
- Causal chain: Startup event → `db_session = next(get_db_session())` → `get_db_session` returns Session → Python raises TypeError → service not initialized → online learning endpoints unusable.

Remediation
- Workaround/Mitigation: Temporarily skip online learning startup or wrap call in try/except to continue without it.
- Proposed permanent fix: Remove `next()` and call `get_db_session()` directly (or change helper to yield). Optionally wrap session creation in context manager to manage lifecycle.
- Risk & rollback considerations: Ensure session is closed on shutdown; adjust tests to handle real sessions.

Validation & Prevention
- Test plan: Add startup test ensuring `app.state.online_learning_service` initializes and can start a dummy session.
- Regression tests to add: Direct unit test on helper verifying it returns a Session; integration test for startup service wiring.
- Monitoring/alerts: Log service readiness metrics so failures surface in dashboards.

Ownership & Next Steps
- Owner(s): ML Platform team
- Dependencies/links: `src/alpha_pulse/api/main.py:399-416`, `src/alpha_pulse/config/database.py:376`
- Checklist:
  - [ ] Reproducible steps verified
  - [ ] Evidence attached/linked
  - [ ] RCA written and reviewed
  - [ ] Fix implemented/validated
  - [ ] Regression tests merged
