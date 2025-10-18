# Bug Report — bug(services/risk-management): Risk budgeting and tail hedging loops run on placeholder data

Summary
- Scope/Area: services/risk-management
- Type: data — Severity: S2
- Environment: macOS 14.6 (arm64), Python 3.11, commit HEAD, `poetry run uvicorn src.alpha_pulse.api.main:app --reload`

Expected vs Actual
- Expected: Risk budgeting should use real market data for allocations and tail hedging should analyze the live portfolio.
- Actual: Startup wires neither a `DataFetcher` nor portfolio feed; `_fetch_market_data` falls back to synthetic random data and `analyze_portfolio_tail_risk()` returns `None` after logging “Portfolio data not provided, using placeholder”.

Steps to Reproduce
1. `poetry run uvicorn src.alpha_pulse.api.main:app --reload`
2. Inspect logs for “Portfolio data not provided, using placeholder”.
3. Hit `/api/v1/risk-budget/analytics` or `/api/v1/hedging/status` and observe unrealistic outputs, confirming services aren’t using live inputs.

Evidence
- Logs: “Portfolio data not provided, using placeholder” from `TailRiskHedgingService`.
- Traces/Metrics: Risk analytics fluctuate randomly between restarts because of synthetic data.
- Screenshots/Attachments: None
- Recent changes considered: `src/alpha_pulse/api/main.py:320-384`, `src/alpha_pulse/services/risk_budgeting_service.py:401-420`, `src/alpha_pulse/services/tail_risk_hedging_service.py:80-170`

Diagnosis Timeline
- t0: During integration audit, risk analytics outputs were incoherent and tail hedging never raised alerts.
- t1: Hypothesized missing dependencies for market/portfolio data.
- t2: Confirmed risk budgeting synthesizes random series when `data_fetcher` is `None` and tail hedging exits early without injected portfolio data.
- t3: Proposed wiring the real data fetcher and portfolio service before restarting background loops.

Root Cause Analysis
- 5 Whys:
  1. Why are analytics unreliable? Services operate on placeholders.
  2. Why placeholders? No data fetcher/portfolio supplied on startup.
  3. Why not provided? Startup code only constructs configs but never injects real fetchers.
  4. Why was the placeholder path left active? Dummy helpers were intended for tests but leaked into production startup.
  5. Why weren’t warnings caught? Logs emit warnings but no health check fails, so regression passed unnoticed.
- Causal chain: Startup → instantiate services with `data_fetcher=None` and no portfolio → background loops call helper → synthetic data or early return → risk outputs meaningless and hedging never triggers alerts.

Remediation
- Workaround/Mitigation: Pause services (`auto_rebalance=False`, disable tail hedging) until real data sources are wired up.
- Proposed permanent fix: Construct and pass the shared market data fetcher and portfolio gateway during startup; update services to raise configuration errors when inputs missing.
- Risk & rollback considerations: Ensure new fetchers respect rate limits and add defensive error handling so loops don’t crash on transient failures.

Validation & Prevention
- Test plan: Integration test that injects a fake `DataFetcher` and portfolio snapshot, asserting analytics/alerts respond to deterministic inputs.
- Regression tests to add: Unit tests verifying dummy data generators only run in explicit test mode; hedging service should alert when portfolio data absent.
- Monitoring/alerts: Add health metrics that flag placeholder modes so ops can detect misconfiguration.

Ownership & Next Steps
- Owner(s): Risk Engineering team
- Dependencies/links: `src/alpha_pulse/api/main.py:320-384`, `src/alpha_pulse/services/risk_budgeting_service.py:401-420`, `src/alpha_pulse/services/tail_risk_hedging_service.py:80-170`
- Checklist:
  - [ ] Reproducible steps verified
  - [ ] Evidence attached/linked
  - [ ] RCA written and reviewed
  - [ ] Fix implemented/validated
  - [ ] Regression tests merged
