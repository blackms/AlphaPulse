# Executive Summary – Perizia Tecnica AlphaPulse

- Purpose: independent appraisal of software IP for capitalization.
- Owner: aigensolutions srl. Lead developer: Alessio Rocchi (socio).
- Baseline: date 2025-09-12; repo ref `0e3ce1c` (HEAD); stack: Python/FastAPI, PostgreSQL, Redis.
- Evidence snapshot: tests present (84 files); current coverage report ≈7.3%; CI workflows present; migrations via Alembic.

## Scope
- Backend AlphaPulse: `src/alpha_pulse/*` (API, backtesting, ML, services), Alembic migrations, ops/CI artifacts.

## Methods Applied (provisional)
- Cost (RCN): LOC-driven with AI productivity, overhead, tooling/data, obsolescence.
- Income (RfR): royalty on indicative revenues (B2C/B2B mix), tax/WACC/g assumptions.
- Market: IP-adjusted multiples on ARR attributable to IP.

## Results (EUR)
- Cost (RCN): point ≈ 198,000; range ≈ 92,000 – 333,000.
- Income (RfR): base ≈ 320,000; range ≈ 260,000 – 380,000.
- Market (comparables): base ≈ 1,200,000; range ≈ 500,000 – 1,700,000.
- Reconciliation (weights: 30% Cost, 40% RfR, 30% Market):
  - Indicative point ≈ 580,000; indicative range ≈ 300,000 – 1,000,000.

## Key Assumptions
- LOC non-test ≈ 141,983; productivity with AI 4,000–8,000 LOC/month (1 dev).
- Dev monthly cost 6,500–9,500; overhead 15%; tooling 8–15k; data 0–10k; obsolescence 20–35%.
- RfR mix: B2C 2,000 × 39 €/mo; B2B 30 × 12k €/y; royalty 5%; tax 24%; WACC 22%; g 2%.
- Market: α (attribution to IP) ≈ 0.6; IP-adjusted multiples 2–4× with IP-only discount.

## Risks & Limitations
- Coverage low vs. test count (possible config/report drift);
- No SBOM/licensing or SAST/deps audit evidence attached yet;
- Financial inputs (forecast, WACC, tax, growth) are indicative;
- Legal chain-of-title and OSS compliance to be provided.

## Next Actions
- Generate evidence: `bash scripts/generate_evidence.sh` (coverage, mypy/flake8, bandit, safety, SBOM).
- Provide financial forecast (3–5y) and confirm pricing/volumes; refine RfR and Market.
- Provide ownership/licensing documents; attach SBOM and security reports to the perizia.
- Confirm valuation date/commit if different; then finalize reconciliation and sensitivity.

Note: This is a provisional, triangulated view for decision support; final perizia depends on the requested evidences and confirmed financials.
