#!/usr/bin/env bash
set -euo pipefail

# Generate technical evidence: tests/coverage, lint, type-check, SAST, SBOM.
# Outputs under: ./results, ./reports, ./sbom

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results reports sbom

has_cmd() { command -v "$1" >/dev/null 2>&1; }

POETRY_RUN=""
if has_cmd poetry; then
  POETRY_RUN="poetry run"
fi

echo "== Python tests + coverage =="
if [ -n "$POETRY_RUN" ]; then
  $POETRY_RUN pytest -m "not slow" --cov=src/alpha_pulse --cov-report=xml:coverage.xml --junitxml=results/junit.xml || true
else
  pytest -m "not slow" --cov=src/alpha_pulse --cov-report=xml:coverage.xml --junitxml=results/junit.xml || true
fi
[ -f coverage.xml ] && cp coverage.xml reports/coverage.xml || true

echo "== Lint/Format checks =="
if [ -n "$POETRY_RUN" ]; then
  $POETRY_RUN black --check . | tee results/black.txt || true
  $POETRY_RUN flake8 | tee results/flake8.txt || true
else
  black --check . | tee results/black.txt || true
  flake8 | tee results/flake8.txt || true
fi

echo "== Type checking =="
if [ -n "$POETRY_RUN" ]; then
  $POETRY_RUN mypy src | tee results/mypy.txt || true
else
  mypy src | tee results/mypy.txt || true
fi

echo "== Security SAST (bandit) =="
if has_cmd bandit; then
  bandit -r src -f json -o results/bandit.json || true
elif [ -n "$POETRY_RUN" ] && $POETRY_RUN bandit --version >/dev/null 2>&1; then
  $POETRY_RUN bandit -r src -f json -o results/bandit.json || true
else
  echo "bandit not installed; skip. Install: poetry add -G dev bandit" | tee -a results/security_missing.txt
fi

echo "== Dependency audit (safety/pip-audit) =="
if has_cmd safety; then
  safety check -o results/safety.txt || true
elif [ -n "$POETRY_RUN" ] && $POETRY_RUN safety --version >/dev/null 2>&1; then
  $POETRY_RUN safety check -o results/safety.txt || true
elif has_cmd pip-audit; then
  pip-audit -r requirements.txt -f json -o results/pip-audit.json || true
else
  echo "safety/pip-audit not installed; skip. Install: poetry add -G dev safety || pipx install pip-audit" | tee -a results/security_missing.txt
fi

echo "== SBOM (CycloneDX) =="
if has_cmd cyclonedx-py; then
  cyclonedx-py -o sbom/sbom.json || true
elif has_cmd cyclonedx; then
  cyclonedx -o sbom/sbom.json || true
elif [ -n "$POETRY_RUN" ] && $POETRY_RUN cyclonedx --help >/dev/null 2>&1; then
  $POETRY_RUN cyclonedx --format json -o sbom/sbom.json || true
else
  echo "CycloneDX tool not installed; skip. Install: poetry self add 'cyclonedx-bom' then 'poetry cyclonedx --format json -o sbom/sbom.json'." | tee -a results/sbom_missing.txt
fi

echo "== Summary =="
if [ -f reports/coverage.xml ]; then
  if has_cmd rg; then
    rate=$(rg -o 'line-rate="([0-9.]+)"' -r '$1' reports/coverage.xml | head -n1 || true)
  else
    rate=$(grep -o 'line-rate="[0-9.]*"' reports/coverage.xml | head -n1 | sed -E 's/.*"([0-9.]+)"/\1/' || true)
  fi
  if [ -n "${rate:-}" ]; then
    pct=$(awk "BEGIN{print $rate*100}")
    printf "Coverage line-rate: %.2f%%\n" "$pct" | tee results/summary.txt
  fi
fi
echo "Artifacts written to ./results, ./reports, ./sbom"

