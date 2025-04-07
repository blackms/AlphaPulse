# Walk-Forward Backtesting Framework Plan

**Date:** 2025-04-07
**Status:** Approved

## I. Goal

Implement a rigorous walk-forward backtesting framework for AlphaPulse strategies. This framework will:
- Optimize strategy parameters on rolling training windows.
- Evaluate performance on subsequent out-of-sample (OOS) periods.
- Ensure strict chronological data handling to prevent lookahead bias.
- Provide statistical validation of results (market condition analysis, Monte Carlo).
- Include benchmark comparison.

## II. Architectural Design & Core Concepts

1.  **Framework:**
    *   A new script (`run_walk_forward.py`) or dedicated class structure (`src/alpha_pulse/walk_forward/`) will manage the process.
    *   Orchestrates data loading, splitting, optimization, backtesting per period, and result aggregation/analysis.

2.  **Lookahead Bias Prevention:**
    *   **Method:** Pre-calculation within each walk-forward step.
    *   **Process:**
        *   Load training data for the current step.
        *   Run parameter optimization (Optuna) *only* on training data.
        *   Use best parameters to calculate indicators/signals *only* for the subsequent OOS data.
        *   Run backtester *only* on OOS data with its pre-calculated inputs.

3.  **Walk-Forward Structure:**
    *   **Type:** Rolling Window.
    *   **Training Length:** 24 months.
    *   **OOS Test Length:** 6 months.
    *   **Step Size:** 6 months.

4.  **Parameter Optimization (Long/Short Strategy Example):**
    *   **Library:** Optuna.
    *   **Objective:** Maximize Sortino Ratio on the training period.
    *   **Parameters & Ranges (Initial Proposal):**
        *   `ma_window`: Int(10, 100)
        *   `rsi_window`: Int(5, 30)
        *   `atr_window`: Int(5, 30)
        *   `vix_threshold`: Float(15.0, 40.0)
        *   `atr_multiplier`: Float(1.0, 5.0)
        *   `trend_weight`: Float(0.0, 1.0)
        *   `mr_weight`: Float(0.0, 1.0)
        *   `vol_weight`: Float(0.0, 1.0) (Constraint: Sum of weights = 1.0)
        *   `long_threshold`: Float(0.0, 0.5)
        *   `short_threshold`: Float(-0.5, 0.0)

5.  **Statistical Validation:**
    *   **Market Conditions:** Report key metrics (Return, Sharpe, Sortino, Max Drawdown) per OOS period.
    *   **Monte Carlo:** Perform simulation (e.g., 1000 shuffles) on combined OOS returns to generate distribution of outcomes and compare actual results.

6.  **Benchmarking:**
    *   Compare against S&P 500 (^GSPC) Buy-and-Hold over the full OOS period and potentially per OOS period.

7.  **Analysis & Reporting:**
    *   Enhance `src/alpha_pulse/analysis/performance_analyzer.py`.
    *   Generate plots showing OOS period metrics over time.
    *   Incorporate benchmark comparison plots.
    *   Include Monte Carlo simulation results/plots.
    *   Generate a consolidated report (Markdown or potentially extending QuantStats HTML if library issues resolved).

## III. Implementation Steps

1.  **Create Walk-Forward Script/Class:** (`run_walk_forward.py` or `src/alpha_pulse/walk_forward/engine.py`).
2.  **Implement Data Splitting Logic:** Generate rolling train/test dates.
3.  **Integrate Optuna:** Define `objective` function for training period optimization.
4.  **Implement Walk-Forward Loop:** Iterate through splits, run Optuna, generate OOS signals, run OOS backtest, store results.
5.  **Aggregate Results:** Combine OOS period results (equity curve, positions). Recalculate overall metrics.
6.  **Implement Benchmark Calculation:** Calculate Buy-and-Hold performance.
7.  **Implement Monte Carlo Simulation:** Add function in `performance_analyzer.py`.
8.  **Enhance Analysis Module:** Update `performance_analyzer.py` to handle aggregated OOS results, benchmark, and Monte Carlo data.
9.  **Update Configuration:** Modify/create config file for walk-forward and optimization settings.
10. **Update Memory Bank:** Log decisions (`decisionLog.md`), add patterns (`systemPatterns.md`).

## IV. Deliverables

*   Modified/New configuration file(s).
*   New/Modified Python scripts (`run_walk_forward.py`, `performance_analyzer.py`, potentially `walk_forward/engine.py`).
*   Updated Memory Bank files.
*   Output directory containing aggregated OOS results, benchmark comparison, Monte Carlo analysis, and relevant plots/reports.