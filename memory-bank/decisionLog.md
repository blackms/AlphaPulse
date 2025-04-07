# Decision Log

This file records architectural and implementation decisions using a list format.
2025-04-07 10:16:42 - Updated with implementation decisions.

2025-04-07 08:45:25 - Updated with new strategy implementation decision.

2025-04-07 02:21:21 - Initial creation of Memory Bank.

## 2025-04-07: Initial Backtest Configuration

### Decision
* Use a 2-year backtest period (2021-2022) instead of the full 2010-2023 range
* Enable only technical and sentiment agents with equal weights (0.5 each)
* Disable the fundamental agent for initial testing
* Disable validation features (walk-forward analysis, Monte Carlo simulations)

### Rationale
* A shorter time period allows for faster iteration and initial testing
* Starting with fewer agents simplifies the analysis of results
* The technical and sentiment agents provide a good balance of price-based and news-based signals
* Validation features can be enabled once the basic strategy is proven effective

### Implementation Details
* Modified config/backtest_sp500_config.yaml to reflect these decisions
* Set technical.weight and sentiment.weight to 0.5
* Set fundamental.enabled to false
* Set validation features enabled to false
* Adjusted start_date to "2021-01-01" and end_date to "2022-12-31"

## 2025-04-07: Agent Strategy Selection

### Decision
* Technical agent: Implement a moving average crossover strategy (SMA50/SMA200)
* Sentiment agent: Use news sentiment analysis with a 7-day lookback period

### Rationale
* Moving average crossover is a well-established technical strategy for trend following
* News sentiment can capture market mood and potentially predict short-term price movements
* The combination of trend following and sentiment analysis provides a balanced approach

### Implementation Details
* Technical agent uses SMA50 and SMA200 to generate BUY signals when price > SMA200 and SMA50 > SMA200
* Technical agent generates SELL signals when price < SMA200 or SMA50 < SMA200
* Sentiment agent analyzes news sentiment over a 7-day period and generates signals based on sentiment score


## 2025-04-07: Long/Short Trading Strategy Implementation

### Decision
* Implement a new long/short trading strategy for the S&P 500 with weekly or monthly rebalancing
* Combine trend-following, mean reversion, and optional ML signals
* Include comprehensive risk management with stop-loss and drawdown controls
* Backtest the strategy over historical data to measure performance metrics

### Rationale
* A long/short strategy can potentially generate returns in both up and down markets
* Combining multiple signal types (trend, mean reversion, ML) can provide more robust trading decisions
* Weekly/monthly rebalancing reduces transaction costs compared to daily trading
* Proper risk management is essential to protect capital during adverse market conditions

### Implementation Details
* Use moving averages (e.g., 40-week MA) for trend-following signals
* Implement RSI for mean reversion signals
* Consider VIX for volatility regime detection
* Optionally add ML model to predict market direction
* Implement position sizing based on signal strength
* Add stop-loss and trailing stop mechanisms
* Monitor drawdown and reduce position size when necessary

* Both agents include confidence scores with their signals to influence position sizing


## 2025-04-07: Long/Short Strategy Implementation Structure

### Decision
* Implement the strategy as a modular system with separate components
* Create a dedicated directory structure under src/strategies/long_short/
* Develop specialized classes for each component of the strategy
* Integrate with the existing backtester framework

### Rationale
* Modular design improves maintainability and testability
* Separation of concerns allows for easier updates and modifications
* Dedicated directory structure keeps the codebase organized
* Integration with existing backtester leverages proven functionality

### Implementation Details
* Created data_handler.py for data fetching and processing
* Created indicators.py for technical indicator calculations
* Created signal_generator.py for signal generation and combination
* Created position_manager.py for position sizing and management
* Created risk_manager.py for risk control implementation
* Created long_short_strategy.py as the main strategy class
* Updated configuration files to support the new strategy



## 2025-04-07: Walk-Forward Backtesting Framework Design

### Decision
* Implement a walk-forward backtesting framework to provide more robust strategy evaluation.
* Utilize rolling training and testing windows.
* Integrate parameter optimization using Optuna within each training window.
* Optimize parameters based on Sortino Ratio.
* Implement strict lookahead bias prevention by calculating signals/indicators only on data available up to the current point in each step (training or OOS).
* Perform Monte Carlo simulation on aggregated OOS results for statistical validation.
* Compare aggregated OOS performance against a Buy-and-Hold benchmark (^GSPC).
* Create a new script (`run_walk_forward.py`) and configuration (`config/walk_forward_config.yaml`) for this process.

### Rationale
* Walk-forward analysis provides a more realistic assessment of strategy performance by simulating how it would adapt to changing market conditions over time.
* Optimizing parameters only on past (training) data prevents overfitting to the entire dataset.
* Sortino Ratio is chosen as the optimization metric as it focuses on downside deviation, which is often more relevant for risk management.
* Strict lookahead bias prevention is crucial for valid backtest results.
* Monte Carlo simulation helps assess the likelihood of the observed performance occurring by chance.
* Benchmark comparison provides context for the strategy's performance.

### Implementation Details
* **Walk-Forward Structure:** Rolling Window (24 months train, 6 months test, 6 months step).
* **Optimization:** Optuna library, maximizing Sortino Ratio, optimizing key strategy parameters (MA, RSI, ATR windows, VIX threshold, ATR multiplier, position thresholds) with defined ranges.
* **Lookahead Prevention:** Data is loaded once, then sliced for each train/test period. Optimization runs only on train slice. Best params used to generate signals/stops on OOS slice (including buffer). Backtest runs only on the OOS test slice (without buffer).
* **Analysis:** Aggregate OOS results, recalculate overall metrics, perform Monte Carlo on combined OOS returns, compare vs benchmark in `analyze_and_save_results`.
* **Files:** `run_walk_forward.py`, `config/walk_forward_config.yaml`, enhancements to `src/alpha_pulse/analysis/performance_analyzer.py`.
