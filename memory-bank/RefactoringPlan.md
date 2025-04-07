# Refactoring Plan: Database-Driven Backtester within AlphaPulse

**Goal:** Refactor the backtesting system to download and store market data (initially SP500/VIX from yfinance) in a PostgreSQL/TimescaleDB database (`backtesting` db, `devuser` user), and run backtests using this data within the existing `src/alpha_pulse` structure.

**Assumptions:**
*   PostgreSQL with TimescaleDB extension is running on `localhost:5432`.
*   Database: `backtesting`, User: `devuser`, Password: `devpassword`. Credentials managed via environment variables (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`).
*   Initial data: SP500 (`^GSPC`), VIX (`^VIX`) from yfinance.
*   Initial timeframes: `1d`, `1h`.
*   Modules should be kept under ~500 lines where feasible.

**Modules & Files (Consolidated within `src/alpha_pulse/`):**

1.  **Database Setup (`migrations/`, `scripts/` - Root Level):**
    *   **Credentials:** Document required environment variables (`DB_*`).
    *   **Alembic Migration (`migrations/versions/XXX_add_ohlcv_hypertable.py`):**
        *   Define `ohlcv_data` table (columns: `timestamp` (pk, timestamptz), `symbol` (pk, text), `timeframe` (pk, text), `open`, `high`, `low`, `close`, `volume` (float)). Use composite primary key (timestamp, symbol, timeframe).
        *   Use `op.execute()` to run `SELECT create_hypertable('ohlcv_data', 'timestamp');`. Consider partitioning by symbol.
    *   **Initialization Script (`scripts/setup_db.sh` or similar):**
        *   Handles DB/user creation (may require manual steps initially).
        *   Runs `alembic upgrade head`.

2.  **Data Pipeline (`src/alpha_pulse/data_pipeline/`):**
    *   **Configuration (`src/alpha_pulse/config/settings.py`):** Ensure it reads DB credentials from environment variables correctly.
    *   **Provider (`providers/yfinance_provider.py` - New/Modify):** Class to fetch OHLCV for specified symbols/timeframes from yfinance.
    *   **Models (`models.py`):** Use existing `OHLCVRecord`.
    *   **Database (`database.py`):** Use existing async session setup.
    *   **Storage/CRUD (`storage/db_writer.py` or similar - New/Modify):** Function(s) to bulk-insert pandas DataFrames (transformed into `OHLCVRecord` instances) into the `ohlcv_data` table using the async session. Handle potential conflicts (e.g., unique constraint violations).
    *   **Manager (`manager.py` - Modify):** Orchestrate fetching using the yfinance provider based on config.
    *   **Runner (`run_downloader.py` - Root Level, New):** Script to initialize and run the data pipeline manager for configured symbols/timeframes.

3.  **Strategies (`src/alpha_pulse/strategies/`):**
    *   **Move:** Relocate `src/strategies/long_short/` to `src/alpha_pulse/strategies/long_short/`.
    *   **Refactor Components:** Modify `indicators.py`, `signal_generator.py`, `position_manager.py`, `risk_manager.py` into stateless functions or classes that accept DataFrames and parameters, returning calculated Series/values. Remove data fetching/resampling. Ensure correct column name handling (including MultiIndex if applicable).

4.  **Backtesting (`src/alpha_pulse/backtesting/`):**
    *   **Data Loader (`data_loader.py` - New):**
        *   `load_backtest_data(symbols: list, timeframe: str, start_dt: datetime, end_dt: datetime) -> Dict[str, pd.DataFrame]`.
        *   Uses `data_pipeline.database` session.
        *   Queries `ohlcv_data` hypertable using SQLAlchemy.
        *   Returns `Dict[str, pd.DataFrame]` (symbol -> OHLCV DataFrame).
    *   **Signal Processor (`signal_processor.py` - New):**
        *   `generate_target_allocation(data: Dict[str, pd.DataFrame], strategy_config: dict) -> pd.Series`.
        *   Takes loaded data and strategy parameters.
        *   Uses refactored `strategies.long_short` components to calculate indicators and composite signal.
        *   Uses `position_manager` logic to determine final target allocation Series (-1.0 to +1.0).
    *   **Backtester Engine (`backtester.py` - Modify):**
        *   Adapt `backtest` method:
            *   Input `signals` Series is now target allocation.
            *   Loop logic: Calculate required trade size to move from current position to target allocation.
            *   Update commission calculation for partial buys/sells.
            *   Integrate stop-loss logic (needs ATR/stop levels).
            *   Handle short selling logic if target allocation is negative.
    *   **Strategy Interface (`strategy.py`):** `BaseStrategy` likely not used directly by the modified backtester for target allocation strategies.
    *   **Models (`models.py`):** Keep `Position`, `BacktestResult`.
    *   **Results (`results.py` - New/Modify):** Function `analyze_results(result: BacktestResult, benchmark_returns: pd.Series)` using QuantStats.

5.  **Runner (`run_backtest.py` - Root Level, Modify):**
    *   Load backtest config.
    *   Use `backtesting.data_loader.load_backtest_data`.
    *   Use `backtesting.signal_processor.generate_target_allocation`.
    *   Extract primary asset price Series.
    *   Instantiate the modified `backtesting.Backtester`.
    *   Run `backtester.backtest(prices=primary_prices, signals=target_allocation_series)`.
    *   Fetch/calculate benchmark returns.
    *   Call results analysis function.

**Implementation Order:**
1.  DB Setup (Migration, Init Script).
2.  Data Pipeline (Provider, Writer, Runner).
3.  Backtester Data Loader.
4.  Strategy Refactoring.
5.  Backtester Signal Processor.
6.  Backtester Engine Modification.
7.  Backtest Runner Modification.
8.  Testing & Refinement.