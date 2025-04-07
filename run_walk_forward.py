#!/usr/bin/env python3
"""
Walk-Forward Backtesting Script for AlphaPulse.

Orchestrates the process of:
1. Loading configuration.
2. Defining walk-forward periods (training and OOS).
3. For each period:
    a. Loading training data.
    b. Optimizing strategy parameters using Optuna on training data.
    c. Loading OOS data.
    d. Generating signals/stops for OOS data using optimized parameters.
    e. Running the backtester on OOS data.
4. Aggregating OOS results.
5. Performing final analysis and reporting (including benchmark and Monte Carlo).
"""

import os
import argparse
# import logging # Replaced with loguru
from loguru import logger # Use loguru
import yaml
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import optuna # Import Optuna
import copy # Import copy for deepcopy
import warnings # Import warnings module

# Import necessary AlphaPulse components
from src.alpha_pulse.backtesting.backtester import Backtester, BacktestResult
from src.alpha_pulse.backtesting.data_loader import load_ohlcv_data
from src.alpha_pulse.strategies.long_short.orchestrator import LongShortOrchestrator
from src.alpha_pulse.strategies.long_short.risk_manager import RiskManager
# Import the analysis function from the correct module
from src.alpha_pulse.analysis.performance_analyzer import analyze_and_save_results
import quantstats as qs # For benchmark fetching and stats calculation

# --- Configuration & Setup ---

# Configure Loguru
logger.remove() # Remove default handler
logger.add("run_walk_forward.log", level="DEBUG", rotation="10 MB", compression="zip", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}") # Log DEBUG+ to file with detailed format
logger.add(lambda msg: print(msg, end=""), level="INFO", format="{message}") # Log INFO+ to console with minimal format

# Set Optuna logging level to WARNING to reduce verbosity during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)
# logger instance is now handled by loguru directly


load_dotenv()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run Walk-Forward Backtest for AlphaPulse')
    parser.add_argument('--config', type=str, default='config/walk_forward_config.yaml', # Expecting a new config file
                        help='Path to the walk-forward configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='./results/walk_forward_backtest',
                        help='Output directory for results')
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except (yaml.YAMLError, IOError) as e:
        logger.error(f"Error loading configuration: {e}")
        raise ValueError(f"Cannot load configuration from {config_path}: {str(e)}")

# --- Helper Functions ---

def generate_walk_forward_periods(
    start_dt: datetime,
    end_dt: datetime,
    train_months: int,
    test_months: int,
    step_months: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]: # Return tuple includes test_start now
    """
    Generates (train_start, train_end, test_start, test_end) tuples for rolling walk-forward periods.
    Ensures timestamps are timezone-aware (UTC) if input is naive.
    """
    periods = []
    current_train_start = pd.Timestamp(start_dt)
    # Ensure timezone awareness (assume UTC if naive)
    if current_train_start.tzinfo is None:
        current_train_start = current_train_start.tz_localize('UTC')
    end_ts = pd.Timestamp(end_dt)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize('UTC')

    while True:
        current_train_end = current_train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1) # End of training day
        current_test_start = current_train_end + pd.Timedelta(days=1) # Start of test day
        current_test_end = current_test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1) # End of test day

        # Ensure the full test period fits within the overall end date
        if current_test_end > end_ts:
            # Adjust the last test_end to not exceed the overall end_dt
            current_test_end = end_ts
            # If the adjusted test period is too short (e.g., less than a month), break
            if (current_test_end - current_test_start).days < 30: # Minimum test period length
                 break

        # Ensure train_end does not exceed test_end (can happen with short data)
        # Also ensure test_start is not after test_end
        if current_train_end >= current_test_end or current_test_start > current_test_end:
            break

        # Add the period: (train_start, train_end, test_start, test_end)
        # Note: train_end is inclusive, test_start is the day after train_end
        periods.append((current_train_start, current_train_end, current_test_start, current_test_end))

        # Move to the next period start
        next_train_start = current_train_start + pd.DateOffset(months=step_months)

        # Break if the next training period would start after the data ends
        # or if the next test period would end after the data ends
        next_test_end_check = next_train_start + pd.DateOffset(months=train_months + test_months) - pd.Timedelta(days=1)
        if next_train_start >= end_ts or next_test_end_check > end_ts:
            break

        current_train_start = next_train_start

    logger.info(f"Generated {len(periods)} walk-forward periods.")
    return periods

def define_optuna_objective(
    train_data: Dict[str, pd.DataFrame],
    strategy_config: Dict,
    backtest_config: Dict,
    price_col_name: str, # Pass the specific price column name
    primary_symbol: str # Pass the primary symbol
) -> callable:
    """
    Factory function to create the Optuna objective function for a given training dataset.
    """
    # Pre-calculate benchmark returns for the training period if needed for objective metric
    # train_start_dt = min(df.index.min() for df in train_data.values() if not df.empty)
    # train_end_dt = max(df.index.max() for df in train_data.values() if not df.empty)
    # benchmark_returns_train = qs.utils.download_returns('^GSPC').loc[train_start_dt:train_end_dt] # Example

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # 1. Suggest parameters based on config ranges
        params_to_optimize = strategy_config.get('optimization_params', {})
        trial_params = {}
        # Use deepcopy to avoid modifying the original config dict across trials
        current_strategy_config = copy.deepcopy(strategy_config)

        for name, details in params_to_optimize.items():
            param_type = details.get('type', 'float')
            low = details.get('low')
            high = details.get('high')
            step = details.get('step', None)
            log = details.get('log', False)
            param_value = None

            if param_type == 'float':
                param_value = trial.suggest_float(name, low, high, step=step, log=log)
            elif param_type == 'int':
                param_value = trial.suggest_int(name, low, high, step=step, log=log)
            # TODO: Add support for categorical if needed
            else:
                 logger.warning(f"Unsupported parameter type '{param_type}' for '{name}'")
                 continue # Skip this parameter

            trial_params[name] = param_value

            # Update nested config - assumes keys like 'indicators.ma_window' or 'risk_manager.atr_multiplier'
            keys = name.split('.')
            d = current_strategy_config
            try:
                for key in keys[:-1]:
                    # Ensure nested dictionaries exist
                    if key not in d or not isinstance(d[key], dict):
                        d[key] = {}
                    d = d[key]
                d[keys[-1]] = param_value
            except (TypeError, KeyError) as e:
                 logger.error(f"Could not set nested parameter '{name}': {e}. Check config structure and parameter names.")
                 # Fail the trial if parameter setting fails
                 return 1e9 # Return large positive penalty for minimization

         # Note: Optimized signal weights ('signal_weights.trend', 'signal_weights.mean_reversion')
         # are used directly as suggested by Optuna. No normalization (e.g., sum to 1)
         # is applied within this objective function, as the signal_generator uses them as is.

         # 3. Run backtest on TRAINING data
        try:
            # a. Generate signals/stops for training period
            # Ensure orchestrator uses the correct primary symbol
            current_strategy_config['primary_symbol'] = primary_symbol
            orchestrator = LongShortOrchestrator(current_strategy_config)
            # Pass only the necessary data slice to the orchestrator for this trial
            orchestrator_results = orchestrator.calculate_signals_and_targets(train_data)

            if orchestrator_results is None:
                logger.warning(f"Trial {trial.number}: Orchestrator failed for params {trial_params}")
                return 1e9 # Penalize failure

            train_data_with_signals, train_target_allocations, train_stop_losses = orchestrator_results

            # b. Extract price series for training period using the passed price_col_name
            if price_col_name not in train_data_with_signals.columns:
                 logger.error(f"Trial {trial.number}: Price column '{price_col_name}' not found in orchestrated data.")
                 return 1e9
            train_price_series = train_data_with_signals[price_col_name]

            # Filter data to the exact training period for backtesting
            # Get actual start/end from the data passed to the objective function
            train_start_dt = min(df.index.min() for df in train_data.values() if not df.empty)
            train_end_dt = max(df.index.max() for df in train_data.values() if not df.empty)

            # Re-introduce filtering based on indicator warmup
            indicator_config = current_strategy_config.get('indicators', {})
            lookback_periods = max(
                indicator_config.get('ma_window', 40),
                indicator_config.get('rsi_window', 14),
                indicator_config.get('atr_window', 14),
                1
            )
            # Find the first valid index after the buffer period in the price series
            first_valid_index = train_price_series.first_valid_index()
            if first_valid_index is None: # Handle case where entire series might be NaN
                 logger.warning(f"Trial {trial.number}: Price series is all NaN.")
                 return 1e9

            # Calculate the actual start date based on lookback
            # Ensure we don't go past the end date
            # Find the index position corresponding to the first valid index + lookback
            try:
                start_loc = train_price_series.index.get_loc(first_valid_index) + lookback_periods
                if start_loc >= len(train_price_series.index):
                     logger.warning(f"Trial {trial.number}: Not enough data for indicator warmup.")
                     return 1e9
                actual_train_start_dt = train_price_series.index[start_loc]
            except KeyError:
                 logger.warning(f"Trial {trial.number}: Could not determine valid start date after warmup.")
                 return 1e9

            if actual_train_start_dt > train_end_dt:
                 logger.warning(f"Trial {trial.number}: Calculated training start date {actual_train_start_dt} is after end date {train_end_dt}. No valid training range.")
                 return 1e9

            train_price_series = train_price_series[actual_train_start_dt:train_end_dt]
            train_target_allocations = train_target_allocations.reindex(train_price_series.index)
            train_stop_losses = train_stop_losses.reindex(train_price_series.index)

            if train_price_series.empty:
                 logger.warning(f"Trial {trial.number}: No price data in training range after filtering/warmup ({actual_train_start_dt} to {train_end_dt}).")
                 return 1e9


            # c. Run backtester
            backtester_params = backtest_config.get('parameters', {})
            # Instantiate RiskManager with the current trial's config
            risk_manager_config = current_strategy_config.get('risk_manager', {})
            risk_manager = RiskManager(config=risk_manager_config)
            # Instantiate Backtester with RiskManager
            backtester = Backtester(
                risk_manager=risk_manager, # Pass the risk manager
                initial_capital=backtester_params.get('initial_capital', 100000.0),
                commission=backtester_params.get('commission', 0.001)
            )
            # Extract necessary data for backtester using prefixed column names (NOTE: Orchestrator currently hardcodes 'SP500_')
            prefix = "SP500_" # Hardcoded prefix from orchestrator
            ohlc_cols = [f"{prefix}Open", f"{prefix}High", f"{prefix}Low", f"{prefix}Close"]
            if not all(col in train_data_with_signals.columns for col in ohlc_cols):
                logger.error(f"Trial {trial.number}: Missing one or more OHLC columns ({ohlc_cols}) in orchestrated data. Available: {train_data_with_signals.columns.tolist()}")
                return 1e9 # Penalize failure
            # Rename columns to standard OHLC for the backtester
            train_ohlc_data = train_data_with_signals[ohlc_cols].loc[actual_train_start_dt:train_end_dt].copy() # Use .copy() to avoid SettingWithCopyWarning
            train_ohlc_data.columns = ['Open', 'High', 'Low', 'Close'] # Rename to standard names

            # Extract ATR series (assuming column name format like ATR_14)
            atr_col_name = f"ATR_{indicator_config.get('atr_window', 14)}"
            train_atr_series = train_data_with_signals[atr_col_name].loc[actual_train_start_dt:train_end_dt] if atr_col_name in train_data_with_signals else None

            train_result = backtester.backtest(
                ohlc_data=train_ohlc_data, # Pass OHLC DataFrame
                signals=train_target_allocations,
                stop_losses=train_stop_losses,
                atr_series=train_atr_series # Pass ATR series
            )

            # 4. Calculate objective metric (Sharpe Ratio)
            daily_returns = train_result.equity_curve.pct_change().fillna(0)
            # Ensure returns index is timezone-naive for QuantStats calculation
            if daily_returns.index.tz is not None:
                daily_returns = daily_returns.tz_localize(None)

            # Use quantstats to calculate Sharpe Ratio, suppressing specific warnings
            sharpe_ratio = np.nan # Default to NaN
            with warnings.catch_warnings():
                # Suppress RuntimeWarning: invalid value encountered in scalar divide
                # Originating from quantstats/stats.py lines 294 and 349 (Sharpe/Sortino)
                warnings.filterwarnings(
                    'ignore',
                    message='invalid value encountered in scalar divide',
                    category=RuntimeWarning,
                    module='quantstats\.stats' # Match the module where the warning originates
                )
                try:
                    sharpe_ratio = qs.stats.sharpe(daily_returns) # Changed from sortino
                except RuntimeWarning as rw:
                    # Log if the specific warning we tried to suppress still occurred (shouldn't happen)
                    if 'invalid value encountered in scalar divide' in str(rw):
                         logger.warning(f"Trial {trial.number}: Suppressed RuntimeWarning occurred during Sharpe calculation: {rw}")
                    else:
                         raise # Re-raise other RuntimeWarnings
                except Exception as e:
                     logger.error(f"Trial {trial.number}: Error during Sharpe calculation: {e}")
                     # Keep sharpe_ratio as NaN

            # Handle NaN or infinite results from Sharpe calculation
            if pd.isna(sharpe_ratio) or np.isinf(sharpe_ratio):
                # Check if it's due to no trades or zero standard deviation
                if train_result.total_trades == 0:
                    logger.debug(f"Trial {trial.number}: No trades executed. Assigning poor Sharpe score (-10).")
                    metric_value = -10.0 # Assign a poor score instead of failing
                elif daily_returns.std() == 0: # Check for zero standard deviation
                    logger.debug(f"Trial {trial.number}: Zero standard deviation in returns. Assigning poor Sharpe score (-10).")
                    metric_value = -10.0 # Assign a poor score
                else:
                    logger.warning(f"Trial {trial.number}: Sharpe calculation resulted in NaN/inf ({sharpe_ratio}) for unknown reason. Returning large penalty.")
                    return 1e9 # Return penalty for other NaN/inf cases
            else:
                 metric_value = sharpe_ratio

            logger.debug(f"Trial {trial.number}: Params={trial_params}, Calculated Sharpe Ratio={metric_value:.4f}") # Changed log message

            # Optuna minimizes by default, but we want to MAXIMIZE Sharpe Ratio.
            # Return negative Sharpe Ratio.
            return_value = -metric_value
            logger.debug(f"Trial {trial.number}: Returning objective value: {return_value}")
            return return_value

        except Exception as e:
            logger.error(f"Trial {trial.number}: Exception during objective evaluation for params {trial_params}: {e}", exc_info=True)
            # Return a large penalty value for Optuna to discourage these parameters
            # Since Optuna minimizes, return positive infinity
            return float('inf') # Keep inf for general exceptions

    return objective # Correctly indented return for the outer function

# --- Main Execution ---

async def main():
    """Main asynchronous function for walk-forward backtesting."""
    args = parse_arguments()
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # --- Database Config ---
    # Get config directly, placeholders removed from YAML
    db_config = config.get('database', {})
    # Log DB connection info (mask password)
    log_db_host = db_config.get('host', 'localhost') # Use .get for safety
    log_db_port = db_config.get('port', 5432)
    log_db_name = db_config.get('database', 'backtesting')
    log_db_user = db_config.get('user', 'devuser')
    logger.info(f"Using DB Config: {log_db_user}@{log_db_host}:{log_db_port}/{log_db_name}")

    # --- Walk-Forward Parameters ---
    wf_config = config.get('walk_forward', {})
    train_months = wf_config.get('train_months', 24)
    test_months = wf_config.get('test_months', 6)
    step_months = wf_config.get('step_months', 6)
    optimization_trials = wf_config.get('optimization_trials', 50) # Number of Optuna trials

    # --- Overall Date Range ---
    backtest_config = config.get('backtest', {})
    overall_start_dt_naive = datetime.fromisoformat(backtest_config['start_date'])
    overall_end_dt_naive = datetime.fromisoformat(backtest_config['end_date'])

    # --- Data Loading Parameters ---
    data_loader_config = config.get('data_loader', {})
    strategy_config = config.get('strategy', {}).get('long_short', {})
    symbols_needed = strategy_config.get('symbols', ['^GSPC', '^VIX'])
    timeframe = strategy_config.get('timeframe', '1d')
    exchange = data_loader_config.get('exchange', 'yfinance')
    primary_symbol = strategy_config.get('primary_symbol', '^GSPC')
    price_col_name = 'SP500_Close' # Based on orchestrator

    # --- Generate Walk-Forward Periods ---
    periods = generate_walk_forward_periods(overall_start_dt_naive, overall_end_dt_naive, train_months, test_months, step_months)
    if not periods:
        logger.error("No valid walk-forward periods generated. Check dates and window lengths.")
        return

    all_oos_results: List[BacktestResult] = []
    all_oos_prices: List[pd.Series] = []
    best_params_per_period = {}

    # --- Load ALL Data Once ---
    # Calculate the necessary lookback buffer based on the max window in the base config
    indicator_config = strategy_config.get('indicators', {})
    lookback_periods = max(
        indicator_config.get('ma_window', 40),
        indicator_config.get('rsi_window', 14),
        indicator_config.get('atr_window', 14),
        1 # Ensure at least 1 period lookback
    )
    buffer_days = int(lookback_periods * 1.5 * (7/5 if timeframe == '1d' else 1)) # Approx calendar days

    # Determine the absolute start date needed for loading data (first train_start - buffer)
    abs_load_start_dt_naive = periods[0][0].to_pydatetime().replace(tzinfo=None) - pd.Timedelta(days=buffer_days)
    abs_load_end_dt_naive = periods[-1][3].to_pydatetime().replace(tzinfo=None) # Load up to the end of the last test period (naive)

    logger.info(f"Loading ALL data from {abs_load_start_dt_naive.date()} to {abs_load_end_dt_naive.date()} for walk-forward...")
    all_loaded_data = await load_ohlcv_data(
        symbols=symbols_needed,
        timeframe=timeframe,
        start_dt=abs_load_start_dt_naive,
        end_dt=abs_load_end_dt_naive,
        exchange=exchange,
        db_config=db_config # Pass db_config here
    )
    if all_loaded_data is None or all_loaded_data.get(primary_symbol) is None or all_loaded_data[primary_symbol].empty:
         logger.error("Failed to load sufficient data for the entire walk-forward period.")
         return
    logger.info("Full data loaded.")


    # --- Walk-Forward Loop ---
    initial_capital_run = backtest_config.get('parameters', {}).get('initial_capital', 100000.0)
    cumulative_equity = initial_capital_run # Start with initial capital

    for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
        logger.info(f"\n=== Walk-Forward Period {i+1}/{len(periods)} ===")
        logger.info(f"Train: {train_start.date()} - {train_end.date()} | Test: {test_start.date()} - {test_end.date()}")

        # --- 1. Select TRAINING Data Subset (including buffer) ---
        current_load_start = train_start - pd.Timedelta(days=buffer_days)
        train_data_period = {}
        data_available = True
        for symbol, df in all_loaded_data.items():
            # Slice the data including the buffer up to the end of the training period
            # Ensure slicing works with potentially timezone-aware index from loader
            df_slice = df.loc[current_load_start:train_end]
            if df_slice.empty:
                 logger.error(f"Training data empty for {symbol} in period {i+1} ({current_load_start} to {train_end}). Skipping period.")
                 data_available = False
                 break
            train_data_period[symbol] = df_slice
        if not data_available:
            continue # Skip to next period

        # Log details of the training data slice before optimization
        if primary_symbol in train_data_period and not train_data_period[primary_symbol].empty:
             train_df_debug = train_data_period[primary_symbol]
             logger.debug(f"Period {i+1} Training Slice ({primary_symbol}): Shape={train_df_debug.shape}, Start={train_df_debug.index.min()}, End={train_df_debug.index.max()}")
        else:
             logger.warning(f"Period {i+1}: Primary symbol data missing or empty in train_data_period before optimization call.")
             continue # Skip if primary data is missing

        # --- 2. Optimize Parameters on Training Data ---
        logger.info("Optimizing parameters using Optuna...")
        study = optuna.create_study(direction="maximize") # Maximize objective (negative Sharpe Ratio)
        objective_func = define_optuna_objective(
             train_data_period, # Pass data including buffer for indicator calculation
             strategy_config,
             backtest_config,
             price_col_name,
             primary_symbol
        )
        study.optimize(objective_func, n_trials=optimization_trials, timeout=wf_config.get('optimization_timeout_seconds', None)) # Add timeout

        # --- DEBUG Optuna Results ---
        try:
            logger.debug(f"Optuna study best value: {study.best_value}")
            logger.debug(f"Optuna study best params: {study.best_params}")
            logger.debug(f"Optuna study best trial object: {study.best_trial}")
        except Exception as log_e:
            logger.error(f"Error logging Optuna results: {log_e}")
        # --- END DEBUG ---

        # Handle cases where optimization might fail completely (e.g., all trials fail or best value is penalty)
        # Optuna minimizes, so best_value should be negative (or zero) for valid Sharpe. abs(value) >= 1e9 indicates failure.
        if not study.best_trial or abs(study.best_value) >= 1e9: # Check if abs(best_value) is the penalty
             logger.error(f"Optuna optimization failed to find any valid trial for period {i+1}. Skipping.")
             continue

        best_params = study.best_params
        best_value = study.best_value # This is the negative Sharpe Ratio
        best_params_per_period[f"Period_{i+1}"] = {'params': best_params, 'metric_value': -best_value} # Store actual metric value (Sharpe)
        logger.info(f"Optimization complete. Best Metric Value (Sharpe Ratio): {-best_value:.4f}") # Updated log message
        logger.info(f"Best Parameters: {best_params}")

        # --- 3. Select OOS Data Subset (including buffer for indicators) ---
        current_oos_load_start = test_start - pd.Timedelta(days=buffer_days)
        oos_data_period = {}
        data_available = True
        for symbol, df in all_loaded_data.items():
             # Slice data including buffer up to the end of the test period
             df_slice = df.loc[current_oos_load_start:test_end]
             if df_slice.empty:
                  logger.error(f"OOS data empty for {symbol} in period {i+1} ({current_oos_load_start} to {test_end}). Skipping period.")
                  data_available = False
                  break
             oos_data_period[symbol] = df_slice
        if not data_available:
            continue # Skip to next period


        # --- 4. Generate Signals/Stops for OOS Period using Best Params ---
        logger.info("Generating signals for OOS period with optimized parameters...")
        # Create strategy config with best params found
        optimized_strategy_config = copy.deepcopy(strategy_config) # Use deepcopy
        # Update nested config robustly
        for name, value in best_params.items():
             keys = name.split('.')
             d = optimized_strategy_config
             try:
                 for key in keys[:-1]:
                     # Ensure nested dictionaries exist
                     if key not in d or not isinstance(d[key], dict):
                          d[key] = {}
                     d = d[key]
                 d[keys[-1]] = value
             except (TypeError, KeyError) as e:
                  logger.error(f"Error applying optimized param '{name}': {e}. Skipping period.")
                  oos_data_period = None # Flag to skip
                  break
        if oos_data_period is None:
            continue # Skip to next period

        oos_orchestrator = LongShortOrchestrator(optimized_strategy_config)
        oos_orchestrator_results = oos_orchestrator.calculate_signals_and_targets(oos_data_period) # Use OOS data slice
        if oos_orchestrator_results is None:
            logger.error("Failed to generate signals for OOS period. Skipping.")
            continue
        oos_data_with_signals, oos_target_allocations, oos_stop_losses = oos_orchestrator_results

        # --- 5. Run Backtest on OOS Period ---
        logger.info("Running backtest on OOS period...")
        # Extract OOS OHLC data and ATR series using prefixed column names (NOTE: Orchestrator currently hardcodes 'SP500_')
        prefix = "SP500_" # Hardcoded prefix from orchestrator
        ohlc_cols = [f"{prefix}Open", f"{prefix}High", f"{prefix}Low", f"{prefix}Close"]
        if not all(col in oos_data_with_signals.columns for col in ohlc_cols):
            logger.error(f"Missing one or more OHLC columns ({ohlc_cols}) in OOS orchestrated data. Available: {oos_data_with_signals.columns.tolist()}. Skipping period.")
            continue
        oos_ohlc_data_full = oos_data_with_signals[ohlc_cols].copy() # Use .copy()
        oos_ohlc_data_full.columns = ['Open', 'High', 'Low', 'Close'] # Rename to standard names

        atr_col_name = f"ATR_{optimized_strategy_config.get('indicators', {}).get('atr_window', 14)}"
        oos_atr_series_full = oos_data_with_signals[atr_col_name] if atr_col_name in oos_data_with_signals else None

        # Filter for the actual OOS test range (test_start to test_end)
        oos_ohlc_data_test = oos_ohlc_data_full.loc[test_start:test_end]
        oos_target_allocations_test = oos_target_allocations.reindex(oos_ohlc_data_test.index)
        oos_stop_losses_test = oos_stop_losses.reindex(oos_ohlc_data_test.index)
        oos_atr_series_test = oos_atr_series_full.reindex(oos_ohlc_data_test.index) if oos_atr_series_full is not None else pd.Series(np.nan, index=oos_ohlc_data_test.index)


        if oos_ohlc_data_test.empty:
             logger.warning("No OHLC data in OOS test range after filtering. Skipping period.")
             continue

        # Use equity from end of previous period as starting capital for this one
        current_initial_capital = cumulative_equity

        # Instantiate RiskManager and Backtester for OOS run
        oos_risk_manager_config = optimized_strategy_config.get('risk_manager', {})
        oos_risk_manager = RiskManager(config=oos_risk_manager_config)
        oos_backtester = Backtester(
            risk_manager=oos_risk_manager, # Pass risk manager
            initial_capital=current_initial_capital, # Use carried-over capital
            commission=backtest_config.get('parameters', {}).get('commission', 0.001) # Get commission from backtest params
        )

        oos_result = oos_backtester.backtest(
            ohlc_data=oos_ohlc_data_test, # Pass OHLC DataFrame
            signals=oos_target_allocations_test,
            stop_losses=oos_stop_losses_test,
            atr_series=oos_atr_series_test # Pass ATR series
        )
        # Update cumulative equity for the next period
        cumulative_equity = oos_result.equity_curve.iloc[-1]

        logger.info(f"OOS Period {i+1} Backtest Complete. Return: {oos_result.total_return:.2%}, End Equity: {cumulative_equity:.2f}")
        all_oos_results.append(oos_result)
        all_oos_prices.append(oos_ohlc_data_test['Close']) # Store the Close prices used for this OOS backtest

    # --- 6. Aggregate & Analyze Results ---
    logger.info("\n=== Aggregating and Analyzing Walk-Forward Results ===")
    if not all_oos_results:
        logger.error("No OOS periods were successfully backtested.")
        return

    # Combine OOS equity curves and positions
    # Adjust equity curves to be cumulative
    combined_equity_curve_list = []
    # initial_capital_overall = backtest_config.get('parameters', {}).get('initial_capital', 100000.0) # Use the initial capital from config
    last_equity = initial_capital_run # Use the capital from the start of the run
    for i, result in enumerate(all_oos_results):
        if i == 0:
            combined_equity_curve_list.append(result.equity_curve)
            last_equity = result.equity_curve.iloc[-1]
        else:
            # Scale subsequent curves by the ratio of start_equity/initial_capital_of_that_run
            # Ensure result.equity_curve.iloc[0] is not zero or close to zero
            start_equity_this_run = result.equity_curve.iloc[0]
            if abs(start_equity_this_run) < 1e-9:
                 logger.warning(f"Initial equity for OOS period {i+1} is near zero ({start_equity_this_run}). Cannot scale accurately. Appending unscaled.")
                 scaled_curve = result.equity_curve # Append unscaled if start is zero
            else:
                 scaling_factor = all_oos_results[i-1].equity_curve.iloc[-1] / start_equity_this_run
                 scaled_curve = result.equity_curve * scaling_factor

            combined_equity_curve_list.append(scaled_curve.iloc[1:]) # Exclude first point to avoid duplication
            last_equity = scaled_curve.iloc[-1]

    combined_equity_curve = pd.concat(combined_equity_curve_list)
    combined_positions = [pos for res in all_oos_results for pos in res.positions]
    # Combine prices carefully, avoiding duplicates if periods overlap (though rolling shouldn't overlap)
    combined_prices = pd.concat(all_oos_prices).sort_index()
    combined_prices = combined_prices[~combined_prices.index.duplicated(keep='first')]


    # --- Recalculate Overall Metrics on Combined OOS Performance ---
    final_equity = combined_equity_curve.iloc[-1]
    overall_total_return = (final_equity / initial_capital_run) - 1 # Use initial capital of the entire run

    combined_daily_returns = combined_equity_curve.pct_change().fillna(0)
    if combined_daily_returns.index.tz is not None:
        combined_daily_returns = combined_daily_returns.tz_localize(None) # For QuantStats

    # Use QuantStats for robust metric calculation on combined returns
    overall_sharpe = qs.stats.sharpe(combined_daily_returns, periods=252) # Assuming daily
    overall_sortino = qs.stats.sortino(combined_daily_returns, periods=252)
    overall_max_drawdown = qs.stats.max_drawdown(combined_equity_curve) # Use equity curve for drawdown

    winning_trades = sum(1 for p in combined_positions if p.pnl is not None and p.pnl > 0)
    losing_trades = sum(1 for p in combined_positions if p.pnl is not None and p.pnl < 0)
    total_trades = len(combined_positions)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    wins = [p.pnl for p in combined_positions if p.pnl is not None and p.pnl > 0]
    losses = [p.pnl for p in combined_positions if p.pnl is not None and p.pnl < 0]
    avg_win_pct = (np.mean(wins) / initial_capital_run) if wins else 0.0
    avg_loss_pct = abs(np.mean(losses) / initial_capital_run) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

    # Create a final BacktestResult object for the combined OOS performance
    combined_oos_result = BacktestResult(
         total_return=overall_total_return,
         sharpe_ratio=overall_sharpe,
         max_drawdown=overall_max_drawdown,
         total_trades=total_trades,
         winning_trades=winning_trades,
         losing_trades=losing_trades,
         win_rate=win_rate,
         avg_win=avg_win_pct,
         avg_loss=avg_loss_pct,
         profit_factor=profit_factor,
         positions=combined_positions, # Combined list of all trades
         equity_curve=combined_equity_curve # Combined equity curve
    )

    logger.info(f"Overall OOS Total Return: {overall_total_return:.2%}")
    logger.info(f"Overall OOS Sharpe Ratio: {overall_sharpe:.2f}")
    logger.info(f"Overall OOS Sortino Ratio: {overall_sortino:.2f}")
    logger.info(f"Overall OOS Max Drawdown: {overall_max_drawdown:.2%}")

    # --- Load Benchmark Data ---
    benchmark_ticker = backtest_config.get('benchmark_ticker', '^GSPC')
    logger.info(f"Loading benchmark data: {benchmark_ticker}")
    try:
        # Download returns for the combined OOS period
        oos_start_dt = combined_equity_curve.index.min()
        oos_end_dt = combined_equity_curve.index.max()
        benchmark_returns = qs.utils.download_returns(benchmark_ticker)
        # Ensure benchmark index is datetime and potentially timezone-aware
        if not isinstance(benchmark_returns.index, pd.DatetimeIndex):
             benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
        if benchmark_returns.index.tz is None:
             # Localize to UTC to match our equity curve if it became tz-aware during concat
             if combined_equity_curve.index.tz is not None:
                  benchmark_returns = benchmark_returns.tz_localize('UTC')
        elif combined_equity_curve.index.tz is not None:
             # Convert benchmark to match equity curve timezone if both are aware
             benchmark_returns = benchmark_returns.tz_convert(combined_equity_curve.index.tz)
        # If equity curve is naive but benchmark is aware, make benchmark naive
        elif benchmark_returns.index.tz is not None:
             benchmark_returns = benchmark_returns.tz_localize(None)


        benchmark_returns = benchmark_returns.loc[oos_start_dt:oos_end_dt]
        benchmark_returns = benchmark_returns.reindex(combined_equity_curve.index).fillna(0) # Align and fill
        logger.info(f"Benchmark data loaded and aligned ({len(benchmark_returns)} points).")
    except Exception as e:
        logger.error(f"Failed to download or align benchmark '{benchmark_ticker}': {e}")
        benchmark_returns = None

    # --- Save aggregated results and perform final analysis ---
    # Pass the combined result object and benchmark, suppressing specific warnings
    logger.info("Performing final analysis on combined OOS results...")
    with warnings.catch_warnings():
        # Suppress RuntimeWarning: invalid value encountered in scalar divide
        # Originating from quantstats/stats.py lines 294 and 349 (Sharpe/Sortino)
        # Note: This will only suppress the warning if it's triggered directly by this call
        # or within quantstats functions called *during* this execution context.
        warnings.filterwarnings(
            'ignore',
            message='invalid value encountered in scalar divide',
            category=RuntimeWarning,
            module='quantstats\.stats' # Match the module where the warning originates
        )
        try:
            analyze_and_save_results(
                output_dir,
                combined_oos_result,
                combined_prices, # Pass combined prices for plotting trades over full OOS period
                benchmark_returns,
                initial_capital_run # Pass initial capital for MC sim scaling
            )
        except RuntimeWarning as rw:
             # Log if the specific warning we tried to suppress still occurred
             if 'invalid value encountered in scalar divide' in str(rw):
                  logger.warning(f"Suppressed RuntimeWarning occurred during final analysis: {rw}")
             else:
                  raise # Re-raise other RuntimeWarnings
        except Exception as e:
             logger.error(f"Error during final analysis execution: {e}")
             # Decide if processing should stop or continue

    logger.info("Walk-forward analysis complete.")
    # Save best parameters found per period
    with open(output_dir / "best_params_per_period.yaml", 'w') as f:
        yaml.dump(best_params_per_period, f, default_flow_style=False)
    logger.info(f"Optimized parameters saved to {output_dir / 'best_params_per_period.yaml'}")


if __name__ == "__main__":
    # Note: Optuna studies can take time. Consider running specific parts if needed.
    asyncio.run(main())