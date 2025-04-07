#!/usr/bin/env python3
"""
Performance Analysis Module for AlphaPulse Backtests.

Provides functions to generate reports and visualizations based on backtest results.
"""

from loguru import logger # Use loguru
import pandas as pd
import numpy as np
import quantstats as qs
from pathlib import Path
from typing import List, Optional
import yaml # Add yaml import

# Assuming BacktestResult and Position are defined elsewhere (e.g., backtester module)
# Adjust import if necessary
try:
    from ..backtesting.backtester import BacktestResult
    from ..backtesting.models import Position
except ImportError:
    # Define dummy classes if imports fail, to allow module loading
    logger.error("Could not import BacktestResult or Position from backtesting module.") # Use logger
    class BacktestResult: pass
    class Position: pass


# Loguru logger is imported directly
# logger = logger.bind(name="PerformanceAnalyzer") # Optional binding

# Import plotting libraries here to keep them contained
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib # Import matplotlib base

# Set default font to avoid Arial warnings on Linux
try:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
except Exception as e:
    logger.warning(f"Could not set default Matplotlib font: {e}")


def generate_quantstats_report(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    output_path: Path = Path("quantstats_report.html"),
    title: str = "Strategy Performance"
):
    """
    Generates a comprehensive performance report using QuantStats.

    Args:
        returns: Series of portfolio returns (daily or other frequency).
                 Index must be DatetimeIndex.
        benchmark_returns: Optional Series of benchmark returns (e.g., S&P 500).
                           Must have the same index as returns.
        output_path: Path object for the output HTML file.
        title: Title for the report.
    """
    if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
        logger.error("Input 'returns' must be a pandas Series with a DatetimeIndex.")
        return
    if returns.empty:
        logger.error("Input 'returns' Series is empty. Cannot generate report.")
        return

    # Ensure returns index is timezone-naive, as QuantStats often expects this
    if returns.index.tz is not None:
        logger.debug("Converting returns index to timezone-naive for QuantStats.")
        returns = returns.tz_localize(None)
        # Try setting frequency explicitly to 'B' (Business Day) as a workaround for potential QuantStats/Pandas issues
        try:
            returns = returns.asfreq('B').fillna(0) # Fill potential NaNs introduced by asfreq
            logger.debug("Set returns frequency to 'B' and filled NaNs.")
        except Exception as e:
            logger.warning(f"Could not set frequency for returns index: {e}")


    # Prepare benchmark if provided
    benchmark = None
    if benchmark_returns is not None:
        if not isinstance(benchmark_returns, pd.Series) or not isinstance(benchmark_returns.index, pd.DatetimeIndex):
            logger.warning("Benchmark returns must be a pandas Series with a DatetimeIndex. Skipping benchmark.")
        else:
            # Align benchmark to returns index
            benchmark_returns = benchmark_returns.reindex(returns.index)
            if benchmark_returns.index.tz is not None:
                 benchmark_returns = benchmark_returns.tz_localize(None)
            # Check for NaNs after reindexing
            if benchmark_returns.isnull().any():
                 logger.warning("Benchmark returns contain NaNs after aligning to strategy returns. Forward-filling.")
                 benchmark_returns = benchmark_returns.ffill().fillna(0) # Fill initial NaNs with 0
            benchmark = benchmark_returns
            logger.info("Using benchmark for QuantStats report.")


    logger.info(f"Generating QuantStats report to: {output_path}")
    try:
        # Extend QuantStats functionality if needed (e.g., custom metrics)
        qs.extend_pandas()

        # Generate the HTML report
        qs.reports.html(
            returns=returns,
            benchmark=benchmark, # Pass the prepared benchmark series
            output=str(output_path), # QuantStats expects string path
            title=title,
            download_filename=str(output_path.name) # Use Path object's name attribute
        )
        logger.info("QuantStats report generated successfully.")
    except ImportError:
         logger.error("QuantStats not installed. Skipping report generation.")
    except Exception as e:
        # Catch specific QuantStats/Pandas errors if possible
        logger.error(f"Error generating QuantStats report: {e}", exc_info=True) # Log traceback


# --- Custom Plotting Functions ---

def plot_equity_curve_and_drawdown(
   equity_curve: pd.Series,
   title: str = "Equity Curve and Drawdown",
   output_path: Optional[Path] = None
):
   """Generates a plot showing the equity curve and drawdown periods."""
   if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
       logger.error("Equity curve is empty or not a Series. Cannot plot.")
       return

   logger.info("Generating Equity Curve and Drawdown plot...")
   try:
       # Ensure index is datetime
       if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve.index = pd.to_datetime(equity_curve.index)

       # Calculate drawdown
       rolling_max = equity_curve.expanding(min_periods=1).max()
       drawdown = (equity_curve - rolling_max) / rolling_max
       drawdown = drawdown * 100 # Convert to percentage

       # Create figure and axes
       fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
                                      gridspec_kw={'height_ratios': [3, 1]})
       fig.suptitle(title, fontsize=16)

       # Plot Equity Curve
       ax1.plot(equity_curve.index, equity_curve, label='Equity Curve', color='blue')
       ax1.set_ylabel('Portfolio Value ($)')
       ax1.set_title('Equity Curve')
       ax1.grid(True, linestyle='--', alpha=0.6)
       ax1.legend()
       ax1.ticklabel_format(style='plain', axis='y') # Avoid scientific notation

       # Plot Drawdown
       ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
       ax2.plot(drawdown.index, drawdown, color='red', linewidth=1) # Outline drawdown
       ax2.set_ylabel('Drawdown (%)')
       ax2.set_title('Drawdown')
       ax2.set_xlabel('Date')
       ax2.grid(True, linestyle='--', alpha=0.6)
       ax2.legend()

       # Format x-axis dates
       fig.autofmt_xdate()
       ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
       # Optional: Adjust date tick frequency if needed
       # ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

       plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

       if output_path:
           plt.savefig(output_path, dpi=300)
           logger.info(f"Equity/Drawdown plot saved to: {output_path}")
       else:
           plt.show()
       plt.close(fig) # Close the figure to free memory

   except Exception as e:
       logger.error(f"Error generating equity/drawdown plot: {e}")


def plot_trades_on_price(
    prices: pd.Series,
    positions: List[Position],
    title: str = "Trades on Price Chart",
    output_path: Optional[Path] = None
):
    """Generates a price chart with markers for trade entries and exits."""
    if not isinstance(prices, pd.Series) or prices.empty:
        logger.error("Price series is empty or not a Series. Cannot plot trades.")
        return
    if not positions:
        logger.warning("No positions provided. Skipping trade plot.")
        return

    logger.info("Generating Trades on Price plot...")
    try:
        # Ensure index is datetime
        if not isinstance(prices.index, pd.DatetimeIndex):
             prices.index = pd.to_datetime(prices.index)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(prices.index, prices, label='Price', color='black', alpha=0.8, linewidth=1)

        # Plot entry/exit markers
        long_entries = [p.entry_time for p in positions if p.size > 0]
        long_exits = [p.exit_time for p in positions if p.size > 0 and p.exit_time is not None]
        short_entries = [p.entry_time for p in positions if p.size < 0]
        short_exits = [p.exit_time for p in positions if p.size < 0 and p.exit_time is not None]

        # Ensure timestamps exist in the price index before plotting
        long_entries_idx = prices.index.intersection(long_entries)
        long_exits_idx = prices.index.intersection(long_exits)
        short_entries_idx = prices.index.intersection(short_entries)
        short_exits_idx = prices.index.intersection(short_exits)

        if not long_entries_idx.empty:
            ax.plot(long_entries_idx, prices.loc[long_entries_idx], '^', markersize=8, color='lime', label='Long Entry', alpha=0.9)
        if not long_exits_idx.empty:
            ax.plot(long_exits_idx, prices.loc[long_exits_idx], 'o', markersize=8, color='green', label='Long Exit', alpha=0.9)
        if not short_entries_idx.empty:
            ax.plot(short_entries_idx, prices.loc[short_entries_idx], 'v', markersize=8, color='red', label='Short Entry', alpha=0.9)
        if not short_exits_idx.empty:
            ax.plot(short_exits_idx, prices.loc[short_exits_idx], 'o', markersize=8, color='maroon', label='Short Exit', alpha=0.9)

        ax.set_title(title)
        ax.set_ylabel('Price')
        ax.set_xlabel('Date')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.ticklabel_format(style='plain', axis='y')

        # Format x-axis dates
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Trades plot saved to: {output_path}")
        else:
            plt.show()
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error generating trades plot: {e}")


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 252, # Default to rolling annual Sharpe
    title: str = "Rolling Sharpe Ratio",
    output_path: Optional[Path] = None
):
    """Generates a plot of the rolling Sharpe ratio."""
    if not isinstance(returns, pd.Series) or returns.empty:
        logger.error("Returns series is empty or not a Series. Cannot plot rolling Sharpe.")
        return
    if len(returns) < window:
        logger.warning(f"Not enough returns ({len(returns)}) for rolling window ({window}). Skipping plot.")
        return

    logger.info(f"Generating Rolling Sharpe Ratio ({window}-period) plot...")
    try:
        # Ensure index is datetime
        if not isinstance(returns.index, pd.DatetimeIndex):
             returns.index = pd.to_datetime(returns.index)

        # Calculate rolling annualized Sharpe ratio (assuming 0 risk-free rate)
        # Note: periods_per_year should match the frequency of returns (e.g., 252 for daily, 52 for weekly)
        # This assumes daily returns for now. Adjust if needed.
        periods_per_year = 252
        rolling_mean = returns.rolling(window=window, min_periods=window).mean() * periods_per_year
        rolling_std = returns.rolling(window=window, min_periods=window).std() * np.sqrt(periods_per_year)
        rolling_sharpe = rolling_mean / rolling_std
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan) # Handle division by zero

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(rolling_sharpe.index, rolling_sharpe, label=f'Rolling Sharpe ({window}-period)', color='purple')
        ax.axhline(0, color='grey', linestyle='--', linewidth=1) # Add zero line

        ax.set_title(title)
        ax.set_ylabel('Annualized Sharpe Ratio')
        ax.set_xlabel('Date')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Format x-axis dates
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Rolling Sharpe plot saved to: {output_path}")
        else:
            plt.show()
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error generating rolling Sharpe plot: {e}")

def run_monte_carlo_simulation(
   returns: pd.Series,
   initial_capital: float = 100000.0,
   simulations: int = 1000,
   output_path: Optional[Path] = None,
   title: str = "Monte Carlo Simulation"
):
   """
   Performs a Monte Carlo simulation by shuffling daily returns.

   Args:
       returns: Series of daily returns.
       initial_capital: Starting capital for simulations.
       simulations: Number of simulation paths to generate.
       output_path: Path object to save the plot.
       title: Title for the plot.
   """
   if not isinstance(returns, pd.Series) or returns.empty:
       logger.error("Returns series is empty or not a Series. Cannot run Monte Carlo.")
       return
   if simulations <= 0:
       logger.error("Number of simulations must be positive.")
       return

   logger.info(f"Running Monte Carlo simulation ({simulations} paths)...")
   try:
       # Ensure returns index is datetime
       if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

       # Collect simulation results in a dictionary first
       sim_data = {}
       returns_array = returns.to_numpy()

       for i in range(simulations):
           # Shuffle daily returns
           shuffled_returns = np.random.choice(returns_array, size=len(returns_array), replace=True)
           # Calculate cumulative equity path
           sim_equity = initial_capital * (1 + pd.Series(shuffled_returns, index=returns.index)).cumprod()
           sim_data[f'Sim_{i+1}'] = sim_equity

       # Create DataFrame from the dictionary in one go
       sim_results = pd.DataFrame(sim_data)

       # Plotting the simulation paths
       fig, ax = plt.subplots(figsize=(12, 7))
       sim_results.plot(ax=ax, legend=False, alpha=0.1, color='grey') # Plot individual simulations faintly

       # Plot the actual equity curve (need to recalculate or pass it)
       actual_equity = initial_capital * (1 + returns).cumprod()
       ax.plot(actual_equity.index, actual_equity, color='red', linewidth=2, label='Actual Equity Curve')

       # Plot median and percentiles (e.g., 5th and 95th)
       median_equity = sim_results.median(axis=1)
       percentile_5 = sim_results.quantile(0.05, axis=1)
       percentile_95 = sim_results.quantile(0.95, axis=1)

       ax.plot(median_equity.index, median_equity, color='blue', linestyle='--', linewidth=2, label='Median Simulation')
       ax.plot(percentile_5.index, percentile_5, color='cyan', linestyle=':', linewidth=1.5, label='5th Percentile')
       ax.plot(percentile_95.index, percentile_95, color='cyan', linestyle=':', linewidth=1.5, label='95th Percentile')


       ax.set_title(title)
       ax.set_ylabel('Portfolio Value ($)')
       ax.set_xlabel('Date')
       ax.grid(True, linestyle='--', alpha=0.6)
       ax.legend()
       ax.ticklabel_format(style='plain', axis='y')

       # Format x-axis dates
       fig.autofmt_xdate()
       ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

       plt.tight_layout()

       if output_path:
           plt.savefig(output_path, dpi=300)
           logger.info(f"Monte Carlo plot saved to: {output_path}")
       else:
           plt.show()
       plt.close(fig)

       # Log some summary stats from the simulation
       final_equity_dist = sim_results.iloc[-1]
       logger.info(f"Monte Carlo Final Equity Summary:")
       logger.info(f"  Actual: {actual_equity.iloc[-1]:,.2f}")
       logger.info(f"  Median: {final_equity_dist.median():,.2f}")
       logger.info(f"  5th Pctl: {final_equity_dist.quantile(0.05):,.2f}")
       logger.info(f"  95th Pctl: {final_equity_dist.quantile(0.95):,.2f}")

   except Exception as e:
       logger.error(f"Error running Monte Carlo simulation: {e}")


# --- End Custom Plotting Functions ---


# --- Main Analysis Function ---
def analyze_and_save_results(
    output_dir: Path,
    result: BacktestResult,
    price_series: pd.Series, # Prices used for the backtest (for plotting trades)
    benchmark_returns: Optional[pd.Series] = None, # Add benchmark returns
    initial_capital: float = 100000.0 # Add initial capital for MC sim
):
    """Salva i risultati del backtest ed esegue l'analisi delle performance."""
    if result is None:
        logger.error("Backtest result is None. Cannot save or analyze.")
        return
    if not isinstance(result, BacktestResult):
        logger.error(f"Expected BacktestResult object, got {type(result)}. Cannot save or analyze.")
        return

    logger.info("--- Analisi e Salvataggio Risultati ---")
    try:
        # --- Salvataggio Dati Grezzi ---
        result.equity_curve.to_csv(output_dir / "equity_curve.csv")
        logger.debug(f"Curva equity salvata in {output_dir / 'equity_curve.csv'}") # DEBUG

        if result.positions:
            positions_df = pd.DataFrame([p.__dict__ for p in result.positions])
            # Convert datetime columns if needed (adjust format as necessary)
            for col in ['entry_time', 'exit_time']:
                 if col in positions_df.columns and positions_df[col].notna().any():
                      # Ensure conversion handles potential NaT values gracefully
                      positions_df[col] = pd.to_datetime(positions_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            positions_df.to_csv(output_dir / "trades_history.csv", index=False)
            logger.debug(f"Storia operazioni salvata in {output_dir / 'trades_history.csv'}") # DEBUG
        else:
            logger.info("Nessuna operazione eseguita, file trades_history.csv non creato.") # Keep INFO for this case

        # --- Salvataggio Riepilogo Metriche ---
        summary = {
            "Total Return": f"{result.total_return:.2%}",
            "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{qs.stats.sortino(result.equity_curve.pct_change()):.2f}", # Calculate Sortino here too
            "Max Drawdown": f"{result.max_drawdown:.2%}",
            "Total Trades": result.total_trades,
            "Winning Trades": result.winning_trades,
            "Losing Trades": result.losing_trades,
            "Win Rate": f"{result.win_rate:.2%}",
            "Avg Win Pct": f"{result.avg_win:.2%}",
            "Avg Loss Pct": f"{result.avg_loss:.2%}",
            "Profit Factor": f"{result.profit_factor:.2f}",
        }
        with open(output_dir / "summary.yaml", 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        logger.debug(f"Riepilogo metriche salvato in {output_dir / 'summary.yaml'}") # DEBUG

        # --- Generazione Report QuantStats (only if trades occurred) ---
        daily_returns = result.equity_curve.pct_change().fillna(0)
        if result.total_trades > 0:
            generate_quantstats_report(
                returns=daily_returns,
                benchmark_returns=benchmark_returns, # Pass benchmark here
                output_path=output_dir / "quantstats_report.html",
                title="Strategy Performance Analysis"
            )
        else:
            logger.warning("Skipping QuantStats report generation as no trades were executed.")


        # --- Generazione Grafici Custom ---
        logger.debug("Generating Equity Curve and Drawdown plot...") # DEBUG
        plot_equity_curve_and_drawdown(
            equity_curve=result.equity_curve,
            output_path=output_dir / "quantstats_report.html",
            title="Strategy Performance Analysis"
        )

        # --- Generazione Grafici Custom ---
        plot_equity_curve_and_drawdown(
            equity_curve=result.equity_curve,
            output_path=output_dir / "equity_drawdown.png"
        )
        logger.debug("Generating Trades on Price plot...") # DEBUG
        plot_trades_on_price(
            prices=price_series, # Pass original price series used for backtest
            positions=result.positions,
            output_path=output_dir / "trades_on_price.png"
        )
        logger.debug("Generating Rolling Sharpe Ratio (252-period) plot...") # DEBUG
        plot_rolling_sharpe(
            returns=daily_returns,
            output_path=output_dir / "rolling_sharpe.png"
        )

        # --- Esecuzione Simulazione Monte Carlo ---
        logger.debug("Running Monte Carlo simulation (1000 paths)...") # DEBUG
        run_monte_carlo_simulation(
            returns=daily_returns,
            initial_capital=initial_capital,
            output_path=output_dir / "monte_carlo_simulation.png"
        )

    except Exception as e:
        logger.exception(f"Errore durante il salvataggio o l'analisi dei risultati: {e}", exc_info=True) # Add traceback


if __name__ == '__main__':
    # Example Usage (requires sample data)
    logger.info("Running performance_analyzer example...")

    # Create dummy returns data
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    dummy_returns = pd.Series(np.random.normal(0.0005, 0.01, size=500), index=dates)
    dummy_benchmark = pd.Series(np.random.normal(0.0003, 0.008, size=500), index=dates)

    # Create dummy equity curve
    initial_equity = 100000
    dummy_equity_curve = initial_equity * (1 + dummy_returns).cumprod()

    # Create dummy positions list (replace with actual Position objects)
    dummy_positions = [
        # Example Position objects would go here if needed for plot_trades_on_price
    ]
    # Create dummy BacktestResult
    dummy_result = BacktestResult(
        total_return=(dummy_equity_curve.iloc[-1] / initial_equity) - 1,
        sharpe_ratio=qs.stats.sharpe(dummy_returns),
        max_drawdown=qs.stats.max_drawdown(dummy_equity_curve),
        total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
        avg_win=0, avg_loss=0, profit_factor=0,
        positions=dummy_positions, equity_curve=dummy_equity_curve
    )


    # Define output directory
    example_output_dir = Path("./temp_analysis_output")
    example_output_dir.mkdir(exist_ok=True)

    # Call the main analysis function
    analyze_and_save_results(
        example_output_dir,
        dummy_result,
        dummy_equity_curve, # Use equity as proxy for price
        dummy_benchmark,
        initial_equity
    )


    logger.info(f"Example analysis complete. Check output in: {example_output_dir}")