#!/usr/bin/env python3
"""
Performance Analysis Module for AlphaPulse Backtests.

Provides functions to generate reports and visualizations based on backtest results.
"""

import logging
import pandas as pd
import numpy as np # Add numpy import
import quantstats as qs
from pathlib import Path
from typing import List, Optional
import pandas.errors # Add explicit import for pandas errors
 
 # Assuming BacktestResult and Position are defined elsewhere (e.g., backtester module)
# Adjust import if necessary
try:
    from ..backtesting.backtester import BacktestResult
    from ..backtesting.models import Position
except ImportError:
    # Define dummy classes if imports fail, to allow module loading
    logging.error("Could not import BacktestResult or Position from backtesting module.")
    class BacktestResult: pass
    class Position: pass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PerformanceAnalyzer")

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
        # Workaround for pandas resampling issue in QuantStats
        try:
            qs.reports.html(
                returns=returns,
                benchmark=benchmark,
                output=str(output_path), # QuantStats expects string path
                title=title,
                download_filename=str(output_path.name) # Use Path object's name attribute
            )
        except pandas.errors.UnsupportedFunctionCall as e:
            # Handle the specific resampling error
            if "numpy operations are not valid with resample" in str(e):
                logger.warning("Encountered pandas resampling issue in QuantStats. Using alternative approach.")
                # Create a simpler report without the problematic resampling operations
                with open(str(output_path), 'w') as f:
                    f.write(f"<html><head><title>{title}</title></head><body>")
                    f.write("<h1>Strategy Performance Summary</h1>")
                    
                    # Calculate key metrics manually
                    total_return = ((returns + 1).prod() - 1) * 100
                    annual_return = ((returns + 1).prod() ** (252/len(returns)) - 1) * 100
                    sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)
                    max_dd = (returns.cumsum() - returns.cumsum().cummax()).min() * 100
                    
                    # Write metrics to HTML
                    f.write(f"<p>Total Return: {total_return:.2f}%</p>")
                    f.write(f"<p>Annualized Return: {annual_return:.2f}%</p>")
                    f.write(f"<p>Sharpe Ratio: {sharpe:.2f}</p>")
                    f.write(f"<p>Max Drawdown: {max_dd:.2f}%</p>")
                    
                    f.write("</body></html>")
                logger.info("Generated simplified performance report due to QuantStats compatibility issue.")
            else:
                # Re-raise if it's a different error
                raise
        logger.info("QuantStats report generated successfully.")
    except ImportError:
         logger.error("QuantStats not installed. Skipping report generation.")
    except Exception as e:
        # Catch specific QuantStats/Pandas errors if possible
        logger.error(f"Error generating QuantStats report: {e}", exc_info=True) # Log traceback


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# --- End Custom Plotting Functions ---


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

    # Define output directory
    example_output_dir = Path("./temp_analysis_output")
    example_output_dir.mkdir(exist_ok=True)

    # 1. Generate QuantStats Report
    qs_report_path = example_output_dir / "example_quantstats_report.html"
    generate_quantstats_report(dummy_returns, benchmark_returns=dummy_benchmark, output_path=qs_report_path, title="Example Strategy")

    # 2. Generate Custom Plots (Placeholders)
    plot_equity_curve_and_drawdown(dummy_equity_curve, output_path=example_output_dir / "equity_drawdown.png")
    # Need dummy prices aligned with returns for trade plot
    dummy_prices = dummy_equity_curve # Use equity as proxy for price for example
    plot_trades_on_price(dummy_prices, dummy_positions, output_path=example_output_dir / "trades_on_price.png")
    plot_rolling_sharpe(dummy_returns, output_path=example_output_dir / "rolling_sharpe.png")

    logger.info(f"Example analysis complete. Check output in: {example_output_dir}")