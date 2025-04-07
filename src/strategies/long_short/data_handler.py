#!/usr/bin/env python3
"""
Data Handler for the Long/Short S&P 500 Strategy.

Fetches and prepares S&P 500 (^GSPC) and VIX (^VIX) data,
handles caching, and provides resampling capabilities.
"""

import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LongShortDataHandler")

class LongShortDataHandler:
    """
    Handles fetching, caching, and resampling of data for the Long/Short strategy.
    """

    def __init__(self, config: Dict):
        """
        Initializes the Data Handler.

        Args:
            config: Configuration dictionary, expects a 'cache_dir' key.
        """
        self.cache_dir = Path(config.get("cache_dir", "./data/cache/long_short"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LongShortDataHandler initialized. Cache directory: {self.cache_dir}")

    def _fetch_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Internal method to fetch data for a given ticker using yfinance and cache it.

        Args:
            ticker: The stock ticker symbol (e.g., '^GSPC', '^VIX').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            A pandas DataFrame with the historical data, or None if fetching fails.
        """
        # Sanitize ticker for filename
        safe_ticker = ticker.replace('^', '')
        cache_file = self.cache_dir / f"{safe_ticker}_{start_date}_{end_date}.parquet"

        # Check cache first
        if cache_file.exists():
            logger.info(f"Loading {ticker} data from cache: {cache_file}")
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load {ticker} from cache {cache_file}: {e}. Refetching.")

        # Fetch from yfinance
        logger.info(f"Fetching {ticker} data from yfinance ({start_date} to {end_date})")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                logger.warning(f"No data returned for {ticker} from yfinance.")
                return None

            # Ensure index is DatetimeIndex and select relevant columns
            data.index = pd.to_datetime(data.index)
            # Keep OHLCV and Adj Close
            cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            data = data[[col for col in cols_to_keep if col in data.columns]]

            # Save to cache
            try:
                data.to_parquet(cache_file)
                logger.info(f"Saved {ticker} data to cache: {cache_file}")
            except Exception as e:
                logger.error(f"Failed to save {ticker} data to cache {cache_file}: {e}")

            return data
        except Exception as e:
            logger.error(f"Error fetching {ticker} data from yfinance: {e}")
            return None

    def fetch_sp500_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches S&P 500 (^GSPC) historical data.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            A pandas DataFrame with S&P 500 data, or None on failure.
        """
        return self._fetch_data("^GSPC", start_date, end_date)

    def fetch_vix_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches VIX (^VIX) historical data.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            A pandas DataFrame with VIX data, or None on failure.
        """
        return self._fetch_data("^VIX", start_date, end_date)

    def get_combined_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches both S&P 500 and VIX data and combines them into a single DataFrame.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            A pandas DataFrame containing 'Adj Close' for S&P 500 and VIX,
            or None if fetching fails for either.
        """
        sp500_data = self.fetch_sp500_data(start_date, end_date)
        vix_data = self.fetch_vix_data(start_date, end_date)

        if sp500_data is None or vix_data is None:
            logger.error("Failed to fetch either S&P 500 or VIX data. Cannot combine.")
            logger.error("Failed to fetch either S&P 500 or VIX data. Cannot combine.")
            return None

        # Rename columns for clarity before merging
        sp500_data = sp500_data.add_prefix('SP500_')
        vix_data = vix_data.add_prefix('VIX_')

        # Combine dataframes using SP500 index as the primary one
        combined_data = pd.merge(sp500_data, vix_data, left_index=True, right_index=True, how='left')

        # Forward-fill VIX data for potential missing days (weekends, holidays mismatch)
        vix_cols = [col for col in combined_data.columns if col.startswith('VIX_')]
        combined_data[vix_cols] = combined_data[vix_cols].ffill()

        # Drop rows where essential SP500 data might be NaN (e.g., at the very start if VIX starts later)
        essential_sp500_cols = ['SP500_Adj Close', 'SP500_High', 'SP500_Low', 'SP500_Close'] # Use Adj Close for strategy? Or Close? Let's assume Adj Close for now.
        combined_data.dropna(subset=['SP500_Adj Close'], inplace=True) # Drop if primary price is missing

        # Optional: Drop rows if VIX is still NaN after ffill (might happen at the beginning of the series)
        # combined_data.dropna(subset=['VIX_Adj Close'], inplace=True) # Decide if VIX is strictly required from day 1

        logger.info(f"Combined S&P 500 and VIX data from {start_date} to {end_date}. Shape: {combined_data.shape}")
        logger.debug(f"Combined columns: {combined_data.columns.tolist()}")
        return combined_data

    def resample_data(self, data: pd.DataFrame, timeframe: str = 'W') -> Optional[pd.DataFrame]:
        """
        Resamples the combined data to the specified timeframe (Weekly or Monthly).

        Args:
            data: The combined daily data DataFrame.
            timeframe: The target timeframe ('W' for weekly, 'M' for monthly). Default is 'W'.

        Returns:
            A pandas DataFrame with the resampled data, or None if input is invalid.
        """
        if data is None or data.empty:
            logger.error("Cannot resample empty or None DataFrame.")
            return None
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be a DatetimeIndex for resampling.")
            return None

        allowed_timeframes = {'W', 'M'}
        if timeframe not in allowed_timeframes:
            logger.error(f"Invalid timeframe '{timeframe}'. Allowed: {allowed_timeframes}")
            return None

        logger.info(f"Resampling data to timeframe: {timeframe}")

        # Define aggregation rules for OHLCV data
        aggregation_rules = {}
        prefixes = ['SP500_', 'VIX_']
        for prefix in prefixes:
            if f'{prefix}Open' in data.columns:
                aggregation_rules[f'{prefix}Open'] = 'first'
            if f'{prefix}High' in data.columns:
                aggregation_rules[f'{prefix}High'] = 'max'
            if f'{prefix}Low' in data.columns:
                aggregation_rules[f'{prefix}Low'] = 'min'
            if f'{prefix}Close' in data.columns:
                aggregation_rules[f'{prefix}Close'] = 'last'
            if f'{prefix}Adj Close' in data.columns:
                aggregation_rules[f'{prefix}Adj Close'] = 'last' # Use last adjusted close for the period
            if f'{prefix}Volume' in data.columns:
                aggregation_rules[f'{prefix}Volume'] = 'sum'

        # Ensure we only try to aggregate columns that actually exist
        valid_aggregation_rules = {k: v for k, v in aggregation_rules.items() if k in data.columns}

        if not valid_aggregation_rules:
            logger.error("No valid columns found for resampling aggregation.")
            return None

        try:
            resampled_data = data.resample(timeframe).agg(valid_aggregation_rules)
            # Drop rows where all values might be NaN after resampling (e.g., start of series)
            resampled_data.dropna(how='all', inplace=True)
            logger.info(f"Resampling complete. New shape: {resampled_data.shape}")
            return resampled_data
        except Exception as e:
            logger.error(f"Error during resampling to {timeframe}: {e}")
            return None


if __name__ == "__main__":
    # Example Usage
    config_example = {"cache_dir": "./data/cache/long_short_test"}
    data_handler = LongShortDataHandler(config_example)

    start = "2020-01-01"
    end = "2023-12-31"

    # Fetch combined daily data
    daily_data = data_handler.get_combined_data(start, end)

    if daily_data is not None:
        print("--- Daily Data ---")
        print(daily_data.head())
        print(daily_data.tail())
        print(f"Shape: {daily_data.shape}")

        # Resample to Weekly
        weekly_data = data_handler.resample_data(daily_data, timeframe='W')
        if weekly_data is not None:
            print("\n--- Weekly Resampled Data ---")
            print(weekly_data.head())
            print(weekly_data.tail())
            print(f"Shape: {weekly_data.shape}")

        # Resample to Monthly
        monthly_data = data_handler.resample_data(daily_data, timeframe='M')
        if monthly_data is not None:
            print("\n--- Monthly Resampled Data ---")
            print(monthly_data.head())
            print(monthly_data.tail())
            print(f"Shape: {monthly_data.shape}")

    else:
        print("Failed to fetch or combine data.")