"""
Data provider implementation for fetching OHLCV data from Yahoo Finance (yfinance).
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta # Import timedelta
from typing import Optional
from loguru import logger

# Map common timeframes to yfinance intervals
# Ref: https://github.com/ranaroussi/yfinance/wiki/Tickers#download
TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}

class YFinanceProvider:
    """
    Provides OHLCV data by fetching from Yahoo Finance using the yfinance library.
    """

    def __init__(self):
        logger.info("Initialized YFinanceProvider.")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches OHLCV data for a given symbol and timeframe from yfinance.

        Args:
            symbol: The ticker symbol (e.g., '^GSPC', 'AAPL').
            timeframe: The timeframe string (e.g., '1d', '1h', '5m').
            start_date: The start datetime for the data range.
            end_date: The end datetime for the data range.

        Returns:
            A pandas DataFrame with OHLCV data and DatetimeIndex,
            or None if fetching fails. Returns columns: Open, High, Low, Close, Volume.
        """
        interval = TIMEFRAME_MAP.get(timeframe)
        if not interval:
            logger.error(f"Unsupported timeframe '{timeframe}' for yfinance.")
            return None

        logger.info(
            f"Fetching {symbol} {timeframe} ({interval}) data from yfinance: "
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        try:
            # yfinance uses start/end dates (end is exclusive for daily/weekly/monthly)
            fetch_start_date = start_date
            fetch_end_date = end_date

            # Adjust start date for intraday data limitations (max 730 days for 1h/30m/15m/5m, 60 days for 1m)
            now_utc = datetime.now(pd.Timestamp.utcnow().tz) # Get current UTC time
            max_start_date = None
            if interval == "1m":
                 max_start_date = now_utc - timedelta(days=60)
            elif interval in ["5m", "15m", "30m", "1h"]:
                 max_start_date = now_utc - timedelta(days=730)

            if max_start_date:
                 # Ensure start_date is timezone-aware for comparison
                 start_date_aware = pd.Timestamp(start_date).tz_localize('UTC') if pd.Timestamp(start_date).tzinfo is None else pd.Timestamp(start_date).tz_convert('UTC')
                 if start_date_aware < max_start_date:
                      logger.warning(f"Requested start date {start_date.strftime('%Y-%m-%d')} for {interval} is beyond yfinance limit. Adjusting start date to {max_start_date.strftime('%Y-%m-%d')}.")
                      fetch_start_date = max_start_date.replace(tzinfo=None) # yfinance expects naive datetime strings

            # Add one day to end_date for daily+ intervals to ensure inclusion
            if interval in ["1d", "1wk", "1mo"]:
                 fetch_end_date = end_date + pd.Timedelta(days=1)

            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=fetch_start_date.strftime('%Y-%m-%d'),
                end=fetch_end_date.strftime('%Y-%m-%d'),
                interval=interval,
                auto_adjust=False, # Keep Open, High, Low, Close, Adj Close, Volume
                actions=False # Exclude dividends and splits actions columns
            )

            if data.empty:
                logger.warning(f"No data returned for {symbol} ({timeframe}) in the specified range.")
                return None

            # Select and rename standard OHLCV columns
            # yfinance column names are capitalized
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                 logger.error(f"Missing expected OHLCV columns in data returned for {symbol}. Columns: {data.columns.tolist()}")
                 # Try to proceed if at least Close exists? Or return None?
                 if 'Close' not in data.columns:
                      return None
                 # Fill missing OHLCV with Close if necessary? Risky. Best to return None.
                 return None


            ohlcv_data = data[required_cols].copy()

            # Ensure index is timezone-aware (UTC)
            if ohlcv_data.index.tz is None:
                ohlcv_data.index = ohlcv_data.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
            else:
                ohlcv_data.index = ohlcv_data.index.tz_convert('UTC')

            # Filter exact date range again after fetching, ensuring timezone comparison
            start_date_utc = pd.Timestamp(start_date).tz_localize('UTC')
            end_date_utc = pd.Timestamp(end_date).tz_localize('UTC')
            ohlcv_data = ohlcv_data[(ohlcv_data.index >= start_date_utc) & (ohlcv_data.index < end_date_utc)]


            if ohlcv_data.empty:
                 logger.warning(f"No data remained for {symbol} ({timeframe}) after final date filtering.")
                 return None

            logger.info(f"Successfully fetched {len(ohlcv_data)} records for {symbol} ({timeframe}).")
            return ohlcv_data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} ({timeframe}) from yfinance: {e}")
            return None

if __name__ == '__main__':
    # Example Usage
    provider = YFinanceProvider()
    start = datetime(2023, 1, 1)
    end = datetime(2024, 1, 1)

    # Daily data
    daily_data = provider.fetch_ohlcv('^GSPC', '1d', start, end)
    if daily_data is not None:
        print("\n--- Daily SP500 Data ---")
        print(daily_data.head())
        print(daily_data.tail())

    # Hourly data
    start_intra = datetime(2023, 12, 1) # Shorter period for intraday
    end_intra = datetime(2023, 12, 5)
    hourly_data = provider.fetch_ohlcv('AAPL', '1h', start_intra, end_intra)
    if hourly_data is not None:
        print("\n--- Hourly AAPL Data ---")
        print(hourly_data.head())
        print(hourly_data.tail())

    # Invalid timeframe
    invalid_data = provider.fetch_ohlcv('MSFT', '1y', start, end)
    print(f"\nInvalid timeframe result: {invalid_data}")

    # Non-existent symbol
    # Note: yfinance might not raise an error but return an empty DataFrame
    nonexistent_data = provider.fetch_ohlcv('NONEXISTENTTICKER', '1d', start, end)
    print(f"\nNon-existent ticker result: {nonexistent_data}")