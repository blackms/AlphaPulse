"""
Historical Data Manager.

This module is responsible for handling historical data retrieval and ensuring that the data exists in storage.
If data for a requested timeframe is missing, it downloads and stores the data.
It also integrates with the exchange data provider.
"""

from datetime import datetime, timedelta
from typing import Optional

from alpha_pulse.data_pipeline.storage import SQLAlchemyStorage
from alpha_pulse.data_pipeline.exchange_data_provider import ExchangeDataProvider
from alpha_pulse.data_pipeline.models import OHLCV

class HistoricalDataManager:
    """
    Class to manage historical data for market data storage.
    """

    def __init__(self, storage: SQLAlchemyStorage, exchange_provider: ExchangeDataProvider):
        """
        Initialize the HistoricalDataManager.

        Args:
            storage (SQLAlchemyStorage): Storage backend for saving and retrieving data.
            exchange_provider (ExchangeDataProvider): Provider to fetch data from exchanges.
        """
        self.storage = storage
        self.exchange_provider = exchange_provider

        # Mapping of supported timeframe strings to their duration.
        self.timeframe_durations = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1)
        }

    def get_historical_data(self, exchange_id: str, symbol: str, timeframe: str,
                            start_time: Optional[datetime]=None, end_time: Optional[datetime]=None):
        """
        Retrieve historical data from storage. If data is missing, download and store it.

        Args:
            exchange_id (str): Exchange identifier.
            symbol (str): Trading pair symbol.
            timeframe (str): Timeframe string (e.g. "1m", "5m", "15m", "1h", "4h", "1d").
            start_time (datetime, optional): Start time for data retrieval.
            end_time (datetime, optional): End time for data retrieval.

        Returns:
            List[OHLCV]: List of OHLCV objects representing the historical data.
        """
        # Retrieve data from storage.
        data = self.storage.get_historical_data(exchange_id, symbol, timeframe, start_time, end_time)
        if not data or len(data) == 0:
            # Download missing data.
            downloaded_data = self.exchange_provider.fetch_historical_data(exchange_id, symbol, timeframe, start_time, end_time)
            if downloaded_data:
                # Save downloaded data to storage.
                self.storage.save_historical_data(exchange_id, symbol, timeframe, downloaded_data)
                # Retrieve data again after saving.
                data = self.storage.get_historical_data(exchange_id, symbol, timeframe, start_time, end_time)
        return data