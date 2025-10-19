"""
Data fetcher implementation.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, UTC
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from abc import ABC, abstractmethod

from alpha_pulse.data_pipeline.providers.yfinance_provider import YFinanceProvider
from alpha_pulse.exchanges.base import OHLCV
from alpha_pulse.exchanges.factories import ExchangeFactory


class IDataStorage(ABC):
    """Interface for data storage."""

    @abstractmethod
    async def save_ohlcv(self, records: List[OHLCV]) -> None:
        """Save OHLCV records."""

    @abstractmethod
    async def get_latest_ohlcv(
        self,
        exchange: str,
        symbol: str,
    ) -> Optional[datetime]:
        """Get latest OHLCV timestamp."""


class DataFetcher:
    """Fetches and stores market and benchmark data."""

    def __init__(
        self,
        exchange_factory: Optional[ExchangeFactory] = None,
        storage: Optional[IDataStorage] = None,
        market_data_provider: Optional[YFinanceProvider] = None,
        concurrent_requests: int = 4,
    ) -> None:
        self.exchange_factory = exchange_factory
        self.storage = storage
        self.market_data_provider = market_data_provider
        self._concurrent_requests = max(1, concurrent_requests)

    async def fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """
        Fetch OHLCV data from an exchange via the configured factory.
        """
        if not self.exchange_factory:
            raise RuntimeError("Exchange factory not configured for OHLCV fetching")

        exchange = await self.exchange_factory.create_exchange(exchange_id)
        try:
            since_ts = int(since.timestamp() * 1000) if since else None
            return await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ts,
                limit=limit,
            )
        finally:
            await exchange.close()

    async def update_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        days_back: int = 30,
    ) -> None:
        """
        Update historical data in storage using the configured exchange and storage backends.
        """
        if not self.storage:
            raise RuntimeError("Storage backend not configured for historical data updates")

        since = datetime.now(UTC) - timedelta(days=days_back)
        data = await self.fetch_ohlcv(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            since=since,
        )

        for record in data:
            record.exchange = exchange_id
            record.symbol = symbol
            record.timeframe = timeframe

        await self.storage.save_ohlcv(data)

    async def fetch_historical_data(
        self,
        symbols: Sequence[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical data for a set of symbols using the configured market data provider.

        Returns:
            DataFrame indexed by timestamp with one column per symbol (closing prices).
        """
        if not self.market_data_provider:
            raise RuntimeError("Market data provider not configured for historical requests")

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_symbols: List[str] = []
        for symbol in symbols:
            if symbol not in seen:
                unique_symbols.append(symbol)
                seen.add(symbol)

        loop = asyncio.get_running_loop()
        sem = asyncio.Semaphore(self._concurrent_requests)

        async def _fetch(symbol: str) -> tuple[str, Optional[pd.Series]]:
            async with sem:
                df = await loop.run_in_executor(
                    None,
                    self.market_data_provider.fetch_ohlcv,
                    symbol,
                    interval,
                    start_date,
                    end_date,
                )
                if df is None or df.empty:
                    return symbol, None
                return symbol, df["Close"].rename(symbol)

        tasks = [_fetch(symbol) for symbol in unique_symbols]
        results = await asyncio.gather(*tasks)

        series_list = [series for _, series in results if series is not None]
        if not series_list:
            raise RuntimeError("No historical data available for requested symbols")

        combined = pd.concat(series_list, axis=1).sort_index()
        combined = combined.loc[
            (combined.index >= pd.Timestamp(start_date).tz_localize("UTC"))
            & (combined.index <= pd.Timestamp(end_date).tz_localize("UTC"))
        ]
        return combined.ffill()


# Type alias for backward compatibility
DataPipeline = DataFetcher
