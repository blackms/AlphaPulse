"""
Script to download the last year of historical data for BTC/USDT from Binance.
This script uses DataFetcher to update and store historical OHLCV data using a 1-day timeframe.
"""

import asyncio
from loguru import logger
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.data_pipeline.storage import SQLAlchemyStorage
from alpha_pulse.exchanges import ExchangeType

async def main():
    storage = SQLAlchemyStorage()
    data_fetcher = DataFetcher(storage)
    logger.info("Starting historical data download for BTC/USDT from Binance...")
    await data_fetcher.update_historical_data(
        exchange_type=ExchangeType.BINANCE,
        symbol="BTC/USDT",
        timeframe="1d",
        days_back=365,
        testnet=False
    )
    logger.info("Historical data download complete.")

if __name__ == "__main__":
    asyncio.run(main())