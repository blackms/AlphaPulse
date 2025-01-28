from data_pipeline.exchange import CCXTExchangeFactory
from data_pipeline.storage import SQLAlchemyStorage
from data_pipeline.data_fetcher import DataFetcher

def main():
    # Create dependencies
    exchange_factory = CCXTExchangeFactory()
    storage = SQLAlchemyStorage()
    
    # Create data fetcher with injected dependencies
    data_fetcher = DataFetcher(exchange_factory, storage)
    
    # Example usage
    data_fetcher.update_historical_data(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        days_back=7
    )

if __name__ == "__main__":
    main()