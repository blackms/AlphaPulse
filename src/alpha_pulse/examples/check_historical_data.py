"""
Script to check if historical OHLCV data is correctly stored in the database.
This script queries the OHLCV records from the database and prints the total count and a sample record.
It also reminds that for intensive timeseries operations, a timeseries optimized database like TimescaleDB is recommended.
"""

from sqlalchemy.orm import Session
from alpha_pulse.data_pipeline.storage import SQLAlchemyStorage
from alpha_pulse.data_pipeline.models import OHLCVRecord

def main():
    # Initialize storage (using the database URL from settings)
    storage = SQLAlchemyStorage()
    
    # Create a new session
    with storage.Session() as session:
        # Query all OHLCV records for BTC/USDT on 1d timeframe as an example
        records = session.query(OHLCVRecord).filter(
            OHLCVRecord.exchange == "binance",
            OHLCVRecord.symbol == "BTC/USDT",
            OHLCVRecord.timeframe == "1d"
        ).all()
        
        total = len(records)
        print(f"Total OHLCV records for BTC/USDT (1d): {total}")
        
        if total > 0:
            sample = records[0]
            print("Sample record:")
            print(f"  Exchange: {sample.exchange}")
            print(f"  Symbol: {sample.symbol}")
            print(f"  Timeframe: {sample.timeframe}")
            print(f"  Timestamp: {sample.timestamp}")
            print(f"  Open: {sample.open}")
            print(f"  High: {sample.high}")
            print(f"  Low: {sample.low}")
            print(f"  Close: {sample.close}")
            print(f"  Volume: {sample.volume}")
        else:
            print("No records found.")
    
    print("\nNote: For high-performance timeseries manipulation, consider using a specialized database like TimescaleDB.")

if __name__ == "__main__":
    main()