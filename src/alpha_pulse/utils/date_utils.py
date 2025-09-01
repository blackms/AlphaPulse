"""Date and time utility functions for trading operations."""
from datetime import datetime, time, timedelta
from typing import List, Tuple
import pytz
import pandas as pd


def get_market_open_close(market: str = "NYSE") -> Tuple[time, time]:
    """Get market open and close times."""
    market_hours = {
        "NYSE": (time(9, 30), time(16, 0)),
        "NASDAQ": (time(9, 30), time(16, 0)),
        "CRYPTO": (time(0, 0), time(23, 59)),  # 24/7
        "FOREX": (time(0, 0), time(23, 59)),  # 24/5
    }
    return market_hours.get(market, (time(9, 30), time(16, 0)))


def is_market_open(market: str = "NYSE", current_time: datetime = None) -> bool:
    """Check if market is currently open."""
    if current_time is None:
        current_time = datetime.now(pytz.UTC)
    
    open_time, close_time = get_market_open_close(market)
    current = current_time.time()
    
    # Handle 24/7 markets
    if market in ["CRYPTO"]:
        return True
    
    # Handle weekends for traditional markets
    if market in ["NYSE", "NASDAQ"] and current_time.weekday() >= 5:
        return False
    
    return open_time <= current <= close_time


def get_trading_days(start: datetime, end: datetime, market: str = "NYSE") -> List[datetime]:
    """Get list of trading days between start and end dates."""
    if market == "CRYPTO":
        # Crypto trades every day
        return pd.date_range(start, end, freq='D').tolist()
    
    # For traditional markets, exclude weekends
    all_days = pd.date_range(start, end, freq='B')  # Business days only
    
    # You could add holiday calendar here
    # For now, just return business days
    return all_days.tolist()


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC."""
    if dt.tzinfo is None:
        # Assume local timezone if not specified
        dt = pytz.timezone('UTC').localize(dt)
    return dt.astimezone(pytz.UTC)


def format_timestamp(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime as string."""
    return dt.strftime(fmt)