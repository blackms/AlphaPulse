"""
Mock market data provider for demo purposes.
"""
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
from typing import List, Optional

from ..interfaces import MarketData


class MockMarketDataProvider:
    """Provides mock market data for demo purposes."""
    
    def __init__(self, **kwargs):
        """Initialize mock provider."""
        self.base_prices = {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2500.0,
            "BNB/USDT": 300.0,
            "XRP/USDT": 0.5,
            "SOL/USDT": 100.0
        }
        
    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d"
    ) -> List[MarketData]:
        """Generate mock historical data."""
        if symbol not in self.base_prices:
            return []
            
        base_price = self.base_prices[symbol]
        data = []
        
        # Generate daily data points
        current_time = start_time
        while current_time <= end_time:
            # Generate random price movement
            price_change = np.random.normal(0, 0.02)  # 2% standard deviation
            price = base_price * (1 + price_change)
            
            # Generate random volume
            volume = np.random.uniform(1000, 10000)
            
            # Create market data point
            data_point = MarketData(
                symbol=symbol,  # Include symbol in data point
                timestamp=current_time,
                open=Decimal(str(price * (1 - 0.005))),  # 0.5% spread
                high=Decimal(str(price * (1 + 0.01))),   # 1% higher
                low=Decimal(str(price * (1 - 0.01))),    # 1% lower
                close=Decimal(str(price)),
                volume=Decimal(str(volume)),
                source="mock"  # Indicate this is mock data
            )
            data.append(data_point)
            
            # Update base price for next iteration
            base_price = price
            current_time += timedelta(days=1)
            
        return data
        
    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current mock price for symbol."""
        if symbol not in self.base_prices:
            return None
            
        base_price = self.base_prices[symbol]
        price_change = np.random.normal(0, 0.01)  # 1% standard deviation
        return Decimal(str(base_price * (1 + price_change)))