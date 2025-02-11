"""
Mock data manager for demonstration purposes.
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class MockDataManager:
    """Mock data manager that provides simulated market data."""
    
    def __init__(self):
        """Initialize mock data manager."""
        self.price_data = {}
        self.volume_data = {}
        self._generate_mock_data()
        
    def _generate_mock_data(self):
        """Generate mock price and volume data."""
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "V", "JNJ"
        ]
        
        # Generate 365 days of data
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        
        for symbol in symbols:
            # Generate price data with trend and volatility
            base_price = np.random.uniform(50, 500)
            trend = np.random.uniform(-0.001, 0.002)
            volatility = np.random.uniform(0.01, 0.03)
            
            prices = []
            current_price = base_price
            
            for _ in range(len(dates)):
                current_price *= (1 + trend + np.random.normal(0, volatility))
                prices.append(current_price)
                
            self.price_data[symbol] = pd.Series(prices, index=dates)
            
            # Generate volume data
            base_volume = np.random.uniform(1e6, 1e7)
            volumes = []
            
            for price in prices:
                volume = base_volume * (1 + np.random.normal(0, 0.3))
                volumes.append(max(volume, 100000))  # Ensure minimum volume
                
            self.volume_data[symbol] = pd.Series(volumes, index=dates)
            
    async def get_historical_prices(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical price data."""
        prices = pd.DataFrame()
        
        for symbol in symbols:
            if symbol in self.price_data:
                mask = (self.price_data[symbol].index >= start_date) & \
                       (self.price_data[symbol].index <= end_date)
                prices[symbol] = self.price_data[symbol][mask]
                
        return prices
        
    async def get_historical_volumes(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical volume data."""
        volumes = pd.DataFrame()
        
        for symbol in symbols:
            if symbol in self.volume_data:
                mask = (self.volume_data[symbol].index >= start_date) & \
                       (self.volume_data[symbol].index <= end_date)
                volumes[symbol] = self.volume_data[symbol][mask]
                
        return volumes
        
    async def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get mock fundamental data."""
        return {
            "market_cap": np.random.uniform(1e9, 1e12),
            "pe_ratio": np.random.uniform(10, 30),
            "pb_ratio": np.random.uniform(1, 5),
            "dividend_yield": np.random.uniform(0, 0.05),
            "revenue_growth": np.random.uniform(0.05, 0.3),
            "profit_margin": np.random.uniform(0.1, 0.3),
            "debt_to_equity": np.random.uniform(0.5, 2),
            "current_ratio": np.random.uniform(1, 3),
            "roe": np.random.uniform(0.1, 0.3),
            "price": self.price_data[symbol].iloc[-1] if symbol in self.price_data else 100,
            "sector": np.random.choice([
                "Technology", "Healthcare", "Finance",
                "Consumer", "Industrial", "Energy"
            ])
        }
        
    async def get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Get mock sentiment data."""
        return {
            "news": [
                {
                    "title": f"Positive news about {symbol}",
                    "content": f"Company {symbol} shows strong performance",
                    "sentiment": np.random.uniform(0.5, 1.0),
                    "source_credibility": np.random.uniform(0.7, 1.0),
                    "timestamp": datetime.now() - timedelta(hours=np.random.randint(1, 24))
                },
                {
                    "title": f"Market analysis of {symbol}",
                    "content": f"Analysts review {symbol} performance",
                    "sentiment": np.random.uniform(-0.3, 0.7),
                    "source_credibility": np.random.uniform(0.6, 0.9),
                    "timestamp": datetime.now() - timedelta(hours=np.random.randint(1, 24))
                }
            ],
            "social_media": {
                "twitter": [
                    {
                        "sentiment": np.random.uniform(-1, 1),
                        "engagement": np.random.randint(100, 10000)
                    }
                    for _ in range(5)
                ],
                "reddit": [
                    {
                        "sentiment": np.random.uniform(-1, 1),
                        "engagement": np.random.randint(50, 5000)
                    }
                    for _ in range(3)
                ]
            },
            "analyst_ratings": {
                "ratings": {
                    "strong_buy": np.random.randint(0, 10),
                    "buy": np.random.randint(5, 15),
                    "hold": np.random.randint(3, 12),
                    "sell": np.random.randint(0, 5),
                    "strong_sell": np.random.randint(0, 3)
                },
                "current_price": self.price_data[symbol].iloc[-1] if symbol in self.price_data else 100,
                "consensus_target": self.price_data[symbol].iloc[-1] * np.random.uniform(0.8, 1.2) \
                    if symbol in self.price_data else 100
            }
        }