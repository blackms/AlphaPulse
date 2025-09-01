"""Data transformation utilities for the data pipeline."""
from typing import Dict, Any, List
import numpy as np


class DataTransformer:
    """Transform raw data into standardized formats."""
    
    def transform_ohlcv(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw OHLCV data to standard format."""
        return {
            "open": raw_data.get("o", raw_data.get("open", 0)),
            "high": raw_data.get("h", raw_data.get("high", 0)),
            "low": raw_data.get("l", raw_data.get("low", 0)),
            "close": raw_data.get("c", raw_data.get("close", 0)),
            "volume": raw_data.get("v", raw_data.get("volume", 0))
        }
    
    def aggregate_ticks(self, ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate tick data into summary statistics."""
        if not ticks:
            return {
                "avg_price": 0,
                "total_volume": 0,
                "vwap": 0,
                "tick_count": 0
            }
        
        prices = [t.get("price", 0) for t in ticks]
        volumes = [t.get("volume", 0) for t in ticks]
        
        total_volume = sum(volumes)
        
        # Calculate VWAP (Volume Weighted Average Price)
        if total_volume > 0:
            vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
        else:
            vwap = np.mean(prices) if prices else 0
        
        return {
            "avg_price": np.mean(prices) if prices else 0,
            "total_volume": total_volume,
            "vwap": vwap,
            "tick_count": len(ticks)
        }
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize trading pair symbols."""
        # Remove special characters and standardize format
        symbol = symbol.upper().replace("-", "/").replace("_", "/")
        
        # Ensure proper format (BASE/QUOTE)
        if "/" not in symbol:
            # Try to guess the split point
            if symbol.endswith("USDT"):
                symbol = symbol[:-4] + "/USDT"
            elif symbol.endswith("USD"):
                symbol = symbol[:-3] + "/USD"
            elif symbol.endswith("BTC"):
                symbol = symbol[:-3] + "/BTC"
        
        return symbol