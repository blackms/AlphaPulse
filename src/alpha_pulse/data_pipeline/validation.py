"""Data validation utilities for the data pipeline."""
from datetime import datetime
from typing import Dict, Any, Optional


class DataValidator:
    """Validator for market data and other inputs."""
    
    def validate_market_data(self, data: Dict[str, Any]) -> bool:
        """Validate market data structure and values."""
        # Check required fields
        required_fields = ["symbol", "price", "volume", "timestamp"]
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate symbol
        if not data["symbol"] or not isinstance(data["symbol"], str):
            return False
        
        # Validate price
        if not isinstance(data["price"], (int, float)) or data["price"] <= 0:
            return False
        
        # Validate volume
        if not isinstance(data["volume"], (int, float)) or data["volume"] <= 0:
            return False
        
        # Validate timestamp
        if data["timestamp"] is None:
            return False
        if not isinstance(data["timestamp"], datetime):
            return False
        
        return True
    
    def validate_order(self, order: Dict[str, Any]) -> bool:
        """Validate order structure."""
        required = ["symbol", "side", "amount", "type"]
        for field in required:
            if field not in order:
                return False
        
        if order["side"] not in ["buy", "sell"]:
            return False
        
        if order["type"] not in ["market", "limit", "stop_loss"]:
            return False
        
        if order["amount"] <= 0:
            return False
        
        return True