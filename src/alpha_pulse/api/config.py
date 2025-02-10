"""
Configuration module for AlphaPulse API.
"""
from typing import Dict
import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ExchangeConfig(BaseModel):
    """Exchange configuration."""
    api_key: str
    api_secret: str
    testnet: bool = True

class ApiConfig(BaseModel):
    """API configuration."""
    # API authentication
    api_keys: Dict[str, str] = {
        "test_key": "test_user"  # For development only
    }
    
    # Exchange configuration
    exchange: ExchangeConfig = ExchangeConfig(
        api_key=os.getenv("BYBIT_API_KEY", ""),
        api_secret=os.getenv("BYBIT_API_SECRET", ""),
        testnet=os.getenv("ALPHA_PULSE_BYBIT_TESTNET", "true").lower() == "true"
    )

# Global configuration instance
config = ApiConfig()