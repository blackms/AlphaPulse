from pydantic_settings import BaseSettings
from typing import Dict, List
from pathlib import Path

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./alpha_pulse.db"
    
    # Exchange settings
    EXCHANGES: List[str] = ["bybit"]
    EXCHANGE_CONFIGS: Dict[str, Dict] = {
        "binance": {
            "enableRateLimit": True,
            # Add your API keys here if needed
            # "apiKey": "",
            # "secret": "",
        },
        "coinbase": {
            "enableRateLimit": True,
        },
        "kraken": {
            "enableRateLimit": True,
        }
    }
    
    # Trading parameters
    DEFAULT_TIMEFRAME: str = "1h"
    TRADING_PAIRS: List[str] = ["BTC/USDT", "ETH/USDT"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    
    class Config:
        env_file = ".env"

settings = Settings()