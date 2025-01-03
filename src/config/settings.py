from pydantic_settings import BaseSettings
from typing import Dict, List
from pathlib import Path

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/crypto_trading"
    
    # Exchange settings
    EXCHANGES: List[str] = ["binance", "coinbase", "kraken"]
    EXCHANGE_CONFIGS: Dict[str, Dict] = {
        "binance": {
            "apiKey": "",
            "secret": "",
            "enableRateLimit": True,
        },
        # Add other exchange configs
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