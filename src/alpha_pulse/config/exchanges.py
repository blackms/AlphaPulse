"""
Exchange configuration settings for AlphaPulse.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ExchangeConfig:
    """Exchange-specific configuration."""
    id: str
    name: str
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    options: Dict[str, Any] = None
    description: str = ""


# Default exchange configurations
EXCHANGE_CONFIGS = {
    "binance": ExchangeConfig(
        id="binance",
        name="Binance",
        description="Binance cryptocurrency exchange",
        options={
            "defaultType": "spot",  # spot, margin, future
            "adjustForTimeDifference": True,
            "recvWindow": 5000,
        }
    ),
    "binance_testnet": ExchangeConfig(
        id="binance",
        name="Binance Testnet",
        testnet=True,
        description="Binance testnet for paper trading",
        options={
            "defaultType": "spot",
            "adjustForTimeDifference": True,
            "recvWindow": 5000,
            "test": True,
        }
    ),
    "coinbase": ExchangeConfig(
        id="coinbase",
        name="Coinbase Pro",
        description="Coinbase Pro cryptocurrency exchange",
        options={
            "adjustForTimeDifference": True,
        }
    ),
    "kraken": ExchangeConfig(
        id="kraken",
        name="Kraken",
        description="Kraken cryptocurrency exchange",
        options={
            "adjustForTimeDifference": True,
        }
    ),
}


def get_exchange_config(exchange_id: str) -> Optional[ExchangeConfig]:
    """
    Get exchange configuration by ID.
    
    Args:
        exchange_id: Exchange identifier
        
    Returns:
        Exchange configuration if found, None otherwise
    """
    return EXCHANGE_CONFIGS.get(exchange_id)


def register_exchange_config(config: ExchangeConfig) -> None:
    """
    Register a new exchange configuration.
    
    Args:
        config: Exchange configuration to register
    """
    EXCHANGE_CONFIGS[config.id] = config


def update_exchange_credentials(
    exchange_id: str,
    api_key: str,
    api_secret: str
) -> None:
    """
    Update API credentials for an exchange.
    
    Args:
        exchange_id: Exchange identifier
        api_key: API key
        api_secret: API secret
    """
    if exchange_id in EXCHANGE_CONFIGS:
        config = EXCHANGE_CONFIGS[exchange_id]
        config.api_key = api_key
        config.api_secret = api_secret