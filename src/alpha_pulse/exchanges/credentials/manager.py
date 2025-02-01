"""
Exchange credentials management system.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


@dataclass
class ExchangeCredentials:
    """Exchange API credentials."""
    api_key: str
    api_secret: str
    testnet: bool = False


class CredentialsManager:
    """Manages exchange API credentials."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize credentials manager.
        
        Args:
            config_path: Path to credentials config file
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to user's home directory
            self.config_path = Path.home() / '.alpha_pulse' / 'exchange_credentials.json'
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty config if file doesn't exist
        if not self.config_path.exists():
            self._save_config({})
    
    def _load_config(self) -> Dict:
        """Load credentials configuration."""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading credentials config: {e}")
            return {}
    
    def _save_config(self, config: Dict) -> None:
        """Save credentials configuration."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving credentials config: {e}")
    
    def get_credentials(self, exchange_id: str) -> Optional[ExchangeCredentials]:
        """Get credentials for an exchange.
        
        First checks environment variables, then falls back to config file.
        Environment variables take precedence over config file.
        
        Environment variable format:
        - ALPHA_PULSE_{EXCHANGE}_API_KEY
        - ALPHA_PULSE_{EXCHANGE}_API_SECRET
        - ALPHA_PULSE_{EXCHANGE}_TESTNET (optional)
        
        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'bybit')
            
        Returns:
            Exchange credentials if found, None otherwise
        """
        # Try environment variables first
        env_prefix = f"ALPHA_PULSE_{exchange_id.upper()}"
        api_key = os.getenv(f"{env_prefix}_API_KEY")
        api_secret = os.getenv(f"{env_prefix}_API_SECRET")
        testnet = os.getenv(f"{env_prefix}_TESTNET", "").lower() == "true"
        
        if api_key and api_secret:
            return ExchangeCredentials(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
        
        # Fall back to config file
        config = self._load_config()
        if exchange_id in config:
            return ExchangeCredentials(**config[exchange_id])
        
        return None
    
    def save_credentials(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False
    ) -> None:
        """Save credentials for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            api_key: API key
            api_secret: API secret
            testnet: Whether to use testnet
        """
        config = self._load_config()
        config[exchange_id] = {
            'api_key': api_key,
            'api_secret': api_secret,
            'testnet': testnet
        }
        self._save_config(config)
        logger.info(f"Saved credentials for {exchange_id}")
    
    def remove_credentials(self, exchange_id: str) -> None:
        """Remove credentials for an exchange.
        
        Args:
            exchange_id: Exchange identifier
        """
        config = self._load_config()
        if exchange_id in config:
            del config[exchange_id]
            self._save_config(config)
            logger.info(f"Removed credentials for {exchange_id}")


# Global credentials manager instance
credentials_manager = CredentialsManager()