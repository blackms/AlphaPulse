"""
Exchange credentials management with secure secret storage integration.
"""
from typing import Dict, Optional, Any
import os
import json
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

# Import the secure secrets manager
from alpha_pulse.utils.secrets_manager import create_secrets_manager


@dataclass
class Credentials:
    """Exchange API credentials."""
    api_key: str
    api_secret: str
    testnet: bool = False
    passphrase: Optional[str] = None  # For exchanges that require it


class CredentialsManager:
    """Manages exchange API credentials with secure secret storage."""

    def __init__(self):
        """Initialize credentials manager with integrated secret management."""
        self._credentials: Dict[str, Credentials] = {}
        self._config_path = os.path.expanduser("~/.alpha_pulse/credentials.json")
        # Initialize secure secrets manager
        self._secrets_manager = create_secrets_manager()
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Load credentials from secure secret storage."""
        try:
            # First try to load from secure secrets manager
            logger.info("Attempting to load credentials from secure secret storage")
            
            # Get list of supported exchanges
            exchanges = ['bybit', 'binance', 'coinbase', 'kraken']  # Add more as needed
            
            for exchange in exchanges:
                try:
                    # Try to get credentials from secrets manager
                    creds_dict = self._secrets_manager.get_exchange_credentials(exchange)
                    
                    if creds_dict and creds_dict.get('api_key') and creds_dict.get('api_secret'):
                        logger.info(f"Successfully loaded credentials for {exchange} from secure storage")
                        
                        self._credentials[exchange] = Credentials(
                            api_key=creds_dict['api_key'],
                            api_secret=creds_dict['api_secret'],
                            testnet=creds_dict.get('testnet', False),
                            passphrase=creds_dict.get('passphrase')
                        )
                except Exception as e:
                    logger.debug(f"No credentials found for {exchange} in secure storage: {e}")
            
            # If no credentials found in secure storage, fall back to legacy methods
            if not self._credentials:
                logger.info("No credentials found in secure storage, checking legacy sources")
                
                # Check legacy JSON file (log warning about security risk)
                config_path = Path(self._config_path)
                if config_path.exists():
                    logger.warning(f"SECURITY WARNING: Loading credentials from plain JSON file at {self._config_path}")
                    logger.warning("Please migrate to secure secret storage using the migration script")
                    
                    with open(config_path, "r") as f:
                        data = json.load(f)
                    
                    for exchange, creds in data.items():
                        self._credentials[exchange] = Credentials(
                            api_key=creds.get("api_key", ""),
                            api_secret=creds.get("api_secret", ""),
                            testnet=creds.get("testnet", False),
                            passphrase=creds.get("passphrase")
                        )
                else:
                    # Final fallback to environment variables
                    logger.info("Checking environment variables as final fallback")
                    api_key = os.environ.get('BYBIT_API_KEY', os.environ.get('EXCHANGE_API_KEY', ''))
                    api_secret = os.environ.get('BYBIT_API_SECRET', os.environ.get('EXCHANGE_API_SECRET', ''))
                    testnet = os.environ.get('EXCHANGE_TESTNET', 'true').lower() == 'true'
                    
                    if api_key and api_secret:
                        logger.info("Found API credentials in environment variables")
                        self._credentials['bybit'] = Credentials(
                            api_key=api_key,
                            api_secret=api_secret,
                            testnet=testnet
                        )

            logger.info(f"Loaded credentials for {len(self._credentials)} exchanges")
            
        except Exception as e:
            logger.error(f"Error loading credentials: {str(e)}")

    def _save_credentials(self) -> None:
        """Save credentials to config file."""
        try:
            # Create directory if it doesn't exist
            config_path = Path(self._config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert credentials to dictionary
            data = {
                exchange: {
                    "api_key": creds.api_key,
                    "api_secret": creds.api_secret,
                    "testnet": creds.testnet,
                    "passphrase": creds.passphrase
                }
                for exchange, creds in self._credentials.items()
            }

            # Save with pretty formatting
            with open(config_path, "w") as f:
                json.dump(data, f, indent=4)

            logger.info(f"Saved credentials to {self._config_path}")

        except Exception as e:
            logger.error(f"Error saving credentials: {str(e)}")

    def get_credentials(self, exchange: str) -> Optional[Credentials]:
        """
        Get credentials for exchange with secure secret storage priority.

        Args:
            exchange: Exchange name

        Returns:
            Credentials if found, None otherwise
        """
        exchange_lower = exchange.lower()
        logger.debug(f"Requesting credentials for exchange: {exchange_lower}")
        
        # First check in loaded credentials (from secure storage)
        creds = self._credentials.get(exchange_lower)
        
        if creds:
            logger.debug(f"Found credentials for {exchange_lower} in credentials manager")
            # Don't log sensitive credential values
            logger.debug(f"Testnet: {creds.testnet}")
            return creds
        
        # If not in cache, try to fetch directly from secure storage
        logger.debug(f"No cached credentials for {exchange_lower}, checking secure storage")
        
        try:
            creds_dict = self._secrets_manager.get_exchange_credentials(exchange_lower)
            
            if creds_dict and creds_dict.get('api_key') and creds_dict.get('api_secret'):
                logger.info(f"Found credentials for {exchange_lower} in secure storage")
                
                # Create and cache the credentials
                credentials = Credentials(
                    api_key=creds_dict['api_key'],
                    api_secret=creds_dict['api_secret'],
                    testnet=creds_dict.get('testnet', False),
                    passphrase=creds_dict.get('passphrase')
                )
                
                # Cache for future use
                self._credentials[exchange_lower] = credentials
                return credentials
                
        except Exception as e:
            logger.debug(f"Failed to get credentials from secure storage: {e}")
        
        # Final fallback to environment variables
        logger.debug(f"Checking environment variables as final fallback")
        
        api_key = os.environ.get(f'{exchange_lower.upper()}_API_KEY', os.environ.get('EXCHANGE_API_KEY', ''))
        api_secret = os.environ.get(f'{exchange_lower.upper()}_API_SECRET', os.environ.get('EXCHANGE_API_SECRET', ''))
        
        if api_key and api_secret:
            logger.info(f"Found API credentials in environment variables for {exchange_lower}")
            
            testnet = os.environ.get('EXCHANGE_TESTNET', 'true').lower() == 'true'
            
            # Create temporary credentials but don't save them
            return Credentials(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
        
        logger.warning(f"No credentials found for {exchange_lower} anywhere")
        return None

    def set_credentials(
        self,
        exchange: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        passphrase: Optional[str] = None
    ) -> None:
        """
        Set credentials for exchange.

        Args:
            exchange: Exchange name
            api_key: API key
            api_secret: API secret
            testnet: Whether to use testnet
            passphrase: Optional passphrase
        """
        self._credentials[exchange.lower()] = Credentials(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            passphrase=passphrase
        )
        self._save_credentials()

    def remove_credentials(self, exchange: str) -> None:
        """
        Remove credentials for exchange.

        Args:
            exchange: Exchange name
        """
        if exchange.lower() in self._credentials:
            del self._credentials[exchange.lower()]
            self._save_credentials()

    def list_exchanges(self) -> list[str]:
        """
        Get list of exchanges with credentials.

        Returns:
            List of exchange names
        """
        return list(self._credentials.keys())


# Global credentials manager instance
credentials_manager = CredentialsManager()