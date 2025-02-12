"""
Exchange credentials management.
"""
from typing import Dict, Optional, Any
import os
import json
from dataclasses import dataclass
from pathlib import Path
from loguru import logger


@dataclass
class Credentials:
    """Exchange API credentials."""
    api_key: str
    api_secret: str
    testnet: bool = False
    passphrase: Optional[str] = None  # For exchanges that require it


class CredentialsManager:
    """Manages exchange API credentials."""

    def __init__(self):
        """Initialize credentials manager."""
        self._credentials: Dict[str, Credentials] = {}
        self._config_path = os.path.expanduser("~/.alpha_pulse/credentials.json")
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Load credentials from config file."""
        try:
            config_path = Path(self._config_path)
            if not config_path.exists():
                logger.info(f"No credentials file found at {self._config_path}")
                return

            with open(config_path, "r") as f:
                data = json.load(f)

            for exchange, creds in data.items():
                self._credentials[exchange] = Credentials(
                    api_key=creds.get("api_key", ""),
                    api_secret=creds.get("api_secret", ""),
                    testnet=creds.get("testnet", False),
                    passphrase=creds.get("passphrase")
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
        Get credentials for exchange.

        Args:
            exchange: Exchange name

        Returns:
            Credentials if found, None otherwise
        """
        return self._credentials.get(exchange.lower())

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