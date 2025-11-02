"""
Configuration loader for AlphaPulse.

This module provides utilities for loading configuration from various sources
including YAML files, environment variables, and default settings.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .settings import Settings


class ConfigLoader:
    """Load and manage configuration from multiple sources."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None
        self._settings: Optional[Settings] = None

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file or defaults.

        Returns:
            Configuration dictionary
        """
        if self._config is not None:
            return self._config

        # Try to load from file if path provided
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return self._config
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")

        # Return empty dict as fallback
        self._config = {}
        return self._config

    def get_settings(self) -> Settings:
        """
        Get Settings instance with loaded configuration.

        Returns:
            Settings instance
        """
        if self._settings is None:
            self._settings = Settings()
        return self._settings

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if self._config is None:
            self.load()

        # Support dot notation (e.g., 'database.url')
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    @staticmethod
    def from_yaml(path: Path) -> 'ConfigLoader':
        """
        Create ConfigLoader from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            ConfigLoader instance
        """
        return ConfigLoader(config_path=path)

    @staticmethod
    def from_env() -> 'ConfigLoader':
        """
        Create ConfigLoader using environment variables.

        Returns:
            ConfigLoader instance
        """
        loader = ConfigLoader()
        loader._config = {
            'database': {
                'url': os.getenv('DATABASE_URL', 'postgresql://localhost/alphapulse'),
            },
            'exchange': {
                'api_key': os.getenv('EXCHANGE_API_KEY', ''),
                'api_secret': os.getenv('EXCHANGE_API_SECRET', ''),
            }
        }
        return loader


def get_config_loader(config_path: Optional[Path] = None) -> ConfigLoader:
    """
    Get singleton ConfigLoader instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path=config_path)
