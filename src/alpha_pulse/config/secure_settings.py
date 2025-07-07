"""
Secure settings management for AlphaPulse.

This module replaces the hardcoded credentials with secure secret management.
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from functools import lru_cache

from ..utils.secrets_manager import create_secrets_manager, SecretsManager


class DatabaseConfig(BaseSettings):
    """Database configuration with secure credential management."""
    
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="alphapulse")
    user: str = Field(default="")
    password: str = Field(default="")
    
    @classmethod
    def from_secrets(cls, secrets_manager: SecretsManager) -> "DatabaseConfig":
        """Create database config from secrets manager."""
        creds = secrets_manager.get_database_credentials()
        return cls(
            host=creds.get("host", "localhost"),
            port=int(creds.get("port", 5432)),
            database=creds.get("database", "alphapulse"),
            user=creds.get("user", ""),
            password=creds.get("password", "")
        )
    
    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        """Get async PostgreSQL connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class ExchangeConfig(BaseSettings):
    """Exchange configuration with secure credential management."""
    
    api_key: str = Field(default="")
    api_secret: str = Field(default="")
    passphrase: Optional[str] = Field(default=None)
    testnet: bool = Field(default=True)
    
    @classmethod
    def from_secrets(cls, exchange: str, secrets_manager: SecretsManager) -> "ExchangeConfig":
        """Create exchange config from secrets manager."""
        creds = secrets_manager.get_exchange_credentials(exchange)
        return cls(
            api_key=creds.get("api_key", ""),
            api_secret=creds.get("api_secret", ""),
            passphrase=creds.get("passphrase"),
            testnet=os.environ.get(f"{exchange.upper()}_TESTNET", "true").lower() == "true"
        )


class DataProviderConfig(BaseSettings):
    """Data provider configuration with secure API key management."""
    
    iex_cloud_api_key: str = Field(default="")
    polygon_api_key: str = Field(default="")
    alpha_vantage_api_key: str = Field(default="")
    finnhub_api_key: str = Field(default="")
    
    @classmethod
    def from_secrets(cls, secrets_manager: SecretsManager) -> "DataProviderConfig":
        """Create data provider config from secrets manager."""
        return cls(
            iex_cloud_api_key=secrets_manager.get_api_key("iex_cloud") or "",
            polygon_api_key=secrets_manager.get_api_key("polygon") or "",
            alpha_vantage_api_key=secrets_manager.get_api_key("alpha_vantage") or "",
            finnhub_api_key=secrets_manager.get_api_key("finnhub") or ""
        )


class SecurityConfig(BaseSettings):
    """Security configuration for authentication and encryption."""
    
    jwt_secret: str = Field(...)
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_minutes: int = Field(default=30)
    encryption_key: bytes = Field(...)
    
    @validator("encryption_key", pre=True)
    def validate_encryption_key(cls, v):
        """Ensure encryption key is bytes."""
        if isinstance(v, str):
            return v.encode()
        return v
    
    @classmethod
    def from_secrets(cls, secrets_manager: SecretsManager) -> "SecurityConfig":
        """Create security config from secrets manager."""
        return cls(
            jwt_secret=secrets_manager.get_jwt_secret(),
            encryption_key=secrets_manager.get_encryption_key()
        )


class Settings(BaseSettings):
    """Main settings class that aggregates all configuration."""
    
    # Application settings
    app_name: str = Field(default="AlphaPulse")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    data_providers: DataProviderConfig = Field(default_factory=DataProviderConfig)
    exchanges: Dict[str, ExchangeConfig] = Field(default_factory=dict)
    
    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    
    # Trading settings
    max_positions: int = Field(default=10)
    risk_per_trade: float = Field(default=0.02)
    max_leverage: float = Field(default=1.0)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @classmethod
    def from_secrets_manager(cls, secrets_manager: SecretsManager) -> "Settings":
        """Create settings from secrets manager."""
        # Load base settings from environment
        settings = cls()
        
        # Override with secure configurations
        settings.database = DatabaseConfig.from_secrets(secrets_manager)
        settings.security = SecurityConfig.from_secrets(secrets_manager)
        settings.data_providers = DataProviderConfig.from_secrets(secrets_manager)
        
        # Load exchange configurations
        exchanges = ["binance", "bybit", "coinbase", "kraken"]
        for exchange in exchanges:
            try:
                settings.exchanges[exchange] = ExchangeConfig.from_secrets(exchange, secrets_manager)
            except Exception as e:
                # Log but don't fail if exchange credentials not found
                print(f"Warning: Could not load {exchange} credentials: {e}")
        
        return settings
    
    def get_exchange_config(self, exchange: str) -> Optional[ExchangeConfig]:
        """Get configuration for a specific exchange."""
        return self.exchanges.get(exchange.lower())
    
    def mask_sensitive_data(self) -> dict:
        """Return settings with sensitive data masked."""
        data = self.dict()
        
        # Mask database password
        if "database" in data and "password" in data["database"]:
            data["database"]["password"] = "***masked***"
        
        # Mask security keys
        if "security" in data:
            data["security"]["jwt_secret"] = "***masked***"
            data["security"]["encryption_key"] = "***masked***"
        
        # Mask API keys
        if "data_providers" in data:
            for key in data["data_providers"]:
                if key.endswith("_api_key"):
                    data["data_providers"][key] = "***masked***"
        
        # Mask exchange credentials
        if "exchanges" in data:
            for exchange, config in data["exchanges"].items():
                if "api_key" in config:
                    config["api_key"] = "***masked***"
                if "api_secret" in config:
                    config["api_secret"] = "***masked***"
                if "passphrase" in config and config["passphrase"]:
                    config["passphrase"] = "***masked***"
        
        return data


# Global settings instance
_settings: Optional[Settings] = None
_secrets_manager: Optional[SecretsManager] = None


@lru_cache()
def get_secrets_manager() -> SecretsManager:
    """Get or create the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = create_secrets_manager()
    return _secrets_manager


@lru_cache()
def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        secrets_manager = get_secrets_manager()
        _settings = Settings.from_secrets_manager(secrets_manager)
    return _settings


def reload_settings():
    """Reload settings and clear caches."""
    global _settings, _secrets_manager
    
    # Clear caches
    get_settings.cache_clear()
    get_secrets_manager.cache_clear()
    
    # Clear global instances
    _settings = None
    _secrets_manager = None
    
    # Clear secrets cache if exists
    if _secrets_manager:
        _secrets_manager.clear_cache()


# Configuration validation
def validate_configuration():
    """Validate that all required configuration is present."""
    settings = get_settings()
    errors = []
    
    # Check database configuration
    if not settings.database.user:
        errors.append("Database user not configured")
    if not settings.database.password:
        errors.append("Database password not configured")
    
    # Check security configuration
    if not settings.security.jwt_secret or settings.security.jwt_secret == "demo_secret_key_for_development_only":
        errors.append("JWT secret not properly configured")
    
    # Check at least one exchange is configured
    configured_exchanges = [
        name for name, config in settings.exchanges.items()
        if config.api_key and config.api_secret
    ]
    if not configured_exchanges:
        errors.append("No exchanges configured with API credentials")
    
    # Check data providers
    configured_providers = []
    if settings.data_providers.iex_cloud_api_key:
        configured_providers.append("iex_cloud")
    if settings.data_providers.polygon_api_key:
        configured_providers.append("polygon")
    if settings.data_providers.alpha_vantage_api_key:
        configured_providers.append("alpha_vantage")
    if settings.data_providers.finnhub_api_key:
        configured_providers.append("finnhub")
    
    if not configured_providers:
        errors.append("No data providers configured with API keys")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# Export key functions and classes
__all__ = [
    "Settings",
    "DatabaseConfig",
    "ExchangeConfig",
    "DataProviderConfig",
    "SecurityConfig",
    "get_settings",
    "get_secrets_manager",
    "reload_settings",
    "validate_configuration"
]