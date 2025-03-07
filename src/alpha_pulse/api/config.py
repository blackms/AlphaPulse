"""
Configuration module for AlphaPulse API.
"""
import os
from typing import List, Optional, Dict
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class AuthConfig(BaseSettings):
    """Authentication configuration."""
    jwt_secret: str = Field(default_factory=lambda: os.getenv("AP_JWT_SECRET", "dev-secret-key"))
    token_expiry: int = 3600  # seconds
    api_keys_enabled: bool = True
    api_keys: Dict[str, str] = {
        "test_key": "test_user"  # For development only
    }


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 120


class CorsConfig(BaseSettings):
    """CORS configuration."""
    allowed_origins: List[str] = ["http://localhost:3000"]
    allow_credentials: bool = True


class CacheConfig(BaseSettings):
    """Cache configuration."""
    type: str = "memory"  # "memory", "redis"
    redis_url: Optional[str] = Field(default=None, env="AP_REDIS_URL")
    default_ttl: int = 300  # seconds


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = "info"
    format: str = "json"


class WebSocketConfig(BaseSettings):
    """WebSocket configuration."""
    max_connections: int = 1000
    ping_interval: int = 30  # seconds


class ExchangeConfig(BaseSettings):
    """Exchange configuration."""
    api_key: str = Field(default="", env="BYBIT_API_KEY")
    api_secret: str = Field(default="", env="BYBIT_API_SECRET")
    testnet: bool = Field(
        default_factory=lambda: os.getenv("ALPHA_PULSE_BYBIT_TESTNET", "true").lower() == "true"
    )


class ApiConfig(BaseSettings):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    auth: AuthConfig = AuthConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    cors: CorsConfig = CorsConfig()
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()
    websocket: WebSocketConfig = WebSocketConfig()
    exchange: ExchangeConfig = ExchangeConfig()

    class Config:
        """Pydantic configuration."""
        env_prefix = "AP_API_"
        env_nested_delimiter = "__"


def load_config() -> ApiConfig:
    """Load API configuration."""
    return ApiConfig()


# Global configuration instance
config = load_config()