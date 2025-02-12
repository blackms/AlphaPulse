"""
Exchange factory for creating exchange instances.
"""
from typing import Optional, Dict, Any
from loguru import logger

from .base import BaseExchange
from .binance import BinanceExchange
from .bybit import BybitExchange
from .types import ExchangeType


class ExchangeFactory:
    """Factory for creating exchange instances."""

    _exchange_map = {
        ExchangeType.BINANCE: BinanceExchange,
        ExchangeType.BYBIT: BybitExchange
    }

    @classmethod
    def create_exchange(
        cls,
        exchange_type: ExchangeType,
        api_key: str,
        api_secret: str,
        **kwargs
    ) -> BaseExchange:
        """
        Create exchange instance.

        Args:
            exchange_type: Type of exchange to create
            api_key: API key
            api_secret: API secret
            **kwargs: Additional exchange-specific parameters

        Returns:
            Exchange instance

        Raises:
            ValueError: If exchange type is not supported
        """
        exchange_class = cls._exchange_map.get(exchange_type)
        if not exchange_class:
            raise ValueError(f"Unsupported exchange type: {exchange_type}")

        logger.info(f"Creating {exchange_type} exchange instance")
        return exchange_class(
            api_key=api_key,
            api_secret=api_secret,
            **kwargs
        )

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> Optional[BaseExchange]:
        """
        Create exchange instance from configuration.

        Args:
            config: Exchange configuration dictionary

        Returns:
            Exchange instance or None if configuration is invalid

        Example config:
        {
            "type": "binance",
            "api_key": "your-api-key",
            "api_secret": "your-api-secret",
            "testnet": true
        }
        """
        try:
            exchange_type = ExchangeType(config.get("type", "").lower())
            api_key = config.get("api_key")
            api_secret = config.get("api_secret")

            if not all([exchange_type, api_key, api_secret]):
                logger.error("Missing required exchange configuration")
                return None

            # Remove known keys and pass rest as kwargs
            kwargs = {
                k: v for k, v in config.items()
                if k not in ["type", "api_key", "api_secret"]
            }

            return cls.create_exchange(
                exchange_type=exchange_type,
                api_key=api_key,
                api_secret=api_secret,
                **kwargs
            )

        except (ValueError, KeyError) as e:
            logger.error(f"Error creating exchange: {str(e)}")
            return None

    @classmethod
    def register_exchange(
        cls,
        exchange_type: ExchangeType,
        exchange_class: type
    ) -> None:
        """
        Register new exchange type.

        Args:
            exchange_type: Exchange type enum value
            exchange_class: Exchange class to register
        """
        if not issubclass(exchange_class, BaseExchange):
            raise ValueError(
                f"Exchange class must inherit from BaseExchange: {exchange_class}"
            )

        cls._exchange_map[exchange_type] = exchange_class
        logger.info(f"Registered exchange type: {exchange_type}")