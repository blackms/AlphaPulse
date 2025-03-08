"""
Exchange factory implementations.
"""
from typing import Dict, Type, Optional
from loguru import logger

from .interfaces import BaseExchange, ExchangeConfiguration, ConfigurationError
from .types import ExchangeType
from .adapters.ccxt_adapter import CCXTAdapter


class ExchangeRegistry:
    """
    Registry for exchange implementations.
    
    This class manages the registration of exchange implementations
    and their associated configurations.
    """
    
    _registry: Dict[ExchangeType, Dict[str, any]] = {}
    
    @classmethod
    def register(
        cls,
        exchange_type: ExchangeType,
        adapter_class: Type[BaseExchange],
        exchange_id: str,
        **defaults
    ) -> None:
        """
        Register an exchange implementation.
        
        Args:
            exchange_type: Type of exchange
            adapter_class: Exchange adapter class
            exchange_id: Exchange identifier (e.g., CCXT exchange id)
            **defaults: Default configuration options
        """
        cls._registry[exchange_type] = {
            'adapter_class': adapter_class,
            'defaults': {
                **defaults,
                '_exchange_id': exchange_id  # Store exchange_id in defaults with a different name
            }
        }
        logger.info(f"Registered exchange type: {exchange_type}")
    
    @classmethod
    def get_implementation(
        cls,
        exchange_type: ExchangeType
    ) -> Optional[Dict[str, any]]:
        """Get registered implementation details."""
        return cls._registry.get(exchange_type)


class ExchangeFactory:
    """
    Factory for creating exchange instances.
    
    This class provides methods for creating configured exchange instances
    while handling the complexity of different exchange types and configurations.
    """
    
    @classmethod
    def create_exchange(
        cls,
        exchange_type: ExchangeType,
        api_key: str,
        api_secret: str,
        extra_options: dict = None,
        **kwargs
    ) -> BaseExchange:
        """
        Create exchange instance.
        
        Args:
            exchange_type: Type of exchange to create
            api_key: API key
            api_secret: API secret
            extra_options: Additional options to pass directly to the exchange
            **kwargs: Additional exchange-specific parameters
        
        Returns:
            Configured exchange instance
        
        Raises:
            ConfigurationError: If exchange type is not supported or configuration is invalid
        """
        # Get registered implementation
        implementation = ExchangeRegistry.get_implementation(exchange_type)
        if not implementation:
            raise ConfigurationError(f"Unsupported exchange type: {exchange_type}")
        
        # Merge default and provided options
        options = {
            **implementation['defaults'],
            **kwargs
        }
        
        # Add extra options if provided
        if extra_options:
            options.update(extra_options)
            logger.debug(f"Added extra options to {exchange_type} exchange: {extra_options}")
        
        # Get implementation details
        adapter_class = implementation['adapter_class']
        
        logger.info(f"Creating {exchange_type} exchange instance")
        
        # Handle different adapter types
        if adapter_class == CCXTAdapter:
            # Create configuration without exchange_id
            config = ExchangeConfiguration(
                api_key=api_key,
                api_secret=api_secret,
                testnet=options.pop('testnet', False),
                options=options
            )
            
            # Create adapter instance
            return adapter_class(config)
        else:
            # For BinanceExchange and other specialized adapters
            return adapter_class(testnet=options.pop('testnet', False))
    
    @classmethod
    def create_from_config(cls, config: Dict[str, any]) -> Optional[BaseExchange]:
        """
        Create exchange instance from configuration dictionary.
        
        Args:
            config: Exchange configuration dictionary
        
        Returns:
            Exchange instance or None if configuration is invalid
        
        Example config:
        {
            "type": "binance",
            "api_key": "your-api-key",
            "api_secret": "your-api-secret",
            "testnet": true,
            "extra_options": {
                "accountType": "UNIFIED",
                "recvWindow": 60000
            }
        }
        """
        try:
            # Make a copy of the config to avoid modifying the original
            config_copy = config.copy()
            
            exchange_type = ExchangeType(config_copy.pop("type", "").lower())
            api_key = config_copy.pop("api_key", None)
            api_secret = config_copy.pop("api_secret", None)
            
            # Extract extra_options if present
            extra_options = config_copy.pop("extra_options", None)
            
            if not all([exchange_type, api_key, api_secret]):
                logger.error("Missing required exchange configuration")
                return None
            
            return cls.create_exchange(
                exchange_type=exchange_type,
                api_key=api_key,
                api_secret=api_secret,
                extra_options=extra_options,
                **config_copy
            )
            
        except (ValueError, KeyError) as e:
            logger.error(f"Error creating exchange: {str(e)}")
            return None


# Register built-in exchanges
ExchangeRegistry.register(
    ExchangeType.BINANCE,
    CCXTAdapter,
    'binance',
    defaultType='spot',
    adjustForTimeDifference=True,
    recvWindow=60000
)

ExchangeRegistry.register(
    ExchangeType.BYBIT,
    CCXTAdapter,
    'bybit',
    defaultType='spot',
    adjustForTimeDifference=True,
    recvWindow=60000,
    createMarketBuyOrderRequiresPrice=True
)