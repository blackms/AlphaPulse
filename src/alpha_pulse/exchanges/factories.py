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
            'exchange_id': exchange_id,
            'defaults': defaults
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
        
        # Create configuration
        config = ExchangeConfiguration(
            api_key=api_key,
            api_secret=api_secret,
            exchange_id=exchange_id,
            testnet=options.pop('testnet', False),
            options=options
        )
        
        # Create and return adapter instance
        adapter_class = implementation['adapter_class']
        exchange_id = implementation['exchange_id']
        
        logger.info(f"Creating {exchange_type} exchange instance")
        return adapter_class(config=config)
    
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
            "testnet": true
        }
        """
        try:
            exchange_type = ExchangeType(config.pop("type", "").lower())
            api_key = config.pop("api_key", None)
            api_secret = config.pop("api_secret", None)
            
            if not all([exchange_type, api_key, api_secret]):
                logger.error("Missing required exchange configuration")
                return None
            
            return cls.create_exchange(
                exchange_type=exchange_type,
                api_key=api_key,
                api_secret=api_secret,
                **config
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