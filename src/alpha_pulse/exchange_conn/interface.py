"""
Exchange connector interface and base classes.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional


@dataclass
class Balance:
    """Represents an asset balance in the exchange."""
    free: Decimal  # Available balance
    locked: Decimal  # Balance in orders
    total: Decimal  # Total balance (free + locked)
    in_base_currency: Decimal  # Value in base currency (e.g., USDT)


class ExchangeConnector(ABC):
    """Base class for exchange connectors."""
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        """Initialize exchange connector.
        
        Args:
            api_key: Exchange API key
            api_secret: Exchange API secret
            testnet: Whether to use testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
    
    @abstractmethod
    async def get_balances(self) -> Dict[str, Balance]:
        """Get balances for all assets.
        
        Returns:
            Dict mapping asset symbols to their balances
        """
        pass
    
    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Current price or None if not available
        """
        pass
    
    @abstractmethod
    async def get_exchange_info(self) -> Dict:
        """Get exchange information including trading rules.
        
        Returns:
            Dict containing exchange information
        """
        pass
    
    @abstractmethod
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees for all symbols.
        
        Returns:
            Dict mapping symbols to their trading fees
        """
        pass
    
    @abstractmethod
    async def validate_api_keys(self) -> bool:
        """Validate API keys by attempting to access private endpoints.
        
        Returns:
            True if keys are valid, False otherwise
        """
        pass