"""
Bybit exchange implementation.
"""
from decimal import Decimal
from typing import Dict
from loguru import logger

from ..adapters.ccxt_adapter import CCXTAdapter
from ..interfaces import Balance, ExchangeConfiguration
from ..credentials.manager import credentials_manager


class BybitExchange(CCXTAdapter):
    """
    Bybit exchange implementation.
    
    This class extends the CCXT adapter with Bybit-specific functionality
    and configuration.
    """
    
    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit exchange.
        
        Args:
            testnet: Whether to use testnet
        """
        # Get credentials from manager
        creds = credentials_manager.get_credentials('bybit')
        if creds:
            api_key = creds.api_key
            api_secret = creds.api_secret
            # Override testnet from credentials if provided
            testnet = testnet or creds.testnet
        else:
            api_key = ""
            api_secret = ""
        
        # Create configuration with Bybit-specific options
        config = ExchangeConfiguration(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            options={
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 60000,
                'createMarketBuyOrderRequiresPrice': True
            }
        )
        
        super().__init__(exchange_id='bybit', config=config)
    
    async def initialize(self) -> None:
        """Initialize Bybit exchange connection."""
        await super().initialize()
        
        if self.exchange and self.config.testnet:
            # Set testnet-specific endpoints
            self.exchange.urls.update({
                'test': {
                    'public': 'https://api-testnet.bybit.com',
                    'private': 'https://api-testnet.bybit.com',
                }
            })
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get balances for all assets."""
        try:
            # Get base implementation
            balances = await super().get_balances()
            
            # Add Bybit-specific processing
            # For example, handle wallet types
            account = await self.exchange.fetch_balance()
            if 'info' in account:
                for wallet in account['info'].get('result', []):
                    asset = wallet.get('coin')
                    if asset in balances:
                        # Add wallet-specific balances
                        locked = Decimal(str(wallet.get('locked', '0')))
                        balances[asset].locked += locked
                        balances[asset].total += locked
            
            return balances
            
        except Exception as e:
            logger.error(f"Error fetching Bybit balances: {e}")
            raise
    
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees for all symbols."""
        try:
            # Get account trading fees
            response = await self.exchange.privateGetAccount()
            
            # Extract commission rates
            fees = {}
            for symbol in await self.exchange.load_markets():
                # Bybit uses tiered fees, we'll use the base tier
                maker_fee = Decimal('0.001')  # 0.1% base maker fee
                taker_fee = Decimal('0.001')  # 0.1% base taker fee
                fees[symbol] = max(maker_fee, taker_fee)  # Use higher fee
            
            return fees
            
        except Exception as e:
            logger.error(f"Error fetching Bybit trading fees: {e}")
            # Fall back to default implementation
            return await super().get_trading_fees()