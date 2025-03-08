"""
Bybit exchange implementation.
"""
import json
from decimal import Decimal
from typing import Dict
import os
from loguru import logger
import ccxt.async_support as ccxt

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
            testnet: Ignored - we always use mainnet
        """
        # Get credentials from manager
        logger.debug(f"BYBIT DEBUG - Requesting credentials from credentials_manager")
        creds = credentials_manager.get_credentials('bybit')
        
        if creds:
            api_key = creds.api_key
            api_secret = creds.api_secret
            
            # Debug the credentials obtained from the manager
            logger.debug(f"BYBIT DEBUG - Credentials obtained from manager:")
            logger.debug(f"BYBIT DEBUG - API Key: {api_key}")
            logger.debug(f"BYBIT DEBUG - API Secret: {api_secret}")
            logger.info("BYBIT INFO - Always using mainnet mode (testnet=False) for Bybit")
        else:
            logger.debug("BYBIT DEBUG - No credentials found in manager!")
            api_key = ""
            api_secret = ""
        
        # Create configuration with Bybit-specific options
        # Always use mainnet (testnet=False)
        config = ExchangeConfiguration(
            api_key=api_key,
            api_secret=api_secret,
            testnet=False,  # Always use mainnet
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
        
        # No testnet-specific endpoints needed as we always use mainnet
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get balances for all assets."""
        # Check if API credentials are available
        if not self.config.api_key or not self.config.api_secret:
            logger.warning("No API credentials available for Bybit. Returning empty balances.")
            return {}
            
        try:
            # Get base implementation
            balances = await super().get_balances()
            
            # If no balances returned, don't try to process further
            if not balances:
                return balances
                
            try:
                # Add Bybit-specific processing
                # For example, handle wallet types
                account = await self.exchange.fetch_balance()
                if 'info' in account:
                    info = account['info']
                    # Handle the case where info might be a string instead of a dict
                    if isinstance(info, str):
                        try:
                            info = json.loads(info)
                        except json.JSONDecodeError:
                            logger.debug(f"Could not parse Bybit info as JSON: {info}")
                            info = {}
                    
                    # Now safely get results, handling both dict and parsed JSON cases
                    result = info.get('result', []) if isinstance(info, dict) else []
                    for wallet in result:
                        asset = wallet.get('coin') if isinstance(wallet, dict) else None
                        if asset and asset in balances:
                            # Add wallet-specific balances
                            locked = Decimal(str(wallet.get('locked', '0')))
                            balances[asset].locked += locked
                            balances[asset].total += locked
            except Exception as wallet_error:
                logger.warning(f"Error processing Bybit wallet info: {wallet_error}")
                # Continue with basic balances if wallet processing fails
            
            return balances
            
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication error fetching Bybit balances: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching Bybit balances: {e}")
            return {}
    
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees for all symbols."""
        # Check if API credentials are available
        if not self.config.api_key or not self.config.api_secret:
            logger.warning("No API credentials available for Bybit. Using default trading fees.")
            return self._get_default_fees()
            
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
            
        except ccxt.AuthenticationError as e:
            logger.warning(f"Authentication error fetching Bybit trading fees: {e}")
            return self._get_default_fees()
        except Exception as e:
            logger.error(f"Error fetching Bybit trading fees: {e}")
            return self._get_default_fees()
            
    def _get_default_fees(self) -> Dict[str, Decimal]:
        """Return default fee structure when API access fails."""
        try:
            # Create default fees for all markets
            fees = {}
            markets = self.exchange.markets if hasattr(self.exchange, 'markets') and self.exchange.markets else {}
            
            for symbol in markets:
                fees[symbol] = Decimal('0.001')  # 0.1% default fee
                
            return fees
        except Exception as e:
            logger.error(f"Error creating default fees: {e}")
            return {'DEFAULT': Decimal('0.001')}