"""
Bybit exchange implementation.
"""
import json
import hmac
import hashlib
import urllib.parse
import time
from decimal import Decimal
from typing import Dict, Any
import os
from loguru import logger
import ccxt.async_support as ccxt
import asyncio

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
                'createMarketBuyOrderRequiresPrice': True,
                # Add custom auth handling for Bybit V5 API
                'auth': {
                    'v5': {
                        'sign': self._generate_v5_signature
                    }
                }
            }
        )
        
        super().__init__(exchange_id='bybit', config=config)
    
    def _generate_v5_signature(self, api_key: str, api_secret: str, timestamp: int,
                              recv_window: int, params: Dict[str, Any] = None) -> str:
        """
        Generate signature for Bybit V5 API.
        
        Args:
            api_key: API key
            api_secret: API secret
            timestamp: Current timestamp in milliseconds
            recv_window: Receive window in milliseconds
            params: Additional parameters to include in signature
            
        Returns:
            Signature string
        """
        # For v5 API, the signature is calculated differently
        param_str = f"{timestamp}{api_key}{recv_window}"
        
        # Add query parameters to the signature if provided
        if params:
            # Sort parameters alphabetically
            sorted_params = sorted(params.items())
            # URL encode parameters
            encoded_params = urllib.parse.urlencode(sorted_params)
            # Append to param_str
            param_str += encoded_params
        
        logger.debug(f"Bybit V5 signature param string: {param_str}")
        
        signature = hmac.new(
            bytes(api_secret, "utf-8"),
            bytes(param_str, "utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    async def initialize(self) -> None:
        """Initialize Bybit exchange connection."""
        try:
            # Add a retry mechanism for initialization
            max_retries = 3
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    # Try to initialize with a timeout
                    await asyncio.wait_for(self._initialize_with_fallback(), timeout=15)
                    # If we get here, initialization was successful
                    logger.info("Successfully initialized Bybit exchange")
                    return
                except asyncio.TimeoutError:
                    retry_count += 1
                    logger.warning(f"Timeout initializing Bybit exchange (attempt {retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        raise TimeoutError(f"Failed to initialize Bybit exchange after {max_retries} attempts due to timeout")
                except Exception as e:
                    retry_count += 1
                    last_error = e
                    logger.warning(f"Error initializing Bybit exchange (attempt {retry_count}/{max_retries}): {str(e)}")
                    if retry_count >= max_retries:
                        raise
                
                # Wait before retrying
                await asyncio.sleep(2 * retry_count)  # Exponential backoff
            
            # If we get here, all retries failed
            if last_error:
                raise last_error
        except Exception as e:
            logger.error(f"Failed to initialize Bybit exchange: {str(e)}")
            raise
    
    async def _initialize_with_fallback(self) -> None:
        """Initialize with fallback for problematic endpoints."""
        try:
            # First try the standard initialization
            await super().initialize()
        except Exception as e:
            error_str = str(e)
            
            # Check if it's the specific query-info endpoint issue
            if "query-info" in error_str:
                logger.warning("Detected query-info endpoint issue, using fallback initialization")
                
                # Override the CCXT sign method to use our custom signature generator for V5 API
                if hasattr(self.exchange, 'sign'):
                    original_sign = self.exchange.sign
                    
                    def custom_sign(path, api='public', method='GET', params={}, headers=None, body=None):
                        # Check if this is a V5 API call
                        if 'v5' in path:
                            logger.debug(f"Using custom V5 signature for path: {path}")
                            # Get the timestamp
                            timestamp = int(time.time() * 1000)
                            recv_window = self.config.options.get('recvWindow', 5000)
                            
                            # Generate the signature
                            signature = self._generate_v5_signature(
                                self.config.api_key,
                                self.config.api_secret,
                                timestamp,
                                recv_window,
                                params
                            )
                            
                            # Set up headers
                            if headers is None:
                                headers = {}
                            
                            headers.update({
                                'X-BAPI-API-KEY': self.config.api_key,
                                'X-BAPI-SIGN': signature,
                                'X-BAPI-TIMESTAMP': str(timestamp),
                                'X-BAPI-RECV-WINDOW': str(recv_window),
                                'Content-Type': 'application/json'
                            })
                            
                            # For GET requests with params, add them to the URL
                            if method == 'GET' and params:
                                path = f"{path}?{urllib.parse.urlencode(params)}"
                                params = {}
                            
                            return {'url': path, 'method': method, 'body': body, 'headers': headers}
                        else:
                            # Use the original sign method for non-V5 API calls
                            return original_sign(path, api, method, params, headers, body)
                    
                    # Replace the sign method
                    self.exchange.sign = custom_sign
                    logger.info("Replaced CCXT sign method with custom implementation for Bybit V5 API")
                
                # Load markets directly instead of using the problematic endpoint
                logger.info("Loading markets directly as part of fallback initialization")
                await self.exchange.load_markets()
            else:
                # If it's not the specific issue we're handling, re-raise
                raise
    
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
            
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current positions with entry price and unrealized PnL.
        
        Overrides the base implementation to add Bybit-specific position data.
        """
        try:
            # Get base positions from parent class
            positions = await super().get_positions()
            
            # If no positions are available, return empty dict
            if not positions:
                logger.warning("No positions available to enhance")
                return positions
                
            # Enhance positions with entry price and unrealized PnL
            for symbol, position in positions.items():
                try:
                    # Get average entry price
                    entry_price = await self.get_average_entry_price(symbol)
                    position['entry_price'] = float(entry_price) if entry_price else None
                    
                    # Calculate unrealized PnL if we have both entry price and current price
                    if entry_price and position['current_price']:
                        quantity = Decimal(str(position['quantity']))
                        current_price = Decimal(str(position['current_price']))
                        unrealized_pnl = quantity * (current_price - entry_price)
                        position['unrealized_pnl'] = float(unrealized_pnl)
                        
                        # Add percentage PnL for convenience
                        if entry_price > 0:
                            pnl_percentage = ((current_price / entry_price) - 1) * 100
                            position['pnl_percentage'] = float(pnl_percentage)
                except Exception as e:
                    logger.warning(f"Error enhancing position data for {symbol}: {str(e)}")
                    # Keep the position but without the enhanced data
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to fetch enhanced positions: {str(e)}")
            return {}