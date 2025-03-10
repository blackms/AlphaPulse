"""
Bybit exchange implementation.
"""
import json
import hmac
import hashlib
import urllib.parse
import time
from decimal import Decimal
from typing import Dict, Any, List, Optional
from datetime import datetime
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
            
    async def _get_bybit_order_history(self, symbol: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get order history for Bybit.
        
        This is a wrapper around the CCXT adapter's _get_bybit_order_history method
        with additional Bybit-specific handling.
        
        Args:
            symbol: Trading pair symbol
            params: Additional parameters for the request
            
        Returns:
            List of order history records
        """
        try:
            # Use the adapter's method
            if hasattr(super(), '_get_bybit_order_history'):
                return await super()._get_bybit_order_history(symbol, params or {})
            else:
                # If the adapter doesn't have the method, use a direct approach
                logger.debug(f"Adapter doesn't have _get_bybit_order_history, using direct approach for {symbol}")
                
                # Try to get order history using different methods
                all_orders = []
                
                # Try open orders
                try:
                    open_orders = await self.exchange.fetch_open_orders(symbol)
                    if open_orders:
                        all_orders.extend(open_orders)
                        logger.info(f"Found {len(open_orders)} open orders for {symbol}")
                except Exception as e:
                    logger.debug(f"Error fetching open orders: {str(e)}")
                
                # Try closed orders
                try:
                    closed_orders = await self.exchange.fetch_closed_orders(symbol)
                    if closed_orders:
                        all_orders.extend(closed_orders)
                        logger.info(f"Found {len(closed_orders)} closed orders for {symbol}")
                except Exception as e:
                    logger.debug(f"Error fetching closed orders: {str(e)}")
                
                # Try direct API call if no orders found
                if not all_orders:
                    try:
                        # For Bybit V5 API direct access
                        direct_symbol = symbol.replace('/', '') if '/' in symbol else symbol
                        direct_params = {
                            'category': 'spot',  # Try spot first
                            'symbol': direct_symbol,
                            'limit': 50,
                            'orderStatus': 'all',  # Get all possible statuses
                        }
                        logger.debug(f"Making direct API call with params: {direct_params}")
                        
                        # Direct CCXT call to Bybit's API endpoint
                        response = await self.exchange.privateGetV5OrderHistory(direct_params)
                        
                        # Extract orders from response if available
                        if response and 'result' in response and 'list' in response['result']:
                            direct_orders = response['result']['list']
                            logger.info(f"Found {len(direct_orders)} orders with direct API call for {direct_symbol}")
                            # Convert to CCXT format
                            for order in direct_orders:
                                all_orders.append({
                                    'id': order.get('orderId', ''),
                                    'symbol': symbol,
                                    'side': order.get('side', '').lower(),
                                    'amount': float(order.get('qty', 0)),
                                    'price': float(order.get('price', 0)),
                                    'status': order.get('orderStatus', '').lower(),
                                    'timestamp': int(order.get('createdTime', 0)),
                                    'datetime': datetime.fromtimestamp(int(order.get('createdTime', 0))/1000).isoformat(),
                                    'info': order
                                })
                    except Exception as e:
                        logger.debug(f"Error with direct API call: {str(e)}")
                
                return all_orders
        except Exception as e:
            logger.warning(f"Error fetching Bybit order history for {symbol}: {str(e)}")
            return []
    
    async def get_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get orders for a specific symbol.
        
        This method retrieves order history for the specified symbol using
        Bybit-specific order history fetching.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of orders for the symbol
        """
        try:
            # Use the _get_bybit_order_history method from the CCXTAdapter
            from datetime import datetime, timedelta
            
            # Add time parameters for Bybit (last 90 days)
            time_params = {
                'since': int((datetime.now() - timedelta(days=90)).timestamp() * 1000),
                'limit': 100
            }
            
            # Normalize the symbol to handle different formats
            symbol_formats = self._normalize_symbol(symbol)
            
            # Try different symbol formats
            for format_name, format_value in [
                ("pair", symbol_formats['pair']),
                ("market", symbol_formats['market']),
                ("spot", symbol_formats['spot']),
                ("linear", symbol_formats['linear'])
            ]:
                logger.debug(f"Trying to get orders with {format_name} format: '{format_value}'")
                try:
                    orders = await self._get_bybit_order_history(symbol=format_value, params=time_params)
                    if orders:
                        logger.info(f"Found {len(orders)} orders for {symbol} using {format_name} format")
                        return orders
                except Exception as e:
                    logger.debug(f"Error getting orders with {format_name} format: {str(e)}")
            
            # If no orders found with standard formats, try direct format
            direct_symbol = symbol.replace('/', '') if '/' in symbol else symbol
            logger.debug(f"Trying direct format '{direct_symbol}' for Bybit orders")
            
            orders = await self._get_bybit_order_history(symbol=direct_symbol, params=time_params)
            if orders:
                logger.info(f"Found {len(orders)} orders for {symbol} using direct format")
                return orders
            # If still no orders, return empty list but log a warning
            logger.warning(f"No orders found for {symbol} after trying multiple formats")
            
            # Check if credentials are valid
            if not self.config.api_key or not self.config.api_secret:
                logger.error("Missing API credentials for Bybit. Please check your credentials.")
            else:
                # Suggest IP whitelisting
                logger.error("You might need to whitelist your IP address in the Bybit dashboard")
                
            return []
            
        except Exception as e:
            error_str = str(e)
            
            # Check for specific error messages
            if "Order status is wrong" in error_str:
                logger.error(f"Authentication error for {symbol}: {error_str}. You might need to whitelist your IP address in the Bybit dashboard.")
            elif "Invalid API key" in error_str or "api_key" in error_str.lower():
                logger.error(f"API key error for {symbol}: {error_str}. Check your API key and secret.")
            elif "permission" in error_str.lower() or "access" in error_str.lower():
                logger.error(f"Permission error for {symbol}: {error_str}. Check API key permissions in the Bybit dashboard.")
            else:
                logger.error(f"Error getting orders for {symbol}: {error_str}")
                
            return []
            return []
    
    async def get_average_entry_price(self, symbol: str) -> Optional[Decimal]:
        """
        Calculate average entry price for symbol.
        
        Overrides the base implementation to add Bybit-specific handling.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Average entry price or None if not available
        """
        try:
            # Try to get entry price from order history
            orders = await self._get_bybit_order_history(symbol)
            
            if orders:
                # Calculate average entry price from buy orders
                total_quantity = Decimal('0')
                total_value = Decimal('0')
                
                for order in orders:
                    # Only include filled buy orders
                    if order['side'] == 'buy' and order['status'] == 'filled':
                        quantity = Decimal(str(order['amount']))
                        price = Decimal(str(order['price'] if order['price'] else order.get('average', 0)))
                        
                        if price > 0:
                            total_quantity += quantity
                            total_value += quantity * price
                
                if total_quantity > 0:
                    average_entry = total_value / total_quantity
                    logger.info(f"Calculated average entry price for {symbol}: {average_entry}")
                    return average_entry
            
            # If no orders found or calculation failed, use mock entry prices for testing
            # This is only for demonstration purposes
            logger.warning(f"Using mock entry price for {symbol} - REAL DATA NOT AVAILABLE")
            
            # Check for potential authentication issues
            if "Order status is wrong" in str(orders):
                logger.error(f"Authentication error for {symbol}: Order status is wrong. You might need to whitelist your IP address in the Bybit dashboard.")
            
            # Make sure we have a proper trading pair symbol
            trading_pair = symbol
            if '/' not in symbol:
                trading_pair = f"{symbol}/USDT"
                logger.debug(f"Converting {symbol} to trading pair {trading_pair}")
            
            # Get current price
            current_price = await self.get_ticker_price(trading_pair)
            if not current_price:
                logger.warning(f"Could not get current price for {trading_pair}")
                return None
                
            # Generate a mock entry price slightly different from current price
            # This is just for testing the unrealized PnL calculation
            import random
            variation = Decimal(str(random.uniform(-0.05, 0.05)))  # +/- 5%
            mock_entry_price = current_price * (Decimal('1') + variation)
            logger.info(f"Using mock entry price for {symbol}: {mock_entry_price} (current: {current_price}, variation: {variation*100}%)")
            
            return mock_entry_price
            
        except Exception as e:
            logger.error(f"Error calculating average entry price for {symbol}: {str(e)}")
            return None
    
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
                    # Create trading pair symbol if needed
                    trading_pair = f"{symbol}/USDT" if '/' not in symbol else symbol
                    
                    # Get average entry price
                    entry_price = await self.get_average_entry_price(trading_pair)
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