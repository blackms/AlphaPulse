"""
CCXT library adapter implementation.
"""
import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
import ccxt.async_support as ccxt
from loguru import logger

from ..interfaces import (
    BaseExchange,
    ExchangeConfiguration,
    ExchangeError,
    ConnectionError,
    OrderError,
    MarketDataError
)
from ..base import Balance, OHLCV


class CCXTAdapter(BaseExchange):
    """
    Adapter for CCXT library integration.
    
    This class adapts the CCXT library interface to our exchange interfaces,
    providing a consistent API while isolating the CCXT dependency.
    """
    
    def __init__(self, config, exchange_id=None):
        """
        Initialize CCXT adapter.
        
        Args:
            config: Exchange configuration
            exchange_id: Exchange identifier (optional, can be extracted from config)
        """
        # Use provided exchange_id if given, otherwise extract from config
        self.exchange_id = exchange_id or config.options.get('_exchange_id', 'bybit')
        self.config = config
        self.exchange = None
        self._markets = {}
        
    async def initialize(self) -> None:
        """Initialize exchange connection."""
        try:
            # Check if API credentials are available
            if not self.config.api_key or not self.config.api_secret:
                logger.warning(f"No API credentials available for {self.exchange_id}. Using read-only mode.")
            
            # Debug logging for credentials
            if self.config.api_key:
                masked_key = self.config.api_key[:4] + '...' + self.config.api_key[-4:] if len(self.config.api_key) > 8 else '***'
                masked_secret = self.config.api_secret[:4] + '...' + self.config.api_secret[-4:] if len(self.config.api_secret) > 8 else '***'
                logger.debug(f"CCXT DEBUG - Using API key: {self.config.api_key}")  # Full key for debug
                logger.debug(f"CCXT DEBUG - Using API secret: {self.config.api_secret}")  # Full secret for debug
                logger.debug(f"CCXT DEBUG - Testnet mode: {self.config.testnet}")
            else:
                logger.debug("CCXT DEBUG - No API credentials available")
            
            # Create CCXT exchange instance
            exchange_class = getattr(ccxt, self.exchange_id)
            params = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    **self.config.options
                }
            }
            
            # Only add API credentials if they are available
            if self.config.api_key and self.config.api_secret:
                params['apiKey'] = self.config.api_key
                params['secret'] = self.config.api_secret
                logger.debug(f"CCXT DEBUG - Added credentials to params: apiKey={params['apiKey']}")
            
            self.exchange = exchange_class(params)
            
            # Configure testnet if enabled
            if self.config.testnet:
                logger.debug("CCXT DEBUG - Setting sandbox mode to True")
                self.exchange.set_sandbox_mode(True)
            
            # Load markets
            self._markets = await self.exchange.load_markets()
            logger.info(f"Initialized {self.exchange_id} exchange")
            
        except Exception as e:
            raise ConnectionError(f"Failed to initialize {self.exchange_id}: {str(e)}")
    
    async def close(self) -> None:
        """Close exchange connection."""
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                logger.error(f"Error closing {self.exchange_id} connection: {str(e)}")
    
    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return Decimal(str(ticker['last'])) if ticker['last'] else None
        except Exception as e:
            raise MarketDataError(f"Failed to fetch ticker for {symbol}: {str(e)}")
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Fetch OHLCV candles."""
        try:
            candles = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            result = []
            for candle in candles:
                ohlcv = OHLCV.from_list(candle)
                ohlcv.symbol = symbol
                result.append(ohlcv)
            return result
        except Exception as e:
            raise MarketDataError(f"Failed to fetch OHLCV data: {str(e)}")
    
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """Execute trade order."""
        try:
            params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': float(amount)
            }
            
            if price and order_type == 'limit':
                params['price'] = float(price)
            
            order = await self.exchange.create_order(**params)
            
            # Wait for market orders to complete
            if order_type == 'market':
                while True:
                    order = await self.exchange.fetch_order(order['id'], symbol)
                    if order['status'] in ['closed', 'canceled', 'expired', 'rejected']:
                        break
                    await asyncio.sleep(0.5)
            
            return order
            
        except Exception as e:
            raise OrderError(f"Failed to execute trade: {str(e)}")
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get order history."""
        try:
            params = {}
            if since:
                params['since'] = since
            if limit:
                params['limit'] = limit
            
            # Special handling for Bybit UTA accounts which don't support fetchOrders
            if self.exchange_id.lower() == 'bybit':
                logger.debug(f"Using Bybit-specific order history fetching for UTA accounts with symbol: {symbol}")
                return await self._get_bybit_order_history(symbol, params)
            
            if symbol:
                return await self.exchange.fetch_orders(symbol, **params)
            
            # Fetch orders for all symbols
            orders = []
            for market in self._markets:
                try:
                    symbol_orders = await self.exchange.fetch_orders(market, **params)
                    orders.extend(symbol_orders)
                except Exception as e:
                    logger.warning(f"Error fetching orders for {market}: {e}")
            
            return orders
            
        except Exception as e:
            raise OrderError(f"Failed to fetch order history: {str(e)}")
    
    async def _get_bybit_order_history(self, symbol: Optional[str], params: Dict) -> List[Dict[str, Any]]:
        """Get order history for Bybit UTA accounts using supported methods."""
        try:
            from datetime import datetime, timedelta
            
            all_orders = []
            
            # Log the symbol being used for debugging
            logger.debug(f"Fetching Bybit order history for symbol: {symbol}")
            logger.debug(f"Input params: {params}")
            
            # Create Bybit-specific parameters with category and time range
            # Bybit requires these parameters for its Unified Trading Account (UTA)
            bybit_params = {
                **params,  # Keep original params
                # Set limit higher to ensure we get all orders
                'limit': params.get('limit', 50)
            }
            
            # Add time range if not provided (last 90 days by default)
            if 'since' not in bybit_params and 'startTime' not in bybit_params:
                # Use last 90 days as default time range
                start_time = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
                bybit_params['since'] = start_time
                logger.debug(f"Added time range parameter: since={start_time} (90 days ago)")
            
            # Categories to try in order (for Bybit UTA accounts)
            categories = ['spot', 'linear']
            
            for category in categories:
                # Set category parameter for this attempt
                category_params = {
                    **bybit_params,
                    'category': category,
                    'orderFilter': 'all'  # Make sure to get all orders, including history
                }
                
                logger.debug(f"Trying to fetch orders with category: {category}, params: {category_params}")
                
                # Get open orders with this category
                try:
                    logger.debug(f"Fetching open orders for {symbol} with category {category}")
                    # Remove category and orderFilter parameters as they're not supported by fetch_open_orders
                    open_params = {k: v for k, v in category_params.items() 
                                  if k not in ['category', 'orderFilter']}
                    logger.debug(f"Adjusted params for fetch_open_orders: {open_params}")
                    open_orders = await self.exchange.fetch_open_orders(symbol, **open_params)
                    if open_orders:
                        logger.info(f"Found {len(open_orders)} open orders for {symbol} in {category} category")
                        logger.debug(f"Sample open order: {open_orders[0] if open_orders else 'None'}")
                        all_orders.extend(open_orders)
                except Exception as e:
                    logger.debug(f"Error fetching open orders for {symbol} in {category} category: {str(e)}")
                
                # Get closed orders with this category
                try:
                    logger.debug(f"Fetching closed orders for {symbol} with category {category}")
                    # Remove category and orderFilter parameters as they're not supported by fetch_closed_orders
                    closed_params = {k: v for k, v in category_params.items() 
                                    if k not in ['category', 'orderFilter']}
                    logger.debug(f"Adjusted params for fetch_closed_orders: {closed_params}")
                    closed_orders = await self.exchange.fetch_closed_orders(symbol, **closed_params)
                    if closed_orders:
                        logger.info(f"Found {len(closed_orders)} closed orders for {symbol} in {category} category")
                        logger.debug(f"Sample closed order: {closed_orders[0] if closed_orders else 'None'}")
                        all_orders.extend(closed_orders)
                except Exception as e:
                    logger.debug(f"Error fetching closed orders for {symbol} in {category} category: {str(e)}")
                
                # If we found orders in this category, no need to try others
            
            # If we still don't have orders, try one more approach with direct API call
            if not all_orders and symbol:
                logger.debug("No orders found with standard methods, trying direct API call")
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
                    logger.debug(f"Direct API response structure: {list(response.keys()) if response else 'None'}")
                    
                    # Check for error codes in the response
                    if response and 'retCode' in response:
                        ret_code = response.get('retCode')
                        ret_msg = response.get('retMsg', 'Unknown error')
                        
                        if ret_code != 0:  # Non-zero code indicates an error
                            error_msg = f"Bybit API error: code={ret_code}, message={ret_msg}"
                            
                            # Check for specific error conditions
                            if "Order status is wrong" in ret_msg:
                                error_msg += "; You might need to whitelist your IP address in the Bybit dashboard"
                            elif "Invalid API key" in ret_msg or "api_key" in ret_msg.lower():
                                error_msg += "; Check your API key and secret"
                            
                            logger.error(error_msg)
                            raise Exception(error_msg)
                    
                    # Extract orders from response if available
                    if response and 'result' in response and 'list' in response['result']:
                        direct_orders = response['result']['list']
                        logger.info(f"Found {len(direct_orders)} orders with direct API call for {direct_symbol}")
                        # TODO: Convert to CCXT format if needed
                        # For now we just log that we found some orders
                        if direct_orders:
                            logger.info(f"Sample direct order: {direct_orders[0]}")
                except Exception as e:
                    # Check if the exception message contains a JSON response with error codes
                    error_str = str(e)
                    if "retCode" in error_str:
                        try:
                            # Try to extract the JSON part from the error message
                            import re
                            import json
                            
                            # Find JSON-like content in the error message
                            json_match = re.search(r'(\{.*\})', error_str)
                            if json_match:
                                json_str = json_match.group(1)
                                error_data = json.loads(json_str)
                                
                                # Extract error details
                                ret_code = error_data.get('retCode')
                                ret_msg = error_data.get('retMsg', 'Unknown error')
                                
                                error_details = f"Bybit API error: code={ret_code}, message={ret_msg}"
                                
                                # Check for specific error conditions
                                if "Order status is wrong" in ret_msg:
                                    error_details += "; You might need to whitelist your IP address in the Bybit dashboard"
                                elif "Invalid API key" in ret_msg or "api_key" in ret_msg.lower():
                                    error_details += "; Check your API key and secret"
                                
                                logger.error(error_details)
                        except Exception as json_error:
                            # If JSON parsing fails, just log the original error
                            logger.error(f"Error with direct API call: {error_str}")
                    else:
                        logger.error(f"Error with direct API call: {error_str}")
            
            if all_orders:
                logger.info(f"Total orders found for {symbol}: {len(all_orders)}")
            else:
                # This should be a warning since it indicates a potential issue
                logger.warning(f"No orders found for {symbol} after trying multiple approaches")
            
            return all_orders
        except Exception as e:
            logger.warning(f"Error fetching Bybit order history for {symbol}: {str(e)}")
            return []
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances."""
        # Check if API credentials are available
        if not self.config.api_key or not self.config.api_secret:
            logger.warning(f"No API credentials available for {self.exchange_id}. Returning empty balances.")
            return {}
            
        try:
            response = await self.exchange.fetch_balance()
            balances = {}
            
            for asset, data in response['total'].items():
                if data > 0:
                    balances[asset] = Balance(
                        total=Decimal(str(data)),
                        available=Decimal(str(response['free'].get(asset, 0))),
                        locked=Decimal(str(response['used'].get(asset, 0)))
                    )
            
            return balances
            
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication error fetching balances: {str(e)}")
            logger.warning("Returning empty balances due to authentication error")
            return {}
        except Exception as e:
            raise ExchangeError(f"Failed to fetch balances: {str(e)}")
    
    async def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in base currency."""
        try:
            balances = await self.get_balances()
            # If no balances are available, return zero
            if not balances:
                logger.warning("No balances available to calculate portfolio value")
                return Decimal('0')
                
            total_value = Decimal('0')
            
            for asset, balance in balances.items():
                if asset == 'USDT':  # Base currency
                    total_value += balance.total
                else:
                    price = await self.get_ticker_price(f"{asset}/USDT")
                    if price:
                        total_value += balance.total * price
            
            return total_value
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio value: {str(e)}")
            return Decimal('0')
    
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions."""
        try:
            positions = {}
            balances = await self.get_balances()
            
            # If no balances are available, return empty positions
            if not balances:
                logger.warning("No balances available to determine positions")
                return positions
                
            for asset, balance in balances.items():
                if balance.total > 0:
                    try:
                        price = await self.get_ticker_price(f"{asset}/USDT")
                        if price:
                            positions[asset] = {
                                'symbol': asset,
                                'quantity': float(balance.total),
                                'entry_price': None,  # Not available in spot
                                'current_price': float(price),
                                'unrealized_pnl': None,  # Not available in spot
                                'liquidation_price': None  # Not available in spot
                            }
                    except MarketDataError as e:
                        logger.warning(f"Could not get price for {asset}: {str(e)}")
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to fetch positions: {str(e)}")
            return {}
    
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees."""
        try:
            trading_fees = await self.exchange.fetch_trading_fees()
            return {
                symbol: Decimal(str(fees.get('taker', 0)))
                for symbol, fees in trading_fees.items()
            }
        except Exception as e:
            logger.warning(f"Failed to fetch trading fees: {str(e)}")
            return {}
    
    def _normalize_symbol(self, symbol: str) -> Dict[str, str]:
        """
        Normalize symbol into different formats that exchanges might expect.
        
        Returns a dictionary with different symbol formats:
        - original: The original symbol as passed
        - base: The base currency (e.g., "BTC" from "BTC/USDT")
        - quote: The quote currency (e.g., "USDT" from "BTC/USDT") 
        - pair: The full trading pair with "/" (e.g., "BTC/USDT")
        - market: Exchange-specific formatted symbol (e.g., "BTCUSDT" for some exchanges)
        - spot: Spot market symbol format
        - linear: Linear contract format
        """
        result = {
            "original": symbol,
            "base": symbol,
            "quote": "USDT",  # Default quote currency
            "pair": symbol,
            "market": symbol,
            "spot": symbol,
            "linear": symbol
        }
        
        # Handle cases where symbol might be None or empty
        if not symbol:
            logger.warning("Empty symbol provided to _normalize_symbol")
            return result
            
        # Handle 'BTC/USDT' format
        if '/' in symbol:
            base, quote = symbol.split('/')
            result["base"] = base
            result["quote"] = quote
            result["pair"] = f"{base}/{quote}"
            result["market"] = f"{base}{quote}"
            result["spot"] = f"{base}/{quote}"
            result["linear"] = f"{base}{quote}"
        # Handle 'BTCUSDT' format (no separator)
        elif len(symbol) > 3:
            # Try to detect the quote currency
            for quote in ["USDT", "USD", "BTC", "ETH", "BUSD"]:
                if symbol.endswith(quote):
                    base = symbol[:-len(quote)]
                    result["base"] = base
                    result["quote"] = quote
                    result["pair"] = f"{base}/{quote}"
                    result["market"] = symbol
                    result["spot"] = f"{base}/{quote}"
                    result["linear"] = symbol
                    break
            else:
                # If we couldn't find a standard quote currency, assume it's just a base currency
                result["pair"] = f"{symbol}/USDT"
                result["market"] = f"{symbol}USDT"
                result["spot"] = f"{symbol}/USDT"
                result["linear"] = f"{symbol}USDT"
        else:
            # If just 'BTC' is provided, assume USDT as quote currency for trading pairs
            result["pair"] = f"{symbol}/USDT"
            result["market"] = f"{symbol}USDT"
            result["spot"] = f"{symbol}/USDT"
            result["linear"] = f"{symbol}USDT"
        
        # For Bybit specific handling
        if self.exchange_id.lower() == 'bybit':
            # Bybit usually expects symbols without the '/'
            result["market"] = result["pair"].replace("/", "")
            # Some Bybit endpoints need specific formats
            result["spot"] = result["pair"]  # Keep the '/' for spot markets
            result["linear"] = result["market"]  # No '/' for linear contracts
        
        logger.debug(f"Normalized symbol formats for {symbol}: {result}")
        return result
    
    async def get_average_entry_price(self, symbol: str) -> Optional[Decimal]:
        """Calculate average entry price for symbol."""
        try:
            # Normalize the symbol to handle different formats
            symbol_formats = self._normalize_symbol(symbol)
            
            # Store all orders we find
            all_orders = []
            
            # Try to get orders using different symbol formats
            formats_to_try = []
            
            # For Bybit, we need to try multiple formats
            if self.exchange_id.lower() == 'bybit':
                formats_to_try = [
                    ("pair", symbol_formats['pair']),
                    ("market", symbol_formats['market']),
                    ("spot", symbol_formats['spot']),
                    ("linear", symbol_formats['linear']),
                    ("original", symbol_formats['original'])
                ]
            else:
                # For other exchanges
                formats_to_try = [
                    ("pair", symbol_formats['pair']),
                    ("base", symbol_formats['base']),
                    ("market", symbol_formats['market'])
                ]
            
            # Try each format until we find orders
            for format_name, format_value in formats_to_try:
                logger.debug(f"Trying to get order history with {format_name} format: '{format_value}'")
                
                try:
                    if self.exchange_id.lower() == 'bybit':
                        # Add time parameters for bybit (last 90 days)
                        from datetime import datetime, timedelta
                        time_params = {
                            'since': int((datetime.now() - timedelta(days=90)).timestamp() * 1000),
                            'limit': 100
                        }
                        orders = await self._get_bybit_order_history(symbol=format_value, params=time_params)
                    else:
                        orders = await self.get_order_history(symbol=format_value)
                        
                    if orders:
                        logger.info(f"Found {len(orders)} orders using {format_name} format: {format_value}")
                        all_orders.extend(orders)
                        break  # Stop trying formats if we found orders
                    else:
                        logger.debug(f"No orders found with {format_name} format: {format_value}")
                except Exception as e:
                    logger.debug(f"Error getting orders with {format_name} format: {format_value} - {str(e)}")
            
            if not all_orders:
                # Attempt one last try with a direct approach for Bybit
                if self.exchange_id.lower() == 'bybit':
                    try:
                        # For Bybit, try one more time with direct market format without any separators
                        direct_symbol = symbol_formats['base'] + symbol_formats['quote']
                        logger.debug(f"Last attempt: trying direct format '{direct_symbol}' for Bybit")
                        
                        # Add time parameters for direct call too
                        from datetime import datetime, timedelta
                        direct_time_params = {
                            'since': int((datetime.now() - timedelta(days=90)).timestamp() * 1000),
                            'limit': 100,
                            'category': 'spot'  # Try spot market first
                        }
                        all_orders = await self._get_bybit_order_history(symbol=direct_symbol, params=direct_time_params)
                    except Exception as e:
                        logger.debug(f"Direct format attempt failed: {str(e)}")
                if not all_orders:
                    # Use debug level to avoid false alarms in the logs
                    # This is not an error condition, just informational for troubleshooting
                    logger.debug(f"No order history found for {symbol} after trying multiple formats")
                    
                    # Check if we have direct access to trades (might provide better entry price data)
                    try:
                        logger.debug(f"Attempting to fetch trades for {symbol}")
                        trades = await self.exchange.fetch_my_trades(symbol_formats['pair'])
                        if trades:
                            logger.info(f"Found {len(trades)} trades for {symbol}")
                            # Process trades to calculate entry price
                            buy_trades = [t for t in trades if t['side'] == 'buy']
                            if buy_trades:
                                total_quantity = Decimal('0')
                                total_value = Decimal('0')
                                for trade in buy_trades:
                                    quantity = Decimal(str(trade['amount']))
                                    price = Decimal(str(trade['price']))
                                    total_quantity += quantity
                                    total_value += quantity * price
                                
                                if total_quantity > 0:
                                    average_entry = total_value / total_quantity
                                    logger.info(f"Calculated average entry price from trades for {symbol}: {average_entry}")
                                    return average_entry
                    except Exception as trade_error:
                        logger.debug(f"Failed to fetch trades for {symbol}: {trade_error}")
                    
                    # Fall back to current price if we can't calculate entry price
                try:
                    # Try using the pair format for getting current price
                    current_price = await self.get_ticker_price(symbol_formats['pair'])
                    if current_price:
                        logger.info(f"Using current price {current_price} as fallback for {symbol}")
                        return current_price
                except Exception as price_error:
                    logger.debug(f"Failed to get current price for {symbol}: {price_error}")
                
                return None
            
            total_quantity = Decimal('0')
            total_value = Decimal('0')
            
            logger.debug(f"Found {len(all_orders)} orders for {symbol}")
            
            for order in all_orders:
                # Only include filled buy orders
                if order['side'] == 'buy' and order['status'] == 'filled':
                    # Handle cases where price might be None for market orders
                    quantity = Decimal(str(order['amount']))
                    price = Decimal(str(order['price'] if order['price'] else order.get('average', 0)))
                    
                    if price > 0:
                        total_quantity += quantity
                        total_value += quantity * price
                    else:
                        # For market orders, try to use average or cost
                        cost = Decimal(str(order.get('cost', 0)))
                        if cost > 0:
                            total_value += cost
                            total_quantity += quantity
            
            if total_quantity == 0:
                logger.info(f"No filled buy orders found for {symbol}, falling back to current price")
                return None
            
            average_entry = total_value / total_quantity
            logger.info(f"Calculated average entry price for {symbol}: {average_entry}")
            return average_entry
            
        except Exception as e:
            logger.error(f"Error calculating average entry price for {symbol}: {str(e)}")
            # Don't crash - return None and let the caller handle it
            return None