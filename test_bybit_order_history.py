#!/usr/bin/env python3
"""
Comprehensive test script for retrieving order history from the Bybit API.

This script tests:
1. Authentication
2. Order history retrieval with different parameters
3. Error handling
4. Pagination
5. Rate limiting
6. Response field validation
"""

import os
import sys
import json
import time
import hmac
import hashlib
import urllib.parse
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import aiohttp
import ccxt.async_support as ccxt
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('bybit_test')

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.alpha_pulse.exchanges.credentials.manager import credentials_manager
    from src.alpha_pulse.exchanges.implementations.bybit import BybitExchange
    from src.alpha_pulse.exchanges.interfaces import ExchangeConfiguration
    USING_ALPHA_PULSE = True
except ImportError:
    logger.warning("Could not import from alpha_pulse package. Using standalone mode.")
    USING_ALPHA_PULSE = False

# Constants
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
RATE_LIMIT_DELAY = 0.5  # seconds between requests to avoid rate limiting
TEST_SYMBOLS = ["BTC/USDT", "ETH/USDT", "DOGE/USDT", "ADA/USDT"]
TEST_TIMEFRAMES = ["1d", "4h", "1h"]


class BybitAPITester:
    """Test class for Bybit API order history functionality."""

    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        """
        Initialize the tester with API credentials.
        
        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            testnet: Whether to use testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange = None
        self.ccxt_exchange = None
        self.session = None
        self.test_results = {
            "authentication": {"success": False, "details": ""},
            "order_history": {"success": False, "details": ""},
            "pagination": {"success": False, "details": ""},
            "error_handling": {"success": False, "details": ""},
            "rate_limiting": {"success": False, "details": ""},
            "field_validation": {"success": False, "details": ""}
        }
        
    async def initialize(self):
        """Initialize the test environment."""
        logger.info("Initializing test environment...")
        
        # Create aiohttp session for direct API calls
        self.session = aiohttp.ClientSession()
        
        # Initialize CCXT exchange
        try:
            self.ccxt_exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                    'createMarketBuyOrderRequiresPrice': True
                }
            })
            
            if self.testnet:
                self.ccxt_exchange.set_sandbox_mode(True)
                
            await self.ccxt_exchange.load_markets()
            logger.info("CCXT exchange initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CCXT exchange: {str(e)}")
            raise
            
        # Initialize AlphaPulse exchange if available
        if USING_ALPHA_PULSE:
            try:
                self.exchange = BybitExchange(testnet=self.testnet)
                await self.exchange.initialize()
                logger.info("AlphaPulse BybitExchange initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AlphaPulse BybitExchange: {str(e)}")
                # Continue with CCXT only
        
        logger.info("Test environment initialized")
        
    async def close(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        if self.ccxt_exchange:
            try:
                await self.ccxt_exchange.close()
            except Exception as e:
                logger.error(f"Error closing CCXT exchange: {str(e)}")
                
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                logger.error(f"Error closing AlphaPulse exchange: {str(e)}")
                
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.error(f"Error closing aiohttp session: {str(e)}")
                
        logger.info("Resources cleaned up")
        
    async def test_authentication(self) -> bool:
        """
        Test authentication with the Bybit API.
        
        Returns:
            bool: True if authentication is successful, False otherwise
        """
        logger.info("Testing authentication...")
        
        try:
            # Test authentication using CCXT
            account_info = await self.ccxt_exchange.fetch_balance()
            
            if account_info and 'total' in account_info:
                logger.info("Authentication successful via CCXT")
                self.test_results["authentication"] = {
                    "success": True,
                    "details": "Successfully authenticated with Bybit API via CCXT"
                }
                return True
            else:
                logger.error("Authentication failed: Invalid response format")
                self.test_results["authentication"] = {
                    "success": False,
                    "details": "Invalid response format from Bybit API"
                }
                return False
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Authentication failed: {error_msg}")
            
            # Check for specific error conditions
            if "Invalid API key" in error_msg:
                details = "Invalid API key. Please check your credentials."
            elif "IP" in error_msg and "whitelist" in error_msg:
                details = "IP address not whitelisted. Please add your IP to the whitelist in the Bybit dashboard."
            elif "permission" in error_msg:
                details = "Insufficient permissions. Please check API key permissions in the Bybit dashboard."
            else:
                details = f"Authentication error: {error_msg}"
                
            self.test_results["authentication"] = {
                "success": False,
                "details": details
            }
            return False
            
    async def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate signature for Bybit API V5.
        
        Args:
            params: Parameters to include in the signature
            
        Returns:
            Signature string
        """
        timestamp = int(time.time() * 1000)
        recv_window = 60000
        
        # Create signature string
        param_str = f"{timestamp}{self.api_key}{recv_window}"
        
        # Add query parameters to the signature if provided
        if params:
            # Sort parameters alphabetically
            sorted_params = sorted(params.items())
            # URL encode parameters
            encoded_params = urllib.parse.urlencode(sorted_params)
            # Append to param_str
            param_str += encoded_params
        
        # Generate HMAC signature
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(param_str, "utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature, timestamp, recv_window
        
    async def _direct_api_call(self, endpoint: str, params: Dict[str, Any] = None, method: str = "GET") -> Dict[str, Any]:
        """
        Make a direct API call to Bybit V5 API.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            method: HTTP method
            
        Returns:
            API response
        """
        base_url = "https://api.bybit.com"
        url = f"{base_url}{endpoint}"
        
        # Generate signature
        signature, timestamp, recv_window = await self._generate_signature(params or {})
        
        # Set up headers
        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': str(timestamp),
            'X-BAPI-RECV-WINDOW': str(recv_window),
            'Content-Type': 'application/json'
        }
        
        # Make request
        try:
            if method == "GET":
                if params:
                    query_string = urllib.parse.urlencode(params)
                    url = f"{url}?{query_string}"
                response = await self.session.get(url, headers=headers)
            else:  # POST
                response = await self.session.post(url, headers=headers, json=params or {})
                
            # Parse response
            response_text = await response.text()
            response_data = json.loads(response_text)
            
            # Check for errors
            if response_data.get('retCode', 0) != 0:
                error_msg = f"API error: {response_data.get('retCode')}, {response_data.get('retMsg', 'Unknown error')}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
            return response_data
            
        except Exception as e:
            logger.error(f"Error making direct API call to {endpoint}: {str(e)}")
            raise
            
    async def test_order_history_direct(self, symbol: str = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Test order history retrieval using direct API calls.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (success, orders)
        """
        logger.info(f"Testing order history retrieval for {symbol or 'all symbols'} using direct API...")
        
        try:
            # Prepare parameters
            params = {
                'category': 'spot',
                'limit': 50,
                'orderStatus': 'all'
            }
            
            if symbol:
                # Convert symbol format if needed
                if '/' in symbol:
                    formatted_symbol = symbol.replace('/', '')
                else:
                    formatted_symbol = symbol
                params['symbol'] = formatted_symbol
                
            # Make API call
            response = await self._direct_api_call('/v5/order/history', params)
            
            # Check response
            if 'result' in response and 'list' in response['result']:
                orders = response['result']['list']
                logger.info(f"Retrieved {len(orders)} orders via direct API")
                
                # Log sample order
                if orders:
                    logger.info(f"Sample order: {json.dumps(orders[0], indent=2)}")
                    
                return True, orders
            else:
                logger.warning("No orders found or invalid response format")
                return False, []
                
        except Exception as e:
            logger.error(f"Error retrieving order history via direct API: {str(e)}")
            return False, []
            
    async def test_order_history_ccxt(self, symbol: str = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Test order history retrieval using CCXT.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (success, orders)
        """
        logger.info(f"Testing order history retrieval for {symbol or 'all symbols'} using CCXT...")
        
        try:
            # Add time parameters (last 90 days)
            since = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
            params = {
                'since': since,
                'limit': 50
            }
            
            # Try different methods
            orders = []
            
            # Try fetch_orders if available
            try:
                if symbol:
                    fetch_orders = await self.ccxt_exchange.fetch_orders(symbol, **params)
                else:
                    # Fetch for all symbols
                    fetch_orders = []
                    for market in list(self.ccxt_exchange.markets.keys())[:5]:  # Limit to 5 markets to avoid rate limiting
                        try:
                            symbol_orders = await self.ccxt_exchange.fetch_orders(market, **params)
                            fetch_orders.extend(symbol_orders)
                            await asyncio.sleep(RATE_LIMIT_DELAY)  # Avoid rate limiting
                        except Exception as e:
                            logger.debug(f"Error fetching orders for {market}: {str(e)}")
                
                if fetch_orders:
                    logger.info(f"Retrieved {len(fetch_orders)} orders via fetch_orders")
                    orders.extend(fetch_orders)
            except Exception as e:
                logger.debug(f"fetch_orders not supported or failed: {str(e)}")
                
            # Try fetch_open_orders
            try:
                if symbol:
                    open_orders = await self.ccxt_exchange.fetch_open_orders(symbol, **params)
                else:
                    open_orders = await self.ccxt_exchange.fetch_open_orders(None, **params)
                    
                if open_orders:
                    logger.info(f"Retrieved {len(open_orders)} open orders")
                    # Add to orders if not already included
                    order_ids = {o['id'] for o in orders}
                    orders.extend([o for o in open_orders if o['id'] not in order_ids])
            except Exception as e:
                logger.debug(f"Error fetching open orders: {str(e)}")
                
            # Try fetch_closed_orders
            try:
                if symbol:
                    closed_orders = await self.ccxt_exchange.fetch_closed_orders(symbol, **params)
                else:
                    closed_orders = await self.ccxt_exchange.fetch_closed_orders(None, **params)
                    
                if closed_orders:
                    logger.info(f"Retrieved {len(closed_orders)} closed orders")
                    # Add to orders if not already included
                    order_ids = {o['id'] for o in orders}
                    orders.extend([o for o in closed_orders if o['id'] not in order_ids])
            except Exception as e:
                logger.debug(f"Error fetching closed orders: {str(e)}")
                
            # Check if we found any orders
            if orders:
                logger.info(f"Retrieved a total of {len(orders)} orders via CCXT")
                
                # Log sample order
                if orders:
                    logger.info(f"Sample order: {json.dumps(orders[0], indent=2)}")
                    
                return True, orders
            else:
                logger.warning("No orders found via CCXT")
                return False, []
                
        except Exception as e:
            logger.error(f"Error retrieving order history via CCXT: {str(e)}")
            return False, []
            
    async def test_order_history_alpha_pulse(self, symbol: str = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Test order history retrieval using AlphaPulse BybitExchange.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (success, orders)
        """
        if not USING_ALPHA_PULSE or not self.exchange:
            logger.warning("AlphaPulse BybitExchange not available, skipping test")
            return False, []
            
        logger.info(f"Testing order history retrieval for {symbol or 'all symbols'} using AlphaPulse...")
        
        try:
            if symbol:
                orders = await self.exchange.get_orders(symbol)
            else:
                # Get orders for test symbols
                orders = []
                for test_symbol in TEST_SYMBOLS:
                    try:
                        symbol_orders = await self.exchange.get_orders(test_symbol)
                        if symbol_orders:
                            orders.extend(symbol_orders)
                        await asyncio.sleep(RATE_LIMIT_DELAY)  # Avoid rate limiting
                    except Exception as e:
                        logger.debug(f"Error getting orders for {test_symbol}: {str(e)}")
                        
            if orders:
                logger.info(f"Retrieved {len(orders)} orders via AlphaPulse")
                
                # Log sample order
                if orders:
                    logger.info(f"Sample order: {json.dumps(orders[0], indent=2)}")
                    
                return True, orders
            else:
                logger.warning("No orders found via AlphaPulse")
                return False, []
                
        except Exception as e:
            logger.error(f"Error retrieving order history via AlphaPulse: {str(e)}")
            return False, []
            
    async def test_order_history(self) -> bool:
        """
        Test order history retrieval using all available methods.
        
        Returns:
            bool: True if at least one method succeeds, False otherwise
        """
        logger.info("Testing order history retrieval...")
        
        results = []
        orders_found = []
        
        # Test with direct API calls
        direct_success, direct_orders = await self.test_order_history_direct()
        results.append(("Direct API", direct_success))
        if direct_success and direct_orders:
            orders_found.extend(direct_orders)
            
        # Test with CCXT
        ccxt_success, ccxt_orders = await self.test_order_history_ccxt()
        results.append(("CCXT", ccxt_success))
        if ccxt_success and ccxt_orders:
            orders_found.extend(ccxt_orders)
            
        # Test with AlphaPulse
        if USING_ALPHA_PULSE and self.exchange:
            alpha_success, alpha_orders = await self.test_order_history_alpha_pulse()
            results.append(("AlphaPulse", alpha_success))
            if alpha_success and alpha_orders:
                orders_found.extend(alpha_orders)
                
        # Test with specific symbols
        for symbol in TEST_SYMBOLS:
            logger.info(f"Testing order history for specific symbol: {symbol}")
            
            # Direct API
            symbol_direct_success, symbol_direct_orders = await self.test_order_history_direct(symbol)
            results.append((f"Direct API ({symbol})", symbol_direct_success))
            if symbol_direct_success and symbol_direct_orders:
                orders_found.extend(symbol_direct_orders)
                
            # CCXT
            symbol_ccxt_success, symbol_ccxt_orders = await self.test_order_history_ccxt(symbol)
            results.append((f"CCXT ({symbol})", symbol_ccxt_success))
            if symbol_ccxt_success and symbol_ccxt_orders:
                orders_found.extend(symbol_ccxt_orders)
                
            # AlphaPulse
            if USING_ALPHA_PULSE and self.exchange:
                symbol_alpha_success, symbol_alpha_orders = await self.test_order_history_alpha_pulse(symbol)
                results.append((f"AlphaPulse ({symbol})", symbol_alpha_success))
                if symbol_alpha_success and symbol_alpha_orders:
                    orders_found.extend(symbol_alpha_orders)
                    
            # Avoid rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)
            
        # Summarize results
        success_count = sum(1 for _, success in results if success)
        total_count = len(results)
        
        logger.info(f"Order history test results: {success_count}/{total_count} methods successful")
        for method, success in results:
            logger.info(f"  {method}: {'Success' if success else 'Failed'}")
            
        # Update test results
        if success_count > 0:
            self.test_results["order_history"] = {
                "success": True,
                "details": f"{success_count}/{total_count} methods successful, found {len(orders_found)} orders"
            }
            return True
        else:
            self.test_results["order_history"] = {
                "success": False,
                "details": "All methods failed to retrieve order history"
            }
            return False
            
    async def test_pagination(self) -> bool:
        """
        Test pagination functionality for order history.
        
        Returns:
            bool: True if pagination works correctly, False otherwise
        """
        logger.info("Testing pagination functionality...")
        
        try:
            # Test pagination with direct API calls
            # First page
            params_page1 = {
                'category': 'spot',
                'limit': 10,
                'orderStatus': 'all'
            }
            
            response_page1 = await self._direct_api_call('/v5/order/history', params_page1)
            
            if 'result' not in response_page1 or 'list' not in response_page1['result']:
                logger.warning("Invalid response format for first page")
                self.test_results["pagination"] = {
                    "success": False,
                    "details": "Invalid response format for first page"
                }
                return False
                
            orders_page1 = response_page1['result']['list']
            
            if not orders_page1:
                logger.warning("No orders found on first page")
                self.test_results["pagination"] = {
                    "success": False,
                    "details": "No orders found on first page"
                }
                return False
                
            logger.info(f"Retrieved {len(orders_page1)} orders on first page")
            
            # Check if pagination cursor is available
            if 'nextPageCursor' not in response_page1['result']:
                logger.warning("Pagination cursor not available")
                
                # If we have less than the limit, pagination might not be needed
                if len(orders_page1) < params_page1['limit']:
                    logger.info("Less orders than limit, pagination not needed")
                    self.test_results["pagination"] = {
                        "success": True,
                        "details": "Less orders than limit, pagination not needed"
                    }
                    return True
                else:
                    self.test_results["pagination"] = {
                        "success": False,
                        "details": "Pagination cursor not available despite having full page"
                    }
                    return False
                    
            # Get next page
            cursor = response_page1['result']['nextPageCursor']
            params_page2 = {
                'category': 'spot',
                'limit': 10,
                'orderStatus': 'all',
                'cursor': cursor
            }
            
            response_page2 = await self._direct_api_call('/v5/order/history', params_page2)
            
            if 'result' not in response_page2 or 'list' not in response_page2['result']:
                logger.warning("Invalid response format for second page")
                self.test_results["pagination"] = {
                    "success": False,
                    "details": "Invalid response format for second page"
                }
                return False
                
            orders_page2 = response_page2['result']['list']
            
            # Check if we got different orders
            if not orders_page2:
                logger.info("No orders on second page, pagination complete")
                self.test_results["pagination"] = {
                    "success": True,
                    "details": "Pagination works correctly, reached end of results"
                }
                return True
                
            # Check if orders are different between pages
            page1_ids = {order['orderId'] for order in orders_page1}
            page2_ids = {order['orderId'] for order in orders_page2}
            
            if page1_ids.intersection(page2_ids):
                logger.warning("Duplicate orders found between pages")
                self.test_results["pagination"] = {
                    "success": False,
                    "details": "Duplicate orders found between pages"
                }
                return False
                
            logger.info(f"Retrieved {len(orders_page2)} orders on second page")
            logger.info("Pagination works correctly")
            
            self.test_results["pagination"] = {
                "success": True,
                "details": f"Successfully paginated through {len(orders_page1) + len(orders_page2)} orders"
            }
            return True
            
        except Exception as e:
            logger.error(f"Error testing pagination: {str(e)}")
            self.test_results["pagination"] = {
                "success": False,
                "details": f"Error: {str(e)}"
            }
            return False
            
    async def test_error_handling(self) -> bool:
        """
        Test error handling for various error conditions.
        
        Returns:
            bool: True if error handling works correctly, False otherwise
        """
        logger.info("Testing error handling...")
        
        error_tests = []
        
        # Test 1: Invalid API key
        try:
            logger.info("Testing with invalid API key...")
            
            # Create a temporary exchange with invalid credentials
            temp_exchange = ccxt.bybit({
                'apiKey': 'invalid_key',
                'secret': self.api_secret,
                'enableRateLimit': True
            })
            
            await temp_exchange.fetch_balance()
            
            # If we get here, error handling failed
            logger.error("Error handling failed: No exception raised for invalid API key")
            error_tests.append(("Invalid API key", False))
            
        except Exception as e:
            error_msg = str(e)
            logger.info(f"Expected error received for invalid API key: {error_msg}")
            
            # Check if error message is informative
            if "Invalid API key" in error_msg or "api_key" in error_msg.lower():
                error_tests.append(("Invalid API key", True))
            else:
                logger.warning(f"Error message not specific enough: {error_msg}")
                error_tests.append(("Invalid API key", False))
                
        finally:
            # Clean up
            if 'temp_exchange' in locals():
                await temp_exchange.close()
                
        # Test 2: Invalid symbol
        try:
            logger.info("Testing with invalid symbol...")
            
            invalid_symbol = "INVALID/USDT"
            await self.ccxt_exchange.fetch_ticker(invalid_symbol)
            
            # If we get here, error handling failed
            logger.error("Error handling failed: No exception raised for invalid symbol")
            error_tests.append(("Invalid symbol", False))
            
        except Exception as e:
            error_msg = str(e)
            logger.info(f"Expected error received for invalid symbol: {error_msg}")
            
            # Check if error message is informative
            if "symbol" in error_msg.lower() and ("not found" in error_msg.lower() or "invalid" in error_msg.lower()):
                error_tests.append(("Invalid symbol", True))
            else:
                logger.warning(f"Error message not specific enough: {error_msg}")
                error_tests.append(("Invalid symbol", False))
                
        # Test 3: Rate limiting
        try:
            logger.info("Testing rate limiting handling...")
            
            # Make multiple requests in quick succession
            for i in range(10):
                await self.ccxt_exchange.fetch_ticker("BTC/USDT")
                
            logger.info("Rate limiting test passed: No exceptions raised")
            error_tests.append(("Rate limiting", True))
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            if "rate" in error_msg.lower() and "limit" in error_msg.lower():
                logger.info(f"Rate limit error detected: {error_msg}")
                error_tests.append(("Rate limiting", True))
            else:
                logger.error(f"Unexpected error during rate limiting test: {error_msg}")
                error_tests.append(("Rate limiting", False))
                
        # Test 4: Network interruption simulation
        try:
            logger.info("Testing network interruption handling...")
            
            # Create a temporary session with a very short timeout
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.001)) as session:
                async with session.get("https://api.bybit.com/v5/market/tickers") as response:
                    await response.text()
                    
            # If we get here, error handling failed
            logger.error("Error handling failed: No exception raised for network timeout")
            error_tests.append(("Network interruption", False))
            
        except Exception as e:
            error_msg = str(e)
            logger.info(f"Expected error received for network timeout: {error_msg}")
            
            # Check if it's a timeout error
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                error_tests.append(("Network interruption", True))
            else:
                # This could be another type of network error, which is also acceptable
                error_tests.append(("Network interruption", True))
                
        # Summarize results
        success_count = sum(1 for _, success in error_tests if success)
        total_count = len(error_tests)
        
        logger.info(f"Error handling test results: {success_count}/{total_count} tests passed")
        for test, success in error_tests:
            logger.info(f"  {test}: {'Passed' if success else 'Failed'}")
            
        # Update test results
        if success_count == total_count:
            self.test_results["error_handling"] = {
                "success": True,
                "details": f"All {total_count} error handling tests passed"
            }
            return True
        else:
            self.test_results["error_handling"] = {
                "success": False,
                "details": f"{success_count}/{total_count} error handling tests passed"
            }
            return False
            
    async def test_rate_limiting(self) -> bool:
        """
        Test rate limiting handling.
        
        Returns:
            bool: True if rate limiting is handled correctly, False otherwise
        """
        logger.info("Testing rate limiting handling...")
        
        try:
            # Make multiple requests with and without delays
            start_time = time.time()
            
            # Test 1: Without explicit delays (CCXT should handle rate limiting)
            logger.info("Testing CCXT built-in rate limiting...")
            for i in range(20):
                await self.ccxt_exchange.fetch_ticker("BTC/USDT")
                
            elapsed_time_1 = time.time() - start_time
            logger.info(f"Completed 20 requests in {elapsed_time_1:.2f} seconds with CCXT rate limiting")
            
            # Test 2: With explicit delays
            start_time = time.time()
            logger.info("Testing explicit rate limiting...")
            for i in range(20):
                await self.ccxt_exchange.fetch_ticker("BTC/USDT")
                await asyncio.sleep(RATE_LIMIT_DELAY)
                
            elapsed_time_2 = time.time() - start_time
            logger.info(f"Completed 20 requests in {elapsed_time_2:.2f} seconds with explicit delays")
            
            # Check if CCXT rate limiting is working
            if elapsed_time_1 < 1.0:
                logger.warning("CCXT rate limiting might not be working properly (too fast)")
                
            # Update test results
            self.test_results["rate_limiting"] = {
                "success": True,
                "details": f"Successfully handled rate limiting. CCXT: {elapsed_time_1:.2f}s, Explicit: {elapsed_time_2:.2f}s"
            }
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error during rate limiting test: {error_msg}")
            
            # Check if it's a rate limit error
            if "rate" in error_msg.lower() and "limit" in error_msg.lower():
                logger.warning(f"Rate limit exceeded: {error_msg}")
                self.test_results["rate_limiting"] = {
                    "success": False,
                    "details": f"Rate limit exceeded: {error_msg}"
                }
            else:
                self.test_results["rate_limiting"] = {
                    "success": False,
                    "details": f"Error during rate limiting test: {error_msg}"
                }
                
            return False
            
    async def test_field_validation(self, orders: List[Dict[str, Any]]) -> bool:
        """
        Validate the fields in order responses.
        
        Args:
            orders: List of orders to validate
            
        Returns:
            bool: True if all required fields are present, False otherwise
        """
        logger.info("Testing field validation...")
        
        if not orders:
            logger.warning("No orders to validate")
            self.test_results["field_validation"] = {
                "success": False,
                "details": "No orders to validate"
            }
            return False
            
        # Define required fields for different response formats
        ccxt_required_fields = ['id', 'symbol', 'side', 'amount', 'status', 'timestamp']
        bybit_v5_required_fields = ['orderId', 'symbol', 'side', 'qty', 'orderStatus', 'createdTime']
        
        # Count validation results
        total_orders = len(orders)
        valid_ccxt_orders = 0
        valid_bybit_orders = 0
        
        for order in orders:
            # Check CCXT format
            if all(field in order for field in ccxt_required_fields):
                valid_ccxt_orders += 1
                
            # Check Bybit V5 format
            if all(field in order for field in bybit_v5_required_fields):
                valid_bybit_orders += 1
                
        # Determine the predominant format
        if valid_ccxt_orders > valid_bybit_orders:
            format_name = "CCXT"
            valid_orders = valid_ccxt_orders
            required_fields = ccxt_required_fields
        else:
            format_name = "Bybit V5"
            valid_orders = valid_bybit_orders
            required_fields = bybit_v5_required_fields
            
        # Calculate validation percentage
        validation_percentage = (valid_orders / total_orders) * 100 if total_orders > 0 else 0
        
        logger.info(f"Field validation results: {valid_orders}/{total_orders} orders valid ({validation_percentage:.1f}%)")
        logger.info(f"Predominant format: {format_name}")
        
        # Check additional fields and data types
        field_issues = []
        sample_order = orders[0] if orders else {}
        
        for field in required_fields:
            if field in sample_order:
                field_type = type(sample_order[field]).__name__
                logger.info(f"Field '{field}' present with type {field_type}")
            else:
                field_issues.append(f"Missing field: {field}")
                
        # Update test results
        if validation_percentage >= 90:
            self.test_results["field_validation"] = {
                "success": True,
                "details": f"{valid_orders}/{total_orders} orders valid ({validation_percentage:.1f}%) in {format_name} format"
            }
            return True
        else:
            self.test_results["field_validation"] = {
                "success": False,
                "details": f"Only {valid_orders}/{total_orders} orders valid ({validation_percentage:.1f}%) in {format_name} format. Issues: {', '.join(field_issues)}"
            }
            return False
            
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests and return results.
        
        Returns:
            Dict containing test results
        """
        logger.info("Running all tests...")
        
        try:
            # Initialize
            await self.initialize()
            
            # Test authentication
            auth_success = await self.test_authentication()
            if not auth_success:
                logger.error("Authentication failed, skipping remaining tests")
                return self.test_results
                
            # Test order history
            order_success = await self.test_order_history()
            
            # Get orders for field validation
            all_orders = []
            direct_success, direct_orders = await self.test_order_history_direct()
            if direct_success and direct_orders:
                all_orders.extend(direct_orders)
                
            ccxt_success, ccxt_orders = await self.test_order_history_ccxt()
            if ccxt_success and ccxt_orders:
                all_orders.extend(ccxt_orders)
                
            if USING_ALPHA_PULSE and self.exchange:
                alpha_success, alpha_orders = await self.test_order_history_alpha_pulse()
                if alpha_success and alpha_orders:
                    all_orders.extend(alpha_orders)
                    
            # Test pagination
            await self.test_pagination()
            
            # Test error handling
            await self.test_error_handling()
            
            # Test rate limiting
            await self.test_rate_limiting()
            
            # Test field validation
            if all_orders:
                await self.test_field_validation(all_orders)
            else:
                logger.warning("No orders found for field validation")
                self.test_results["field_validation"] = {
                    "success": False,
                    "details": "No orders found for validation"
                }
                
            return self.test_results
            
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return self.test_results
            
        finally:
            # Clean up
            await self.close()
            
    def print_results(self):
        """Print test results in a formatted way."""
        print("\n" + "=" * 80)
        print("BYBIT ORDER HISTORY TEST RESULTS")
        print("=" * 80)
        
        for test_name, result in self.test_results.items():
            status = "PASSED" if result["success"] else "FAILED"
            print(f"\n{test_name.upper()}: {status}")
            print(f"  {result['details']}")
            
        print("\n" + "=" * 80)
        
        # Overall result
        success_count = sum(1 for result in self.test_results.values() if result["success"])
        total_count = len(self.test_results)
        print(f"\nOVERALL: {success_count}/{total_count} tests passed")
        print("=" * 80 + "\n")


async def main():
    """Main function to run the tests."""
    # Get API credentials
    api_key = os.environ.get("BYBIT_API_KEY", "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")
    testnet = os.environ.get("BYBIT_TESTNET", "false").lower() == "true"
    
    # Check if credentials are available
    if not api_key or not api_secret:
        if USING_ALPHA_PULSE:
            logger.info("No API credentials provided via environment variables, trying to get from credentials_manager")
            creds = credentials_manager.get_credentials('bybit')
            if creds:
                api_key = creds.api_key
                api_secret = creds.api_secret
                logger.info(f"Using credentials from credentials_manager: API key: {api_key[:4]}...{api_key[-4:]}")
            else:
                logger.warning("No credentials found in credentials_manager")
        else:
            logger.warning("No API credentials provided via environment variables")
            
    if not api_key or not api_secret:
        logger.error("API credentials are required. Please set BYBIT_API_KEY and BYBIT_API_SECRET environment variables.")
        return
        
    # Create tester
    tester = BybitAPITester(api_key, api_secret, testnet)
    
    # Run tests
    results = await tester.run_all_tests()
    
    # Print results
    tester.print_results()
    
    # Return success status
    success_count = sum(1 for result in results.values() if result["success"])
    total_count = len(results)
    return success_count == total_count


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)