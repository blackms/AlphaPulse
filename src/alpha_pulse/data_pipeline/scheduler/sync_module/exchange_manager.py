"""
Exchange management module for the data synchronization system.

This module handles the creation, initialization, and management of exchange instances.
"""
import os
import time
import asyncio
import threading
from typing import Dict, Optional, Any

from loguru import logger
import ccxt.async_support as ccxt

from alpha_pulse.exchanges.factories import ExchangeFactory
from alpha_pulse.exchanges.interfaces import BaseExchange, ConnectionError, ExchangeError
from alpha_pulse.exchanges.types import ExchangeType


class ExchangeManager:
    """
    Manages exchange instances for the data synchronization system.
    
    This class is responsible for creating, initializing, and caching exchange instances.
    It also implements circuit breaker patterns to prevent repeated failures.
    """
    
    def __init__(self):
        """Initialize the exchange manager."""
        self._exchange_instances: Dict[str, BaseExchange] = {}
        # Circuit breaker state
        self._circuit_breakers: Dict[str, int] = {}
        self._circuit_breaker_times: Dict[str, float] = {}
    
    async def get_exchange(self, exchange_id: str) -> Optional[BaseExchange]:
        """
        Get or create an exchange instance.
        
        Args:
            exchange_id: Exchange identifier
            
        Returns:
            Exchange instance or None if failed
        """
        # Check if circuit breaker is active
        circuit_breaker_count = self._circuit_breakers.get(exchange_id, 0)
        circuit_breaker_time = self._circuit_breaker_times.get(exchange_id, 0)
        
        # If circuit breaker is active and cooldown period hasn't expired, return None
        if circuit_breaker_count >= 5 and circuit_breaker_time > 0:
            cooldown_period = 600  # 10 minutes in seconds
            current_time = time.time()
            if current_time - circuit_breaker_time < cooldown_period:
                remaining_time = int(cooldown_period - (current_time - circuit_breaker_time))
                logger.warning(f"Circuit breaker active for {exchange_id}. Will not attempt initialization for {remaining_time} more seconds.")
                return None
            else:
                # Reset circuit breaker after cooldown period
                logger.info(f"Circuit breaker cooldown period expired for {exchange_id}. Resetting circuit breaker.")
                self._circuit_breakers[exchange_id] = 0
                self._circuit_breaker_times[exchange_id] = 0
        
        # Return cached instance if available
        if exchange_id in self._exchange_instances:
            logger.debug(f"Using cached exchange instance for {exchange_id}")
            return self._exchange_instances[exchange_id]
        
        try:
            # Get API credentials from environment
            api_key = os.environ.get(f'{exchange_id.upper()}_API_KEY',
                                    os.environ.get('EXCHANGE_API_KEY', ''))
            api_secret = os.environ.get(f'{exchange_id.upper()}_API_SECRET',
                                      os.environ.get('EXCHANGE_API_SECRET', ''))
            testnet = os.environ.get(f'{exchange_id.upper()}_TESTNET',
                                   os.environ.get('EXCHANGE_TESTNET', 'true')).lower() == 'true'
            
            # Log credential information (masked for security)
            if api_key:
                masked_key = api_key[:4] + '...' + api_key[-4:] if len(api_key) > 8 else '***'
                masked_secret = api_secret[:4] + '...' + api_secret[-4:] if len(api_secret) > 8 else '***'
                logger.info(f"Initializing {exchange_id} exchange with API key: {masked_key}")
                logger.debug(f"API secret (masked): {masked_secret}")
            else:
                logger.warning(f"No API key found for {exchange_id}. Check your environment variables.")
                logger.info(f"Expected environment variables: {exchange_id.upper()}_API_KEY or EXCHANGE_API_KEY")
            
            logger.info(f"Using testnet mode: {testnet}")
            
            # For Bybit, always use mainnet (testnet=False)
            if exchange_id.lower() == 'bybit':
                testnet = False
                logger.info("Using mainnet mode for Bybit (testnet=False)")
                
                # Get account type for Bybit (UNIFIED or SPOT)
                account_type = os.environ.get(f'{exchange_id.upper()}_ACCOUNT_TYPE',
                                           os.environ.get('EXCHANGE_ACCOUNT_TYPE', 'UNIFIED'))
                logger.info(f"Using account type for Bybit: {account_type}")
            
            # Determine exchange type
            exchange_type = None
            if exchange_id.lower() == 'binance':
                exchange_type = ExchangeType.BINANCE
            elif exchange_id.lower() == 'bybit':
                exchange_type = ExchangeType.BYBIT
            else:
                logger.warning(f"Unknown exchange type: {exchange_id}, assuming Binance")
                exchange_type = ExchangeType.BINANCE
            
            # Create exchange instance with additional options for Bybit
            logger.info(f"Creating exchange of type {exchange_type.value} with testnet={testnet}")
            
            # Add extra options for Bybit
            extra_options = {}
            if exchange_id.lower() == 'bybit':
                extra_options = {
                    'accountType': account_type,  # Add account type for Bybit
                    'recvWindow': 60000,  # Increase receive window for better reliability
                }
                logger.debug(f"Adding extra options for Bybit: {extra_options}")
            
            exchange = ExchangeFactory.create_exchange(
                exchange_type,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                extra_options=extra_options
            )
            
            # Initialize the exchange with retry logic
            max_retries = 3
            base_retry_delay = 2
            retry_delay = base_retry_delay
            last_error = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Initializing {exchange_id} exchange (attempt {attempt}/{max_retries})")
                    
                    # Add timeout to prevent hanging indefinitely
                    init_timeout = 15  # 15 seconds timeout for initialization
                    thread_id = threading.get_ident()
                    
                    logger.debug(f"[THREAD {thread_id}] Initializing {exchange_id} exchange with timeout {init_timeout}s")
                    
                    # Create a task with timeout
                    try:
                        # Get the current event loop
                        current_loop = None
                        try:
                            current_loop = asyncio.get_event_loop()
                            logger.debug(f"[THREAD {thread_id}] Got current event loop for initialization: {current_loop}")
                        except RuntimeError as loop_error:
                            logger.warning(f"[THREAD {thread_id}] Error getting current event loop for initialization: {str(loop_error)}")
                            logger.debug(f"[THREAD {thread_id}] Creating new event loop for initialization")
                            current_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(current_loop)
                            logger.debug(f"[THREAD {thread_id}] Set new event loop for initialization: {current_loop}")
                        
                        # Create a separate task for initialization to better handle cancellation
                        logger.debug(f"[THREAD {thread_id}] Creating initialization task")
                        init_task = asyncio.create_task(exchange.initialize())
                        logger.debug(f"[THREAD {thread_id}] Initialization task created: {init_task}")
                        
                        # Wait for the task with timeout
                        logger.debug(f"[THREAD {thread_id}] Waiting for initialization task with timeout {init_timeout}s")
                        await asyncio.wait_for(init_task, timeout=init_timeout)
                        logger.info(f"[THREAD {thread_id}] Successfully initialized {exchange_id} exchange")
                        break
                    except asyncio.TimeoutError:
                        logger.warning(f"[THREAD {thread_id}] Initialization timed out after {init_timeout} seconds (attempt {attempt}/{max_retries})")
                        
                        # Cancel the task if it's still running
                        if not init_task.done():
                            logger.warning(f"[THREAD {thread_id}] Cancelling initialization task for {exchange_id}")
                            init_task.cancel()
                            try:
                                logger.debug(f"[THREAD {thread_id}] Waiting for cancelled task to complete")
                                await init_task
                                logger.debug(f"[THREAD {thread_id}] Cancelled task completed without exception")
                            except asyncio.CancelledError:
                                logger.info(f"[THREAD {thread_id}] Successfully cancelled initialization task for {exchange_id}")
                            except Exception as cancel_error:
                                logger.warning(f"[THREAD {thread_id}] Error while cancelling initialization task: {str(cancel_error)}")
                                logger.warning(f"[THREAD {thread_id}] Exception type: {type(cancel_error).__name__}")
                                logger.warning(f"[THREAD {thread_id}] Exception details: {repr(cancel_error)}")
                        else:
                            logger.debug(f"[THREAD {thread_id}] Task was already done when timeout occurred")
                        
                        # Raise a connection error to trigger retry
                        logger.debug(f"[THREAD {thread_id}] Raising ConnectionError to trigger retry")
                        raise ConnectionError(f"Initialization timed out after {init_timeout} seconds")
                    
                except ccxt.NetworkError as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(f"Network error initializing {exchange_id} exchange (attempt {attempt}/{max_retries}): {str(e)}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                except ccxt.ExchangeNotAvailable as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(f"Exchange not available error initializing {exchange_id} exchange (attempt {attempt}/{max_retries}): {str(e)}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                except ccxt.RequestTimeout as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(f"Request timeout initializing {exchange_id} exchange (attempt {attempt}/{max_retries}): {str(e)}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                except Exception as init_error:
                    last_error = init_error
                    if attempt < max_retries:
                        logger.warning(f"Failed to initialize {exchange_id} exchange (attempt {attempt}/{max_retries}): {str(init_error)}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
            
            # Check if all attempts failed
            if last_error is not None and attempt == max_retries:
                logger.error(f"Failed to initialize {exchange_id} exchange after {max_retries} attempts: {str(last_error)}")
                raise last_error
            
            # Cache the instance
            self._exchange_instances[exchange_id] = exchange
            
            return exchange
            
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication error creating exchange {exchange_id}: {str(e)}")
            logger.info("This may be due to invalid API credentials or insufficient permissions")
            logger.info("Check your API key and secret, and ensure they have the necessary permissions")
            
            # Add more detailed troubleshooting information
            logger.info("Troubleshooting steps for authentication issues:")
            logger.info("1. Verify that your API key and secret are correct")
            logger.info("2. Check if your API key has expired or been revoked")
            logger.info("3. Ensure your API key has the necessary permissions (read access at minimum)")
            logger.info("4. Verify that you're using the correct testnet/mainnet setting")
            logger.info("5. Check if there are IP restrictions on your API key")
            logger.info(f"6. Run 'python debug_bybit_auth.py' to test authentication specifically")
            
            # Add information about environment variables
            logger.info("Environment variables to check:")
            logger.info(f"- {exchange_id.upper()}_API_KEY or EXCHANGE_API_KEY")
            logger.info(f"- {exchange_id.upper()}_API_SECRET or EXCHANGE_API_SECRET")
            logger.info(f"- {exchange_id.upper()}_TESTNET or EXCHANGE_TESTNET")
            
            return None
            
        except ConnectionError as e:
            logger.error(f"Connection error creating exchange {exchange_id}: {str(e)}")
            logger.info("This may be due to network connectivity issues or API endpoint problems")
            logger.info("Check your internet connection and firewall settings")
            
            # Add more detailed troubleshooting information
            logger.info("Troubleshooting steps for connection issues:")
            logger.info("1. Verify that you can access the exchange API in your browser")
            logger.info("2. Check if your network has any firewall rules blocking the exchange API")
            logger.info("3. Try using a different network connection")
            logger.info("4. Check if the exchange API status page reports any outages")
            logger.info(f"5. Run 'python debug_bybit_api.py' to diagnose API connectivity issues")
            
            # Add information about the specific endpoint that failed
            if "query-info" in str(e):
                logger.info("The asset/coin/query-info endpoint is failing, which is used during initialization")
                logger.info("This endpoint may be temporarily unavailable or rate-limited")
            
            return None
            
        except Exception as e:
            error_msg = f"Error creating exchange {exchange_id}: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Exception type: {type(e).__name__}")
            logger.debug(f"Exception details: {repr(e)}")
            
            # Implement circuit breaker pattern
            self._circuit_breakers[exchange_id] = self._circuit_breakers.get(exchange_id, 0) + 1
            
            # If we've failed too many times, implement a circuit breaker
            max_failures = 5
            if self._circuit_breakers[exchange_id] >= max_failures:
                self._circuit_breaker_times[exchange_id] = time.time()
                logger.warning(f"Circuit breaker activated for {exchange_id} after {self._circuit_breakers[exchange_id]} failures")
                logger.warning(f"Will not attempt to initialize {exchange_id} for 10 minutes")
                
                # Add detailed troubleshooting information
                logger.info("Troubleshooting steps for persistent exchange errors:")
                logger.info("1. Check the exchange status page for any reported issues")
                logger.info("2. Verify all API credentials and permissions")
                logger.info("3. Check your network connectivity to the exchange API")
                logger.info("4. Run the diagnostic tools in the DEBUG_TOOLS.md file")
                logger.info("5. Consider creating new API credentials if issues persist")
            else:
                # Add general troubleshooting information
                logger.info("General troubleshooting steps:")
                logger.info("1. Check the logs for specific error details")
                logger.info("2. Verify your exchange configuration")
                logger.info("3. Run 'python debug_exchange_connection.py' for more diagnostics")
            
            return None