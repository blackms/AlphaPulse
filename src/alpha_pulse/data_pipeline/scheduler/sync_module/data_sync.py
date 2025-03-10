"""
Data synchronization module for exchange data.

This module handles the synchronization of different types of exchange data
such as balances, positions, orders, and prices.
"""
from typing import Dict, Any, Optional, List
from loguru import logger
import asyncpg
import asyncio
from functools import wraps

from alpha_pulse.exchanges.interfaces import BaseExchange
from alpha_pulse.data_pipeline.database.exchange_cache_fixed import ExchangeCacheRepository


class DataSynchronizer:
    """
    Handles the synchronization of different types of exchange data.
    
    This class is responsible for fetching data from exchanges and storing it in the database.
    """
    
    def db_operation_with_retry(max_retries=3, retry_delay=1.0):
        """
        Decorator for database operations that need retry logic.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(1, max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except asyncpg.InterfaceError as e:
                        error_msg = str(e)
                        if "connection has been released back to the pool" in error_msg or "connection is closed" in error_msg:
                            logger.error(f"Database connection error in {func.__name__} (attempt {attempt}/{max_retries}): {error_msg}")
                            if attempt < max_retries:
                                logger.info(f"Retrying operation in {retry_delay} seconds...")
                                await asyncio.sleep(retry_delay)
                                last_exception = e
                            else:
                                logger.error(f"Max retries ({max_retries}) reached for {func.__name__}")
                                raise
                        else:
                            raise
                    except Exception as e:
                        raise
                raise last_exception  # This should never be reached, but just in case
            return wrapper
        return decorator
    
    async def sync_balances(self, exchange_id: str, exchange: BaseExchange, repo: ExchangeCacheRepository) -> bool:
        """
        Synchronize balance data.
        
        Args:
            exchange_id: Exchange identifier
            exchange: Exchange instance
            repo: Repository for storing data
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Syncing balances for {exchange_id}")
        
        @self.db_operation_with_retry(max_retries=3, retry_delay=1.0)
        async def _store_balances(exchange_id, balance_dict, repo):
            try:
                await repo.store_balances(exchange_id, balance_dict)
                logger.info(f"Successfully synced balances for {exchange_id}")
                return True
            except asyncpg.InterfaceError as db_error:
                # Let the decorator handle this
                raise
            except Exception as e:
                logger.error(f"Error storing balances for {exchange_id}: {str(e)}")
                raise
        
        try:
            # Get balances from exchange
            balances = await exchange.get_balances()
            
            # Convert Balance objects to dictionaries
            balance_dict = {}
            for currency, balance in balances.items():
                balance_dict[currency] = {
                    'available': float(balance.available),
                    'locked': float(balance.locked),
                    'total': float(balance.total)
                }
            
            # Store in database with retry logic
            return await _store_balances(exchange_id, balance_dict, repo)
            
        except Exception as e:
            logger.error(f"Error in sync_balances for {exchange_id}: {str(e)}")
            raise
            return False
    
    async def sync_positions(self, exchange_id: str, exchange: BaseExchange, repo: ExchangeCacheRepository) -> bool:
        """
        Synchronize position data.
        
        Args:
            exchange_id: Exchange identifier
            exchange: Exchange instance
            repo: Repository for storing data
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Syncing positions for {exchange_id}")
        
        @self.db_operation_with_retry(max_retries=3, retry_delay=1.0)
        async def _store_positions(exchange_id, positions_dict, repo):
            try:
                await repo.store_positions(exchange_id, positions_dict)
                logger.info(f"Successfully synced positions for {exchange_id}")
                return True
            except asyncpg.InterfaceError as db_error:
                # Let the decorator handle this
                raise
            except Exception as e:
                logger.error(f"Error storing positions for {exchange_id}: {str(e)}")
                raise
        
        try:
            # Get positions from exchange
            positions = await exchange.get_positions()
            
            # Convert positions to dictionary format expected by repository
            positions_dict = {}
            
            # Handle different return types
            if isinstance(positions, dict):
                # If positions is already a dictionary
                positions_dict = positions
            else:
                # If positions is a list of objects
                for position in positions:
                    # Skip if position is a string
                    if isinstance(position, str):
                        logger.warning(f"Unexpected position format (string): {position}")
                        continue
                        
                    # Skip positions with zero quantity
                    try:
                        if hasattr(position, 'quantity') and position.quantity == 0:
                            continue
                            
                        if hasattr(position, 'symbol'):
                            symbol = position.symbol
                            positions_dict[symbol] = {
                                "symbol": symbol,
                                "quantity": float(position.quantity) if hasattr(position, 'quantity') else 0,
                                "entry_price": float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') and position.avg_entry_price else None,
                                "current_price": float(position.current_price) if hasattr(position, 'current_price') else None,
                                "unrealized_pnl": float(position.unrealized_pnl) if hasattr(position, 'unrealized_pnl') else None,
                            }
                    except Exception as pos_error:
                        logger.warning(f"Error processing position: {str(pos_error)}")
                        continue
            
            # Store in database with retry logic
            return await _store_positions(exchange_id, positions_dict, repo)
            
        except Exception as e:
            logger.error(f"Error in sync_positions for {exchange_id}: {str(e)}")
            raise
    
    async def sync_orders(self, exchange_id: str, exchange: BaseExchange, repo: ExchangeCacheRepository) -> bool:
        """
        Synchronize order data.
        
        Args:
            exchange_id: Exchange identifier
            exchange: Exchange instance
            repo: Repository for storing data
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Syncing orders for {exchange_id}")
        
        @self.db_operation_with_retry(max_retries=3, retry_delay=1.0)
        async def _store_orders(exchange_id, all_orders, repo):
            try:
                await repo.store_orders(exchange_id, all_orders)
                logger.info(f"Successfully synced {len(all_orders)} orders for {exchange_id}")
                return True
            except asyncpg.InterfaceError as db_error:
                # Let the decorator handle this
                raise
            except Exception as e:
                logger.error(f"Error storing orders for {exchange_id}: {str(e)}")
                raise
        
        try:
            # Get all symbols with positions
            positions = await exchange.get_positions()
            symbols = set()
            
            # Handle different return types for positions
            if isinstance(positions, dict):
                symbols = set(positions.keys())
            else:
                for position in positions:
                    if isinstance(position, str):
                        continue
                    if hasattr(position, 'symbol') and hasattr(position, 'quantity'):
                        if position.quantity != 0:
                            symbols.add(position.symbol)
            
            # Add some default symbols
            symbols.add("BTC/USDT")
            symbols.add("ETH/USDT")
            
            # Get orders for each symbol
            all_orders = []
            for symbol in symbols:
                try:
                    # Get orders for this symbol
                    symbol_orders = await exchange.get_orders(symbol)
                    
                    # Handle different return types
                    if isinstance(symbol_orders, list):
                        all_orders.extend(symbol_orders)
                        logger.info(f"Got {len(symbol_orders)} orders for {symbol}")
                    elif isinstance(symbol_orders, dict):
                        all_orders.append(symbol_orders)
                        logger.info(f"Got 1 order for {symbol}")
                    else:
                        logger.warning(f"Unexpected order format for {symbol}: {type(symbol_orders)}")
                except Exception as e:
                    logger.error(f"Error getting orders for {symbol}: {str(e)}")
            
            # Store in database with retry logic
            return await _store_orders(exchange_id, all_orders, repo)
            
        except Exception as e:
            logger.error(f"Error in sync_orders for {exchange_id}: {str(e)}")
            raise
    
    async def sync_prices(self, exchange_id: str, exchange: BaseExchange, repo: ExchangeCacheRepository) -> bool:
        """
        Synchronize price data.
        
        Args:
            exchange_id: Exchange identifier
            exchange: Exchange instance
            repo: Repository for storing data
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Syncing prices for {exchange_id}")
        
        @self.db_operation_with_retry(max_retries=3, retry_delay=1.0)
        async def _store_price(exchange_id, base_currency, quote_currency, price, repo):
            try:
                await repo.store_price(
                    exchange_id,
                    base_currency,
                    quote_currency,
                    price
                )
                return True
            except asyncpg.InterfaceError as db_error:
                # Let the decorator handle this
                raise
            except Exception as e:
                raise
        
        try:
            # Get all symbols with positions
            positions = await exchange.get_positions()
            symbols = set()
            
            # Handle different return types for positions
            if isinstance(positions, dict):
                symbols = set(positions.keys())
            else:
                for position in positions:
                    if isinstance(position, str):
                        continue
                    if hasattr(position, 'symbol') and hasattr(position, 'quantity'):
                        if position.quantity != 0:
                            symbols.add(position.symbol)
            
            # Add some default symbols
            symbols.add("BTC/USDT")
            symbols.add("ETH/USDT")
            
            success_count = 0
            # Get prices for each symbol
            for symbol in symbols:
                try:
                    # Get price for this symbol
                    ticker = await exchange.get_ticker(symbol)
                    
                    # Handle different return types
                    if isinstance(ticker, dict) and 'last' in ticker:
                        price = float(ticker['last'])
                        quote_currency = symbol.split('/')[-1] if '/' in symbol else "USDT"
                        try:
                            result = await _store_price(
                                exchange_id,
                                symbol.split('/')[0] if '/' in symbol else symbol,
                                quote_currency,
                                price,
                                repo
                            )
                            if result:
                                success_count += 1
                                logger.info(f"Got price for {symbol}: {price}")
                        except Exception as e:
                            logger.error(f"Error storing price for {symbol}: {str(e)}")
                    elif hasattr(ticker, 'last') and ticker.last is not None:
                        logger.info(f"Got price for {symbol}: {price}")
                    else:
                        logger.warning(f"Unexpected ticker format for {symbol}: {ticker}")
                except Exception as e:
                    logger.error(f"Error getting price for {symbol}: {str(e)}")
            
            logger.info(f"Successfully synced prices for {exchange_id}")
            return success_count > 0
        except Exception as e:
            raise