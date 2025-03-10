"""
Data synchronization module for exchange data.

This module handles the synchronization of different types of exchange data
such as balances, positions, orders, and prices.
"""
from typing import Dict, Any, Optional
from loguru import logger
import asyncpg

from alpha_pulse.exchanges.interfaces import BaseExchange
from alpha_pulse.data_pipeline.database.exchange_cache import ExchangeCacheRepository


class DataSynchronizer:
    """
    Handles the synchronization of different types of exchange data.
    
    This class is responsible for fetching data from exchanges and storing it in the database.
    """
    
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
            
            # Store in database
            try:
                await repo.store_balances(exchange_id, balance_dict)
                logger.info(f"Successfully synced balances for {exchange_id}")
                return True
            except asyncpg.InterfaceError as db_error:
                error_msg = str(db_error)
                if "connection has been released back to the pool" in error_msg:
                    logger.error(f"Database connection error in sync_balances for {exchange_id}: {error_msg}")
                    logger.error(f"This is likely due to a connection pool issue. The operation will be retried.")
                raise
            except Exception as e:
                logger.error(f"Error storing balances for {exchange_id}: {str(e)}")
                raise
            
        except Exception as e:
            raise
    
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
            
            # Store in database
            try:
                await repo.store_positions(exchange_id, positions_dict)
                logger.info(f"Successfully synced positions for {exchange_id}")
                return True
            except asyncpg.InterfaceError as db_error:
                error_msg = str(db_error)
                if "connection has been released back to the pool" in error_msg:
                    logger.error(f"Database connection error in sync_positions for {exchange_id}: {error_msg}")
                    logger.error(f"This is likely due to a connection pool issue. The operation will be retried.")
                raise
            except Exception as e:
                logger.error(f"Error storing positions for {exchange_id}: {str(e)}")
                raise
            
        except Exception as e:
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
            
            # Store in database
            try:
                await repo.store_orders(exchange_id, all_orders)
                logger.info(f"Successfully synced {len(all_orders)} orders for {exchange_id}")
                return True
            except asyncpg.InterfaceError as db_error:
                error_msg = str(db_error)
                if "connection has been released back to the pool" in error_msg:
                    logger.error(f"Database connection error in sync_orders for {exchange_id}: {error_msg}")
                    logger.error(f"This is likely due to a connection pool issue. The operation will be retried.")
                raise
            except Exception as e:
                logger.error(f"Error storing orders for {exchange_id}: {str(e)}")
                raise
            
        except Exception as e:
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
                            await repo.store_price(
                                exchange_id,
                                symbol.split('/')[0] if '/' in symbol else symbol,
                                quote_currency,
                                price
                            )
                            logger.info(f"Got price for {symbol}: {price}")
                        except asyncpg.InterfaceError as db_error:
                            error_msg = str(db_error)
                            if "connection has been released back to the pool" in error_msg:
                                logger.error(f"Database connection error storing price for {symbol}: {error_msg}")
                                logger.error(f"This is likely due to a connection pool issue. The operation will be retried.")
                            raise
                        except Exception as e:
                            logger.error(f"Error storing price for {symbol}: {str(e)}")
                    elif hasattr(ticker, 'last'):
                        # Handle object with 'last' attribute
                        price = float(ticker.last)
                        quote_currency = symbol.split('/')[-1] if '/' in symbol else "USDT"
                        await repo.store_price(
                            exchange_id,
                            symbol.split('/')[0] if '/' in symbol else symbol,
                            quote_currency,
                            price
                        )
                        logger.info(f"Got price for {symbol}: {price}")
                    else:
                        logger.warning(f"Unexpected ticker format for {symbol}: {ticker}")
                except Exception as e:
                    logger.error(f"Error getting price for {symbol}: {str(e)}")
            
            logger.info(f"Successfully synced prices for {exchange_id}")
            return True
        except Exception as e:
            raise