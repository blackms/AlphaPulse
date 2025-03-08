"""
Exchange data cache repository.

This module provides a repository for caching exchange data in a database.
It is used by the exchange data synchronizer to store fetched data and by
the portfolio API to retrieve cached data.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import json

from loguru import logger

from enum import Enum


class DataType(str, Enum):
    """Types of data that can be cached."""
    BALANCES = "balances"
    POSITIONS = "positions"
    ORDERS = "orders"
    PRICES = "prices"

class ExchangeCacheRepository:
    """
    Repository for caching exchange data.
    
    This class provides methods for storing and retrieving exchange data
    such as balances, positions, orders, and prices.
    """
    
    def __init__(self, connection):
        """
        Initialize the repository.
        
        Args:
            connection: Database connection
        """
        self.conn = connection
    
    async def get_all_sync_status(self) -> List[Dict[str, Any]]:
        """
        Get the sync status for all exchanges and data types.
        
        Returns:
            List of sync status records
        """
        query = """
            SELECT 
                exchange_id, 
                data_type, 
                status, 
                last_sync, 
                next_sync, 
                error_message,
                created_at,
                updated_at
            FROM sync_status
        """
        return await self.conn.fetch(query)
    
    async def get_sync_status(self, exchange_id: str, data_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the sync status for a specific exchange and data type.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data
            
        Returns:
            Sync status record or None if not found
        """
        query = """
            SELECT 
                exchange_id, 
                data_type, 
                status, 
                last_sync, 
                next_sync, 
                error_message,
                created_at,
                updated_at
            FROM sync_status
            WHERE exchange_id = $1 AND data_type = $2
        """
        record = await self.conn.fetchrow(query, exchange_id, data_type)
        if record:
            return dict(record)
        return None
    
    async def update_sync_status(
        self, 
        exchange_id: str, 
        data_type: str, 
        status: str, 
        next_sync: Optional[datetime] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update the sync status for a specific exchange and data type.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data
            status: Sync status
            next_sync: Next sync time (optional)
            error_message: Error message if sync failed (optional)
            
        Returns:
            Updated sync status record
        """
        now = datetime.now(timezone.utc)
        
        # Check if record exists
        query = """
            SELECT id FROM sync_status
            WHERE exchange_id = $1 AND data_type = $2
        """
        record = await self.conn.fetchrow(query, exchange_id, data_type)
        
        if record:
            # Update existing record
            query = """
                UPDATE sync_status
                SET 
                    status = $3,
                    last_sync = $4,
                    next_sync = $5,
                    error_message = $6,
                    updated_at = $7
                WHERE exchange_id = $1 AND data_type = $2
                RETURNING 
                    exchange_id, 
                    data_type, 
                    status, 
                    last_sync, 
                    next_sync, 
                    error_message,
                    created_at,
                    updated_at
            """
            record = await self.conn.fetchrow(
                query, 
                exchange_id, 
                data_type, 
                status, 
                now, 
                next_sync, 
                error_message,
                now
            )
        else:
            # Insert new record
            query = """
                INSERT INTO sync_status (
                    exchange_id, 
                    data_type, 
                    status, 
                    last_sync, 
                    next_sync, 
                    error_message,
                    created_at,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
                RETURNING 
                    exchange_id, 
                    data_type, 
                    status, 
                    last_sync, 
                    next_sync, 
                    error_message,
                    created_at,
                    updated_at
            """
            record = await self.conn.fetchrow(
                query, 
                exchange_id, 
                data_type, 
                status, 
                now, 
                next_sync, 
                error_message,
                now
            )
        
        return dict(record)
    
    async def store_balances(self, exchange_id: str, balances: Dict[str, Dict[str, Any]]) -> int:
        """
        Store balance data for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            balances: Dictionary of balances
            
        Returns:
            Number of balances stored
        """
        now = datetime.now(timezone.utc)
        count = 0
        
        for currency, balance in balances.items():
            # Extract values
            available = float(balance.get('available', 0))
            locked = float(balance.get('locked', 0))
            total = float(balance.get('total', 0))
            
            # Skip if all values are zero
            if available == 0 and locked == 0 and total == 0:
                continue
            
            # Check if record exists
            query = """
                SELECT id FROM exchange_balances
                WHERE exchange_id = $1 AND currency = $2
            """
            record = await self.conn.fetchrow(query, exchange_id, currency)
            
            if record:
                # Update existing record
                query = """
                    UPDATE exchange_balances
                    SET 
                        available = $3,
                        locked = $4,
                        total = $5,
                        updated_at = $6
                    WHERE exchange_id = $1 AND currency = $2
                """
                await self.conn.execute(
                    query, 
                    exchange_id, 
                    currency, 
                    available, 
                    locked, 
                    total,
                    now
                )
            else:
                # Insert new record
                query = """
                    INSERT INTO exchange_balances (
                        exchange_id, 
                        currency, 
                        available, 
                        locked, 
                        total,
                        created_at,
                        updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $6)
                """
                await self.conn.execute(
                    query, 
                    exchange_id, 
                    currency, 
                    available, 
                    locked, 
                    total,
                    now
                )
            
            count += 1
        
        logger.info(f"Stored {count} balances for {exchange_id}")
        return count
    
    async def get_balances(self, exchange_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get balance data for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            
        Returns:
            Dictionary of balances
        """
        query = """
            SELECT 
                currency, 
                available, 
                locked, 
                total,
                updated_at
            FROM exchange_balances
            WHERE exchange_id = $1
        """
        records = await self.conn.fetch(query, exchange_id)
        
        balances = {}
        for record in records:
            balances[record['currency']] = {
                'available': float(record['available']),
                'locked': float(record['locked']),
                'total': float(record['total']),
                'updated_at': record['updated_at']
            }
        
        return balances
    
    async def store_positions(self, exchange_id: str, positions: Dict[str, Dict[str, Any]]) -> int:
        """
        Store position data for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            positions: Dictionary of positions
            
        Returns:
            Number of positions stored
        """
        now = datetime.now(timezone.utc)
        count = 0
        
        for symbol, position in positions.items():
            # Extract values
            quantity = float(position.get('quantity', 0))
            entry_price = float(position.get('entry_price', 0)) if position.get('entry_price') is not None else None
            current_price = float(position.get('current_price', 0)) if position.get('current_price') is not None else None
            unrealized_pnl = float(position.get('unrealized_pnl', 0)) if position.get('unrealized_pnl') is not None else None
            
            # Check if record exists
            query = """
                SELECT id FROM exchange_positions
                WHERE exchange_id = $1 AND symbol = $2
            """
            record = await self.conn.fetchrow(query, exchange_id, symbol)
            
            if record:
                # Update existing record
                query = """
                    UPDATE exchange_positions
                    SET 
                        quantity = $3,
                        entry_price = $4,
                        current_price = $5,
                        unrealized_pnl = $6,
                        updated_at = $7
                    WHERE exchange_id = $1 AND symbol = $2
                """
                await self.conn.execute(
                    query, 
                    exchange_id, 
                    symbol, 
                    quantity, 
                    entry_price, 
                    current_price,
                    unrealized_pnl,
                    now
                )
            else:
                # Insert new record
                query = """
                    INSERT INTO exchange_positions (
                        exchange_id, 
                        symbol, 
                        quantity, 
                        entry_price, 
                        current_price,
                        unrealized_pnl,
                        created_at,
                        updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
                """
                await self.conn.execute(
                    query, 
                    exchange_id, 
                    symbol, 
                    quantity, 
                    entry_price, 
                    current_price,
                    unrealized_pnl,
                    now
                )
            
            count += 1
        
        logger.info(f"Stored {count} positions for {exchange_id}")
        return count
    
    async def get_positions(self, exchange_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get position data for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            
        Returns:
            Dictionary of positions
        """
        query = """
            SELECT 
                symbol, 
                quantity, 
                entry_price, 
                current_price,
                unrealized_pnl,
                updated_at
            FROM exchange_positions
            WHERE exchange_id = $1
        """
        records = await self.conn.fetch(query, exchange_id)
        
        positions = {}
        for record in records:
            positions[record['symbol']] = {
                'symbol': record['symbol'],
                'quantity': float(record['quantity']),
                'entry_price': float(record['entry_price']) if record['entry_price'] is not None else None,
                'current_price': float(record['current_price']) if record['current_price'] is not None else None,
                'unrealized_pnl': float(record['unrealized_pnl']) if record['unrealized_pnl'] is not None else None,
                'updated_at': record['updated_at']
            }
        
        return positions
    
    async def store_orders(self, exchange_id: str, orders: List[Dict[str, Any]]) -> int:
        """
        Store order data for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            orders: List of order dictionaries
            
        Returns:
            Number of orders stored
        """
        now = datetime.now(timezone.utc)
        count = 0
        
        for order in orders:
            # Extract values
            order_id = str(order.get('id', ''))
            symbol = order.get('symbol', '')
            order_type = order.get('type', '')
            side = order.get('side', '')
            price = float(order.get('price')) if order.get('price') is not None else None
            amount = float(order.get('amount')) if order.get('amount') is not None else None
            filled = float(order.get('filled')) if order.get('filled') is not None else None
            status = order.get('status', '')
            timestamp = order.get('timestamp', now)
            
            # Check if record exists
            query = """
                SELECT id FROM exchange_orders
                WHERE exchange_id = $1 AND order_id = $2
            """
            record = await self.conn.fetchrow(query, exchange_id, order_id)
            
            if record:
                # Update existing record
                query = """
                    UPDATE exchange_orders
                    SET 
                        symbol = $3,
                        order_type = $4,
                        side = $5,
                        price = $6,
                        amount = $7,
                        filled = $8,
                        status = $9,
                        timestamp = $10,
                        updated_at = $11
                    WHERE exchange_id = $1 AND order_id = $2
                """
                await self.conn.execute(
                    query, 
                    exchange_id, 
                    order_id,
                    symbol,
                    order_type,
                    side,
                    price,
                    amount,
                    filled,
                    status,
                    timestamp,
                    now
                )
            else:
                # Insert new record
                query = """
                    INSERT INTO exchange_orders (
                        exchange_id, 
                        order_id,
                        symbol,
                        order_type,
                        side,
                        price,
                        amount,
                        filled,
                        status,
                        timestamp,
                        created_at,
                        updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $11)
                """
                await self.conn.execute(
                    query, 
                    exchange_id, 
                    order_id,
                    symbol,
                    order_type,
                    side,
                    price,
                    amount,
                    filled,
                    status,
                    timestamp,
                    now
                )
            
            count += 1
        
        logger.info(f"Stored {count} orders for {exchange_id}")
        return count
    
    async def get_orders(self, exchange_id: str, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get order data for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            symbol: Optional symbol filter
            
        Returns:
            List of orders
        """
        if symbol:
            query = """
                SELECT 
                    order_id,
                    symbol,
                    order_type,
                    side,
                    price,
                    amount,
                    filled,
                    status,
                    timestamp,
                    updated_at
                FROM exchange_orders
                WHERE exchange_id = $1 AND symbol = $2
                ORDER BY timestamp DESC
            """
            records = await self.conn.fetch(query, exchange_id, symbol)
        else:
            query = """
                SELECT 
                    order_id,
                    symbol,
                    order_type,
                    side,
                    price,
                    amount,
                    filled,
                    status,
                    timestamp,
                    updated_at
                FROM exchange_orders
                WHERE exchange_id = $1
                ORDER BY timestamp DESC
            """
            records = await self.conn.fetch(query, exchange_id)
        
        orders = []
        for record in records:
            orders.append({
                'id': record['order_id'],
                'symbol': record['symbol'],
                'type': record['order_type'],
                'side': record['side'],
                'price': float(record['price']) if record['price'] is not None else None,
                'amount': float(record['amount']) if record['amount'] is not None else None,
                'filled': float(record['filled']) if record['filled'] is not None else None,
                'status': record['status'],
                'timestamp': record['timestamp'],
                'updated_at': record['updated_at']
            })
        
        return orders
    
    async def store_price(
        self, 
        exchange_id: str, 
        base_currency: str, 
        quote_currency: str, 
        price: float
    ) -> bool:
        """
        Store price data for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            base_currency: Base currency
            quote_currency: Quote currency
            price: Price
            
        Returns:
            True if stored, False otherwise
        """
        now = datetime.now(timezone.utc)
        
        # Check if record exists
        query = """
            SELECT id FROM exchange_prices
            WHERE exchange_id = $1 AND base_currency = $2 AND quote_currency = $3
        """
        record = await self.conn.fetchrow(query, exchange_id, base_currency, quote_currency)
        
        if record:
            # Update existing record
            query = """
                UPDATE exchange_prices
                SET 
                    price = $4,
                    timestamp = $5,
                    updated_at = $5
                WHERE exchange_id = $1 AND base_currency = $2 AND quote_currency = $3
            """
            await self.conn.execute(
                query, 
                exchange_id, 
                base_currency, 
                quote_currency, 
                price,
                now
            )
        else:
            # Insert new record
            query = """
                INSERT INTO exchange_prices (
                    exchange_id, 
                    base_currency, 
                    quote_currency, 
                    price,
                    timestamp,
                    created_at,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $5, $5)
            """
            await self.conn.execute(
                query, 
                exchange_id, 
                base_currency, 
                quote_currency, 
                price,
                now
            )
        
        logger.debug(f"Stored price for {base_currency}/{quote_currency} on {exchange_id}: {price}")
        return True
    
    async def get_price(
        self, 
        exchange_id: str, 
        base_currency: str, 
        quote_currency: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get price data for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            base_currency: Base currency
            quote_currency: Quote currency
            
        Returns:
            Price data or None if not found
        """
        query = """
            SELECT 
                price,
                timestamp,
                updated_at
            FROM exchange_prices
            WHERE exchange_id = $1 AND base_currency = $2 AND quote_currency = $3
        """
        record = await self.conn.fetchrow(query, exchange_id, base_currency, quote_currency)
        
        if record:
            return {
                'price': float(record['price']),
                'timestamp': record['timestamp'],
                'updated_at': record['updated_at']
            }
        
        return None
    
    async def get_all_prices(self, exchange_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all price data for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            
        Returns:
            Dictionary of prices
        """
        query = """
            SELECT 
                base_currency,
                quote_currency,
                price,
                timestamp,
                updated_at
            FROM exchange_prices
            WHERE exchange_id = $1
        """
        records = await self.conn.fetch(query, exchange_id)
        
        prices = {}
        for record in records:
            key = f"{record['base_currency']}/{record['quote_currency']}"
            prices[key] = {
                'price': float(record['price']),
                'timestamp': record['timestamp'],
                'updated_at': record['updated_at']
            }
        
        return prices