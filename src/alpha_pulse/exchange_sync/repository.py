"""
Database repository for exchange synchronization data.

This module provides a simple, reliable approach to database operations
using single connections per operation rather than complex connection pooling.
"""
import logging
import asyncpg
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any

from .models import PortfolioItem, OrderData, SyncResult
from .config import get_database_config


class DatabaseError(Exception):
    """Exception raised for database-related errors."""
    pass


class PortfolioRepository:
    """
    Repository for portfolio data storage and retrieval.
    
    This class handles all database operations related to portfolio data,
    using a simple connection-per-operation approach that is more reliable
    than complex connection pooling.
    """
    
    def __init__(self):
        """Initialize the repository with database configuration."""
        self.db_config = get_database_config()
        self.logger = logging.getLogger(__name__)
    
    async def get_connection(self):
        """
        Get a single database connection.
        
        This method establishes a direct connection to the database
        without using connection pooling, avoiding the complexity
        that led to issues in the previous implementation.
        
        Returns:
            An asyncpg connection
        """
        try:
            return await asyncpg.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise DatabaseError(f"Database connection error: {str(e)}")
    
    async def initialize_tables(self):
        """
        Initialize database tables if they don't exist.
        
        This method creates the necessary tables for storing portfolio data
        if they don't already exist in the database.
        """
        conn = None
        try:
            conn = await self.get_connection()
            
            # Create portfolio_items table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_items (
                    id SERIAL PRIMARY KEY,
                    exchange_id VARCHAR(50) NOT NULL,
                    asset VARCHAR(50) NOT NULL,
                    quantity NUMERIC NOT NULL,
                    current_price NUMERIC,
                    avg_entry_price NUMERIC,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    UNIQUE(exchange_id, asset)
                )
            """)
            
            # Create sync_history table to track sync operations
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_history (
                    id SERIAL PRIMARY KEY,
                    exchange_id VARCHAR(50) NOT NULL,
                    sync_type VARCHAR(50) NOT NULL,
                    items_processed INTEGER NOT NULL,
                    items_synced INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP NOT NULL
                )
            """)
            
            self.logger.info("Database tables initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing database tables: {str(e)}")
            raise DatabaseError(f"Failed to initialize tables: {str(e)}")
        finally:
            if conn:
                await conn.close()
    
    async def save_portfolio_item(self, exchange_id: str, item: PortfolioItem) -> bool:
        """
        Save a portfolio item to the database.
        
        Args:
            exchange_id: The exchange identifier
            item: The portfolio item to save
            
        Returns:
            True if the operation was successful
            
        Raises:
            DatabaseError: If the operation fails
        """
        conn = None
        try:
            conn = await self.get_connection()
            
            # Check if the item exists
            exists = await conn.fetchval("""
                SELECT 1 FROM portfolio_items 
                WHERE exchange_id = $1 AND asset = $2
            """, exchange_id, item.asset)
            
            now = datetime.now()
            
            if exists:
                # Update existing record
                await conn.execute("""
                    UPDATE portfolio_items
                    SET quantity = $3, 
                        current_price = $4,
                        avg_entry_price = $5,
                        updated_at = $6
                    WHERE exchange_id = $1 AND asset = $2
                """, 
                    exchange_id, 
                    item.asset, 
                    item.quantity, 
                    item.current_price, 
                    item.avg_entry_price,
                    now
                )
                self.logger.debug(f"Updated portfolio item: {exchange_id} - {item.asset}")
            else:
                # Insert new record
                await conn.execute("""
                    INSERT INTO portfolio_items
                    (exchange_id, asset, quantity, current_price, avg_entry_price, 
                     created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $6)
                """,
                    exchange_id, 
                    item.asset, 
                    item.quantity,
                    item.current_price, 
                    item.avg_entry_price,
                    now
                )
                self.logger.debug(f"Inserted new portfolio item: {exchange_id} - {item.asset}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio item {item.asset}: {str(e)}")
            raise DatabaseError(f"Failed to save portfolio item: {str(e)}")
        finally:
            if conn:
                await conn.close()
    
    async def get_portfolio_items(self, exchange_id: str) -> List[PortfolioItem]:
        """
        Get all portfolio items for an exchange.
        
        Args:
            exchange_id: The exchange identifier
            
        Returns:
            List of portfolio items
            
        Raises:
            DatabaseError: If the operation fails
        """
        conn = None
        try:
            conn = await self.get_connection()
            
            rows = await conn.fetch("""
                SELECT asset, quantity, current_price, avg_entry_price, updated_at
                FROM portfolio_items
                WHERE exchange_id = $1
            """, exchange_id)
            
            result = []
            for row in rows:
                item = PortfolioItem(
                    asset=row['asset'],
                    quantity=float(row['quantity']),
                    current_price=float(row['current_price']) if row['current_price'] is not None else None,
                    avg_entry_price=float(row['avg_entry_price']) if row['avg_entry_price'] is not None else None,
                    updated_at=row['updated_at']
                )
                result.append(item)
            
            self.logger.debug(f"Retrieved {len(result)} portfolio items for {exchange_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio items for {exchange_id}: {str(e)}")
            raise DatabaseError(f"Failed to retrieve portfolio items: {str(e)}")
        finally:
            if conn:
                await conn.close()
    
    async def save_sync_result(self, exchange_id: str, sync_type: str, result: SyncResult) -> bool:
        """
        Save a synchronization result to the database.
        
        Args:
            exchange_id: The exchange identifier
            sync_type: The type of synchronization (e.g., 'portfolio', 'orders')
            result: The synchronization result
            
        Returns:
            True if the operation was successful
            
        Raises:
            DatabaseError: If the operation fails
        """
        conn = None
        try:
            conn = await self.get_connection()
            
            now = datetime.now()
            error_message = None
            
            if result.has_errors:
                error_message = "; ".join(result.errors)
            
            await conn.execute("""
                INSERT INTO sync_history
                (exchange_id, sync_type, items_processed, items_synced, 
                 success, start_time, end_time, error_message, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                exchange_id,
                sync_type,
                result.items_processed,
                result.items_synced,
                result.success,
                result.start_time,
                result.end_time,
                error_message,
                now
            )
            
            self.logger.debug(f"Saved sync result for {exchange_id} - {sync_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving sync result for {exchange_id}: {str(e)}")
            raise DatabaseError(f"Failed to save sync result: {str(e)}")
        finally:
            if conn:
                await conn.close()