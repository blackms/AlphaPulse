"""
Portfolio synchronization service.

This module provides functionality to synchronize portfolio data from exchanges.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

from loguru import logger

from .models import PortfolioItem, SyncResult
from .repository import PortfolioRepository
from .exchange_client import ExchangeClient


class PortfolioService:
    """
    Service for synchronizing portfolio data from exchanges.
    
    This class handles the synchronization of portfolio data from exchanges,
    storing it in the database, and providing access to the data.
    """
    
    def __init__(self, exchange_id: str = None):
        """
        Initialize the portfolio service.
        
        Args:
            exchange_id: Optional exchange identifier to use as default
        """
        self.repository = PortfolioRepository()
        self.default_exchange_id = exchange_id
    
    async def sync_portfolio(self, exchange_ids: List[str]) -> Dict[str, SyncResult]:
        """
        Synchronize portfolio data from multiple exchanges.
        
        Args:
            exchange_ids: List of exchange identifiers to synchronize
            
        Returns:
            Dictionary mapping exchange IDs to sync results
        """
        logger.info(f"Starting portfolio sync for {len(exchange_ids)} exchanges: {', '.join(exchange_ids)}")
        
        results = {}
        for exchange_id in exchange_ids:
            try:
                result = await self.sync_exchange_portfolio(exchange_id)
                results[exchange_id] = result
            except Exception as e:
                logger.error(f"Error syncing portfolio for {exchange_id}: {str(e)}")
                # Create a failed result
                results[exchange_id] = SyncResult(
                    items_processed=0,
                    items_synced=0,
                    success=False,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    errors=[f"Failed to sync portfolio: {str(e)}"]
                )
        
        logger.info(f"Completed portfolio sync for all exchanges")
        return results
    
    async def sync_exchange_portfolio(self, exchange_id: str) -> SyncResult:
        """
        Synchronize portfolio data from a single exchange.
        
        Args:
            exchange_id: Exchange identifier
            
        Returns:
            Synchronization result
        """
        logger.info(f"Syncing portfolio for {exchange_id}...")
        
        start_time = datetime.now()
        client = ExchangeClient(exchange_id)
        errors = []
        
        try:
            # Connect to the exchange
            await client.connect()
            
            # Fetch portfolio data
            logger.info(f"Fetching portfolio data from {exchange_id}...")
            portfolio_items = await client.get_portfolio()
            
            logger.info(f"Retrieved {len(portfolio_items)} portfolio items from {exchange_id}")
            
            # Save each item to the database
            saved_count = 0
            for item in portfolio_items:
                try:
                    await self.repository.save_portfolio_item(exchange_id, item)
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving portfolio item {item.asset}: {str(e)}")
                    errors.append(f"Failed to save {item.asset}: {str(e)}")
            
            logger.info(f"Successfully saved {saved_count} of {len(portfolio_items)} items")
            
            # Create the result
            result = SyncResult(
                items_processed=len(portfolio_items),
                items_synced=saved_count,
                success=saved_count > 0 and len(errors) == 0,
                start_time=start_time,
                end_time=datetime.now(),
                errors=errors
            )
            
            # Save the sync result
            await self.repository.save_sync_result(exchange_id, "portfolio", result)
            
            logger.info(f"Successfully synced portfolio for {exchange_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error syncing portfolio for {exchange_id}: {str(e)}")
            
            # Create a failed result
            result = SyncResult(
                items_processed=0,
                items_synced=0,
                success=False,
                start_time=start_time,
                end_time=datetime.now(),
                errors=[f"Failed to sync portfolio: {str(e)}"]
            )
            
            # Try to save the sync result
            try:
                await self.repository.save_sync_result(exchange_id, "portfolio", result)
            except Exception as save_error:
                logger.error(f"Error saving sync result: {str(save_error)}")
            
            return result
        finally:
            # Always disconnect from the exchange
            try:
                await client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from {exchange_id}: {str(e)}")
    
    async def get_portfolio(self, exchange_id: str = None) -> List[PortfolioItem]:
        """
        Get portfolio data for an exchange.
        
        Args:
            exchange_id: Exchange identifier
            
        Returns:
            List of portfolio items
        """
        try:
            # Use the default exchange_id if none is provided
            if exchange_id is None:
                exchange_id = self.default_exchange_id
                
            items = await self.repository.get_portfolio_items(exchange_id)
            logger.debug(f"Retrieved {len(items)} portfolio items for {exchange_id} from database")
            return items
        except Exception as e:
            logger.error(f"Error retrieving portfolio for {exchange_id}: {str(e)}")
            raise