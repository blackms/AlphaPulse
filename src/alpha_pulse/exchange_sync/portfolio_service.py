"""
Portfolio service for exchange data synchronization.

This service coordinates the exchange client and repository to synchronize
portfolio data, implementing the core business logic of the synchronization process.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from .models import PortfolioItem, SyncResult
from .exchange_client import ExchangeClient, ExchangeError
from .repository import PortfolioRepository, DatabaseError
from .config import get_exchange_config


class PortfolioService:
    """
    Service for portfolio data synchronization.
    
    This class orchestrates the interaction between the exchange client and
    the database repository, implementing the business logic for portfolio
    data synchronization.
    """
    
    def __init__(self, exchange_id: str):
        """
        Initialize the portfolio service.
        
        Args:
            exchange_id: Identifier of the exchange (e.g., 'bybit', 'binance')
        """
        self.exchange_id = exchange_id
        self.logger = logging.getLogger(__name__)
        self.exchange_client = ExchangeClient(exchange_id)
        self.repository = PortfolioRepository()
    
    async def initialize(self) -> bool:
        """
        Initialize the service and ensure the database tables exist.
        
        Returns:
            True if initialization was successful
        """
        try:
            await self.repository.initialize_tables()
            return True
        except DatabaseError as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            return False
    
    async def sync_portfolio(self) -> SyncResult:
        """
        Synchronize portfolio data from exchange to database.
        
        This method fetches portfolio data from the exchange and saves it to
        the database, handling errors and retries appropriately.
        
        Returns:
            SyncResult with information about the synchronization outcome
        """
        result = SyncResult(
            success=True,
            start_time=datetime.now()
        )
        
        try:
            # Connect to the exchange
            await self.exchange_client.connect()
            
            # Get portfolio data from the exchange
            self.logger.info(f"Fetching portfolio data from {self.exchange_id}...")
            portfolio_items = await self.exchange_client.get_portfolio()
            
            result.items_processed = len(portfolio_items)
            
            if not portfolio_items:
                self.logger.warning(f"No portfolio items returned from {self.exchange_id}")
                result.end_time = datetime.now()
                await self.repository.save_sync_result(
                    self.exchange_id, "portfolio", result
                )
                return result
            
            self.logger.info(f"Retrieved {len(portfolio_items)} portfolio items from {self.exchange_id}")
            
            # Save each portfolio item to the database
            successful_items = 0
            for item in portfolio_items:
                try:
                    await self.repository.save_portfolio_item(
                        self.exchange_id, item
                    )
                    successful_items += 1
                except DatabaseError as e:
                    error_msg = f"Error storing portfolio item {item.asset}: {str(e)}"
                    self.logger.error(error_msg)
                    result.add_error(error_msg)
            
            result.items_synced = successful_items
            self.logger.info(f"Successfully saved {successful_items} of {len(portfolio_items)} items")
            
            # Save the sync result
            result.end_time = datetime.now()
            await self.repository.save_sync_result(
                self.exchange_id, "portfolio", result
            )
            
            return result
            
        except ExchangeError as e:
            error_msg = f"Exchange error during portfolio sync: {str(e)}"
            self.logger.error(error_msg)
            result.add_error(error_msg)
        except DatabaseError as e:
            error_msg = f"Database error during portfolio sync: {str(e)}"
            self.logger.error(error_msg)
            result.add_error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during portfolio sync: {str(e)}"
            self.logger.error(error_msg)
            result.add_error(error_msg)
        finally:
            # Ensure the exchange client is disconnected
            await self.exchange_client.disconnect()
            
            # Ensure we have an end time even if there was an error
            if result.end_time is None:
                result.end_time = datetime.now()
                
            # Try to save the sync result, but don't throw if it fails
            try:
                await self.repository.save_sync_result(
                    self.exchange_id, "portfolio", result
                )
            except Exception as e:
                self.logger.error(f"Error saving sync result: {str(e)}")
        
        return result
    
    async def get_portfolio(self) -> List[PortfolioItem]:
        """
        Get the current portfolio data from the database.
        
        Returns:
            List of portfolio items
        """
        try:
            items = await self.repository.get_portfolio_items(self.exchange_id)
            self.logger.info(f"Retrieved {len(items)} portfolio items from database")
            return items
        except DatabaseError as e:
            self.logger.error(f"Error retrieving portfolio items: {str(e)}")
            return []
    
    @classmethod
    async def sync_all_exchanges(cls) -> Dict[str, SyncResult]:
        """
        Synchronize portfolio data for all configured exchanges.
        
        This class method creates a PortfolioService for each configured exchange
        and synchronizes its portfolio data.
        
        Returns:
            Dictionary mapping exchange IDs to their sync results
        """
        from .config import get_sync_config
        
        logger = logging.getLogger(__name__)
        sync_config = get_sync_config()
        exchanges = sync_config['exchanges']
        
        logger.info(f"Starting portfolio sync for {len(exchanges)} exchanges: {', '.join(exchanges)}")
        
        results = {}
        
        for exchange_id in exchanges:
            try:
                service = cls(exchange_id)
                await service.initialize()
                
                logger.info(f"Syncing portfolio for {exchange_id}...")
                result = await service.sync_portfolio()
                
                results[exchange_id] = result
                
                if result.success:
                    logger.info(f"Successfully synced portfolio for {exchange_id}")
                else:
                    logger.warning(f"Portfolio sync for {exchange_id} completed with errors: {', '.join(result.errors)}")
                    
            except Exception as e:
                logger.error(f"Error during portfolio sync for {exchange_id}: {str(e)}")
                result = SyncResult(success=False)
                result.add_error(str(e))
                results[exchange_id] = result
        
        logger.info(f"Completed portfolio sync for all exchanges")
        return results