# Exchange Synchronization Refactoring Plan

## Problem Statement

The current exchange data synchronization system has multiple issues:

1. Complex threading and connection pool management causing synchronization problems
2. Scattered error handling making debugging difficult
3. High coupling between components
4. Files exceeding 200 lines of code
5. Reliability issues due to complex connection handling

## Proposed Solution

Create a simplified, dedicated module for exchange data synchronization that:

1. Runs as a separate process or background task every 30 minutes
2. Uses simpler database connection management
3. Has clear separation of concerns
4. Keeps files under 200 lines for maintainability
5. Follows SOLID principles while minimizing complexity

## Architecture Components

![Architecture Diagram](https://user-images.githubusercontent.com/placeholder/diagram.png)

### Components Overview

1. **Scheduler** (~100 lines)
   - Simple timer-based scheduler
   - Runs sync job every 30 minutes
   - Handles process lifecycle

2. **Exchange Client** (~150 lines)
   - Abstracts exchange API communication
   - Handles authentication and rate limiting
   - Retrieves portfolio and order data
   - Maps exchange-specific data to internal models

3. **Portfolio Service** (~150 lines)
   - Contains business logic for portfolio data
   - Calculates average entry prices when possible
   - Coordinates fetching and saving data
   - Handles error scenarios with clear logging

4. **Repository** (~150 lines)
   - Handles database operations
   - Uses simple connection management (one connection per operation)
   - CRUD operations for portfolio data
   - No complex connection pooling

5. **Models** (~100 lines)
   - Simple data classes for portfolio items
   - Mapping between exchange and internal formats
   - No complex inheritance hierarchies

6. **Config** (~50 lines)
   - Configuration loading and validation
   - Environment variable handling
   - Default configuration values

## Key Design Principles

### Simplicity
- Each file has a single responsibility
- Clear, linear flow of data
- Explicit error handling with informative logs
- No complex concurrency patterns

### Reliability
- Each database operation uses a fresh connection
- No connection pooling complexities
- Built-in retries for transient failures
- Clear metrics for monitoring

### Maintainability
- Files under 200 lines
- Consistent naming conventions
- Comprehensive error logging
- Clear data flow

## Implementation Approach

### Phase 1: Core Structure
1. Create basic package structure
2. Implement models and configuration
3. Set up basic scheduler framework

### Phase 2: Exchange Integration
1. Implement exchange client
2. Add portfolio data retrieval
3. Implement average price calculation

### Phase 3: Database Integration
1. Implement database repository
2. Add data persistence logic
3. Integrate with existing database schema

### Phase 4: Integration
1. Connect all components
2. Add error handling and logging
3. Implement metrics for monitoring

## Component Specifications

### Scheduler

```python
# Simple example of scheduler approach
import time
import logging
from datetime import datetime, timedelta

class ExchangeSyncScheduler:
    def __init__(self, interval_minutes=30):
        self.interval_minutes = interval_minutes
        self.portfolio_service = PortfolioService()
        self.running = False
        
    def start(self):
        """Start the scheduler loop"""
        self.running = True
        last_run = datetime.min
        
        while self.running:
            now = datetime.now()
            if now - last_run > timedelta(minutes=self.interval_minutes):
                logging.info("Running scheduled exchange sync")
                try:
                    self.portfolio_service.sync_portfolio_data()
                    last_run = now
                except Exception as e:
                    logging.error(f"Error during sync: {str(e)}")
            
            # Sleep for 1 minute before checking again
            time.sleep(60)
    
    def stop(self):
        """Stop the scheduler loop"""
        self.running = False
```

### Exchange Client

```python
# Simple example of exchange client approach
import ccxt
import logging
from typing import Dict, List, Optional
from .models import PortfolioItem, OrderData

class ExchangeClient:
    def __init__(self, exchange_id, api_key, api_secret):
        self.exchange_id = exchange_id
        self.exchange = None
        self.api_key = api_key
        self.api_secret = api_secret
    
    async def connect(self):
        """Connect to exchange API"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True
            })
            logging.info(f"Connected to {self.exchange_id}")
        except Exception as e:
            logging.error(f"Failed to connect to {self.exchange_id}: {str(e)}")
            raise
    
    async def get_portfolio(self) -> List[PortfolioItem]:
        """Get current portfolio balances"""
        if not self.exchange:
            await self.connect()
            
        try:
            balances = await self.exchange.fetch_balance()
            result = []
            
            for asset, data in balances['total'].items():
                if data <= 0:
                    continue
                    
                # Get current price for asset
                ticker = None
                try:
                    ticker = await self.exchange.fetch_ticker(f"{asset}/USDT")
                except Exception:
                    logging.warning(f"Could not fetch price for {asset}")
                
                # Create portfolio item
                portfolio_item = PortfolioItem(
                    asset=asset,
                    quantity=data,
                    current_price=ticker['last'] if ticker else None
                )
                
                # Try to get average entry price
                try:
                    portfolio_item.avg_entry_price = await self.get_average_entry_price(asset)
                except Exception:
                    logging.warning(f"Could not calculate average entry price for {asset}")
                
                result.append(portfolio_item)
                
            return result
        except Exception as e:
            logging.error(f"Error getting portfolio from {self.exchange_id}: {str(e)}")
            raise
    
    async def get_average_entry_price(self, asset: str) -> Optional[float]:
        """Calculate average entry price from order history if available"""
        try:
            # This is implementation-specific to each exchange
            # Just provide a simplified example
            symbol = f"{asset}/USDT"
            orders = await self.exchange.fetch_orders(symbol)
            
            if not orders:
                return None
                
            buy_orders = [o for o in orders if o['side'] == 'buy' and o['status'] == 'closed']
            if not buy_orders:
                return None
                
            total_quantity = sum(o['amount'] for o in buy_orders)
            total_cost = sum(o['amount'] * o['price'] for o in buy_orders)
            
            if total_quantity > 0:
                return total_cost / total_quantity
            return None
        except Exception as e:
            logging.warning(f"Error calculating average entry price for {asset}: {str(e)}")
            return None
```

### Portfolio Service

```python
# Simple example of portfolio service approach
import logging
from typing import List
from .models import PortfolioItem
from .exchange_client import ExchangeClient
from .repository import PortfolioRepository

class PortfolioService:
    def __init__(self, exchange_id="bybit", api_key=None, api_secret=None):
        self.exchange_client = ExchangeClient(exchange_id, api_key, api_secret)
        self.repository = PortfolioRepository()
    
    async def sync_portfolio_data(self):
        """Synchronize portfolio data from exchange to database"""
        logging.info(f"Starting portfolio sync for {self.exchange_client.exchange_id}")
        
        try:
            # Get portfolio data from exchange
            portfolio_items = await self.exchange_client.get_portfolio()
            
            if not portfolio_items:
                logging.warning("No portfolio items returned from exchange")
                return
                
            logging.info(f"Retrieved {len(portfolio_items)} portfolio items")
            
            # Store each portfolio item
            for item in portfolio_items:
                try:
                    await self.repository.save_portfolio_item(
                        self.exchange_client.exchange_id, 
                        item
                    )
                except Exception as e:
                    logging.error(f"Error storing portfolio item {item.asset}: {str(e)}")
            
            logging.info("Portfolio sync completed successfully")
            return True
        except Exception as e:
            logging.error(f"Portfolio sync failed: {str(e)}")
            raise
```

### Repository

```python
# Simple example of repository approach
import logging
import asyncpg
from datetime import datetime
from typing import List, Optional
from .models import PortfolioItem
from .config import get_database_config

class PortfolioRepository:
    def __init__(self):
        self.db_config = get_database_config()
    
    async def get_connection(self):
        """Get a database connection"""
        return await asyncpg.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            database=self.db_config['database']
        )
    
    async def save_portfolio_item(self, exchange_id: str, item: PortfolioItem):
        """Save a portfolio item to the database"""
        conn = None
        try:
            conn = await self.get_connection()
            
            # Check if the item exists
            exists = await conn.fetchval(
                """
                SELECT 1 FROM portfolio_items 
                WHERE exchange_id = $1 AND asset = $2
                """,
                exchange_id, item.asset
            )
            
            if exists:
                # Update existing record
                await conn.execute(
                    """
                    UPDATE portfolio_items
                    SET quantity = $3, 
                        current_price = $4,
                        avg_entry_price = $5,
                        updated_at = $6
                    WHERE exchange_id = $1 AND asset = $2
                    """,
                    exchange_id, item.asset, item.quantity, 
                    item.current_price, item.avg_entry_price,
                    datetime.now()
                )
            else:
                # Insert new record
                await conn.execute(
                    """
                    INSERT INTO portfolio_items
                    (exchange_id, asset, quantity, current_price, avg_entry_price, 
                     created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $6)
                    """,
                    exchange_id, item.asset, item.quantity,
                    item.current_price, item.avg_entry_price,
                    datetime.now()
                )
                
            logging.info(f"Saved portfolio item: {exchange_id} - {item.asset}")
        except Exception as e:
            logging.error(f"Error saving portfolio item: {str(e)}")
            raise
        finally:
            if conn:
                await conn.close()
    
    async def get_portfolio_items(self, exchange_id: str) -> List[PortfolioItem]:
        """Get all portfolio items for an exchange"""
        conn = None
        try:
            conn = await self.get_connection()
            
            rows = await conn.fetch(
                """
                SELECT asset, quantity, current_price, avg_entry_price, updated_at
                FROM portfolio_items
                WHERE exchange_id = $1
                """,
                exchange_id
            )
            
            result = []
            for row in rows:
                item = PortfolioItem(
                    asset=row['asset'],
                    quantity=row['quantity'],
                    current_price=row['current_price'],
                    avg_entry_price=row['avg_entry_price']
                )
                result.append(item)
                
            return result
        except Exception as e:
            logging.error(f"Error getting portfolio items: {str(e)}")
            raise
        finally:
            if conn:
                await conn.close()
```

### Models

```python
# Simple example of models approach
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class PortfolioItem:
    """Portfolio item data model"""
    asset: str
    quantity: float
    current_price: Optional[float] = None
    avg_entry_price: Optional[float] = None
    
    @property
    def value(self) -> Optional[float]:
        """Calculate current value of the portfolio item"""
        if self.current_price is None:
            return None
        return self.quantity * self.current_price
    
    @property
    def profit_loss(self) -> Optional[float]:
        """Calculate profit/loss amount"""
        if self.current_price is None or self.avg_entry_price is None:
            return None
        return self.quantity * (self.current_price - self.avg_entry_price)
    
    @property
    def profit_loss_percentage(self) -> Optional[float]:
        """Calculate profit/loss percentage"""
        if self.current_price is None or self.avg_entry_price is None or self.avg_entry_price == 0:
            return None
        return ((self.current_price / self.avg_entry_price) - 1) * 100

@dataclass
class OrderData:
    """Order data model"""
    order_id: str
    asset: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    status: str
    timestamp: datetime
```

### Config

```python
# Simple example of config approach
import os
import logging
from typing import Dict, Any

def get_database_config() -> Dict[str, Any]:
    """Get database configuration from environment variables"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASS', 'postgres'),
        'database': os.getenv('DB_NAME', 'alphapulse')
    }

def get_exchange_config(exchange_id: str) -> Dict[str, Any]:
    """Get exchange configuration from environment variables"""
    return {
        'api_key': os.getenv(f'{exchange_id.upper()}_API_KEY', ''),
        'api_secret': os.getenv(f'{exchange_id.upper()}_API_SECRET', ''),
        'testnet': os.getenv(f'{exchange_id.upper()}_TESTNET', 'false').lower() == 'true'
    }

def configure_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('exchange_sync.log')
        ]
    )
```

## Implementation Plan

1. Create the basic directory structure and empty files
2. Implement the models and configuration modules first
3. Implement the repository layer with simple database access
4. Implement the exchange client for API interaction
5. Implement the portfolio service for business logic
6. Implement the scheduler for running the sync process
7. Add comprehensive logging and error handling
8. Test the entire flow with a test exchange

This approach will yield a much simpler, more maintainable system that follows SOLID principles while avoiding the complexity that led to the current issues.