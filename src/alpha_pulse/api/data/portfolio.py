"""Portfolio data access module."""
from typing import Dict, Optional, List
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import os
import yaml
from pathlib import Path
from loguru import logger

# Import portfolio manager and exchange
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.exchanges.factories import ExchangeFactory
from alpha_pulse.exchanges.types import ExchangeType
from alpha_pulse.data_pipeline.database.connection import get_pg_connection
from alpha_pulse.data_pipeline.database.exchange_cache import ExchangeCacheRepository
from alpha_pulse.data_pipeline.scheduler import exchange_data_synchronizer, DataType


class PortfolioDataAccessor:
    """Access portfolio data."""
    
    def __init__(self):
        """Initialize portfolio accessor."""
        self._portfolio_manager = None
        self._exchange = None
        self._initialized = False
        self._exchange_type = None
        self._exchange_id = None
        
    async def _initialize(self):
        """Initialize portfolio manager and exchange."""
        if self._initialized:
            return
            
        try:
            # Load portfolio config
            config_path = os.environ.get(
                'PORTFOLIO_CONFIG_PATH',
                os.path.join(os.path.dirname(__file__), '../../../../config/portfolio_config.yaml')
            )
            
            # Initialize portfolio manager
            logger.info(f"Initializing portfolio manager with config from {config_path}")
            self._portfolio_manager = PortfolioManager(config_path)
            
            # Initialize exchange
            exchange_type_str = os.environ.get('EXCHANGE_TYPE', 'bybit')
            self._exchange_id = exchange_type_str.lower()
            logger.info(f"Creating exchange of type {exchange_type_str}")
            
            # Get API credentials from environment or config
            api_key = os.environ.get('BYBIT_API_KEY', os.environ.get('EXCHANGE_API_KEY', ''))
            api_secret = os.environ.get('BYBIT_API_SECRET', os.environ.get('EXCHANGE_API_SECRET', ''))
            
            # For Bybit, we need to be careful with the testnet setting
            # Our debug testing showed that the API key only works in mainnet mode
            if exchange_type_str.lower() == 'bybit':
                self._exchange_type = ExchangeType.BYBIT
                # Only use testnet if explicitly set in the environment
                # This matches the logic in the Bybit exchange implementation
                if 'BYBIT_TESTNET' in os.environ:
                    testnet = os.environ.get('BYBIT_TESTNET', '').lower() == 'true'
                    logger.info(f"Using Bybit-specific testnet setting from BYBIT_TESTNET: {testnet}")
                elif 'EXCHANGE_TESTNET' in os.environ:
                    testnet = os.environ.get('EXCHANGE_TESTNET', '').lower() == 'true'
                    logger.info(f"Using generic testnet setting from EXCHANGE_TESTNET: {testnet}")
                else:
                    # Default to mainnet (false) for Bybit as that's what works with our API key
                    testnet = False
                    logger.info("No testnet environment variable found for Bybit, defaulting to mainnet (testnet=False)")
            else:
                # For other exchanges, use the normal behavior
                self._exchange_type = ExchangeType.BINANCE if exchange_type_str.lower() == 'binance' else ExchangeType.BYBIT
                testnet = os.environ.get('EXCHANGE_TESTNET', 'true').lower() == 'true'
                logger.info(f"Using testnet setting for {exchange_type_str}: {testnet}")
            
            # Log API credentials status (without revealing actual keys)
            if api_key:
                logger.info(f"API Key found, length: {len(api_key)}")
            else:
                logger.warning("No API Key found in environment variables")
                
            if api_secret:
                logger.info(f"API Secret found, length: {len(api_secret)}")
            else:
                logger.warning("No API Secret found in environment variables")
            
            try:
                if self._exchange_type == ExchangeType.BINANCE:
                    logger.info(f"Creating Binance exchange with testnet={testnet}")
                    self._exchange = ExchangeFactory.create_exchange(
                        ExchangeType.BINANCE,
                        api_key=api_key,
                        api_secret=api_secret,
                        testnet=testnet
                    )
                elif self._exchange_type == ExchangeType.BYBIT:
                    logger.info(f"Creating Bybit exchange with testnet={testnet}")
                    
                    # Add debug logging to show actual keys being used
                    logger.debug(f"DEBUG - BYBIT API KEY BEING USED: {api_key}")
                    logger.debug(f"DEBUG - BYBIT API SECRET BEING USED: {api_secret}")
                    logger.debug(f"DEBUG - BYBIT TESTNET SETTING: {testnet}")
                    
                    self._exchange = ExchangeFactory.create_exchange(
                        ExchangeType.BYBIT,
                        api_key=api_key,
                        api_secret=api_secret,
                        testnet=testnet
                    )
                else:
                    logger.warning(f"Unsupported exchange type: {exchange_type_str}, using Binance")
                    self._exchange = ExchangeFactory.create_exchange(
                        ExchangeType.BINANCE,
                        api_key=api_key,
                        api_secret=api_secret,
                        testnet=testnet
                    )
                
                logger.info("Exchange instance created, initializing connection...")
                await self._exchange.initialize()
                logger.info("Exchange connection initialized successfully")
                self._initialized = True
                logger.info("Portfolio accessor initialized successfully")
            except Exception as e:
                logger.error(f"Error creating or initializing exchange: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error initializing portfolio accessor: {str(e)}")
            # Fall back to mock data if initialization fails
            self._initialized = False
            
    async def reload_data(self, data_type: Optional[DataType] = None) -> Dict:
        """
        Force a reload of exchange data.
        
        Args:
            data_type: Type of data to reload (optional)
            
        Returns:
            Status information
        """
        try:
            # Initialize if needed
            await self._initialize()
            
            if not self._initialized:
                return {"status": "error", "message": "Exchange not initialized"}
            
            # If no specific data type, reload all data
            if not data_type:
                data_type = DataType.ALL
                
            # Trigger the synchronizer to reload data
            if exchange_data_synchronizer:
                exchange_data_synchronizer.trigger_sync(self._exchange_id, data_type)
            
            return {
                "status": "success", 
                "message": f"Data reload triggered for {self._exchange_id}, type: {data_type}",
                "details": {
                    "exchange_id": self._exchange_id,
                    "data_type": data_type,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error triggering data reload: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_portfolio(self, include_history: bool = False, use_cache: bool = True) -> Dict:
        """
        Get current portfolio data.
        
        Args:
            include_history: Whether to include historical data
            use_cache: Whether to use cached data (if available)
            
        Returns:
            Portfolio data
        """
        try:
            # Try to initialize if not already done
            await self._initialize()
            
            # If initialization failed, use mock data
            if not self._initialized:
                logger.warning("Using mock data because initialization failed")
                return await self._get_mock_portfolio(include_history)
            
            # Always try to get data from cache first unless explicitly disabled
            if use_cache:
                logger.info("Attempting to use cached exchange data")
                cached_data = await self._get_portfolio_from_cache(include_history)
                if cached_data:
                    # Check if the cached data is fresh enough (last hour)
                    cache_timestamp = cached_data.get("cached_at", "")
                    if cache_timestamp:
                        try:
                            cached_time = datetime.fromisoformat(cache_timestamp)
                            now = datetime.now(timezone.utc)
                            # If cache is less than 1 hour old, use it
                            if (now - cached_time).total_seconds() < 3600:
                                logger.info("Using cached exchange data (less than 1 hour old)")
                                return cached_data
                            else:
                                logger.info(f"Cached data is {(now - cached_time).total_seconds() / 60:.1f} minutes old, fetching fresh data")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error parsing cache timestamp: {e}")
                    else:
                        # If no timestamp, still use cache
                        logger.info("Using cached exchange data (timestamp unknown)")
                        return cached_data
                else:
                    logger.info("No cached data available, fetching from exchange")
            else:
                logger.info("Cache usage disabled, fetching directly from exchange")
                
            # If cache is disabled, stale, or no cached data available,
            # use direct exchange API call and cache the result for future use
            data = await self._get_portfolio_from_exchange(include_history)
            
            # Trigger a background sync to refresh the cache for next time
            # This happens asynchronously so it won't slow down this request
            from alpha_pulse.data_pipeline.scheduler import exchange_data_synchronizer, DataType
            if exchange_data_synchronizer:
                exchange_data_synchronizer.trigger_sync(self._exchange_id, DataType.ALL)
                
            return data
                
        except Exception as e:
            logger.error(f"Error retrieving portfolio data: {str(e)}")
            # Fall back to mock data if real data retrieval fails
            logger.warning("Falling back to mock data due to error")
            return await self._get_mock_portfolio(include_history)
            
    async def _get_portfolio_from_cache(self, include_history: bool = False) -> Optional[Dict]:
        """Get portfolio data from cache."""
        try:
            async with get_pg_connection() as conn:
                repository = ExchangeCacheRepository(conn)
                
                # Get cached balances
                balances = await repository.get_balances(self._exchange_id)
                if not balances:
                    logger.warning("No cached balances found")
                    return None
                
                # Get cached positions
                positions_dict = await repository.get_positions(self._exchange_id)
                
                # Format positions for API response
                positions = []
                total_value = Decimal('0')
                
                for symbol, position_data in positions_dict.items():
                    quantity = Decimal(str(position_data['quantity']))
                    current_price = Decimal(str(position_data['current_price'])) if position_data['current_price'] else Decimal('0')
                    entry_price = Decimal(str(position_data['entry_price'])) if position_data['entry_price'] else current_price
                    
                    # Calculate values
                    value = quantity * current_price
                    total_value += value
                    
                    # Calculate PnL
                    unrealized_pnl = Decimal(str(position_data['unrealized_pnl'])) if position_data['unrealized_pnl'] else (value - (quantity * entry_price))
                    pnl_percentage = ((current_price / entry_price) - 1) * 100 if entry_price and entry_price > 0 else Decimal('0')
                    
                    positions.append({
                        "symbol": symbol,
                        "quantity": float(quantity),
                        "entry_price": float(entry_price),
                        "current_price": float(current_price),
                        "value": float(value),
                        "pnl": float(unrealized_pnl),
                        "pnl_percentage": float(pnl_percentage)
                    })
                
                # Get cash balance (USDT)
                cash = Decimal('0')
                for asset, balance in balances.items():
                    if asset == 'USDT':
                        cash = Decimal(str(balance.get('total', 0)))
                
                # Create portfolio response
                result = {
                    "total_value": float(total_value + cash),
                    "cash": float(cash),
                    "positions": positions,
                    "metrics": {
                        "sharpe_ratio": 0,  # Not available from cache
                        "sortino_ratio": 0,  # Not available from cache
                        "max_drawdown": 0,   # Not available from cache
                        "volatility": 0,     # Not available from cache
                        "return_since_inception": 0  # Not available from cache
                    },
                    "data_source": "cache",
                    "cached_at": datetime.now(timezone.utc).isoformat()
                }
                
                # TODO: Add history if requested
                if include_history:
                    # For now, we don't have historical data in the cache
                    # This would require additional tables and logic
                    pass
                
                return result
                
        except Exception as e:
            logger.error(f"Error retrieving portfolio data from cache: {str(e)}")
            return None
            
    async def _get_portfolio_from_exchange(self, include_history: bool = False) -> Dict:
        """Get portfolio data directly from exchange and cache it for future use."""
        # Get portfolio data from exchange
        portfolio_data = await self._portfolio_manager.get_portfolio_data(self._exchange)
        
        # Cache the data for future requests
        await self._cache_portfolio_data(portfolio_data)
        
        # Format positions for API response
        positions = []
        for position in portfolio_data.positions:
            # Handle both Position and PortfolioPosition types
            if hasattr(position, 'symbol'):
                # It's a Position object
                symbol = position.symbol
                quantity = float(position.quantity)
                entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price) if hasattr(position, 'current_price') else float(position.avg_entry_price)
                unrealized_pnl = float(position.unrealized_pnl) if hasattr(position, 'unrealized_pnl') else 0.0
                
                value = float(quantity * current_price)
                pnl = unrealized_pnl
                pnl_percentage = float((pnl / (entry_price * quantity)) * 100) if entry_price and quantity else 0
            elif hasattr(position, 'asset_id'):
                # It's a PortfolioPosition object
                symbol = position.asset_id
                quantity = float(position.quantity)
                entry_price = float(position.current_price) - (float(position.profit_loss) / quantity) if quantity else 0.0
                current_price = float(position.current_price)
                
                value = float(position.market_value)
                pnl = float(position.profit_loss)
                pnl_percentage = (pnl / (value - pnl)) * 100 if (value - pnl) else 0
            else:
                # Unknown position type, skip
                logger.warning(f"Unknown position type encountered: {type(position)}")
                continue
            
            positions.append({
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "value": value,
                "pnl": pnl,
                "pnl_percentage": pnl_percentage
            })
        
        # Create portfolio response
        result = {
            "total_value": float(portfolio_data.total_value),
            "cash": float(portfolio_data.cash_balance),
            "positions": positions,
            "metrics": {
                "sharpe_ratio": portfolio_data.risk_metrics.get('sharpe_ratio', 0),
                "sortino_ratio": portfolio_data.risk_metrics.get('sortino_ratio', 0),
                "max_drawdown": portfolio_data.risk_metrics.get('max_drawdown_limit', 0),
                "volatility": portfolio_data.risk_metrics.get('volatility_target', 0),
                "return_since_inception": 0  # Not available from portfolio data
            },
            "data_source": "exchange",
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Add history if requested
        if include_history:
            # Get historical data from exchange
            assets = [p.asset_id for p in portfolio_data.positions]
            historical_data = await self._portfolio_manager._fetch_historical_data(
                self._exchange, assets
            )
            
            # Format historical data for API response
            history = []
            for timestamp in historical_data.index:
                # Calculate total value for each day
                day_value = sum(
                    historical_data.loc[timestamp, asset] *
                    next((p.quantity for p in portfolio_data.positions if p.symbol == asset), 0)
                    for asset in assets if asset in historical_data.columns
                )
                
                history.append({
                    "timestamp": timestamp.isoformat(),
                    "total_value": float(day_value + portfolio_data.cash_balance),
                    "cash": float(portfolio_data.cash_balance),
                    "positions_value": float(day_value)
                })
                
            result["history"] = history
        
        return result
        
    async def _cache_portfolio_data(self, portfolio_data):
        """
        Cache portfolio data for future use.
        
        Args:
            portfolio_data: Portfolio data from the exchange
        """
        try:
            async with get_pg_connection() as conn:
                repository = ExchangeCacheRepository(conn)
                
                # Cache balances
                balances = {}
                for balance in portfolio_data.balances:
                    if hasattr(balance, 'asset') and hasattr(balance, 'free') and hasattr(balance, 'locked'):
                        balances[balance.asset] = {
                            'available': float(balance.free),
                            'locked': float(balance.locked),
                            'total': float(balance.free) + float(balance.locked)
                        }
                    elif hasattr(balance, 'currency') and hasattr(balance, 'available') and hasattr(balance, 'locked'):
                        balances[balance.currency] = {
                            'available': float(balance.available),
                            'locked': float(balance.locked),
                            'total': float(balance.available) + float(balance.locked)
                        }
                
                if balances:
                    await repository.store_balances(self._exchange_id, balances)
                    logger.info(f"Cached {len(balances)} balances for {self._exchange_id}")
                
                # Cache positions
                positions = {}
                for position in portfolio_data.positions:
                    if hasattr(position, 'symbol'):
                        # Handle Position type
                        symbol = position.symbol
                        quantity = float(position.quantity)
                        entry_price = float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') else None
                        current_price = float(position.current_price) if hasattr(position, 'current_price') else None
                        unrealized_pnl = float(position.unrealized_pnl) if hasattr(position, 'unrealized_pnl') else None
                        
                        positions[symbol] = {
                            'quantity': quantity,
                            'entry_price': entry_price,
                            'current_price': current_price,
                            'unrealized_pnl': unrealized_pnl
                        }
                    elif hasattr(position, 'asset_id'):
                        # Handle PortfolioPosition type
                        symbol = position.asset_id
                        quantity = float(position.quantity)
                        current_price = float(position.current_price)
                        entry_price = current_price - (float(position.profit_loss) / quantity) if quantity else 0.0
                        
                        positions[symbol] = {
                            'quantity': quantity,
                            'entry_price': entry_price,
                            'current_price': current_price,
                            'unrealized_pnl': float(position.profit_loss)
                        }
                
                if positions:
                    await repository.store_positions(self._exchange_id, positions)
                    logger.info(f"Cached {len(positions)} positions for {self._exchange_id}")
                
                # Cache orders
                if hasattr(portfolio_data, 'orders') and portfolio_data.orders:
                    await repository.store_orders(self._exchange_id, portfolio_data.orders)
                    logger.info(f"Cached {len(portfolio_data.orders)} orders for {self._exchange_id}")
                
                # Update the sync status
                now = datetime.now(timezone.utc)
                next_sync = now + timedelta(hours=1)  # Schedule next sync in 1 hour
                
                for data_type in [dt.value for dt in DataType]:
                    if data_type != DataType.ALL.value:
                        await repository.update_sync_status(
                            self._exchange_id,
                            data_type,
                            "completed",
                            next_sync
                        )
                
                logger.info(f"Updated sync status for {self._exchange_id}")
                
        except Exception as e:
            logger.error(f"Error caching portfolio data: {str(e)}")
            
    async def _get_mock_portfolio(self, include_history: bool = False) -> Dict:
        """Get mock portfolio data for demo purposes."""
        logger.info("Using mock portfolio data")
        
        positions = [
            {
                "symbol": "BTC-USD",
                "quantity": 1.5,
                "entry_price": 45000.0,
                "current_price": 47000.0,
                "value": 70500.0,
                "pnl": 3000.0,
                "pnl_percentage": 6.67
            },
            {
                "symbol": "ETH-USD",
                "quantity": 10.0,
                "entry_price": 2500.0,
                "current_price": 2800.0,
                "value": 28000.0,
                "pnl": 3000.0,
                "pnl_percentage": 12.0
            },
            {
                "symbol": "SOL-USD",
                "quantity": 100.0,
                "entry_price": 100.0,
                "current_price": 120.0,
                "value": 12000.0,
                "pnl": 2000.0,
                "pnl_percentage": 20.0
            }
        ]
        
        # Calculate total value
        total_value = sum(p["value"] for p in positions)
        cash = 50000.0  # Mock cash balance
        
        # Create portfolio response
        result = {
            "total_value": total_value + cash,
            "cash": cash,
            "positions": positions,
            "metrics": {
                "sharpe_ratio": 1.8,
                "sortino_ratio": 2.2,
                "max_drawdown": 0.15,
                "volatility": 0.25,
                "return_since_inception": 0.35
            },
            "data_source": "mock"
        }
        
        # Add history if requested
        if include_history:
            # Generate demo history data
            history = []
            now = datetime.now(timezone.utc)
            for i in range(30):
                day = now.replace(day=now.day - i)
                history.append({
                    "timestamp": day.isoformat(),
                    "total_value": 1000000.0 + i * 10000.0,
                    "cash": 200000.0 - i * 5000.0,
                    "positions_value": 800000.0 + i * 15000.0
                })
            result["history"] = history
        
        return result