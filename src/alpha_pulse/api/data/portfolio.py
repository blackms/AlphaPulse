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
from alpha_pulse.data_pipeline.database.exchange_cache_fixed import ExchangeCacheRepository

# Import the new exchange_sync module
from alpha_pulse.exchange_sync.portfolio_service import PortfolioService


class PortfolioDataAccessor:
    """Access portfolio data."""
    
    def __init__(self):
        """Initialize portfolio accessor."""
        self._portfolio_manager = None
        self._exchange = None
        self._initialized = False
        self._exchange_type = None
        self._exchange_id = "bybit"  # Default exchange ID
        self._exchange_sync_service = None
    
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
    
    async def reload_data(self, data_type: Optional[str] = None) -> Dict:
        """
        Force a reload of exchange data.
        
        Args:
            data_type: Type of data to reload (optional)
            
        Returns:
            Status information
        """
        try:
            # Use the new exchange_sync module to trigger a sync
            from alpha_pulse.api.exchange_sync_integration import trigger_exchange_sync
            
            result = await trigger_exchange_sync(self._exchange_id)
            return result
        except Exception as e:
            logger.error(f"Error triggering data reload: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_portfolio(self, include_history: bool = False, refresh: bool = False) -> Dict:
        """
        Get current portfolio data.
        
        Args:
            include_history: Whether to include historical data
            refresh: Whether to force a refresh from the exchange
            
        Returns:
            Portfolio data
        """
        try:
            # If refresh is requested, get data directly from exchange
            if refresh:
                return await self._get_portfolio_from_exchange()
            
            # Try to initialize if not already done
            await self._initialize()
            
            # If initialization failed, use mock data
            if not self._initialized:
                logger.warning("Using mock data because initialization failed")
                return await self._get_mock_portfolio(include_history)
            
            # Get data from exchange
            data = await self._get_portfolio_from_exchange(include_history)
            return data
                
        except Exception as e:
            logger.error(f"Error retrieving portfolio data: {str(e)}")
            # Fall back to mock data if real data retrieval fails
            logger.warning("Falling back to mock data due to error")
            return await self._get_mock_portfolio(include_history)
    
    async def _get_portfolio_from_exchange(self, include_history: bool = False) -> Dict:
        """
        Get portfolio data directly from exchange using the new exchange_sync module.
        
        This method provides a more reliable way to get portfolio data by using
        the simplified exchange_sync module instead of the legacy connection manager.
        
        Args:
            include_history: Whether to include historical data
            
        Returns:
            Portfolio data dictionary
        """
        try:
            # Initialize the exchange_sync service if needed
            if self._exchange_sync_service is None:
                self._exchange_sync_service = PortfolioService(self._exchange_id)
                await self._exchange_sync_service.initialize()
            
            # Get portfolio items from the exchange
            portfolio_items = await self._exchange_sync_service.get_portfolio()
            
            if not portfolio_items:
                logger.warning(f"No portfolio items found for {self._exchange_id}")
                return {
                    "total_value": 0,
                    "cash": 0,
                    "positions": []
                }
            
            # Format the portfolio items
            positions = []
            total_value = 0
            
            for item in portfolio_items:
                position = {
                    "symbol": item.asset,
                    "quantity": item.quantity,
                    "entry_price": item.avg_entry_price or 0,
                    "current_price": item.current_price or 0,
                    "value": item.value or 0,
                    "pnl": item.profit_loss or 0,
                    "pnl_percentage": item.profit_loss_percentage or 0
                }
                positions.append(position)
                total_value += position["value"] or 0
            
            # Assume cash is 20% of total value for demo purposes
            cash = total_value * 0.2
            
            result = {
                "total_value": total_value + cash,
                "cash": cash,
                "positions": positions,
                "metrics": {
                    "sharpe_ratio": 1.8,  # Demo values
                    "sortino_ratio": 2.2,
                    "max_drawdown": 0.15,
                    "volatility": 0.25,
                    "return_since_inception": 0.35
                },
                "data_source": "exchange_sync",
                "fetched_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Add history if requested
            if include_history:
                # Generate demo history data for now
                # In a real implementation, we would fetch historical data from the exchange
                history = []
                now = datetime.now(timezone.utc)
                for i in range(30):
                    day = now.replace(day=now.day - i)
                    history.append({
                        "timestamp": day.isoformat(),
                        "total_value": total_value * (1 + i * 0.01),
                        "cash": cash * (1 - i * 0.01),
                        "positions_value": total_value * (1 + i * 0.015)
                    })
                result["history"] = history
            
            return result
        except Exception as e:
            logger.error(f"Error getting portfolio from exchange: {e}")
            return {"error": str(e), "total_value": 0, "cash": 0, "positions": []}
    
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