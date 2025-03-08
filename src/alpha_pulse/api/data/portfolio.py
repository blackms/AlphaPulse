"""Portfolio data access module."""
from typing import Dict, Optional, List
from datetime import datetime, timezone
from decimal import Decimal
import os
import yaml
from pathlib import Path
from loguru import logger

# Import portfolio manager and exchange
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.exchanges.factories import ExchangeFactory
from alpha_pulse.exchanges.types import ExchangeType


class PortfolioDataAccessor:
    """Access portfolio data."""
    
    def __init__(self):
        """Initialize portfolio accessor."""
        self._portfolio_manager = None
        self._exchange = None
        self._initialized = False
        
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
            exchange_type = os.environ.get('EXCHANGE_TYPE', 'bybit')
            logger.info(f"Creating exchange of type {exchange_type}")
            
            # Get API credentials from environment or config
            api_key = os.environ.get('BYBIT_API_KEY', os.environ.get('EXCHANGE_API_KEY', ''))
            api_secret = os.environ.get('BYBIT_API_SECRET', os.environ.get('EXCHANGE_API_SECRET', ''))
            testnet = os.environ.get('EXCHANGE_TESTNET', 'true').lower() == 'true'
            
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
                if exchange_type.lower() == 'binance':
                    logger.info(f"Creating Binance exchange with testnet={testnet}")
                    self._exchange = ExchangeFactory.create_exchange(
                        ExchangeType.BINANCE,
                        api_key=api_key,
                        api_secret=api_secret,
                        testnet=testnet
                    )
                elif exchange_type.lower() == 'bybit':
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
                    logger.warning(f"Unsupported exchange type: {exchange_type}, using Binance")
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
    
    async def get_portfolio(self, include_history: bool = False) -> Dict:
        """
        Get current portfolio data.
        
        Args:
            include_history: Whether to include historical data
            
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
                
            # Get portfolio data from exchange
            portfolio_data = await self._portfolio_manager.get_portfolio_data(self._exchange)
            
            # Format positions for API response
            positions = []
            for position in portfolio_data.positions:
                positions.append({
                    "symbol": position.asset_id,
                    "quantity": float(position.quantity),
                    "entry_price": float(position.entry_price),
                    "current_price": float(position.current_price),
                    "value": float(position.quantity * position.current_price),
                    "pnl": float((position.current_price - position.entry_price) * position.quantity),
                    "pnl_percentage": float((position.current_price / position.entry_price - 1) * 100) if position.entry_price else 0
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
                }
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
                        next((p.quantity for p in portfolio_data.positions if p.asset_id == asset), 0)
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
            
        except Exception as e:
            logger.error(f"Error retrieving portfolio data: {str(e)}")
            # Fall back to mock data if real data retrieval fails
            logger.warning("Falling back to mock data due to error")
            return await self._get_mock_portfolio(include_history)
            
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
            }
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