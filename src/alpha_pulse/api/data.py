"""
Data access classes for the API.

These classes provide access to the underlying data sources.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio

from ..data_pipeline.database import (
    connection_manager,
    UserRepository,
    PortfolioRepository,
    PositionRepository,
    TradeRepository,
    AlertRepository,
    MetricRepository,
    Metric
)

# Import the new exchange_sync module
from ..exchange_sync.portfolio_service import PortfolioService as ExchangeSyncPortfolioService

logger = logging.getLogger(__name__)


class MetricsDataAccessor:
    """Accessor for metric data."""
    
    def __init__(self):
        """Initialize the accessor."""
        self.metric_repo = MetricRepository()
    
    async def get_metrics(
        self,
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation: str = 'avg'
    ) -> List[Dict[str, Any]]:
        """Get metrics data."""
        try:
            # Set default time range if not provided
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(days=1)
            
            
            # Otherwise, return raw data
            return await self.metric_repo.find_by_name(
                metric_name=metric_type,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return []
    
    async def get_latest_metrics(self, metric_type: str) -> Dict[str, Any]:
        """Get the latest metric value."""
        try:
            return await self.metric_repo.find_latest_by_name(metric_type)
        except Exception as e:
            logger.error(f"Error getting latest metric: {e}")
            return None


class AlertDataAccessor:
    """Accessor for alert data."""
    
    def __init__(self, alert_manager=None):
        """Initialize the accessor.
        
        Args:
            alert_manager: Optional AlertManager instance for direct alert access
        """
        self.alert_repo = AlertRepository()
        self.alert_manager = alert_manager
    
    async def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Get alerts data."""
        try:
            # Build criteria from filters
            criteria = {}
            if filters:
                for key, value in filters.items():
                    criteria[key] = value
            
            # If unacknowledged filter is set
            if 'acknowledged' in criteria and criteria['acknowledged'] is False:
                return await self.alert_repo.find_unacknowledged()
            
            # If severity filter is set
            if 'severity' in criteria:
                return await self.alert_repo.find_by_severity(criteria['severity'])
            
            # Otherwise, use general criteria
            return await self.alert_repo.find_by_criteria(criteria)
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: int, user: str) -> Dict[str, Any]:
        """Acknowledge an alert."""
        try:
            result = await self.alert_repo.acknowledge(alert_id, user)
            if result:
                return {
                    "success": True,
                    "alert": result
                }
            return {
                "success": False,
                "error": "Alert not found or already acknowledged"
            }
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class PortfolioDataAccessor:
    """Accessor for portfolio data."""
    
    def __init__(self):
        """Initialize the accessor."""
        self.portfolio_repo = PortfolioRepository()
        self.position_repo = PositionRepository()
        self._exchange_id = "bybit"  # Default exchange ID
        self._exchange_sync_service = None
    
    async def get_portfolio(self, include_history: bool = False) -> Dict[str, Any]:
        """Get portfolio data."""
        try:
            # Get all portfolios with positions
            portfolios = await self.portfolio_repo.find_all_with_positions()
            
            if not portfolios:
                return {
                    "total_value": 0,
                    "cash": 0,
                    "positions": [],
                    "metrics": {}
                }
            
            # Use the first portfolio (main portfolio)
            portfolio = portfolios[0]
            positions = portfolio.get('positions', [])
            
            # Calculate total value
            positions_value = sum(
                float(p['quantity']) * float(p['current_price'])
                for p in positions
                if p.get('current_price') is not None
            )
            
            # Assume cash is 20% of total value for demo purposes
            cash = positions_value * 0.2
            total_value = positions_value + cash
            
            # Format positions with additional data
            formatted_positions = []
            for position in positions:
                quantity = float(position['quantity'])
                entry_price = float(position['entry_price'])
                current_price = float(position['current_price']) if position.get('current_price') else None
                
                if current_price:
                    value = quantity * current_price
                    entry_value = quantity * entry_price
                    pnl = value - entry_value
                    pnl_percentage = (pnl / entry_value) * 100 if entry_value else 0
                else:
                    value = None
                    pnl = None
                    pnl_percentage = None
                
                formatted_positions.append({
                    "symbol": position['symbol'],
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "value": value,
                    "pnl": pnl,
                    "pnl_percentage": pnl_percentage
                })
            
            # Create portfolio response
            result = {
                "total_value": total_value,
                "cash": cash,
                "positions": formatted_positions,
                "metrics": {
                    "sharpe_ratio": 1.8,  # Demo values
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
                now = datetime.now()
                for i in range(30):
                    timestamp = now - timedelta(days=i)
                    history.append({
                        "timestamp": timestamp.isoformat(),
                        "total_value": 1000000.0 + i * 10000.0,
                        "cash": 200000.0 - i * 5000.0,
                        "positions_value": 800000.0 + i * 15000.0
                    })
                result["history"] = history
            
            return result
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return {
                "error": str(e),
                "total_value": 0,
                "cash": 0,
                "positions": []
            }
    
    async def _get_portfolio_from_exchange(self) -> Dict[str, Any]:
        """
        Get portfolio data directly from the exchange using the new exchange_sync module.
        
        This method provides a more reliable way to get portfolio data by using
        the simplified exchange_sync module instead of the legacy connection manager.
        
        Returns:
            Portfolio data dictionary
        """
        try:
            # Initialize the exchange_sync service if needed
            if self._exchange_sync_service is None:
                self._exchange_sync_service = ExchangeSyncPortfolioService(self._exchange_id)
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
            
            return {"total_value": total_value, "positions": positions, "cash": total_value * 0.2}
        except Exception as e:
            logger.error(f"Error getting portfolio from exchange: {e}")
            return {"error": str(e), "total_value": 0, "cash": 0, "positions": []}


class TradeDataAccessor:
    """Accessor for trade data."""
    
    def __init__(self):
        """Initialize the accessor."""
        self.trade_repo = TradeRepository()
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get trades data."""
        try:
            # If symbol is provided, filter by symbol
            if symbol:
                return await self.trade_repo.find_by_symbol(symbol)
            
            # If time range is provided, filter by time
            if start_time and end_time:
                return await self.trade_repo.find_by_time_range(start_time, end_time)
            
            # Otherwise, return all trades
            return await self.trade_repo.find_all()
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []


class SystemDataAccessor:
    """Accessor for system data."""
    
    def __init__(self):
        """Initialize the accessor."""
        pass
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            # In a real implementation, this would get actual system metrics
            # For now, return demo data
            return {
                "cpu": {
                    "usage_percent": 45.2,
                    "cores": 8
                },
                "memory": {
                    "total_mb": 16384,
                    "used_mb": 8192,
                    "percent": 50.0
                },
                "disk": {
                    "total_gb": 500,
                    "used_gb": 250,
                    "percent": 50.0
                },
                "process": {
                    "pid": 12345,
                    "memory_mb": 512,
                    "threads": 16,
                    "uptime_seconds": 86400
                }
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                "error": str(e)
            }