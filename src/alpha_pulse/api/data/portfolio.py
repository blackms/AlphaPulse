"""Portfolio data access module."""
from typing import Dict, Optional
import logging

from alpha_pulse.portfolio.portfolio_manager import PortfolioManager


class PortfolioDataAccessor:
    """Access portfolio data."""
    
    def __init__(self):
        """Initialize portfolio accessor."""
        self.logger = logging.getLogger("alpha_pulse.api.data.portfolio")
        self.portfolio_manager = PortfolioManager.get_instance()
    
    async def get_portfolio(self, include_history: bool = False) -> Dict:
        """
        Get current portfolio data.
        
        Args:
            include_history: Whether to include historical data
            
        Returns:
            Portfolio data
        """
        try:
            # Get current portfolio
            portfolio = self.portfolio_manager.get_portfolio_data()
            
            # Transform to API format
            result = {
                "total_value": portfolio.total_value,
                "cash": portfolio.cash,
                "positions": []
            }
            
            # Add positions
            for position in portfolio.positions:
                result["positions"].append({
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "value": position.value,
                    "pnl": position.pnl,
                    "pnl_percentage": position.pnl_percentage
                })
                
            # Add performance metrics
            result["metrics"] = {
                "sharpe_ratio": portfolio.metrics.sharpe_ratio,
                "sortino_ratio": portfolio.metrics.sortino_ratio,
                "max_drawdown": portfolio.metrics.max_drawdown,
                "volatility": portfolio.metrics.volatility,
                "return_since_inception": portfolio.metrics.return_since_inception
            }
            
            # Add historical data if requested
            if include_history:
                history = self.portfolio_manager.get_portfolio_history()
                result["history"] = []
                
                for entry in history:
                    result["history"].append({
                        "timestamp": entry.timestamp.isoformat(),
                        "total_value": entry.total_value,
                        "cash": entry.cash,
                        "positions_value": entry.positions_value
                    })
            
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving portfolio data: {str(e)}")
            return {
                "total_value": 0,
                "cash": 0,
                "positions": [],
                "metrics": {},
                "error": str(e)
            }