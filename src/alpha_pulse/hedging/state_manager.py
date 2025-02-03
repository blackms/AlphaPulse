"""
State management for grid hedging strategy.
"""
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional
from loguru import logger

from .interfaces import StateManager
from .models import GridMetrics, GridState, MarketState, PositionState


class GridStateManager(StateManager):
    """State management implementation for grid strategy."""
    
    def __init__(
        self,
        symbol: str,
        initial_state: GridState = GridState.INITIALIZING
    ):
        """
        Initialize state manager.
        
        Args:
            symbol: Trading symbol
            initial_state: Initial grid state
        """
        self.symbol = symbol
        self.state = initial_state
        self.position = PositionState.create_empty()
        self.metrics = GridMetrics.create_empty()
        self.last_update = datetime.now()
        self.last_rebalance: Optional[datetime] = None
        self.last_funding_check: Optional[datetime] = None
        
        logger.info(
            f"Initialized state manager for {symbol} "
            f"(state: {initial_state.value})"
        )
        
    def update_position(
        self,
        current: PositionState,
        **kwargs
    ) -> PositionState:
        """
        Update position state.
        
        Args:
            current: Current position state
            **kwargs: Position state updates
            
        Returns:
            Updated position state
        """
        try:
            # Create new position state
            new_position = current.update(**kwargs)
            
            # Log significant changes
            if new_position.spot_quantity != current.spot_quantity:
                logger.info(
                    f"Position size changed: "
                    f"{current.spot_quantity} -> {new_position.spot_quantity}"
                )
                
            if new_position.unrealized_pnl != current.unrealized_pnl:
                logger.info(
                    f"Unrealized PnL changed: "
                    f"{current.unrealized_pnl:.2f} -> {new_position.unrealized_pnl:.2f}"
                )
                
            # Update state
            self.position = new_position
            self.last_update = datetime.now()
            
            return new_position
            
        except Exception as e:
            logger.error(f"Error updating position state: {str(e)}")
            return current
            
    def update_metrics(
        self,
        current: GridMetrics,
        **kwargs
    ) -> GridMetrics:
        """
        Update metrics state.
        
        Args:
            current: Current metrics state
            **kwargs: Metrics updates
            
        Returns:
            Updated metrics
        """
        try:
            # Create new metrics state
            new_metrics = current.update(**kwargs)
            
            # Log significant changes
            if new_metrics.total_trades != current.total_trades:
                win_rate = (
                    new_metrics.successful_trades /
                    max(new_metrics.total_trades, 1)
                )
                logger.info(
                    f"Trade metrics updated - "
                    f"Total: {new_metrics.total_trades}, "
                    f"Win Rate: {win_rate:.2%}"
                )
                
            if new_metrics.max_drawdown != current.max_drawdown:
                logger.warning(
                    f"Max drawdown updated: {new_metrics.max_drawdown:.2%}"
                )
                
            # Update state
            self.metrics = new_metrics
            self.last_update = datetime.now()
            
            return new_metrics
            
        except Exception as e:
            logger.error(f"Error updating metrics state: {str(e)}")
            return current
            
    def update_state(self, new_state: GridState) -> None:
        """
        Update grid state.
        
        Args:
            new_state: New grid state
        """
        try:
            if new_state != self.state:
                logger.info(
                    f"State changed: {self.state.value} -> {new_state.value}"
                )
                self.state = new_state
                self.last_update = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating grid state: {str(e)}")
            
    def record_rebalance(self) -> None:
        """Record grid rebalance."""
        try:
            self.last_rebalance = datetime.now()
            logger.debug("Recorded grid rebalance")
            
        except Exception as e:
            logger.error(f"Error recording rebalance: {str(e)}")
            
    def record_funding_check(self) -> None:
        """Record funding rate check."""
        try:
            self.last_funding_check = datetime.now()
            logger.debug("Recorded funding check")
            
        except Exception as e:
            logger.error(f"Error recording funding check: {str(e)}")
            
    def get_status(self) -> Dict:
        """
        Get current strategy status.
        
        Returns:
            Dict containing current status information
        """
        try:
            return {
                "symbol": self.symbol,
                "state": self.state.value,
                "position": {
                    "spot": float(self.position.spot_quantity),
                    "futures": float(self.position.futures_quantity),
                    "entry_price": float(self.position.avg_entry_price),
                    "unrealized_pnl": float(self.position.unrealized_pnl),
                    "realized_pnl": float(self.position.realized_pnl),
                    "funding_paid": float(self.position.funding_paid)
                },
                "metrics": {
                    "total_trades": self.metrics.total_trades,
                    "win_rate": (
                        self.metrics.successful_trades /
                        max(self.metrics.total_trades, 1)
                    ),
                    "avg_profit": float(self.metrics.avg_profit_per_trade),
                    "max_drawdown": float(self.metrics.max_drawdown),
                    "sharpe_ratio": self.metrics.sharpe_ratio,
                    "sortino_ratio": self.metrics.sortino_ratio,
                    "profit_factor": self.metrics.profit_factor
                },
                "last_update": {
                    "state": self.last_update.isoformat(),
                    "rebalance": (
                        self.last_rebalance.isoformat()
                        if self.last_rebalance else None
                    ),
                    "funding_check": (
                        self.last_funding_check.isoformat()
                        if self.last_funding_check else None
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return {
                "symbol": self.symbol,
                "state": GridState.ERROR.value,
                "error": str(e)
            }
            
    def _validate_state(self) -> bool:
        """
        Validate internal state consistency.
        
        Returns:
            True if state is valid
        """
        try:
            # Check position values
            if self.position.spot_quantity < Decimal('0'):
                logger.error("Invalid negative spot quantity")
                return False
                
            # Check metrics values
            if self.metrics.total_trades < 0:
                logger.error("Invalid negative trade count")
                return False
                
            if self.metrics.successful_trades > self.metrics.total_trades:
                logger.error("Successful trades exceeds total trades")
                return False
                
            # Check timestamps
            now = datetime.now()
            if self.last_update > now:
                logger.error("Last update timestamp in future")
                return False
                
            if self.last_rebalance and self.last_rebalance > now:
                logger.error("Last rebalance timestamp in future")
                return False
                
            if self.last_funding_check and self.last_funding_check > now:
                logger.error("Last funding check timestamp in future")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating state: {str(e)}")
            return False