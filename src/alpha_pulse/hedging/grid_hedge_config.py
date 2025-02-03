"""
Configuration for grid hedging strategy.
"""
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class GridDirection(Enum):
    """Grid trading direction."""
    LONG = "LONG"  # Grid trades long when price falls
    SHORT = "SHORT"  # Grid trades short when price rises
    BOTH = "BOTH"  # Grid trades in both directions


@dataclass
class GridLevel:
    """Represents a single grid level."""
    price: float
    quantity: float
    is_long: bool = True  # True for long, False for short
    order_id: Optional[str] = None  # Filled when order is placed


@dataclass
class GridHedgeConfig:
    """Configuration for grid hedging strategy."""
    
    # Grid parameters
    symbol: str
    grid_direction: GridDirection
    grid_levels: List[GridLevel]
    
    # Price range
    upper_price: float
    lower_price: float
    grid_spacing: float  # Price difference between grid levels
    
    # Position parameters
    max_position_size: float
    position_step_size: float
    
    # Risk parameters
    stop_loss_pct: Optional[float] = None  # Stop loss percentage from entry
    take_profit_pct: Optional[float] = None  # Take profit percentage from entry
    max_orders: Optional[int] = None  # Maximum number of open orders
    
    # Execution parameters
    rebalance_interval: int = 60  # Seconds between grid rebalances
    order_timeout: int = 60  # Seconds to wait for order fill
    
    @classmethod
    def create_symmetric_grid(
        cls,
        symbol: str,
        center_price: float,
        grid_spacing: float,
        num_levels: int,
        position_step_size: float,
        max_position_size: float,
        grid_direction: GridDirection = GridDirection.BOTH,
    ) -> 'GridHedgeConfig':
        """
        Create a symmetric grid around a center price.
        
        Args:
            symbol: Trading symbol
            center_price: Price to center the grid around
            grid_spacing: Price difference between levels
            num_levels: Number of levels above and below center
            position_step_size: Position size per grid level
            max_position_size: Maximum total position size
            grid_direction: Direction of grid trades
            
        Returns:
            GridHedgeConfig instance
        """
        upper_price = center_price + (grid_spacing * num_levels)
        lower_price = center_price - (grid_spacing * num_levels)
        
        grid_levels = []
        
        # Create grid levels based on direction
        if grid_direction in [GridDirection.LONG, GridDirection.BOTH]:
            # Long levels below center
            for i in range(num_levels):
                price = center_price - (grid_spacing * (i + 1))
                grid_levels.append(GridLevel(
                    price=price,
                    quantity=position_step_size,
                    is_long=True
                ))
                
        if grid_direction in [GridDirection.SHORT, GridDirection.BOTH]:
            # Short levels above center
            for i in range(num_levels):
                price = center_price + (grid_spacing * (i + 1))
                grid_levels.append(GridLevel(
                    price=price,
                    quantity=position_step_size,
                    is_long=False
                ))
        
        return cls(
            symbol=symbol,
            grid_direction=grid_direction,
            grid_levels=grid_levels,
            upper_price=upper_price,
            lower_price=lower_price,
            grid_spacing=grid_spacing,
            max_position_size=max_position_size,
            position_step_size=position_step_size
        )
        
    def validate(self) -> bool:
        """
        Validate the grid configuration.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.grid_levels:
            raise ValueError("No grid levels defined")
            
        if self.upper_price <= self.lower_price:
            raise ValueError("Upper price must be greater than lower price")
            
        if self.grid_spacing <= 0:
            raise ValueError("Grid spacing must be positive")
            
        if self.max_position_size < self.position_step_size:
            raise ValueError("Max position size must be greater than step size")
            
        total_possible_position = sum(
            level.quantity for level in self.grid_levels
        )
        if total_possible_position > self.max_position_size:
            raise ValueError(
                f"Total possible position ({total_possible_position}) exceeds "
                f"max position size ({self.max_position_size})"
            )
            
        return True
