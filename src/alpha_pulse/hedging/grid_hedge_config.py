"""
Configuration for grid hedging strategy.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from loguru import logger

from alpha_pulse.risk_management.analysis import RiskAnalyzer
from alpha_pulse.risk_management.position_sizing import VolatilityBasedSizer


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
class GridParameters:
    """Grid strategy parameters."""
    grid_spacing: float
    position_step: float
    stop_loss: float
    take_profit: float


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

    @staticmethod
    async def calculate_grid_parameters(
        current_price: float,
        spot_quantity: float,
        volatility: float,
        risk_analyzer: RiskAnalyzer,
        position_sizer: VolatilityBasedSizer,
        portfolio_value: float,
        historical_returns=None,
        num_levels: int = 5,
    ) -> GridParameters:
        """
        Calculate grid parameters based on risk metrics.
        
        Args:
            current_price: Current market price
            spot_quantity: Spot position size
            volatility: Market volatility estimate
            risk_analyzer: Risk analyzer instance
            position_sizer: Position sizer instance
            portfolio_value: Total portfolio value
            historical_returns: Historical returns data (optional)
            num_levels: Number of grid levels
            
        Returns:
            GridParameters instance
        """
        # Calculate position size per grid level using volatility-based sizing
        position_result = position_sizer.calculate_position_size(
            symbol="BTCUSDT",
            current_price=current_price,
            portfolio_value=portfolio_value,
            volatility=volatility,
            signal_strength=1.0,  # Full signal for hedging
            historical_returns=historical_returns
        )
        
        # Calculate grid spacing based on volatility
        # Use 0.5 standard deviations for grid spacing
        grid_spacing = current_price * volatility * 0.5
        
        # Calculate position step size (ensure total matches spot position)
        position_step = min(
            spot_quantity / num_levels,  # Even distribution
            position_result.size / current_price  # Risk-based limit
        )
        
        # Calculate stop loss and take profit using VaR
        if historical_returns is not None:
            metrics = risk_analyzer.calculate_metrics(historical_returns)
            stop_loss = metrics.var_95  # Use 95% VaR for stop loss
            take_profit = -stop_loss * 1.5  # Target 1.5x risk/reward
        else:
            # Default to volatility-based levels if no historical data
            stop_loss = volatility * 2  # 2 standard deviations
            take_profit = stop_loss * 1.5
        
        return GridParameters(
            grid_spacing=grid_spacing,
            position_step=position_step,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    @classmethod
    async def create_hedge_grid(
        cls,
        symbol: str,
        current_price: float,
        spot_quantity: float,
        volatility: float,
        portfolio_value: float,
        risk_analyzer: Optional[RiskAnalyzer] = None,
        position_sizer: Optional[VolatilityBasedSizer] = None,
        historical_returns=None,
        num_levels: int = 5,
    ) -> 'GridHedgeConfig':
        """
        Create a grid configuration for hedging.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            spot_quantity: Spot position size to hedge
            volatility: Market volatility estimate
            portfolio_value: Total portfolio value
            risk_analyzer: Risk analyzer instance (optional)
            position_sizer: Position sizer instance (optional)
            historical_returns: Historical returns data (optional)
            num_levels: Number of grid levels
            
        Returns:
            GridHedgeConfig instance
        """
        # Initialize risk components if not provided
        if not risk_analyzer:
            risk_analyzer = RiskAnalyzer(
                rolling_window=20,  # 20-day window
                var_confidence=0.95,
            )
        
        if not position_sizer:
            position_sizer = VolatilityBasedSizer(
                target_volatility=0.01,  # 1% daily target
                max_size_pct=0.2,  # Maximum 20% of portfolio per grid level
                volatility_lookback=20,
            )
        
        # Calculate grid parameters
        params = await cls.calculate_grid_parameters(
            current_price=current_price,
            spot_quantity=spot_quantity,
            volatility=volatility,
            risk_analyzer=risk_analyzer,
            position_sizer=position_sizer,
            portfolio_value=portfolio_value,
            historical_returns=historical_returns,
            num_levels=num_levels
        )
        
        # Create grid levels
        grid_levels = []
        for i in range(num_levels):
            price = current_price + (params.grid_spacing * (i + 1))
            grid_levels.append(GridLevel(
                price=price,
                quantity=params.position_step,
                is_long=False  # Short for hedging
            ))
        
        logger.info(
            f"Creating hedge grid for {symbol} - "
            f"Spot: {spot_quantity:.8f}, "
            f"Price: {current_price:.2f}, "
            f"Grid Spacing: {params.grid_spacing:.2f}, "
            f"Position Step: {params.position_step:.8f}, "
            f"Stop Loss: {params.stop_loss:.2%}, "
            f"Take Profit: {params.take_profit:.2%}"
        )
        
        return cls(
            symbol=symbol,
            grid_direction=GridDirection.SHORT,
            grid_levels=grid_levels,
            upper_price=current_price + (params.grid_spacing * num_levels),
            lower_price=current_price,
            grid_spacing=params.grid_spacing,
            max_position_size=spot_quantity,
            position_step_size=params.position_step,
            stop_loss_pct=params.stop_loss,
            take_profit_pct=params.take_profit
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
