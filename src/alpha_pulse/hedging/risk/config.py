from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional

@dataclass
class HedgeConfig:
    """Configuration parameters for the hedging system."""
    
    # Target hedge ratio (0.0 = fully hedged, 1.0 = no hedge)
    hedge_ratio_target: Decimal
    
    # Maximum allowed leverage for futures positions
    max_leverage: Decimal
    
    # Maximum percentage of account value to use as margin
    max_margin_usage: Decimal
    
    # Minimum position size for placing hedge orders
    min_position_size: Dict[str, Decimal]
    
    # Maximum position size for placing hedge orders
    max_position_size: Dict[str, Decimal]
    
    # Grid bot configuration
    grid_bot_enabled: bool = False
    grid_bot_params: Optional[Dict[str, Dict[str, Decimal]]] = None
    
    # Rebalancing thresholds
    # How far the hedge ratio can deviate before triggering an adjustment
    hedge_ratio_threshold: Decimal = Decimal('0.05')
    
    # Risk management parameters
    max_drawdown: Decimal = Decimal('0.10')  # Maximum allowed drawdown
    stop_loss_threshold: Decimal = Decimal('0.15')  # Stop loss threshold
    
    # Execution parameters
    execution_delay: int = 0  # Delay in seconds between trades
    max_slippage: Decimal = Decimal('0.001')  # Maximum allowed slippage
    
    @classmethod
    def default_config(cls) -> 'HedgeConfig':
        """Create a default configuration instance."""
        return cls(
            hedge_ratio_target=Decimal('0.0'),  # Fully hedged
            max_leverage=Decimal('3.0'),
            max_margin_usage=Decimal('0.8'),
            min_position_size={'BTC': Decimal('0.001'), 'ETH': Decimal('0.01')},
            max_position_size={'BTC': Decimal('10.0'), 'ETH': Decimal('100.0')},
            grid_bot_enabled=False
        )
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.hedge_ratio_target <= 1:
            raise ValueError("hedge_ratio_target must be between 0 and 1")
        
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        
        if not 0 < self.max_margin_usage <= 1:
            raise ValueError("max_margin_usage must be between 0 and 1")
        
        if self.grid_bot_enabled and not self.grid_bot_params:
            raise ValueError("grid_bot_params must be provided when grid_bot_enabled is True")
        
        for symbol, size in self.min_position_size.items():
            if size <= 0:
                raise ValueError(f"min_position_size for {symbol} must be positive")
            if symbol not in self.max_position_size:
                raise ValueError(f"max_position_size missing for {symbol}")
            if self.max_position_size[symbol] <= size:
                raise ValueError(f"max_position_size must be greater than min_position_size for {symbol}")