"""
Grid level calculation and management.
"""
from decimal import Decimal
from typing import List, Optional
from loguru import logger

from .interfaces import GridCalculator
from .models import GridLevel, MarketState, PositionState


class DefaultGridCalculator(GridCalculator):
    """Default implementation of grid calculations."""
    
    def __init__(
        self,
        grid_spacing_pct: Decimal,
        num_levels: int,
        min_price_distance: Decimal,
        max_position_size: Decimal
    ):
        """
        Initialize grid calculator.
        
        Args:
            grid_spacing_pct: Grid spacing as percentage
            num_levels: Number of grid levels
            min_price_distance: Minimum distance between levels
            max_position_size: Maximum position size
        """
        self.grid_spacing_pct = grid_spacing_pct
        self.num_levels = num_levels
        self.min_price_distance = min_price_distance
        self.max_position_size = max_position_size
        
        logger.info(
            f"Initialized grid calculator with {num_levels} levels, "
            f"{grid_spacing_pct:.2%} spacing"
        )
        
    def calculate_grid_levels(
        self,
        market: MarketState,
        position: PositionState
    ) -> List[GridLevel]:
        """
        Calculate grid levels based on current market state.
        
        Args:
            market: Current market state
            position: Current position state
            
        Returns:
            List of calculated grid levels
        """
        try:
            current_price = market.current_price
            
            # Calculate base grid spacing
            grid_spacing = current_price * self.grid_spacing_pct
            
            # Adjust spacing based on volatility
            if market.volatility > Decimal('0'):
                volatility_spacing = current_price * market.volatility * Decimal('2')
                grid_spacing = max(grid_spacing, volatility_spacing)
            
            # Ensure minimum spacing
            grid_spacing = max(grid_spacing, self.min_price_distance)
            
            # Calculate level quantity based on position size
            base_quantity = min(
                position.spot_quantity / Decimal(str(self.num_levels)),
                self.max_position_size / Decimal(str(self.num_levels))
            )
            
            logger.info(
                f"Calculating grid with spacing {grid_spacing:.2f}, "
                f"base quantity {base_quantity:.8f}"
            )
            
            # Generate levels
            levels = []
            
            # Short levels above current price
            for i in range(self.num_levels):
                price = current_price + (grid_spacing * Decimal(str(i + 1)))
                level = GridLevel(
                    price=price,
                    quantity=base_quantity,
                    is_long=False
                )
                levels.append(level)
                
            # Long levels below current price
            for i in range(self.num_levels):
                price = current_price - (grid_spacing * Decimal(str(i + 1)))
                level = GridLevel(
                    price=price,
                    quantity=base_quantity,
                    is_long=True
                )
                levels.append(level)
                
            return self.validate_levels(levels, market)
            
        except Exception as e:
            logger.error(f"Error calculating grid levels: {str(e)}")
            return []
            
    def adjust_for_funding(
        self,
        levels: List[GridLevel],
        funding_rate: Decimal
    ) -> List[GridLevel]:
        """
        Adjust grid levels based on funding rate.
        
        Args:
            levels: Current grid levels
            funding_rate: Current funding rate
            
        Returns:
            Adjusted grid levels
        """
        try:
            if abs(funding_rate) < Decimal('0.001'):  # 0.1% threshold
                return levels
                
            # Calculate hedge ratio based on funding
            if funding_rate > 0:
                # Positive funding - reduce hedge
                hedge_ratio = Decimal('0.8')  # 80% hedge
            else:
                # Negative funding - increase hedge
                hedge_ratio = Decimal('1.2')  # 120% hedge
                
            # Adjust quantities
            adjusted_levels = []
            for level in levels:
                adjusted_level = level.update(
                    quantity=level.quantity * hedge_ratio
                )
                adjusted_levels.append(adjusted_level)
                
            logger.info(
                f"Adjusted grid for funding rate {funding_rate:.4%} "
                f"(hedge ratio: {hedge_ratio:.2f})"
            )
            
            return adjusted_levels
            
        except Exception as e:
            logger.error(f"Error adjusting for funding: {str(e)}")
            return levels
            
    def validate_levels(
        self,
        levels: List[GridLevel],
        market: MarketState
    ) -> List[GridLevel]:
        """
        Validate and filter grid levels.
        
        Args:
            levels: Grid levels to validate
            market: Current market state
            
        Returns:
            Filtered and validated levels
        """
        try:
            valid_levels = []
            current_price = market.current_price
            
            for level in levels:
                # Skip invalid prices
                if level.price <= Decimal('0'):
                    continue
                    
                # Validate direction
                if level.is_long and level.price >= current_price:
                    continue
                if not level.is_long and level.price <= current_price:
                    continue
                    
                # Validate quantity
                if level.quantity <= Decimal('0'):
                    continue
                if level.quantity > self.max_position_size:
                    continue
                    
                valid_levels.append(level)
                
            return valid_levels
            
        except Exception as e:
            logger.error(f"Error validating levels: {str(e)}")
            return []
            
    def _calculate_dynamic_spacing(
        self,
        base_spacing: Decimal,
        market: MarketState
    ) -> Decimal:
        """
        Calculate dynamic grid spacing based on market conditions.
        
        Args:
            base_spacing: Base grid spacing
            market: Current market state
            
        Returns:
            Adjusted grid spacing
        """
        try:
            # Start with base spacing
            spacing = base_spacing
            
            # Adjust for volatility
            if market.volatility > Decimal('0'):
                volatility_spacing = market.current_price * market.volatility * Decimal('2')
                spacing = max(spacing, volatility_spacing)
                
            # Adjust for volume
            if market.volume > Decimal('0'):
                volume_factor = Decimal('1')
                if market.volume > Decimal('1000000'):  # High volume
                    volume_factor = Decimal('0.8')  # Tighter spacing
                elif market.volume < Decimal('100000'):  # Low volume
                    volume_factor = Decimal('1.2')  # Wider spacing
                spacing *= volume_factor
                
            # Ensure minimum spacing
            spacing = max(spacing, self.min_price_distance)
            
            return spacing
            
        except Exception as e:
            logger.error(f"Error calculating dynamic spacing: {str(e)}")
            return base_spacing