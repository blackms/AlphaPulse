from decimal import Decimal
from typing import List, Dict, Optional
from loguru import logger

from .interfaces import IHedgeAnalyzer
from .models import (
    SpotPosition,
    FuturesPosition,
    HedgeRecommendation,
    HedgeAdjustment,
    GridBotParams
)
from .hedge_config import HedgeConfig

class BasicFuturesHedgeAnalyzer(IHedgeAnalyzer):
    """Basic implementation of futures-based hedging analysis."""
    
    def __init__(self, config: HedgeConfig):
        """
        Initialize the analyzer with configuration parameters.
        
        Args:
            config: HedgeConfig instance containing hedging parameters
        """
        self.config = config
        config.validate()
    
    def calculate_net_exposure(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> Decimal:
        """Calculate net exposure across all positions."""
        # Sum spot positions
        spot_exposure = sum(
            (pos.quantity * pos.current_price)
            for pos in spot_positions
            if pos.current_price is not None
        )
        
        # Sum futures positions (short positions reduce exposure)
        futures_exposure = sum(
            (pos.quantity * pos.current_price * (1 if pos.side == "LONG" else -1))
            for pos in futures_positions
            if pos.current_price is not None
        )
        
        return Decimal(str(spot_exposure + futures_exposure))
    
    def evaluate_hedge_effectiveness(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> Dict[str, Decimal]:
        """Evaluate hedge effectiveness metrics."""
        net_exposure = self.calculate_net_exposure(spot_positions, futures_positions)
        
        # Calculate total portfolio value for relative metrics
        total_portfolio_value = sum(
            pos.market_value for pos in spot_positions if pos.market_value is not None
        )
        
        if total_portfolio_value == 0:
            return {
                "net_exposure": net_exposure,
                "hedge_ratio": Decimal('0'),
                "margin_usage": Decimal('0')
            }
        
        # Calculate current hedge ratio
        current_hedge_ratio = net_exposure / total_portfolio_value
        
        # Calculate margin usage
        total_margin_used = sum(
            pos.margin_used for pos in futures_positions
        )
        margin_usage_ratio = total_margin_used / total_portfolio_value
        
        return {
            "net_exposure": net_exposure,
            "hedge_ratio": current_hedge_ratio,
            "margin_usage": margin_usage_ratio
        }
    
    def _generate_hedge_adjustments(
        self,
        current_hedge_ratio: Decimal,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> List[HedgeAdjustment]:
        """Generate list of recommended hedge adjustments."""
        adjustments = []
        
        # Calculate target exposure
        total_spot_value = sum(
            pos.market_value for pos in spot_positions if pos.market_value is not None
        )
        target_exposure = total_spot_value * self.config.hedge_ratio_target
        current_exposure = self.calculate_net_exposure(spot_positions, futures_positions)
        exposure_difference = current_exposure - target_exposure
        
        # If exposure difference is within threshold, no adjustments needed
        if abs(exposure_difference) <= (total_spot_value * self.config.hedge_ratio_threshold):
            return []
        
        # Only generate adjustments for positions that have futures hedges
        for futures_pos in futures_positions:
            # Extract base symbol (remove USDT suffix)
            base_symbol = futures_pos.symbol.replace("USDT", "")
            
            # Find corresponding spot position
            spot_pos = next(
                (s for s in spot_positions if s.symbol == base_symbol),
                None
            )
            
            if not spot_pos or spot_pos.market_value is None:
                continue
            
            # For target hedge ratio of 0%, recommend closing the entire futures position
            if self.config.hedge_ratio_target == 0:
                adjustments.append(
                    HedgeAdjustment(
                        symbol=futures_pos.symbol,
                        desired_delta=futures_pos.quantity,  # Close entire position
                        side="LONG" if futures_pos.side == "SHORT" else "SHORT",  # Opposite side to close
                        recommendation=f"Close hedge for {base_symbol} by {futures_pos.quantity:.8f}",
                        priority="HIGH" if abs(futures_pos.notional_value or 0) > (spot_pos.market_value * Decimal('0.1')) else "MEDIUM"
                    )
                )
            else:
                # Calculate required adjustment for non-zero target hedge ratio
                spot_value = spot_pos.market_value
                target_hedge = spot_value * (1 - self.config.hedge_ratio_target)
                current_hedge = futures_pos.notional_value * (-1 if futures_pos.side == "SHORT" else 1)
                
                hedge_difference = target_hedge - current_hedge
                
                if abs(hedge_difference) > 0:
                    # Determine adjustment side
                    side = "SHORT" if hedge_difference < 0 else "LONG"
                    
                    # Calculate position size
                    desired_delta = abs(hedge_difference) / (spot_pos.current_price or Decimal('1'))
                    min_size = self.config.min_position_size.get(spot_pos.symbol, Decimal('0'))
                    max_size = self.config.max_position_size.get(spot_pos.symbol, Decimal('inf'))
                    
                    if desired_delta < min_size:
                        continue
                    
                    desired_delta = min(desired_delta, max_size)
                    
                    adjustments.append(
                        HedgeAdjustment(
                            symbol=futures_pos.symbol,
                            desired_delta=desired_delta,
                            side=side,
                            recommendation=f"{'Increase' if side == 'SHORT' else 'Decrease'} hedge for {base_symbol} by {desired_delta:.8f}",
                            priority="HIGH" if abs(hedge_difference) > (spot_value * Decimal('0.1')) else "MEDIUM"
                        )
                    )
        
        return adjustments
    
    def analyze(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> HedgeRecommendation:
        """Analyze positions and generate hedge recommendations."""
        # Evaluate current hedge effectiveness
        metrics = self.evaluate_hedge_effectiveness(spot_positions, futures_positions)
        current_hedge_ratio = metrics["hedge_ratio"]
        
        # Generate hedge adjustments
        adjustments = self._generate_hedge_adjustments(
            current_hedge_ratio,
            spot_positions,
            futures_positions
        )
        
        # Generate commentary
        commentary = (
            f"Current hedge ratio: {current_hedge_ratio:.2%} "
            f"(target: {self.config.hedge_ratio_target:.2%})\n"
            f"Net exposure: {metrics['net_exposure']:.2f} USD\n"
            f"Margin usage: {metrics['margin_usage']:.2%}\n"
        )
        
        if not adjustments:
            commentary += "No hedge adjustments required at this time."
        else:
            commentary += f"Generated {len(adjustments)} hedge adjustment recommendations."
        
        return HedgeRecommendation(
            adjustments=adjustments,
            current_net_exposure=metrics["net_exposure"],
            target_net_exposure=metrics["net_exposure"] * self.config.hedge_ratio_target,
            commentary=commentary,
            risk_metrics=metrics
        )