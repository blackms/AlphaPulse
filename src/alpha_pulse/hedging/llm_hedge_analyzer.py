"""
LLM-enhanced hedge analyzer that provides detailed explanations of hedging decisions.
"""
from decimal import Decimal
from typing import List, Dict, Optional
import json
from loguru import logger
import openai

from .interfaces import IHedgeAnalyzer
from .models import (
    SpotPosition,
    FuturesPosition,
    HedgeRecommendation,
    HedgeAdjustment
)
from .hedge_config import HedgeConfig

class LLMHedgeAnalyzer(IHedgeAnalyzer):
    """
    Enhanced hedge analyzer that uses LLM to explain hedging decisions
    and provide market context for recommendations.
    """
    
    def __init__(self, config: HedgeConfig, openai_api_key: str):
        """
        Initialize the analyzer.
        
        Args:
            config: Hedging configuration
            openai_api_key: OpenAI API key for LLM analysis
        """
        self.config = config
        config.validate()
        openai.api_key = openai_api_key
    
    def _format_positions_for_llm(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> str:
        """Format position data for LLM input."""
        spot_data = []
        for pos in spot_positions:
            if pos.current_price:
                market_value = pos.quantity * pos.current_price
                pnl = market_value - (pos.quantity * pos.avg_price)
                spot_data.append({
                    "asset": pos.symbol,
                    "quantity": float(pos.quantity),
                    "avgPrice": float(pos.avg_price),
                    "currentPrice": float(pos.current_price),
                    "marketValue": float(market_value),
                    "unrealizedPnL": float(pnl)
                })
        
        futures_data = []
        for pos in futures_positions:
            if pos.current_price:
                notional = pos.quantity * pos.current_price
                pnl = pos.pnl if pos.pnl else Decimal('0')
                futures_data.append({
                    "asset": pos.symbol,
                    "quantity": float(pos.quantity),
                    "side": pos.side,
                    "entryPrice": float(pos.entry_price),
                    "currentPrice": float(pos.current_price),
                    "leverage": float(pos.leverage),
                    "notionalValue": float(notional),
                    "marginUsed": float(pos.margin_used),
                    "unrealizedPnL": float(pnl)
                })
        
        return json.dumps({
            "spotPositions": spot_data,
            "futuresPositions": futures_data,
            "hedgeConfig": {
                "targetHedgeRatio": float(self.config.hedge_ratio_target),
                "maxLeverage": float(self.config.max_leverage),
                "maxMarginUsage": float(self.config.max_margin_usage)
            }
        }, indent=2)
    
    async def _get_llm_analysis(
        self,
        positions_data: str,
        metrics: Dict[str, Decimal],
        adjustments: List[HedgeAdjustment]
    ) -> str:
        """Get LLM analysis of the hedging situation."""
        try:
            # Prepare the prompt
            prompt = f"""You are a hedge analysis expert. Analyze the following portfolio and hedging data and provide detailed explanations of the current situation and recommended actions.

Portfolio Data:
{positions_data}

Key Metrics:
- Current Net Exposure: ${float(metrics['net_exposure']):,.2f}
- Current Hedge Ratio: {float(metrics['hedge_ratio'])*100:.2f}%
- Target Hedge Ratio: {float(self.config.hedge_ratio_target)*100:.2f}%
- Current Margin Usage: {float(metrics['margin_usage'])*100:.2f}%

Recommended Adjustments:
{json.dumps([{
    'symbol': adj.symbol,
    'side': adj.side,
    'quantity': float(adj.desired_delta),
    'priority': adj.priority
} for adj in adjustments], indent=2)}

Provide a detailed analysis including:
1. Current portfolio state and risk exposure
2. Explanation of why each adjustment is recommended
3. Market impact considerations
4. Risk management implications
5. Alternative strategies to consider

Keep the analysis concise but informative, focusing on key insights and actionable recommendations."""

            # Get LLM analysis
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a hedge analysis expert providing detailed explanations of hedging decisions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            return "LLM analysis unavailable"
    
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
        
        # Generate adjustments for each spot position
        for spot_pos in spot_positions:
            if spot_pos.market_value is None:
                continue
                
            # Find corresponding futures position
            futures_pos = next(
                (f for f in futures_positions if f.symbol.startswith(spot_pos.symbol)),
                None
            )
            
            # Calculate required adjustment
            spot_value = spot_pos.market_value
            target_hedge = spot_value * (1 - self.config.hedge_ratio_target)
            current_hedge = Decimal('0')
            if futures_pos and futures_pos.notional_value is not None:
                current_hedge = futures_pos.notional_value * (-1 if futures_pos.side == "SHORT" else 1)
            
            hedge_difference = target_hedge - current_hedge
            
            if abs(hedge_difference) > 0:
                # Determine adjustment side
                side = "SHORT" if hedge_difference < 0 else "LONG"
                symbol = f"{spot_pos.symbol}USDT"  # Assuming USDT pairs
                
                # Check position size limits
                desired_delta = abs(hedge_difference) / (spot_pos.current_price or Decimal('1'))
                min_size = self.config.min_position_size.get(spot_pos.symbol, Decimal('0'))
                max_size = self.config.max_position_size.get(spot_pos.symbol, Decimal('inf'))
                
                if desired_delta < min_size:
                    continue
                
                desired_delta = min(desired_delta, max_size)
                
                adjustments.append(
                    HedgeAdjustment(
                        symbol=symbol,
                        desired_delta=desired_delta,
                        side=side,
                        recommendation=f"{'Increase' if side == 'SHORT' else 'Decrease'} hedge for {spot_pos.symbol} by {desired_delta:.8f}",
                        priority="HIGH" if abs(hedge_difference) > (spot_value * Decimal('0.1')) else "MEDIUM"
                    )
                )
        
        return adjustments
    
    async def analyze(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> HedgeRecommendation:
        """Analyze positions and generate hedge recommendations with LLM insights."""
        # Evaluate current hedge effectiveness
        metrics = self.evaluate_hedge_effectiveness(spot_positions, futures_positions)
        current_hedge_ratio = metrics["hedge_ratio"]
        
        # Generate hedge adjustments
        adjustments = self._generate_hedge_adjustments(
            current_hedge_ratio,
            spot_positions,
            futures_positions
        )
        
        # Format data for LLM
        positions_data = self._format_positions_for_llm(spot_positions, futures_positions)
        
        # Get LLM analysis
        llm_analysis = await self._get_llm_analysis(positions_data, metrics, adjustments)
        
        # Generate base commentary
        base_commentary = (
            f"Current hedge ratio: {current_hedge_ratio:.2%} "
            f"(target: {self.config.hedge_ratio_target:.2%})\n"
            f"Net exposure: {metrics['net_exposure']:.2f} USD\n"
            f"Margin usage: {metrics['margin_usage']:.2%}\n\n"
            f"LLM Analysis:\n{llm_analysis}"
        )
        
        return HedgeRecommendation(
            adjustments=adjustments,
            current_net_exposure=metrics["net_exposure"],
            target_net_exposure=metrics["net_exposure"] * self.config.hedge_ratio_target,
            commentary=base_commentary,
            risk_metrics=metrics
        )