"""
LLM-enhanced hedge analyzer that provides detailed explanations of hedging decisions.
Uses Langchain and o3-mini model for analysis and comparison.
"""
from decimal import Decimal
from typing import List, Dict, Optional
import json
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from alpha_pulse.hedging.common.interfaces import RiskManager as IHedgeAnalyzer
from alpha_pulse.hedging.common.types import (
    SpotPosition,
    FuturesPosition,
    HedgeRecommendation,
    HedgeAdjustment
)
from alpha_pulse.hedging.risk.config import HedgeConfig
from alpha_pulse.hedging.risk.analyzers.basic import BasicFuturesHedgeAnalyzer

class LLMHedgeAnalyzer(IHedgeAnalyzer):
    """
    Enhanced hedge analyzer that uses Langchain and o3-mini model for:
    1. Running basic strategy analysis
    2. Getting LLM recommendations
    3. Comparing and evaluating both approaches
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
        self.basic_analyzer = BasicFuturesHedgeAnalyzer(config)
        self.llm = ChatOpenAI(
            model="o3-mini",
            openai_api_key=openai_api_key,
            model_kwargs={"seed": 42}  # For reproducibility
        )
    
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
    
    async def _get_llm_recommendations(
        self,
        positions_data: str,
        metrics: Dict[str, Decimal]
    ) -> str:
        """Get LLM hedge recommendations."""
        try:
            messages = [
                SystemMessage(content="""You are a hedge analysis expert. Your task is to analyze the portfolio data and suggest hedging recommendations.
Focus on:
1. Position sizing for each asset
2. Optimal hedge ratios
3. Risk management considerations
4. Market impact analysis"""),
                HumanMessage(content=f"""Analyze this portfolio and provide hedging recommendations:

Portfolio Data:
{positions_data}

Key Metrics:
- Current Net Exposure: ${float(metrics['net_exposure']):,.2f}
- Current Hedge Ratio: {float(metrics['hedge_ratio'])*100:.2f}%
- Target Hedge Ratio: {float(self.config.hedge_ratio_target)*100:.2f}%
- Current Margin Usage: {float(metrics['margin_usage'])*100:.2f}%

Provide specific recommendations for each asset, including:
1. Suggested position adjustments
2. Entry/exit levels
3. Risk management stops
4. Market timing considerations""")
            ]
            
            # Get LLM recommendations
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error getting LLM recommendations: {e}")
            return "LLM recommendations unavailable"
    
    async def _evaluate_strategies(
        self,
        positions_data: str,
        basic_recommendation: HedgeRecommendation,
        llm_analysis: str
    ) -> str:
        """Compare and evaluate both strategies."""
        try:
            messages = [
                SystemMessage(content="""You are a hedge strategy evaluator. Compare and evaluate two different hedging approaches:
1. A quantitative strategy based on hedge ratios and position sizing
2. An LLM-based strategy using market analysis

Evaluate strengths, weaknesses, and potential synergies."""),
                HumanMessage(content=f"""Compare these two hedging approaches:

Portfolio Data:
{positions_data}

Quantitative Strategy Recommendations:
{basic_recommendation.commentary}

LLM Strategy Recommendations:
{llm_analysis}

Provide:
1. Comparative analysis of both approaches
2. Strengths and weaknesses of each
3. Suggested synthesis of both strategies
4. Risk considerations for each approach
5. Final recommendation on which aspects of each strategy to use""")
            ]
            
            # Get evaluation
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error evaluating strategies: {e}")
            return "Strategy evaluation unavailable"
    
    def calculate_net_exposure(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> Decimal:
        """Calculate net exposure across all positions."""
        return self.basic_analyzer.calculate_net_exposure(
            spot_positions,
            futures_positions
        )
    
    def evaluate_hedge_effectiveness(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> Dict[str, Decimal]:
        """Evaluate hedge effectiveness metrics."""
        return self.basic_analyzer.evaluate_hedge_effectiveness(
            spot_positions,
            futures_positions
        )
    
    async def analyze(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> HedgeRecommendation:
        """
        Three-step analysis process:
        1. Get basic strategy recommendations
        2. Get LLM recommendations
        3. Compare and evaluate both approaches
        """
        # Step 1: Get basic strategy recommendations
        basic_recommendation = self.basic_analyzer.analyze(
            spot_positions,
            futures_positions
        )
        
        # Get metrics and format data
        metrics = self.evaluate_hedge_effectiveness(spot_positions, futures_positions)
        positions_data = self._format_positions_for_llm(spot_positions, futures_positions)
        
        # Step 2: Get LLM recommendations
        llm_analysis = await self._get_llm_recommendations(positions_data, metrics)
        
        # Step 3: Compare and evaluate
        evaluation = await self._evaluate_strategies(
            positions_data,
            basic_recommendation,
            llm_analysis
        )
        
        # Combine all analyses
        combined_commentary = (
            f"Current hedge ratio: {metrics['hedge_ratio']:.2%} "
            f"(target: {self.config.hedge_ratio_target:.2%})\n"
            f"Net exposure: {metrics['net_exposure']:.2f} USD\n"
            f"Margin usage: {metrics['margin_usage']:.2%}\n\n"
            f"Basic Strategy Analysis:\n"
            f"{basic_recommendation.commentary}\n\n"
            f"LLM Strategy Analysis:\n{llm_analysis}\n\n"
            f"Strategy Evaluation:\n{evaluation}"
        )
        
        return HedgeRecommendation(
            adjustments=basic_recommendation.adjustments,
            current_net_exposure=metrics["net_exposure"],
            target_net_exposure=metrics["net_exposure"] * self.config.hedge_ratio_target,
            commentary=combined_commentary,
            risk_metrics=metrics
        )