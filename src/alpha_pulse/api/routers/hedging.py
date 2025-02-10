"""
Router for hedging-related endpoints.
"""
import os
from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from decimal import Decimal
from loguru import logger
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from alpha_pulse.exchanges.base import BaseExchange
from alpha_pulse.hedging.risk.config import HedgeConfig
from alpha_pulse.hedging.risk.analyzers.llm import LLMHedgeAnalyzer
from alpha_pulse.hedging.risk.manager import HedgeManager
from alpha_pulse.hedging.execution.position_fetcher import ExchangePositionFetcher
from alpha_pulse.hedging.execution.order_executor import ExchangeOrderExecutor
from alpha_pulse.hedging.execution.strategy import BasicExecutionStrategy
from ..dependencies import get_exchange_client, verify_api_key

router = APIRouter()

class HedgeAnalysisResponse(BaseModel):
    """Hedge analysis response model."""
    commentary: str
    adjustments: List[Dict[str, str]]
    current_net_exposure: float
    target_net_exposure: float
    risk_metrics: Dict[str, float]

class HedgeExecutionResponse(BaseModel):
    """Hedge execution response model."""
    status: str
    executed_trades: List[Dict[str, str]]
    message: str

@router.get(
    "/analysis",
    response_model=HedgeAnalysisResponse,
    dependencies=[Depends(verify_api_key)]
)
async def analyze_hedge_positions(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Analyze current hedge positions and provide recommendations."""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize components
        config = HedgeConfig(
            hedge_ratio_target=Decimal('0.0'),  # Fully hedged
            max_leverage=Decimal('3.0'),
            max_margin_usage=Decimal('0.8'),
            min_position_size={'BTC': Decimal('0.001'), 'ETH': Decimal('0.01')},
            max_position_size={'BTC': Decimal('1.0'), 'ETH': Decimal('10.0')},
            grid_bot_enabled=False
        )
        
        position_fetcher = ExchangePositionFetcher(exchange)
        hedge_analyzer = LLMHedgeAnalyzer(config, openai_api_key)
        
        # Get current positions
        spot_positions = await position_fetcher.get_spot_positions()
        futures_positions = await position_fetcher.get_futures_positions()
        
        # Get analysis
        recommendation = await hedge_analyzer.analyze(spot_positions, futures_positions)
        
        return HedgeAnalysisResponse(
            commentary=recommendation.commentary,
            adjustments=[
                {
                    "asset": adj.asset,
                    "action": adj.recommendation,
                    "priority": str(adj.priority)
                }
                for adj in recommendation.adjustments
            ],
            current_net_exposure=float(recommendation.current_net_exposure),
            target_net_exposure=float(recommendation.target_net_exposure),
            risk_metrics={
                k: float(v) for k, v in recommendation.risk_metrics.items()
            }
        )
    except Exception as e:
        logger.error(f"Error analyzing hedge positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze hedge positions"
        )

@router.post(
    "/execute",
    response_model=HedgeExecutionResponse,
    dependencies=[Depends(verify_api_key)]
)
async def execute_hedge_adjustments(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Execute recommended hedge adjustments."""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize components
        config = HedgeConfig(
            hedge_ratio_target=Decimal('0.0'),  # Fully hedged
            max_leverage=Decimal('3.0'),
            max_margin_usage=Decimal('0.8'),
            min_position_size={'BTC': Decimal('0.001'), 'ETH': Decimal('0.01')},
            max_position_size={'BTC': Decimal('1.0'), 'ETH': Decimal('10.0')},
            grid_bot_enabled=False
        )
        
        position_fetcher = ExchangePositionFetcher(exchange)
        order_executor = ExchangeOrderExecutor(exchange)
        hedge_analyzer = LLMHedgeAnalyzer(config, openai_api_key)
        execution_strategy = BasicExecutionStrategy()
        
        # Create hedge manager
        manager = HedgeManager(
            hedge_analyzer=hedge_analyzer,
            position_fetcher=position_fetcher,
            execution_strategy=execution_strategy,
            order_executor=order_executor,
            execute_hedge=True
        )
        
        # Execute hedge adjustments
        result = await manager.manage_hedge()
        
        if result is None:
            return HedgeExecutionResponse(
                status="completed",
                executed_trades=[],
                message="No hedge adjustments needed"
            )
            
        return HedgeExecutionResponse(
            status="completed",
            executed_trades=[
                {
                    "asset": trade.asset,
                    "side": trade.side,
                    "amount": str(trade.amount),
                    "price": str(trade.price)
                }
                for trade in result.executed_trades
            ],
            message=result.message
        )
    except Exception as e:
        logger.error(f"Error executing hedge adjustments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute hedge adjustments"
        )

@router.post(
    "/close",
    response_model=HedgeExecutionResponse,
    dependencies=[Depends(verify_api_key)]
)
async def close_all_hedges(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Close all hedge positions."""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize components
        config = HedgeConfig(
            hedge_ratio_target=Decimal('0.0'),
            max_leverage=Decimal('3.0'),
            max_margin_usage=Decimal('0.8'),
            min_position_size={'BTC': Decimal('0.001'), 'ETH': Decimal('0.01')},
            max_position_size={'BTC': Decimal('1.0'), 'ETH': Decimal('10.0')},
            grid_bot_enabled=False
        )
        
        position_fetcher = ExchangePositionFetcher(exchange)
        order_executor = ExchangeOrderExecutor(exchange)
        hedge_analyzer = LLMHedgeAnalyzer(config, openai_api_key)
        execution_strategy = BasicExecutionStrategy()
        
        # Create hedge manager
        manager = HedgeManager(
            hedge_analyzer=hedge_analyzer,
            position_fetcher=position_fetcher,
            execution_strategy=execution_strategy,
            order_executor=order_executor,
            execute_hedge=True
        )
        
        # Close all hedges
        result = await manager.close_all_hedges()
        
        if result is None:
            return HedgeExecutionResponse(
                status="completed",
                executed_trades=[],
                message="No hedge positions to close"
            )
            
        return HedgeExecutionResponse(
            status="completed",
            executed_trades=[
                {
                    "asset": trade.asset,
                    "side": trade.side,
                    "amount": str(trade.amount),
                    "price": str(trade.price)
                }
                for trade in result.executed_trades
            ],
            message=result.message
        )
    except Exception as e:
        logger.error(f"Error closing hedge positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to close hedge positions"
        )