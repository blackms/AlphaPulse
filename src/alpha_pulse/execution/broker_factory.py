"""
Factory for creating broker instances based on trading mode.
"""
from typing import Optional, Dict, Any
from loguru import logger

from .broker_interface import BrokerInterface
from .paper_broker import PaperBroker
from .real_broker import RealBroker
from .recommendation_broker import RecommendationOnlyBroker
from .liquidity_aware_executor import LiquidityAwareExecutor
from alpha_pulse.services.liquidity_risk_service import LiquidityRiskService


class TradingMode:
    """Available trading modes."""
    REAL = "REAL"
    PAPER = "PAPER"
    RECOMMENDATION = "RECOMMENDATION"


def create_broker(
    trading_mode: str,
    exchange_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = False,
    initial_balance: float = 100000.0,
    enable_liquidity_checks: bool = True,
    liquidity_config: Optional[Dict[str, Any]] = None,
) -> BrokerInterface:
    """
    Create a broker instance based on trading mode with optional liquidity awareness.
    
    Args:
        trading_mode: One of REAL, PAPER, or RECOMMENDATION
        exchange_name: Required for REAL mode, one of 'binance' or 'bybit'
        api_key: Required for REAL mode
        api_secret: Required for REAL mode
        testnet: Use testnet for REAL mode
        initial_balance: Initial balance for PAPER mode
        enable_liquidity_checks: Enable liquidity risk assessment
        liquidity_config: Configuration for liquidity checks
        
    Returns:
        BrokerInterface implementation (optionally wrapped with liquidity checks)
        
    Raises:
        ValueError: If trading_mode is invalid or required parameters are missing
    """
    trading_mode = trading_mode.upper()
    
    # Create base broker
    base_broker = None
    
    if trading_mode == TradingMode.REAL:
        if not all([exchange_name, api_key, api_secret]):
            raise ValueError(
                "exchange_name, api_key, and api_secret are required for REAL mode"
            )
        
        exchange_name = exchange_name.lower()
        if exchange_name == "binance":
            logger.info("Creating Binance real broker")
            base_broker = RealBroker.create_binance(api_key, api_secret, testnet)
        elif exchange_name == "bybit":
            logger.info("Creating Bybit real broker")
            base_broker = RealBroker.create_bybit(api_key, api_secret, testnet)
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
            
    elif trading_mode == TradingMode.PAPER:
        logger.info("Creating paper trading broker")
        base_broker = PaperBroker(initial_balance=initial_balance)
        
    elif trading_mode == TradingMode.RECOMMENDATION:
        logger.info("Creating recommendation-only broker")
        base_broker = RecommendationOnlyBroker()
        
    else:
        raise ValueError(
            f"Invalid trading_mode: {trading_mode}. "
            f"Must be one of: {TradingMode.REAL}, {TradingMode.PAPER}, "
            f"or {TradingMode.RECOMMENDATION}"
        )
    
    # Wrap with liquidity checks if enabled
    if enable_liquidity_checks and trading_mode != TradingMode.RECOMMENDATION:
        try:
            logger.info("Wrapping broker with liquidity awareness")
            
            # Create liquidity risk service
            liquidity_service = LiquidityRiskService(
                exchange=base_broker,  # Use broker as exchange interface
                config=liquidity_config
            )
            
            # Wrap broker
            return LiquidityAwareExecutor(
                liquidity_service=liquidity_service,
                base_executor=base_broker,
                config=liquidity_config
            )
        except Exception as e:
            logger.warning(f"Failed to enable liquidity checks: {e}")
            logger.info("Continuing without liquidity awareness")
    
    return base_broker