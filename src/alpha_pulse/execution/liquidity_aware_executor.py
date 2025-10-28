"""
Liquidity-aware order execution wrapper.

This module integrates liquidity risk assessment into the order execution flow,
ensuring orders are evaluated for market impact before placement.
"""
from typing import Dict, Optional, Any, Tuple
from decimal import Decimal
from loguru import logger

from alpha_pulse.services.liquidity_risk_service import LiquidityRiskService
from alpha_pulse.execution.broker_interface import OrderType, OrderSide


class LiquidityAwareExecutor:
    """Wraps order execution with liquidity risk assessment."""
    
    def __init__(
        self,
        liquidity_service: LiquidityRiskService,
        base_executor: Any,  # Real or paper broker
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize liquidity-aware executor.
        
        Args:
            liquidity_service: Liquidity risk service instance
            base_executor: Underlying executor (real or paper broker)
            config: Configuration parameters
        """
        self.liquidity_service = liquidity_service
        self.base_executor = base_executor
        self.config = config or {}
        
        # Configuration
        self.max_impact_threshold = self.config.get('max_impact_threshold', 0.01)  # 1% max impact
        self.use_adaptive_sizing = self.config.get('use_adaptive_sizing', True)
        self.force_optimal_execution = self.config.get('force_optimal_execution', False)
        
        logger.info(f"Initialized LiquidityAwareExecutor with {type(base_executor).__name__}")
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Place an order with liquidity assessment.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            amount: Order amount
            order_type: Order type
            price: Limit price (optional)
            
        Returns:
            Order result with liquidity metadata
        """
        try:
            # Get current market price for assessment
            if order_type == OrderType.MARKET:
                market_price = await self.base_executor.get_ticker_price(symbol)
            else:
                market_price = price or await self.base_executor.get_ticker_price(symbol)
            
            # Assess liquidity risk
            liquidity_assessment = await self.liquidity_service.assess_position_liquidity(
                symbol=symbol,
                position_size=float(amount),
                current_price=float(market_price)
            )
            
            # Check market impact
            market_impact = liquidity_assessment.get('market_impact', {})
            estimated_impact = market_impact.get('total_impact_pct', 0)
            
            logger.info(f"Liquidity assessment for {symbol}: impact={estimated_impact:.2%}")
            
            # Decide on execution strategy
            if estimated_impact > self.max_impact_threshold:
                logger.warning(
                    f"High market impact detected: {estimated_impact:.2%} > {self.max_impact_threshold:.2%}"
                )
                
                if self.use_adaptive_sizing:
                    # Reduce order size to stay within impact threshold
                    adjusted_amount = await self._calculate_optimal_size(
                        symbol, side, amount, market_price
                    )
                    logger.info(f"Adjusted order size from {amount} to {adjusted_amount}")
                    amount = adjusted_amount
                
                if self.force_optimal_execution:
                    # Use optimal execution strategy
                    return await self._execute_with_optimal_strategy(
                        symbol, side, amount, order_type, price, liquidity_assessment
                    )
            
            # Calculate expected slippage
            slippage_model = liquidity_assessment.get('slippage_models', {})
            expected_slippage = slippage_model.get('ensemble_prediction', 0.001)  # Default 0.1%
            
            # Execute order with base executor
            order_result = await self.base_executor.place_order(
                symbol=symbol,
                side=side,
                amount=amount,
                order_type=order_type,
                price=price
            )
            
            # Enhance result with liquidity metadata
            order_result['liquidity_metadata'] = {
                'market_impact': estimated_impact,
                'expected_slippage': expected_slippage,
                'liquidity_score': liquidity_assessment.get('liquidity_score', 0),
                'execution_recommendation': liquidity_assessment.get('recommendation', {})
            }
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error in liquidity-aware order placement: {e}")
            # Fall back to base executor
            return await self.base_executor.place_order(
                symbol=symbol,
                side=side,
                amount=amount,
                order_type=order_type,
                price=price
            )
    
    async def _calculate_optimal_size(
        self,
        symbol: str,
        side: OrderSide,
        requested_amount: Decimal,
        price: Decimal
    ) -> Decimal:
        """
        Calculate optimal order size given liquidity constraints.
        
        Args:
            symbol: Trading symbol
            side: Order side
            requested_amount: Requested order amount
            price: Current price
            
        Returns:
            Optimal order amount
        """
        try:
            # Binary search for optimal size
            min_size = Decimal('0')
            max_size = requested_amount
            optimal_size = requested_amount
            
            while max_size - min_size > requested_amount * Decimal('0.01'):  # 1% precision
                test_size = (min_size + max_size) / 2
                
                assessment = await self.liquidity_service.assess_position_liquidity(
                    symbol=symbol,
                    position_size=float(test_size),
                    current_price=float(price)
                )
                
                impact = assessment.get('market_impact', {}).get('total_impact_pct', 0)
                
                if impact <= self.max_impact_threshold:
                    optimal_size = test_size
                    min_size = test_size
                else:
                    max_size = test_size
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal size: {e}")
            return requested_amount
    
    async def _execute_with_optimal_strategy(
        self,
        symbol: str,
        side: OrderSide,
        amount: Decimal,
        order_type: OrderType,
        price: Optional[Decimal],
        liquidity_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute order using optimal execution strategy.
        
        Args:
            symbol: Trading symbol
            side: Order side
            amount: Order amount
            order_type: Order type
            price: Limit price
            liquidity_assessment: Liquidity assessment results
            
        Returns:
            Execution result
        """
        # Get execution recommendation
        recommendation = liquidity_assessment.get('recommendation', {})
        strategy = recommendation.get('execution_strategy', 'immediate')
        
        if strategy == 'split_order':
            # Split into smaller orders
            num_splits = recommendation.get('splits', 3)
            split_amount = amount / num_splits
            
            results = []
            for i in range(num_splits):
                result = await self.base_executor.place_order(
                    symbol=symbol,
                    side=side,
                    amount=split_amount,
                    order_type=order_type,
                    price=price
                )
                results.append(result)
                
                # Wait between orders if recommended
                if i < num_splits - 1:
                    import asyncio
                    await asyncio.sleep(recommendation.get('delay_seconds', 5))
            
            # Aggregate results
            return {
                'order_id': f"split_{results[0]['order_id']}",
                'status': 'executed',
                'execution_type': 'split_order',
                'sub_orders': results,
                'total_amount': amount,
                'liquidity_metadata': {
                    'strategy': strategy,
                    'splits': num_splits
                }
            }
        
        else:
            # Execute immediately with warning
            result = await self.base_executor.place_order(
                symbol=symbol,
                side=side,
                amount=amount,
                order_type=order_type,
                price=price
            )
            result['liquidity_warning'] = 'High market impact expected'
            return result
    
    # Delegate other methods to base executor
    def __getattr__(self, name):
        """Delegate unknown attributes to base executor."""
        return getattr(self.base_executor, name)