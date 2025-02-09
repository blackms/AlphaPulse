"""
Risk management for hedging strategies.
"""
from decimal import Decimal
from typing import Dict, Optional, List
from loguru import logger

from ..common.interfaces import RiskManager
from ..common.types import MarketState, PositionState


class HedgeManager:
    """Coordinates hedge analysis, position management, and execution."""
    
    def __init__(
        self,
        hedge_analyzer,
        position_fetcher,
        execution_strategy,
        order_executor,
        execute_hedge: bool = False
    ):
        """
        Initialize hedge manager.
        
        Args:
            hedge_analyzer: Component for analyzing and recommending hedges
            position_fetcher: Component for fetching current positions
            execution_strategy: Component for executing hedge adjustments
            order_executor: Component for placing orders
            execute_hedge: Whether to execute hedge trades (False for analysis only)
        """
        self.hedge_analyzer = hedge_analyzer
        self.position_fetcher = position_fetcher
        self.execution_strategy = execution_strategy
        self.order_executor = order_executor
        self.execute_hedge = execute_hedge
        
        logger.info(
            f"Initialized hedge manager "
            f"(execute_hedge: {execute_hedge})"
        )
    
    async def manage_hedge(self) -> None:
        """Analyze and adjust hedge positions."""
        try:
            # Get current positions
            spot_positions = await self.position_fetcher.get_spot_positions()
            futures_positions = await self.position_fetcher.get_futures_positions()
            
            # Get hedge recommendations
            recommendation = await self.hedge_analyzer.analyze(
                spot_positions,
                futures_positions
            )
            
            if not self.execute_hedge:
                logger.info("Skipping hedge execution (dry run)")
                return
                
            # Execute recommended adjustments
            if recommendation.adjustments:
                for adj in recommendation.adjustments:
                    await self.execution_strategy.execute_adjustment(
                        adjustment=adj,
                        order_executor=self.order_executor
                    )
            else:
                logger.info("No hedge adjustments needed")
                
        except Exception as e:
            logger.error(f"Error managing hedge: {str(e)}")
            raise
    
    async def close_all_hedges(self) -> None:
        """Close all hedge positions."""
        try:
            if not self.execute_hedge:
                logger.info("Skipping hedge closure (dry run)")
                return
                
            # Get current futures positions
            futures_positions = await self.position_fetcher.get_futures_positions()
            
            # Close each position
            for pos in futures_positions:
                await self.order_executor.close_position(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    side=pos.side
                )
                logger.info(f"Closed hedge position for {pos.symbol}")
                
        except Exception as e:
            logger.error(f"Error closing hedges: {str(e)}")
            raise


class GridRiskManager(RiskManager):
    """Risk management implementation for grid strategy."""
    
    def __init__(
        self,
        max_position_size: Decimal,
        max_drawdown: Decimal,
        base_stop_loss: Decimal,
        var_limit: Decimal,
        risk_free_rate: Decimal = Decimal('0.02')  # 2% annual
    ):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size
            max_drawdown: Maximum allowed drawdown
            base_stop_loss: Base stop loss percentage
            var_limit: Value at Risk limit
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.base_stop_loss = base_stop_loss
        self.var_limit = var_limit
        self.risk_free_rate = risk_free_rate
        
        logger.info(
            f"Initialized risk manager with max size {max_position_size}, "
            f"max drawdown {max_drawdown:.2%}"
        )
        
    def calculate_position_size(
        self,
        price: Decimal,
        volatility: Decimal,
        available_margin: Decimal
    ) -> Decimal:
        """
        Calculate safe position size based on risk parameters.
        
        Args:
            price: Current price
            volatility: Current volatility
            available_margin: Available margin
            
        Returns:
            Safe position size
        """
        try:
            # Start with maximum size
            size = self.max_position_size
            
            # Adjust for volatility
            if volatility > Decimal('0'):
                # Reduce size as volatility increases
                volatility_factor = Decimal('1') / (volatility * Decimal('10'))
                size = min(size, self.max_position_size * volatility_factor)
                
            # Adjust for available margin
            margin_limit = available_margin * Decimal('0.8')  # Use 80% of margin
            size_from_margin = margin_limit / price
            size = min(size, size_from_margin)
            
            # Apply absolute maximum
            size = min(size, self.max_position_size)
            
            logger.debug(
                f"Calculated position size {size:.8f} "
                f"(volatility: {volatility:.2%}, margin: {available_margin:.2f})"
            )
            
            return size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return Decimal('0')
            
    def validate_risk_limits(
        self,
        position: PositionState,
        market: MarketState
    ) -> bool:
        """
        Check if position meets risk limits.
        
        Args:
            position: Current position state
            market: Current market state
            
        Returns:
            True if position meets risk limits
        """
        try:
            # Check absolute position size
            total_exposure = abs(position.spot_quantity) + abs(position.futures_quantity)
            if total_exposure > self.max_position_size:
                logger.warning(
                    f"Position size {total_exposure:.8f} exceeds "
                    f"limit {self.max_position_size:.8f}"
                )
                return False
                
            # Check drawdown
            if position.unrealized_pnl < Decimal('0'):
                drawdown = abs(position.unrealized_pnl) / (
                    position.spot_quantity * market.current_price
                )
                if drawdown > self.max_drawdown:
                    logger.warning(
                        f"Drawdown {drawdown:.2%} exceeds "
                        f"limit {self.max_drawdown:.2%}"
                    )
                    return False
                    
            # Check VaR if volatility available
            if market.volatility > Decimal('0'):
                position_value = position.spot_quantity * market.current_price
                var_estimate = position_value * market.volatility * Decimal('2.33')
                if var_estimate > self.var_limit:
                    logger.warning(
                        f"VaR {var_estimate:.2f} exceeds "
                        f"limit {self.var_limit:.2f}"
                    )
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating risk limits: {str(e)}")
            return False
            
    def calculate_stop_loss(
        self,
        position: PositionState,
        market: MarketState
    ) -> Decimal:
        """
        Calculate dynamic stop loss level.
        
        Args:
            position: Current position state
            market: Current market state
            
        Returns:
            Stop loss price level
        """
        try:
            # Start with base stop loss
            stop_loss_pct = self.base_stop_loss
            
            # Adjust for volatility
            if market.volatility > Decimal('0'):
                volatility_stop = market.volatility * Decimal('2')  # 2 standard deviations
                stop_loss_pct = max(stop_loss_pct, volatility_stop)
                
            # Adjust for drawdown
            if position.unrealized_pnl < Decimal('0'):
                current_drawdown = abs(position.unrealized_pnl) / (
                    position.spot_quantity * market.current_price
                )
                # Tighten stop loss if approaching max drawdown
                if current_drawdown > self.max_drawdown * Decimal('0.8'):
                    stop_loss_pct = min(stop_loss_pct, self.max_drawdown)
                    
            # Calculate price level
            if position.avg_entry_price > Decimal('0'):
                stop_price = position.avg_entry_price * (Decimal('1') - stop_loss_pct)
            else:
                stop_price = market.current_price * (Decimal('1') - stop_loss_pct)
                
            logger.info(
                f"Calculated stop loss at {stop_price:.2f} "
                f"({stop_loss_pct:.2%} from entry)"
            )
            
            return stop_price
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return Decimal('0')
            
    def _calculate_position_risk(
        self,
        position: PositionState,
        market: MarketState
    ) -> Dict[str, Decimal]:
        """
        Calculate comprehensive position risk metrics.
        
        Args:
            position: Current position state
            market: Current market state
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            position_value = position.spot_quantity * market.current_price
            
            # Calculate VaR
            var_95 = Decimal('0')
            if market.volatility > Decimal('0'):
                var_95 = position_value * market.volatility * Decimal('1.645')  # 95% confidence
                
            # Calculate expected shortfall
            es_95 = var_95 * Decimal('1.2')  # Approximate ES from VaR
            
            # Calculate leverage
            leverage = Decimal('1')
            if position.futures_quantity != Decimal('0'):
                leverage = (abs(position.spot_quantity) + abs(position.futures_quantity)) / \
                          abs(position.spot_quantity)
                          
            return {
                'position_value': position_value,
                'var_95': var_95,
                'es_95': es_95,
                'leverage': leverage,
                'current_drawdown': (
                    abs(position.unrealized_pnl) / position_value 
                    if position.unrealized_pnl < Decimal('0') else Decimal('0')
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {str(e)}")
            return {}