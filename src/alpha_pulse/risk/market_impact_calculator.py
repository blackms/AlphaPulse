"""
Market impact calculator for pre-trade and post-trade analysis.

Provides comprehensive market impact estimation including temporary,
permanent impact and optimal execution strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize_scalar
from dataclasses import dataclass

from alpha_pulse.models.slippage_estimates import (
    MarketImpactEstimate, ExecutionStrategy, ImpactType,
    ExecutionPlan, OptimalExecutionParams, RealizedSlippage
)
from alpha_pulse.models.liquidity_metrics import LiquidityMetrics
from alpha_pulse.risk.slippage_models import SlippageModelEnsemble

logger = logging.getLogger(__name__)


class MarketImpactCalculator:
    """Calculates and analyzes market impact for trades."""
    
    def __init__(
        self,
        slippage_ensemble: Optional[SlippageModelEnsemble] = None,
        impact_half_life: float = 30.0,  # Minutes for temporary impact decay
        permanent_impact_factor: float = 0.5
    ):
        """Initialize market impact calculator."""
        self.slippage_ensemble = slippage_ensemble or SlippageModelEnsemble()
        self.impact_half_life = impact_half_life
        self.permanent_impact_factor = permanent_impact_factor
        
        # Cache for calculations
        self._impact_cache = {}
        
    def estimate_market_impact(
        self,
        symbol: str,
        order_size: float,
        side: str,
        liquidity_metrics: LiquidityMetrics,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.VWAP,
        execution_duration: float = 60.0,  # Minutes
        urgency: str = "medium"
    ) -> MarketImpactEstimate:
        """Estimate market impact for a potential trade."""
        logger.info(f"Estimating market impact for {symbol} order size {order_size}")
        
        # Prepare market data
        market_data = self._prepare_market_data(liquidity_metrics)
        market_data['symbol'] = symbol
        market_data['side'] = side
        
        # Execution parameters
        execution_params = {
            'duration_minutes': execution_duration,
            'urgency': self._urgency_to_numeric(urgency),
            'strategy': execution_strategy
        }
        
        # Get slippage estimate
        slippage_estimate = self.slippage_ensemble.estimate_slippage(
            order_size, market_data, execution_params
        )
        
        # Break down impact components
        impact_breakdown = self._calculate_impact_components(
            order_size,
            liquidity_metrics,
            execution_duration,
            slippage_estimate.expected_slippage_bps
        )
        
        # Calculate dollar impact
        price = market_data.get('price', 100)
        total_impact_dollars = order_size * price * slippage_estimate.expected_slippage_bps / 10000
        
        # Create impact estimate
        return MarketImpactEstimate(
            symbol=symbol,
            order_size=order_size,
            side=side,
            spread_cost=impact_breakdown['spread_cost'],
            temporary_impact=impact_breakdown['temporary_impact'],
            permanent_impact=impact_breakdown['permanent_impact'],
            timing_risk=impact_breakdown['timing_risk'],
            total_impact_bps=slippage_estimate.expected_slippage_bps,
            total_impact_dollars=total_impact_dollars,
            execution_strategy=execution_strategy,
            execution_duration=execution_duration,
            participation_rate=slippage_estimate.recommended_participation,
            impact_std_dev=impact_breakdown.get('impact_std_dev'),
            confidence_interval_95=(
                slippage_estimate.expected_slippage_bps * 0.8,
                slippage_estimate.worst_case_slippage_bps
            ),
            model_used=slippage_estimate.models_used[0] if slippage_estimate.models_used else None,
            model_confidence=slippage_estimate.estimation_confidence
        )
    
    def _prepare_market_data(self, liquidity_metrics: LiquidityMetrics) -> Dict[str, Any]:
        """Prepare market data dictionary from liquidity metrics."""
        return {
            'spread_bps': liquidity_metrics.quoted_spread or 10,
            'average_daily_volume': liquidity_metrics.daily_volume or 1e6,
            'volatility': 0.02,  # Would get from price data
            'amihud_illiquidity': liquidity_metrics.amihud_illiquidity or 0.1,
            'liquidity_score': liquidity_metrics.liquidity_score or 50,
            'depth_imbalance': liquidity_metrics.depth_imbalance or 0,
            'price': 100  # Would get from market data
        }
    
    def _urgency_to_numeric(self, urgency: str) -> float:
        """Convert urgency string to numeric value."""
        urgency_map = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5,
            'critical': 2.0
        }
        return urgency_map.get(urgency, 1.0)
    
    def _calculate_impact_components(
        self,
        order_size: float,
        liquidity_metrics: LiquidityMetrics,
        execution_duration: float,
        total_impact_bps: float
    ) -> Dict[str, float]:
        """Break down total impact into components."""
        # Spread cost (half-spread)
        spread_cost = (liquidity_metrics.quoted_spread or 10) / 2
        
        # Estimate temporary vs permanent split
        # Longer execution = more temporary, less permanent
        temp_ratio = min(0.8, 0.5 + execution_duration / 240)  # Cap at 80% temporary
        
        # Remaining impact after spread
        remaining_impact = max(0, total_impact_bps - spread_cost)
        
        # Temporary impact (decays over time)
        temporary_impact = remaining_impact * temp_ratio
        
        # Permanent impact (price movement)
        permanent_impact = remaining_impact * (1 - temp_ratio)
        
        # Timing risk (volatility cost)
        volatility = 0.02  # Annual volatility
        daily_vol = volatility / np.sqrt(252)
        execution_days = execution_duration / 390  # Trading minutes per day
        timing_risk = daily_vol * np.sqrt(execution_days) * 10000 * 0.5
        
        # Impact standard deviation
        impact_std_dev = temporary_impact * 0.3  # 30% uncertainty
        
        return {
            'spread_cost': spread_cost,
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'timing_risk': timing_risk,
            'impact_std_dev': impact_std_dev
        }
    
    def create_execution_plan(
        self,
        symbol: str,
        order_size: float,
        side: str,
        market_impact: MarketImpactEstimate,
        start_time: datetime,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """Create detailed execution plan based on impact estimate."""
        logger.info(f"Creating execution plan for {symbol}")
        
        # Default constraints
        if constraints is None:
            constraints = {}
        
        min_slice_size = constraints.get('min_slice_size', order_size * 0.01)
        max_participation = constraints.get('max_participation', 0.3)
        
        # Determine number of slices based on strategy
        if market_impact.execution_strategy == ExecutionStrategy.TWAP:
            n_slices = int(market_impact.execution_duration / 5)  # 5-minute slices
        elif market_impact.execution_strategy == ExecutionStrategy.VWAP:
            n_slices = int(market_impact.execution_duration / 10)  # 10-minute slices
        else:
            n_slices = max(1, int(market_impact.execution_duration / 15))
        
        n_slices = max(1, min(n_slices, int(order_size / min_slice_size)))
        
        # Generate time slices
        time_slices = []
        slice_duration = market_impact.execution_duration / n_slices
        
        for i in range(n_slices):
            slice_start = start_time + timedelta(minutes=i * slice_duration)
            slice_end = start_time + timedelta(minutes=(i + 1) * slice_duration)
            time_slices.append((slice_start, slice_end))
        
        # Determine slice sizes based on strategy
        if market_impact.execution_strategy == ExecutionStrategy.TWAP:
            # Equal slices
            slice_sizes = [order_size / n_slices] * n_slices
        elif market_impact.execution_strategy == ExecutionStrategy.VWAP:
            # Volume-weighted (simplified - would use actual volume curve)
            weights = self._get_vwap_weights(n_slices)
            slice_sizes = [order_size * w for w in weights]
        elif market_impact.execution_strategy == ExecutionStrategy.IS:
            # Front-loaded for implementation shortfall
            weights = self._get_is_weights(n_slices)
            slice_sizes = [order_size * w for w in weights]
        else:
            # Default to equal
            slice_sizes = [order_size / n_slices] * n_slices
        
        # Calculate participation rates
        slice_participation_rates = [
            min(max_participation, size / (order_size * 0.1))  # Simplified
            for size in slice_sizes
        ]
        
        # Estimate costs for each slice
        estimated_costs = []
        cumulative_impact = 0
        
        for i, size in enumerate(slice_sizes):
            # Simplified cost model - would be more sophisticated
            slice_impact = market_impact.total_impact_bps * (size / order_size)
            
            # Add cumulative permanent impact
            cumulative_impact += slice_impact * self.permanent_impact_factor
            
            slice_cost = (slice_impact + cumulative_impact) * size / order_size
            estimated_costs.append(slice_cost)
        
        total_estimated_cost = sum(estimated_costs)
        
        # Calculate execution risk
        execution_risk = market_impact.timing_risk * np.sqrt(n_slices)
        
        # Completion risk (simplified)
        completion_risk = 0.01 if n_slices <= 20 else 0.05
        
        # Benchmark prices
        arrival_price = 100  # Would get from market data
        expected_vwap = arrival_price * (1 + market_impact.permanent_impact / 10000)
        expected_twap = arrival_price * (1 + market_impact.total_impact_bps / 20000)
        
        return ExecutionPlan(
            order_id=f"plan_{datetime.now().timestamp()}",
            symbol=symbol,
            total_size=order_size,
            time_slices=time_slices,
            slice_sizes=slice_sizes,
            slice_participation_rates=slice_participation_rates,
            primary_strategy=market_impact.execution_strategy,
            adaptive_parameters={
                'min_participation': 0.01,
                'max_participation': max_participation,
                'urgency_factor': 1.0
            },
            estimated_costs=estimated_costs,
            total_estimated_cost=total_estimated_cost,
            execution_risk=execution_risk,
            completion_risk=completion_risk,
            arrival_price=arrival_price,
            expected_vwap=expected_vwap,
            expected_twap=expected_twap
        )
    
    def _get_vwap_weights(self, n_slices: int) -> List[float]:
        """Get VWAP-style weights for slices."""
        # U-shaped volume curve (simplified)
        x = np.linspace(0, 1, n_slices)
        weights = 1.5 - np.abs(x - 0.5)
        weights = weights / weights.sum()
        return weights.tolist()
    
    def _get_is_weights(self, n_slices: int) -> List[float]:
        """Get implementation shortfall weights (front-loaded)."""
        # Exponentially decreasing weights
        weights = np.exp(-np.linspace(0, 3, n_slices))
        weights = weights / weights.sum()
        return weights.tolist()
    
    def analyze_realized_slippage(
        self,
        order_id: str,
        symbol: str,
        execution_data: pd.DataFrame,
        market_data: pd.DataFrame,
        estimated_impact: MarketImpactEstimate
    ) -> RealizedSlippage:
        """Analyze actual slippage after execution."""
        logger.info(f"Analyzing realized slippage for order {order_id}")
        
        # Extract execution details
        execution_start = execution_data.index[0]
        execution_end = execution_data.index[-1]
        
        # Calculate prices
        arrival_price = market_data.loc[execution_start, 'price']
        avg_execution_price = (
            execution_data['price'] * execution_data['size']
        ).sum() / execution_data['size'].sum()
        
        # Get closing price (30 minutes after execution)
        close_time = execution_end + timedelta(minutes=30)
        if close_time in market_data.index:
            closing_price = market_data.loc[close_time, 'price']
        else:
            closing_price = market_data.iloc[-1]['price']
        
        # Calculate slippage components
        side_multiplier = 1 if execution_data['side'].iloc[0] == 'buy' else -1
        
        # Implementation shortfall
        implementation_shortfall = (
            (avg_execution_price - arrival_price) / arrival_price * 
            side_multiplier * 10000
        )
        
        # Realized spread
        mid_prices = (market_data['bid'] + market_data['ask']) / 2
        execution_mids = mid_prices.loc[execution_data.index]
        realized_spread = abs(
            (execution_data['price'] - execution_mids).mean() / 
            execution_mids.mean() * 10000
        )
        
        # Effective spread
        quoted_spreads = (
            (market_data['ask'] - market_data['bid']) / 
            mid_prices * 10000
        )
        effective_spread = quoted_spreads.loc[execution_data.index].mean()
        
        # Price improvement (negative if worse than expected)
        price_improvement = estimated_impact.total_impact_bps - implementation_shortfall
        
        # Market impact decomposition
        total_impact_bps = implementation_shortfall
        
        # Temporary impact (price reversion)
        reversion_10min = self._calculate_reversion(
            execution_end, 10, market_data, closing_price, arrival_price
        )
        reversion_30min = self._calculate_reversion(
            execution_end, 30, market_data, closing_price, arrival_price
        )
        
        temporary_impact = max(reversion_10min, reversion_30min)
        permanent_impact = total_impact_bps - temporary_impact
        
        # Execution quality metrics
        total_size = execution_data['size'].sum()
        participation_rate = total_size / market_data.loc[
            execution_start:execution_end, 'volume'
        ].sum()
        
        fill_rate = total_size / estimated_impact.order_size
        
        # Model accuracy
        model_accuracy = 1 - abs(
            total_impact_bps - estimated_impact.total_impact_bps
        ) / max(estimated_impact.total_impact_bps, 1)
        
        return RealizedSlippage(
            order_id=order_id,
            symbol=symbol,
            execution_start=execution_start,
            execution_end=execution_end,
            order_size=total_size,
            side=execution_data['side'].iloc[0],
            arrival_price=arrival_price,
            average_execution_price=avg_execution_price,
            closing_price=closing_price,
            implementation_shortfall=implementation_shortfall,
            realized_spread=realized_spread,
            effective_spread=effective_spread,
            price_improvement=price_improvement,
            temporary_impact_realized=temporary_impact,
            permanent_impact_realized=permanent_impact,
            total_impact_bps=total_impact_bps,
            participation_rate_achieved=participation_rate,
            fill_rate=fill_rate,
            reversion_10min=reversion_10min,
            reversion_30min=reversion_30min,
            estimated_slippage=estimated_impact.total_impact_bps,
            slippage_variance=total_impact_bps - estimated_impact.total_impact_bps,
            model_accuracy=model_accuracy
        )
    
    def _calculate_reversion(
        self,
        execution_end: datetime,
        minutes_after: int,
        market_data: pd.DataFrame,
        closing_price: float,
        arrival_price: float
    ) -> float:
        """Calculate price reversion after execution."""
        reversion_time = execution_end + timedelta(minutes=minutes_after)
        
        if reversion_time in market_data.index:
            reversion_price = market_data.loc[reversion_time, 'price']
        else:
            # Use closest available price
            future_data = market_data[market_data.index > execution_end]
            if len(future_data) > 0:
                reversion_price = future_data.iloc[0]['price']
            else:
                reversion_price = closing_price
        
        # Calculate reversion as percentage of total move
        total_move = closing_price - arrival_price
        price_at_reversion = reversion_price - arrival_price
        
        if abs(total_move) > 0:
            reversion = (total_move - price_at_reversion) / total_move
            return abs(reversion) * abs(total_move) / arrival_price * 10000
        
        return 0.0
    
    def optimize_execution_schedule(
        self,
        symbol: str,
        order_size: float,
        side: str,
        liquidity_metrics: LiquidityMetrics,
        risk_aversion: float = 1.0,
        max_duration: float = 390.0,  # Full trading day
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimalExecutionParams:
        """Optimize execution schedule using Almgren-Chriss framework."""
        logger.info(f"Optimizing execution schedule for {symbol}")
        
        # Market parameters
        daily_volume = liquidity_metrics.average_daily_volume or 1e6
        daily_volatility = 0.02  # Would calculate from price data
        spread = (liquidity_metrics.quoted_spread or 10) / 10000
        
        # Optimization parameters
        params = OptimalExecutionParams(
            symbol=symbol,
            order_size=order_size,
            risk_aversion=risk_aversion,
            min_participation_rate=0.01,
            max_participation_rate=0.30,
            max_duration_minutes=max_duration,
            daily_volatility=daily_volatility,
            daily_volume=daily_volume,
            bid_ask_spread=spread * 10000  # Back to bps
        )
        
        # Objective function: minimize expected cost + risk
        def objective(duration_minutes):
            # Calculate expected cost and variance
            T = duration_minutes / 390  # Fraction of day
            
            # Temporary impact cost (simplified)
            temp_cost = self._calculate_temp_impact_cost(
                order_size, daily_volume, T
            )
            
            # Permanent impact cost
            perm_cost = self._calculate_perm_impact_cost(
                order_size, daily_volume
            )
            
            # Timing risk (variance)
            variance = daily_volatility**2 * order_size**2 * T
            
            # Risk-adjusted cost
            total_cost = temp_cost + perm_cost + risk_aversion * np.sqrt(variance)
            
            return total_cost
        
        # Optimize duration
        result = minimize_scalar(
            objective,
            bounds=(5, max_duration),  # At least 5 minutes
            method='bounded'
        )
        
        optimal_duration = result.x
        
        # Calculate optimal trajectory
        n_slices = max(1, int(optimal_duration / 5))
        trajectory = self._calculate_optimal_trajectory(
            order_size, n_slices, risk_aversion
        )
        
        # Update params with results
        params.optimal_duration = optimal_duration
        params.optimal_trajectory = trajectory
        params.expected_cost = self._calculate_total_expected_cost(
            order_size, daily_volume, optimal_duration / 390, spread
        )
        params.cost_variance = daily_volatility**2 * order_size**2 * optimal_duration / 390
        
        return params
    
    def _calculate_temp_impact_cost(
        self,
        order_size: float,
        daily_volume: float,
        time_fraction: float
    ) -> float:
        """Calculate temporary impact cost."""
        # Simplified square-root model
        participation = order_size / (daily_volume * time_fraction)
        temp_impact = 0.1 * np.sqrt(participation)  # 10% coefficient
        return temp_impact * order_size
    
    def _calculate_perm_impact_cost(
        self,
        order_size: float,
        daily_volume: float
    ) -> float:
        """Calculate permanent impact cost."""
        participation = order_size / daily_volume
        perm_impact = 0.05 * participation  # 5% coefficient
        return perm_impact * order_size
    
    def _calculate_optimal_trajectory(
        self,
        order_size: float,
        n_slices: int,
        risk_aversion: float
    ) -> np.ndarray:
        """Calculate optimal execution trajectory."""
        # Simplified - linear trajectory
        # Full implementation would use Almgren-Chriss solution
        if risk_aversion > 1.5:
            # Risk-averse: front-loaded
            weights = np.exp(-np.linspace(0, 2, n_slices))
        elif risk_aversion < 0.5:
            # Risk-seeking: back-loaded
            weights = np.exp(np.linspace(-2, 0, n_slices))
        else:
            # Neutral: uniform
            weights = np.ones(n_slices)
        
        weights = weights / weights.sum()
        return order_size * weights
    
    def _calculate_total_expected_cost(
        self,
        order_size: float,
        daily_volume: float,
        time_fraction: float,
        spread: float
    ) -> float:
        """Calculate total expected execution cost."""
        # Spread cost
        spread_cost = spread * order_size
        
        # Temporary impact
        temp_cost = self._calculate_temp_impact_cost(
            order_size, daily_volume, time_fraction
        )
        
        # Permanent impact
        perm_cost = self._calculate_perm_impact_cost(
            order_size, daily_volume
        )
        
        return spread_cost + temp_cost + perm_cost