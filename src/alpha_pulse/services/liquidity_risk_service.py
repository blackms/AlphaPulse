"""
Liquidity risk management service.

Orchestrates liquidity analysis, slippage prediction, and execution optimization
for comprehensive liquidity risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dataclasses import dataclass

from alpha_pulse.models.liquidity_metrics import (
    LiquidityMetrics, IntradayLiquidity, VolumeProfile,
    LiquidityEvent, LiquidityTier
)
from alpha_pulse.models.slippage_estimates import (
    SlippageEstimate, MarketImpactEstimate, ExecutionPlan,
    RealizedSlippage, ExecutionStrategy, OptimalExecutionParams
)
from alpha_pulse.risk.liquidity_analyzer import LiquidityAnalyzer
from alpha_pulse.risk.slippage_models import SlippageModelEnsemble
from alpha_pulse.risk.market_impact_calculator import MarketImpactCalculator
from alpha_pulse.utils.liquidity_indicators import LiquidityIndicators
from alpha_pulse.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class LiquidityRiskAssessment:
    """Comprehensive liquidity risk assessment for a position."""
    symbol: str
    timestamp: datetime
    position_size: float
    position_value: float
    
    # Liquidity metrics
    liquidity_score: float
    liquidity_tier: LiquidityTier
    
    # Liquidation estimates
    immediate_liquidation_cost: float  # Cost to liquidate now
    gradual_liquidation_cost: float    # Cost to liquidate over time
    liquidation_time_days: float       # Estimated days to liquidate
    
    # Risk metrics
    liquidity_risk_score: float        # 0-100 scale
    concentration_risk: float          # Position as % of ADV
    market_impact_risk: float          # Expected permanent impact
    
    # Recommendations
    max_position_size: float           # Based on liquidity constraints
    recommended_holding_period: float  # Days
    risk_warnings: List[str]


class LiquidityRiskService:
    """Main service for liquidity risk management."""
    
    def __init__(
        self,
        liquidity_analyzer: Optional[LiquidityAnalyzer] = None,
        slippage_ensemble: Optional[SlippageModelEnsemble] = None,
        impact_calculator: Optional[MarketImpactCalculator] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        max_workers: int = 4
    ):
        """Initialize liquidity risk service."""
        self.liquidity_analyzer = liquidity_analyzer or LiquidityAnalyzer()
        self.slippage_ensemble = slippage_ensemble or SlippageModelEnsemble()
        self.impact_calculator = impact_calculator or MarketImpactCalculator(
            slippage_ensemble=self.slippage_ensemble
        )
        self.liquidity_indicators = LiquidityIndicators()
        self.metrics_collector = metrics_collector
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Cache for recent calculations
        self._liquidity_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Risk thresholds
        self.risk_thresholds = {
            'max_concentration': 0.20,      # Max 20% of ADV
            'max_impact_bps': 50,           # Max 50 bps impact
            'min_liquidity_score': 30,      # Min liquidity score
            'critical_spread_bps': 100      # Critical spread threshold
        }
        
    async def assess_position_liquidity_risk(
        self,
        symbol: str,
        position_size: float,
        market_data: pd.DataFrame,
        order_book_data: Optional[Dict[str, Any]] = None,
        liquidation_urgency: str = "medium"
    ) -> LiquidityRiskAssessment:
        """Assess liquidity risk for a position."""
        logger.info(f"Assessing liquidity risk for {symbol} position size {position_size}")
        
        # Get liquidity metrics
        liquidity_metrics = await self._get_or_calculate_liquidity_metrics(
            symbol, market_data, order_book_data
        )
        
        # Estimate liquidation costs
        immediate_cost = await self._estimate_immediate_liquidation_cost(
            symbol, position_size, liquidity_metrics, market_data
        )
        
        gradual_cost = await self._estimate_gradual_liquidation_cost(
            symbol, position_size, liquidity_metrics, market_data, liquidation_urgency
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            position_size, liquidity_metrics, immediate_cost, gradual_cost
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            symbol, position_size, liquidity_metrics, risk_metrics
        )
        
        # Track metrics
        if self.metrics_collector:
            self.metrics_collector.record_gauge(
                "liquidity_risk_score",
                risk_metrics['liquidity_risk_score'],
                {"symbol": symbol}
            )
        
        return LiquidityRiskAssessment(
            symbol=symbol,
            timestamp=datetime.now(),
            position_size=position_size,
            position_value=position_size * market_data['close'].iloc[-1],
            liquidity_score=liquidity_metrics.liquidity_score or 50,
            liquidity_tier=self._determine_liquidity_tier(liquidity_metrics),
            immediate_liquidation_cost=immediate_cost['total_cost_bps'],
            gradual_liquidation_cost=gradual_cost['total_cost_bps'],
            liquidation_time_days=gradual_cost['liquidation_days'],
            liquidity_risk_score=risk_metrics['liquidity_risk_score'],
            concentration_risk=risk_metrics['concentration_risk'],
            market_impact_risk=risk_metrics['market_impact_risk'],
            max_position_size=recommendations['max_position_size'],
            recommended_holding_period=recommendations['recommended_holding_period'],
            risk_warnings=recommendations['warnings']
        )
    
    async def _get_or_calculate_liquidity_metrics(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        order_book_data: Optional[Dict[str, Any]] = None
    ) -> LiquidityMetrics:
        """Get liquidity metrics from cache or calculate."""
        cache_key = f"{symbol}_liquidity"
        
        # Check cache
        if cache_key in self._liquidity_cache:
            cached_data, timestamp = self._liquidity_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self._cache_ttl):
                return cached_data
        
        # Calculate new metrics
        metrics = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.liquidity_analyzer.calculate_liquidity_metrics,
            symbol,
            market_data,
            order_book_data
        )
        
        # Update cache
        self._liquidity_cache[cache_key] = (metrics, datetime.now())
        
        return metrics
    
    async def _estimate_immediate_liquidation_cost(
        self,
        symbol: str,
        position_size: float,
        liquidity_metrics: LiquidityMetrics,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Estimate cost of immediate liquidation."""
        # Prepare market data for slippage model
        model_market_data = {
            'symbol': symbol,
            'side': 'sell',  # Assume liquidation is selling
            'price': market_data['close'].iloc[-1],
            'average_daily_volume': liquidity_metrics.average_daily_volume or 1e6,
            'spread_bps': liquidity_metrics.quoted_spread or 10,
            'volatility': market_data['close'].pct_change().std() * np.sqrt(252),
            'liquidity_score': liquidity_metrics.liquidity_score or 50
        }
        
        # Estimate with aggressive execution
        execution_params = {
            'duration_minutes': 30,  # 30 minutes for immediate
            'urgency': 2.0,  # High urgency
            'strategy': ExecutionStrategy.AGGRESSIVE
        }
        
        slippage_estimate = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.slippage_ensemble.estimate_slippage,
            position_size,
            model_market_data,
            execution_params
        )
        
        return {
            'total_cost_bps': slippage_estimate.expected_slippage_bps,
            'worst_case_bps': slippage_estimate.worst_case_slippage_bps,
            'spread_cost': liquidity_metrics.quoted_spread / 2 if liquidity_metrics.quoted_spread else 5,
            'market_impact': slippage_estimate.expected_slippage_bps - liquidity_metrics.quoted_spread / 2
        }
    
    async def _estimate_gradual_liquidation_cost(
        self,
        symbol: str,
        position_size: float,
        liquidity_metrics: LiquidityMetrics,
        market_data: pd.DataFrame,
        urgency: str = "medium"
    ) -> Dict[str, float]:
        """Estimate cost of gradual liquidation."""
        # Determine optimal liquidation period
        adv = liquidity_metrics.average_daily_volume or 1e6
        max_participation = 0.15  # 15% of volume
        
        min_days = position_size / (adv * max_participation)
        liquidation_days = max(1, min(10, min_days))  # 1-10 days
        
        # Prepare execution parameters
        execution_params = {
            'duration_minutes': liquidation_days * 390,  # Trading minutes per day
            'urgency': self._urgency_to_numeric(urgency),
            'strategy': ExecutionStrategy.VWAP
        }
        
        # Get market impact estimate
        impact_estimate = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.impact_calculator.estimate_market_impact,
            symbol,
            position_size,
            'sell',
            liquidity_metrics,
            ExecutionStrategy.VWAP,
            execution_params['duration_minutes'],
            urgency
        )
        
        return {
            'total_cost_bps': impact_estimate.total_impact_bps,
            'spread_cost': impact_estimate.spread_cost,
            'temporary_impact': impact_estimate.temporary_impact,
            'permanent_impact': impact_estimate.permanent_impact,
            'timing_risk': impact_estimate.timing_risk,
            'liquidation_days': liquidation_days,
            'daily_participation': position_size / (liquidation_days * adv)
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
    
    def _calculate_risk_metrics(
        self,
        position_size: float,
        liquidity_metrics: LiquidityMetrics,
        immediate_cost: Dict[str, float],
        gradual_cost: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate liquidity risk metrics."""
        # Concentration risk
        adv = liquidity_metrics.average_daily_volume or 1e6
        concentration_risk = position_size / adv
        
        # Market impact risk
        market_impact_risk = gradual_cost.get('permanent_impact', 0) / 10000
        
        # Liquidity risk score (0-100, higher is worse)
        risk_components = []
        
        # Concentration component (0-40 points)
        concentration_score = min(40, concentration_risk / self.risk_thresholds['max_concentration'] * 40)
        risk_components.append(concentration_score)
        
        # Impact component (0-30 points)
        impact_score = min(30, immediate_cost['total_cost_bps'] / self.risk_thresholds['max_impact_bps'] * 30)
        risk_components.append(impact_score)
        
        # Liquidity score component (0-30 points)
        if liquidity_metrics.liquidity_score:
            liquidity_score_inv = max(0, 1 - liquidity_metrics.liquidity_score / 100) * 30
        else:
            liquidity_score_inv = 15  # Default medium risk
        risk_components.append(liquidity_score_inv)
        
        liquidity_risk_score = sum(risk_components)
        
        return {
            'liquidity_risk_score': liquidity_risk_score,
            'concentration_risk': concentration_risk,
            'market_impact_risk': market_impact_risk,
            'immediate_impact_bps': immediate_cost['total_cost_bps'],
            'gradual_impact_bps': gradual_cost['total_cost_bps']
        }
    
    def _determine_liquidity_tier(self, liquidity_metrics: LiquidityMetrics) -> LiquidityTier:
        """Determine liquidity tier based on metrics."""
        score = liquidity_metrics.liquidity_score or 50
        
        if score >= 80:
            return LiquidityTier.ULTRA_LIQUID
        elif score >= 60:
            return LiquidityTier.HIGHLY_LIQUID
        elif score >= 40:
            return LiquidityTier.LIQUID
        elif score >= 20:
            return LiquidityTier.MODERATELY_LIQUID
        else:
            return LiquidityTier.ILLIQUID
    
    def _generate_recommendations(
        self,
        symbol: str,
        position_size: float,
        liquidity_metrics: LiquidityMetrics,
        risk_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate liquidity risk recommendations."""
        warnings = []
        adv = liquidity_metrics.average_daily_volume or 1e6
        
        # Maximum position size recommendation
        max_position_size = adv * self.risk_thresholds['max_concentration']
        
        # Recommended holding period based on liquidation time
        if risk_metrics['concentration_risk'] > 0.1:
            recommended_holding_period = max(5, risk_metrics['concentration_risk'] * 20)
        else:
            recommended_holding_period = 1  # Can liquidate in a day
        
        # Generate warnings
        if risk_metrics['concentration_risk'] > self.risk_thresholds['max_concentration']:
            warnings.append(
                f"Position exceeds {self.risk_thresholds['max_concentration']*100}% "
                f"of average daily volume"
            )
        
        if risk_metrics['immediate_impact_bps'] > self.risk_thresholds['max_impact_bps']:
            warnings.append(
                f"Immediate liquidation would incur {risk_metrics['immediate_impact_bps']:.1f} bps "
                f"in slippage"
            )
        
        if liquidity_metrics.liquidity_score and \
           liquidity_metrics.liquidity_score < self.risk_thresholds['min_liquidity_score']:
            warnings.append(
                f"Low liquidity score ({liquidity_metrics.liquidity_score:.0f}) "
                f"indicates potential execution difficulties"
            )
        
        if liquidity_metrics.quoted_spread and \
           liquidity_metrics.quoted_spread > self.risk_thresholds['critical_spread_bps']:
            warnings.append(
                f"Wide spread ({liquidity_metrics.quoted_spread:.1f} bps) "
                f"will increase transaction costs"
            )
        
        return {
            'max_position_size': max_position_size,
            'recommended_holding_period': recommended_holding_period,
            'warnings': warnings
        }
    
    async def create_optimal_execution_plan(
        self,
        symbol: str,
        order_size: float,
        side: str,
        market_data: pd.DataFrame,
        constraints: Optional[Dict[str, Any]] = None,
        risk_aversion: float = 1.0
    ) -> ExecutionPlan:
        """Create optimal execution plan for an order."""
        logger.info(f"Creating optimal execution plan for {symbol} order size {order_size}")
        
        # Get liquidity metrics
        liquidity_metrics = await self._get_or_calculate_liquidity_metrics(
            symbol, market_data
        )
        
        # Optimize execution parameters
        optimal_params = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.impact_calculator.optimize_execution_schedule,
            symbol,
            order_size,
            side,
            liquidity_metrics,
            risk_aversion,
            390.0,  # Max duration (full day)
            constraints
        )
        
        # Get market impact estimate
        impact_estimate = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.impact_calculator.estimate_market_impact,
            symbol,
            order_size,
            side,
            liquidity_metrics,
            ExecutionStrategy.ADAPTIVE,
            optimal_params.optimal_duration,
            "medium"
        )
        
        # Create execution plan
        start_time = datetime.now()
        execution_plan = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.impact_calculator.create_execution_plan,
            symbol,
            order_size,
            side,
            impact_estimate,
            start_time,
            constraints
        )
        
        # Track metrics
        if self.metrics_collector:
            self.metrics_collector.record_histogram(
                "optimal_execution_duration",
                optimal_params.optimal_duration,
                {"symbol": symbol, "side": side}
            )
        
        return execution_plan
    
    async def analyze_execution_quality(
        self,
        order_id: str,
        symbol: str,
        execution_data: pd.DataFrame,
        market_data: pd.DataFrame,
        estimated_impact: MarketImpactEstimate
    ) -> RealizedSlippage:
        """Analyze execution quality post-trade."""
        logger.info(f"Analyzing execution quality for order {order_id}")
        
        # Calculate realized slippage
        realized_slippage = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.impact_calculator.analyze_realized_slippage,
            order_id,
            symbol,
            execution_data,
            market_data,
            estimated_impact
        )
        
        # Track metrics
        if self.metrics_collector:
            self.metrics_collector.record_histogram(
                "realized_slippage_bps",
                realized_slippage.total_impact_bps,
                {"symbol": symbol}
            )
            
            self.metrics_collector.record_gauge(
                "model_accuracy",
                realized_slippage.model_accuracy,
                {"symbol": symbol}
            )
        
        return realized_slippage
    
    async def monitor_intraday_liquidity(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame],
        alert_threshold: float = 0.3
    ) -> Dict[str, IntradayLiquidity]:
        """Monitor intraday liquidity for multiple symbols."""
        logger.info(f"Monitoring intraday liquidity for {len(symbols)} symbols")
        
        # Analyze liquidity for each symbol concurrently
        tasks = []
        for symbol in symbols:
            if symbol in market_data:
                task = asyncio.create_task(
                    self._analyze_symbol_intraday_liquidity(
                        symbol, market_data[symbol]
                    )
                )
                tasks.append((symbol, task))
        
        # Collect results
        results = {}
        alerts = []
        
        for symbol, task in tasks:
            try:
                intraday_liquidity = await task
                results[symbol] = intraday_liquidity
                
                # Check for alerts
                if intraday_liquidity.liquidity_factor < alert_threshold:
                    alerts.append({
                        'symbol': symbol,
                        'liquidity_factor': intraday_liquidity.liquidity_factor,
                        'reason': 'Low intraday liquidity detected'
                    })
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol} intraday liquidity: {e}")
        
        # Send alerts if needed
        if alerts and self.metrics_collector:
            for alert in alerts:
                self.metrics_collector.record_counter(
                    "liquidity_alerts",
                    1,
                    {"symbol": alert['symbol'], "reason": alert['reason']}
                )
        
        return results
    
    async def _analyze_symbol_intraday_liquidity(
        self,
        symbol: str,
        intraday_data: pd.DataFrame
    ) -> IntradayLiquidity:
        """Analyze intraday liquidity for a single symbol."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.liquidity_analyzer.analyze_intraday_liquidity,
            symbol,
            intraday_data
        )
    
    def calculate_portfolio_liquidity_risk(
        self,
        positions: Dict[str, float],
        liquidity_metrics: Dict[str, LiquidityMetrics],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Calculate portfolio-level liquidity risk."""
        logger.info("Calculating portfolio liquidity risk")
        
        total_risk = 0
        portfolio_value = sum(positions.values())
        
        # Individual position risks
        position_risks = {}
        for symbol, position_value in positions.items():
            if symbol not in liquidity_metrics:
                continue
            
            metrics = liquidity_metrics[symbol]
            adv_value = metrics.average_daily_volume * 100  # Assume $100 avg price
            concentration = position_value / adv_value if adv_value > 0 else 1.0
            
            # Simple liquidity risk score
            risk_score = concentration * (100 - (metrics.liquidity_score or 50)) / 100
            position_risks[symbol] = risk_score
            total_risk += risk_score * (position_value / portfolio_value)
        
        # Correlation adjustment (if provided)
        if correlation_matrix is not None:
            # Higher correlation = higher joint liquidation risk
            avg_correlation = correlation_matrix.values.mean()
            correlation_adjustment = 1 + avg_correlation * 0.5
            total_risk *= correlation_adjustment
        
        # Calculate time to liquidate portfolio
        total_liquidation_days = 0
        for symbol, position_value in positions.items():
            if symbol in liquidity_metrics:
                metrics = liquidity_metrics[symbol]
                adv_value = metrics.average_daily_volume * 100
                days_to_liquidate = position_value / (adv_value * 0.15)  # 15% participation
                total_liquidation_days = max(total_liquidation_days, days_to_liquidate)
        
        return {
            'portfolio_liquidity_risk': total_risk * 100,  # 0-100 scale
            'position_risks': position_risks,
            'liquidation_days': total_liquidation_days,
            'concentration_herfindahl': sum(
                (pos / portfolio_value) ** 2 for pos in positions.values()
            )
        }
    
    async def stress_test_liquidity(
        self,
        positions: Dict[str, float],
        scenarios: List[Dict[str, Any]],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """Stress test portfolio liquidity under various scenarios."""
        logger.info(f"Running liquidity stress tests for {len(scenarios)} scenarios")
        
        results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', 'unnamed')
            logger.info(f"Running scenario: {scenario_name}")
            
            # Apply scenario shocks
            shocked_metrics = await self._apply_liquidity_shocks(
                positions, market_data, scenario
            )
            
            # Calculate liquidation costs under stress
            scenario_costs = {}
            total_cost = 0
            
            for symbol, position_value in positions.items():
                if symbol in shocked_metrics:
                    cost = await self._estimate_stressed_liquidation_cost(
                        symbol,
                        position_value,
                        shocked_metrics[symbol],
                        scenario
                    )
                    scenario_costs[symbol] = cost
                    total_cost += cost * position_value
            
            results[scenario_name] = {
                'total_cost_bps': total_cost / sum(positions.values()) * 10000,
                'position_costs': scenario_costs,
                'worst_position': max(scenario_costs.items(), key=lambda x: x[1])[0],
                'scenario_severity': scenario.get('severity', 'medium')
            }
        
        return results
    
    async def _apply_liquidity_shocks(
        self,
        positions: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        scenario: Dict[str, Any]
    ) -> Dict[str, LiquidityMetrics]:
        """Apply scenario shocks to liquidity metrics."""
        shocked_metrics = {}
        
        # Scenario parameters
        spread_multiplier = scenario.get('spread_multiplier', 2.0)
        volume_multiplier = scenario.get('volume_multiplier', 0.5)
        depth_reduction = scenario.get('depth_reduction', 0.7)
        
        for symbol in positions:
            if symbol not in market_data:
                continue
            
            # Get base metrics
            base_metrics = await self._get_or_calculate_liquidity_metrics(
                symbol, market_data[symbol]
            )
            
            # Apply shocks
            shocked = LiquidityMetrics(
                symbol=symbol,
                timestamp=datetime.now(),
                quoted_spread=(base_metrics.quoted_spread or 10) * spread_multiplier,
                effective_spread=(base_metrics.effective_spread or 15) * spread_multiplier,
                daily_volume=(base_metrics.daily_volume or 1e6) * volume_multiplier,
                average_daily_volume=(base_metrics.average_daily_volume or 1e6) * volume_multiplier,
                amihud_illiquidity=(base_metrics.amihud_illiquidity or 0.1) / volume_multiplier,
                liquidity_score=max(0, (base_metrics.liquidity_score or 50) - 30)
            )
            
            shocked_metrics[symbol] = shocked
        
        return shocked_metrics
    
    async def _estimate_stressed_liquidation_cost(
        self,
        symbol: str,
        position_value: float,
        shocked_metrics: LiquidityMetrics,
        scenario: Dict[str, Any]
    ) -> float:
        """Estimate liquidation cost under stressed conditions."""
        # Use impact calculator with stressed parameters
        impact_estimate = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.impact_calculator.estimate_market_impact,
            symbol,
            position_value / 100,  # Convert to shares (assume $100 price)
            'sell',
            shocked_metrics,
            ExecutionStrategy.AGGRESSIVE,
            60.0,  # 1 hour liquidation
            "high"
        )
        
        # Add stress premium
        stress_premium = scenario.get('stress_premium', 1.5)
        
        return impact_estimate.total_impact_bps * stress_premium
    
    def get_liquidity_risk_summary(
        self,
        positions: Dict[str, float],
        assessments: Dict[str, LiquidityRiskAssessment]
    ) -> Dict[str, Any]:
        """Get portfolio liquidity risk summary."""
        if not assessments:
            return {
                'status': 'no_data',
                'total_positions': len(positions),
                'assessed_positions': 0
            }
        
        # Aggregate metrics
        total_value = sum(positions.values())
        weighted_risk_score = sum(
            assessments[symbol].liquidity_risk_score * positions[symbol] / total_value
            for symbol in assessments
            if symbol in positions
        )
        
        # Find problematic positions
        high_risk_positions = [
            symbol for symbol, assessment in assessments.items()
            if assessment.liquidity_risk_score > 70
        ]
        
        illiquid_positions = [
            symbol for symbol, assessment in assessments.items()
            if assessment.liquidity_tier == LiquidityTier.ILLIQUID
        ]
        
        # Calculate aggregate liquidation metrics
        max_liquidation_days = max(
            assessment.liquidation_time_days
            for assessment in assessments.values()
        )
        
        avg_immediate_cost = np.mean([
            assessment.immediate_liquidation_cost
            for assessment in assessments.values()
        ])
        
        return {
            'portfolio_liquidity_risk_score': weighted_risk_score,
            'high_risk_positions': high_risk_positions,
            'illiquid_positions': illiquid_positions,
            'max_liquidation_days': max_liquidation_days,
            'avg_immediate_liquidation_cost_bps': avg_immediate_cost,
            'total_warnings': sum(
                len(assessment.risk_warnings)
                for assessment in assessments.values()
            ),
            'risk_distribution': {
                'low': len([a for a in assessments.values() if a.liquidity_risk_score < 30]),
                'medium': len([a for a in assessments.values() if 30 <= a.liquidity_risk_score < 70]),
                'high': len([a for a in assessments.values() if a.liquidity_risk_score >= 70])
            }
        }
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        logger.info("Liquidity risk service closed")