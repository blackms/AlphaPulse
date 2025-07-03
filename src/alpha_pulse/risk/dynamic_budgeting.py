"""
Dynamic risk budgeting system based on market regime detection.

Implements adaptive risk allocation, volatility targeting, and regime-based
portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
import cvxpy as cp

from alpha_pulse.models.risk_budget import (
    RiskBudget, RiskBudgetType, AllocationMethod,
    RiskAllocation, RiskBudgetRebalancing, VolatilityTarget,
    RiskBudgetSnapshot, RiskBudgetMonitoring
)
from alpha_pulse.models.market_regime import (
    MarketRegime, RegimeType, RegimeDetectionResult
)
from alpha_pulse.models.portfolio import Portfolio, Position
from alpha_pulse.risk.correlation_analyzer import CorrelationAnalyzer

logger = logging.getLogger(__name__)


class DynamicRiskBudgetManager:
    """Manages dynamic risk budgets based on market regimes."""
    
    def __init__(
        self,
        base_volatility_target: float = 0.15,
        max_leverage: float = 2.0,
        rebalancing_frequency: str = "daily",
        correlation_analyzer: Optional[CorrelationAnalyzer] = None
    ):
        """Initialize dynamic risk budget manager."""
        self.base_volatility_target = base_volatility_target
        self.max_leverage = max_leverage
        self.rebalancing_frequency = rebalancing_frequency
        
        self.correlation_analyzer = correlation_analyzer or CorrelationAnalyzer()
        
        # Risk budgets by type
        self.risk_budgets: Dict[RiskBudgetType, RiskBudget] = {}
        
        # Volatility targeting
        self.volatility_target = VolatilityTarget(
            target_id="main",
            base_target=base_volatility_target,
            current_target=base_volatility_target,
            max_leverage=max_leverage
        )
        
        # Monitoring
        self.monitoring = RiskBudgetMonitoring(
            budget_id="portfolio",
            monitoring_frequency="real-time"
        )
        
        # History tracking
        self.budget_history: List[RiskBudgetSnapshot] = []
        self.rebalancing_history: List[RiskBudgetRebalancing] = []
        
    def create_regime_based_budget(
        self,
        portfolio: Portfolio,
        regime_result: RegimeDetectionResult,
        market_data: pd.DataFrame
    ) -> RiskBudget:
        """Create risk budget based on current market regime."""
        logger.info(f"Creating risk budget for {regime_result.current_regime.regime_type} regime")
        
        # Get regime-specific parameters
        regime_params = self._get_regime_parameters(regime_result.current_regime)
        
        # Calculate risk metrics
        returns = market_data.pct_change().dropna()
        correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(returns)
        
        # Create base budget
        budget = RiskBudget(
            budget_id=f"regime_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            budget_type=RiskBudgetType.TOTAL,
            total_risk_limit=regime_params["risk_limit"],
            allocation_method=regime_params["allocation_method"],
            target_volatility=regime_params["target_volatility"],
            regime_multiplier=regime_params["regime_multiplier"],
            regime_type=regime_result.current_regime.regime_type.value
        )
        
        # Create allocations based on method
        if budget.allocation_method == AllocationMethod.RISK_PARITY:
            allocations = self._create_risk_parity_allocations(
                portfolio, returns, correlation_matrix
            )
        elif budget.allocation_method == AllocationMethod.REGIME_BASED:
            allocations = self._create_regime_based_allocations(
                portfolio, regime_result, returns
            )
        else:
            allocations = self._create_equal_weight_allocations(portfolio)
        
        # Apply regime adjustments
        allocations = self._apply_regime_adjustments(
            allocations, regime_result.current_regime
        )
        
        budget.allocations = allocations
        
        # Update current utilization
        budget.current_utilization = self._calculate_current_utilization(
            portfolio, allocations, returns
        )
        
        # Store budget
        self.risk_budgets[RiskBudgetType.TOTAL] = budget
        
        return budget
    
    def update_volatility_target(
        self,
        current_volatility: float,
        forecast_volatility: float,
        regime: MarketRegime
    ) -> float:
        """Update volatility target based on regime and market conditions."""
        # Get regime-adjusted target
        regime_target = self.volatility_target.get_regime_adjusted_target(
            regime.regime_type.value
        )
        
        # Update target
        self.volatility_target.current_target = regime_target
        self.volatility_target.realized_volatility = current_volatility
        self.volatility_target.forecast_volatility = forecast_volatility
        
        # Calculate target leverage
        target_leverage = self.volatility_target.calculate_target_leverage()
        
        # Smooth leverage changes
        if hasattr(self, '_leverage_history'):
            self._leverage_history.append(target_leverage)
            if len(self._leverage_history) > self.volatility_target.leverage_smoothing:
                self._leverage_history.pop(0)
            
            # Use EMA if configured
            if self.volatility_target.use_exponential_weighting:
                weights = np.exp(np.linspace(-2, 0, len(self._leverage_history)))
                weights /= weights.sum()
                smoothed_leverage = np.sum(
                    np.array(self._leverage_history) * weights
                )
            else:
                smoothed_leverage = np.mean(self._leverage_history)
            
            target_leverage = smoothed_leverage
        else:
            self._leverage_history = [target_leverage]
        
        # Update current leverage
        self.volatility_target.current_leverage = target_leverage
        self.volatility_target.last_updated = datetime.utcnow()
        
        logger.info(f"Updated volatility target: {regime_target:.1%}, "
                   f"Leverage: {target_leverage:.2f}")
        
        return target_leverage
    
    def check_rebalancing_triggers(
        self,
        portfolio: Portfolio,
        current_regime: MarketRegime,
        market_data: pd.DataFrame
    ) -> Optional[RiskBudgetRebalancing]:
        """Check if rebalancing is needed and create recommendation."""
        if not self.risk_budgets:
            return None
        
        budget = self.risk_budgets[RiskBudgetType.TOTAL]
        triggers = []
        
        # 1. Regime change trigger
        if budget.regime_type != current_regime.regime_type.value:
            triggers.append({
                "type": "regime_change",
                "old_regime": budget.regime_type,
                "new_regime": current_regime.regime_type.value
            })
        
        # 2. Risk budget breach trigger
        current_util = self._calculate_current_utilization(
            portfolio, budget.allocations, market_data.pct_change().dropna()
        )
        
        if current_util > budget.total_risk_limit * 1.1:  # 10% breach
            triggers.append({
                "type": "risk_breach",
                "current_utilization": current_util,
                "limit": budget.total_risk_limit
            })
        
        # 3. Volatility target deviation
        current_vol = self._calculate_portfolio_volatility(
            portfolio, market_data.pct_change().dropna()
        )
        
        if not budget.is_within_volatility_target():
            triggers.append({
                "type": "volatility_deviation",
                "current_volatility": current_vol,
                "target": budget.target_volatility
            })
        
        # 4. Allocation drift trigger
        drift_threshold = 0.10  # 10% drift
        for asset, allocation in budget.allocations.items():
            if abs(allocation.current_utilization - allocation.allocated_risk) > drift_threshold:
                triggers.append({
                    "type": "allocation_drift",
                    "asset": asset,
                    "drift": allocation.current_utilization - allocation.allocated_risk
                })
                break
        
        # Create rebalancing recommendation if triggers exist
        if triggers:
            return self._create_rebalancing_recommendation(
                budget, portfolio, current_regime, triggers
            )
        
        return None
    
    def execute_rebalancing(
        self,
        portfolio: Portfolio,
        rebalancing: RiskBudgetRebalancing
    ) -> Dict[str, float]:
        """Execute risk budget rebalancing."""
        logger.info(f"Executing rebalancing {rebalancing.rebalancing_id}")
        
        position_adjustments = {}
        
        # Calculate position changes
        for asset, target_allocation in rebalancing.target_allocations.items():
            current_allocation = rebalancing.current_allocations.get(asset, 0.0)
            change = target_allocation - current_allocation
            
            if abs(change) > 0.001:  # Minimum change threshold
                # Convert allocation change to position adjustment
                position_adjustment = self._allocation_to_position_adjustment(
                    portfolio, asset, change
                )
                position_adjustments[asset] = position_adjustment
        
        # Mark as executed
        rebalancing.executed = True
        rebalancing.execution_timestamp = datetime.utcnow()
        
        # Update monitoring
        self.monitoring.last_rebalance_time = datetime.utcnow()
        
        # Store in history
        self.rebalancing_history.append(rebalancing)
        
        return position_adjustments
    
    def optimize_risk_allocation(
        self,
        portfolio: Portfolio,
        regime: MarketRegime,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Optimize risk allocation using regime-aware optimization."""
        logger.info(f"Optimizing risk allocation for {regime.regime_type} regime")
        
        positions = list(portfolio.positions.values())
        n_assets = len(positions)
        
        if n_assets == 0:
            return {}
        
        # Get expected returns and covariance
        expected_returns = self._get_regime_expected_returns(positions, regime)
        cov_matrix = self._estimate_covariance_matrix(positions, regime)
        
        # Define optimization problem
        weights = cp.Variable(n_assets)
        
        # Objective: Maximize risk-adjusted returns
        portfolio_return = expected_returns @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Regime-specific risk aversion
        risk_aversion = self._get_regime_risk_aversion(regime)
        
        objective = cp.Maximize(
            portfolio_return - risk_aversion * portfolio_variance
        )
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= 0  # Long only (can be relaxed)
        ]
        
        # Add regime-specific constraints
        if regime.regime_type == RegimeType.CRISIS:
            # Tighter position limits in crisis
            constraints_list.append(weights <= 0.05)
        else:
            constraints_list.append(weights <= regime.max_position_size)
        
        # Add custom constraints
        if constraints:
            if "max_positions" in constraints:
                # Cardinality constraint (approximate)
                constraints_list.append(
                    cp.sum(weights > 0.001) <= constraints["max_positions"]
                )
            
            if "sector_limits" in constraints:
                # Sector concentration limits
                for sector, limit in constraints["sector_limits"].items():
                    sector_mask = [p.sector == sector for p in positions]
                    constraints_list.append(
                        cp.sum(weights[sector_mask]) <= limit
                    )
        
        # Solve optimization
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.OSQP)
            
            if problem.status == cp.OPTIMAL:
                # Convert to allocation dictionary
                allocations = {}
                for i, position in enumerate(positions):
                    if weights.value[i] > 0.001:  # Minimum allocation
                        allocations[position.symbol] = float(weights.value[i])
                
                return allocations
            else:
                logger.warning(f"Optimization failed: {problem.status}")
                return self._get_fallback_allocations(positions, regime)
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return self._get_fallback_allocations(positions, regime)
    
    def get_risk_budget_analytics(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get comprehensive analytics for risk budgeting."""
        if not self.risk_budgets:
            return {}
        
        budget = self.risk_budgets[RiskBudgetType.TOTAL]
        returns = market_data.pct_change().dropna()
        
        analytics = {
            "current_utilization": budget.current_utilization,
            "utilization_ratio": budget.current_utilization / budget.total_risk_limit,
            "concentration_ratio": budget.get_concentration_ratio(),
            "effective_allocations": budget.get_effective_allocations(),
            "volatility_metrics": {
                "realized": self._calculate_portfolio_volatility(portfolio, returns),
                "target": budget.target_volatility,
                "forecast": self.volatility_target.forecast_volatility
            },
            "leverage_metrics": {
                "current": self.volatility_target.current_leverage,
                "target": self.volatility_target.calculate_target_leverage(),
                "max": self.volatility_target.max_leverage
            },
            "regime_info": {
                "current_regime": budget.regime_type,
                "regime_multiplier": budget.regime_multiplier
            }
        }
        
        # Add allocation details
        allocation_details = []
        for asset, allocation in budget.allocations.items():
            allocation_details.append({
                "asset": asset,
                "allocated": allocation.allocated_risk,
                "utilized": allocation.current_utilization,
                "utilization_ratio": allocation.utilization_ratio,
                "within_limits": allocation.is_within_limits
            })
        
        analytics["allocation_details"] = sorted(
            allocation_details,
            key=lambda x: x["allocated"],
            reverse=True
        )
        
        # Add performance metrics
        if self.budget_history:
            recent_history = self.budget_history[-20:]  # Last 20 snapshots
            
            analytics["performance_metrics"] = {
                "avg_utilization": np.mean([s.total_risk_utilization for s in recent_history]),
                "utilization_stability": np.std([s.total_risk_utilization for s in recent_history]),
                "avg_sharpe": np.mean([s.sharpe_ratio for s in recent_history]),
                "rebalancing_frequency": len(self.rebalancing_history) / max(len(self.budget_history), 1)
            }
        
        return analytics
    
    def _get_regime_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get risk budget parameters for specific regime."""
        params = {
            RegimeType.BULL: {
                "risk_limit": self.base_volatility_target * 1.2,
                "target_volatility": self.base_volatility_target * 1.2,
                "allocation_method": AllocationMethod.VOLATILITY_WEIGHTED,
                "regime_multiplier": 1.2,
                "concentration_limit": 0.40
            },
            RegimeType.BEAR: {
                "risk_limit": self.base_volatility_target * 0.7,
                "target_volatility": self.base_volatility_target * 0.7,
                "allocation_method": AllocationMethod.RISK_PARITY,
                "regime_multiplier": 0.7,
                "concentration_limit": 0.25
            },
            RegimeType.SIDEWAYS: {
                "risk_limit": self.base_volatility_target,
                "target_volatility": self.base_volatility_target,
                "allocation_method": AllocationMethod.EQUAL_WEIGHT,
                "regime_multiplier": 1.0,
                "concentration_limit": 0.35
            },
            RegimeType.CRISIS: {
                "risk_limit": self.base_volatility_target * 0.4,
                "target_volatility": self.base_volatility_target * 0.4,
                "allocation_method": AllocationMethod.HIERARCHICAL,
                "regime_multiplier": 0.4,
                "concentration_limit": 0.15
            },
            RegimeType.RECOVERY: {
                "risk_limit": self.base_volatility_target * 0.8,
                "target_volatility": self.base_volatility_target * 0.8,
                "allocation_method": AllocationMethod.REGIME_BASED,
                "regime_multiplier": 0.8,
                "concentration_limit": 0.30
            }
        }
        
        return params.get(regime.regime_type, {
            "risk_limit": self.base_volatility_target,
            "target_volatility": self.base_volatility_target,
            "allocation_method": AllocationMethod.EQUAL_WEIGHT,
            "regime_multiplier": 1.0,
            "concentration_limit": 0.35
        })
    
    def _create_risk_parity_allocations(
        self,
        portfolio: Portfolio,
        returns: pd.DataFrame,
        correlation_matrix: Any
    ) -> Dict[str, RiskAllocation]:
        """Create risk parity allocations."""
        positions = list(portfolio.positions.values())
        allocations = {}
        
        # Calculate volatilities
        volatilities = returns.std() * np.sqrt(252)
        
        # Risk parity weights (inverse volatility)
        inv_vols = 1 / volatilities
        weights = inv_vols / inv_vols.sum()
        
        for i, position in enumerate(positions):
            symbol = position.symbol
            if symbol in weights.index:
                weight = weights[symbol]
                
                allocation = RiskAllocation(
                    asset_or_category=symbol,
                    allocated_risk=weight,
                    current_utilization=0.0,  # Will be updated
                    risk_contribution=weight * volatilities[symbol],
                    min_allocation=0.02,
                    max_allocation=0.30,
                    target_allocation=weight,
                    expected_volatility=volatilities[symbol],
                    realized_volatility=volatilities[symbol]
                )
                
                allocations[symbol] = allocation
        
        return allocations
    
    def _create_regime_based_allocations(
        self,
        portfolio: Portfolio,
        regime_result: RegimeDetectionResult,
        returns: pd.DataFrame
    ) -> Dict[str, RiskAllocation]:
        """Create allocations based on regime preferences."""
        allocations = {}
        regime = regime_result.current_regime
        
        # Get regime-preferred characteristics
        preferred_strategies = regime.preferred_strategies
        avoided_strategies = regime.avoided_strategies
        
        # Score each position
        position_scores = {}
        for position in portfolio.positions.values():
            score = self._score_position_for_regime(
                position, regime, preferred_strategies, avoided_strategies
            )
            position_scores[position.symbol] = score
        
        # Normalize scores to weights
        total_score = sum(position_scores.values())
        if total_score > 0:
            weights = {k: v/total_score for k, v in position_scores.items()}
        else:
            # Fallback to equal weight
            n = len(position_scores)
            weights = {k: 1/n for k in position_scores}
        
        # Create allocations
        for symbol, weight in weights.items():
            if symbol in returns.columns:
                vol = returns[symbol].std() * np.sqrt(252)
                
                allocation = RiskAllocation(
                    asset_or_category=symbol,
                    allocated_risk=weight,
                    current_utilization=0.0,
                    risk_contribution=weight * vol,
                    min_allocation=0.01,
                    max_allocation=regime.max_position_size,
                    target_allocation=weight,
                    expected_volatility=vol,
                    realized_volatility=vol
                )
                
                allocations[symbol] = allocation
        
        return allocations
    
    def _create_equal_weight_allocations(
        self,
        portfolio: Portfolio
    ) -> Dict[str, RiskAllocation]:
        """Create equal weight allocations."""
        allocations = {}
        n_positions = len(portfolio.positions)
        
        if n_positions == 0:
            return allocations
        
        weight = 1.0 / n_positions
        
        for position in portfolio.positions.values():
            allocation = RiskAllocation(
                asset_or_category=position.symbol,
                allocated_risk=weight,
                current_utilization=0.0,
                risk_contribution=weight,
                min_allocation=0.0,
                max_allocation=0.50,
                target_allocation=weight
            )
            
            allocations[position.symbol] = allocation
        
        return allocations
    
    def _apply_regime_adjustments(
        self,
        allocations: Dict[str, RiskAllocation],
        regime: MarketRegime
    ) -> Dict[str, RiskAllocation]:
        """Apply regime-specific adjustments to allocations."""
        # Adjust position limits
        for allocation in allocations.values():
            allocation.max_allocation = min(
                allocation.max_allocation,
                regime.max_position_size
            )
            
            # Apply stop loss multiplier
            allocation.metadata["stop_loss_multiplier"] = regime.stop_loss_multiplier
        
        # Reduce allocations in crisis
        if regime.regime_type == RegimeType.CRISIS:
            crisis_reduction = 0.5
            for allocation in allocations.values():
                allocation.allocated_risk *= crisis_reduction
                allocation.target_allocation *= crisis_reduction
        
        # Normalize to ensure sum = 1
        total_allocated = sum(a.allocated_risk for a in allocations.values())
        if total_allocated > 0:
            for allocation in allocations.values():
                allocation.allocated_risk /= total_allocated
                allocation.target_allocation /= total_allocated
        
        return allocations
    
    def _calculate_current_utilization(
        self,
        portfolio: Portfolio,
        allocations: Dict[str, RiskAllocation],
        returns: pd.DataFrame
    ) -> float:
        """Calculate current risk utilization."""
        total_risk = 0.0
        
        for position in portfolio.positions.values():
            if position.symbol in allocations and position.symbol in returns.columns:
                # Position weight
                position_value = position.quantity * position.current_price
                weight = position_value / portfolio.total_value
                
                # Position volatility
                vol = returns[position.symbol].std() * np.sqrt(252)
                
                # Risk contribution
                risk_contrib = weight * vol
                total_risk += risk_contrib
                
                # Update allocation
                allocations[position.symbol].current_utilization = weight
                allocations[position.symbol].risk_contribution = risk_contrib
                allocations[position.symbol].realized_volatility = vol
        
        return total_risk
    
    def _calculate_portfolio_volatility(
        self,
        portfolio: Portfolio,
        returns: pd.DataFrame
    ) -> float:
        """Calculate portfolio volatility."""
        weights = []
        symbols = []
        
        for position in portfolio.positions.values():
            if position.symbol in returns.columns:
                position_value = position.quantity * position.current_price
                weight = position_value / portfolio.total_value
                weights.append(weight)
                symbols.append(position.symbol)
        
        if not weights:
            return 0.0
        
        # Portfolio returns
        portfolio_returns = (returns[symbols] * weights).sum(axis=1)
        
        # Annualized volatility
        return portfolio_returns.std() * np.sqrt(252)
    
    def _create_rebalancing_recommendation(
        self,
        budget: RiskBudget,
        portfolio: Portfolio,
        new_regime: MarketRegime,
        triggers: List[Dict[str, Any]]
    ) -> RiskBudgetRebalancing:
        """Create rebalancing recommendation."""
        rebalancing_id = f"rebal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Get current allocations
        current_allocations = {
            asset: alloc.current_utilization
            for asset, alloc in budget.allocations.items()
        }
        
        # Calculate target allocations
        if triggers[0]["type"] == "regime_change":
            # Full reoptimization for regime change
            target_allocations = self.optimize_risk_allocation(
                portfolio, new_regime
            )
        else:
            # Incremental adjustments for other triggers
            target_allocations = {
                asset: alloc.target_allocation
                for asset, alloc in budget.allocations.items()
            }
        
        # Calculate changes
        allocation_changes = {}
        for asset in set(current_allocations.keys()) | set(target_allocations.keys()):
            current = current_allocations.get(asset, 0.0)
            target = target_allocations.get(asset, 0.0)
            allocation_changes[asset] = target - current
        
        # Estimate impact
        turnover = sum(abs(change) for change in allocation_changes.values()) / 2
        transaction_cost = turnover * 0.001  # 10 bps assumption
        
        rebalancing = RiskBudgetRebalancing(
            rebalancing_id=rebalancing_id,
            budget_id=budget.budget_id,
            timestamp=datetime.utcnow(),
            trigger_type=triggers[0]["type"],
            trigger_details=triggers[0],
            current_allocations=current_allocations,
            target_allocations=target_allocations,
            allocation_changes=allocation_changes,
            transaction_cost_estimate=transaction_cost,
            priority_order=sorted(
                allocation_changes.keys(),
                key=lambda x: abs(allocation_changes[x]),
                reverse=True
            ),
            execution_risk="medium" if turnover > 0.20 else "low"
        )
        
        return rebalancing
    
    def _allocation_to_position_adjustment(
        self,
        portfolio: Portfolio,
        asset: str,
        allocation_change: float
    ) -> float:
        """Convert allocation change to position adjustment."""
        # Get current position
        position = next(
            (p for p in portfolio.positions.values() if p.symbol == asset),
            None
        )
        
        if not position:
            # New position
            target_value = portfolio.total_value * allocation_change
            # Return number of shares (simplified)
            return target_value / 100  # Placeholder price
        
        # Calculate position change
        current_weight = (position.quantity * position.current_price) / portfolio.total_value
        target_weight = current_weight + allocation_change
        
        # Convert to shares
        target_value = portfolio.total_value * target_weight
        target_shares = target_value / position.current_price
        
        return target_shares - position.quantity
    
    def _score_position_for_regime(
        self,
        position: Position,
        regime: MarketRegime,
        preferred_strategies: List[str],
        avoided_strategies: List[str]
    ) -> float:
        """Score position suitability for current regime."""
        score = 1.0  # Base score
        
        # Check if position matches preferred strategies
        position_characteristics = self._get_position_characteristics(position)
        
        for strategy in preferred_strategies:
            if strategy in position_characteristics:
                score *= 1.5
        
        for strategy in avoided_strategies:
            if strategy in position_characteristics:
                score *= 0.5
        
        # Adjust for regime-specific factors
        if regime.regime_type == RegimeType.CRISIS:
            # Prefer defensive, liquid assets
            if position.sector in ["utilities", "consumer_staples", "healthcare"]:
                score *= 1.5
            elif position.sector in ["technology", "discretionary"]:
                score *= 0.7
        
        elif regime.regime_type == RegimeType.BULL:
            # Prefer growth, momentum
            if position.sector in ["technology", "discretionary"]:
                score *= 1.3
        
        return max(score, 0.1)  # Minimum score
    
    def _get_position_characteristics(self, position: Position) -> List[str]:
        """Get characteristics/strategies associated with position."""
        # Simplified - in practice would use more sophisticated classification
        characteristics = []
        
        # Sector-based
        if position.sector in ["technology", "discretionary"]:
            characteristics.extend(["growth", "momentum"])
        elif position.sector in ["utilities", "staples"]:
            characteristics.extend(["defensive", "value"])
        elif position.sector in ["financials"]:
            characteristics.append("cyclical")
        
        # Size-based (if available)
        if hasattr(position, 'market_cap'):
            if position.market_cap > 100e9:
                characteristics.append("large_cap")
            elif position.market_cap < 2e9:
                characteristics.append("small_cap")
        
        return characteristics
    
    def _get_regime_expected_returns(
        self,
        positions: List[Position],
        regime: MarketRegime
    ) -> np.ndarray:
        """Get expected returns for positions given regime."""
        expected_returns = []
        
        # Base expected returns by regime
        regime_base_returns = {
            RegimeType.BULL: 0.12,
            RegimeType.BEAR: -0.08,
            RegimeType.SIDEWAYS: 0.04,
            RegimeType.CRISIS: -0.15,
            RegimeType.RECOVERY: 0.08
        }
        
        base_return = regime_base_returns.get(regime.regime_type, 0.06)
        
        for position in positions:
            # Adjust based on position characteristics
            pos_return = base_return
            
            # Sector adjustments
            if regime.regime_type == RegimeType.BULL:
                if position.sector == "technology":
                    pos_return *= 1.5
                elif position.sector == "utilities":
                    pos_return *= 0.7
            elif regime.regime_type == RegimeType.BEAR:
                if position.sector == "utilities":
                    pos_return *= 0.5  # Less negative
                elif position.sector == "technology":
                    pos_return *= 1.5  # More negative
            
            expected_returns.append(pos_return)
        
        return np.array(expected_returns)
    
    def _estimate_covariance_matrix(
        self,
        positions: List[Position],
        regime: MarketRegime
    ) -> np.ndarray:
        """Estimate covariance matrix for regime."""
        n = len(positions)
        
        # Simplified covariance estimation
        # In practice, would use historical data with regime adjustment
        
        # Base volatilities by sector
        sector_vols = {
            "technology": 0.25,
            "financials": 0.20,
            "utilities": 0.12,
            "healthcare": 0.15,
            "consumer_staples": 0.10,
            "energy": 0.30,
            "default": 0.18
        }
        
        # Get volatilities
        vols = []
        for position in positions:
            sector = getattr(position, 'sector', 'default')
            vol = sector_vols.get(sector, sector_vols['default'])
            
            # Adjust for regime
            if regime.volatility_level == "extreme":
                vol *= 2.0
            elif regime.volatility_level == "high":
                vol *= 1.5
            elif regime.volatility_level == "low":
                vol *= 0.7
            
            vols.append(vol)
        
        vols = np.array(vols)
        
        # Correlation matrix (simplified)
        if regime.regime_type == RegimeType.CRISIS:
            # High correlation in crisis
            corr = np.full((n, n), 0.8)
        elif regime.regime_type == RegimeType.SIDEWAYS:
            # Low correlation in sideways market
            corr = np.full((n, n), 0.3)
        else:
            # Normal correlation
            corr = np.full((n, n), 0.5)
        
        np.fill_diagonal(corr, 1.0)
        
        # Covariance = diag(vol) @ corr @ diag(vol)
        cov = np.outer(vols, vols) * corr
        
        return cov
    
    def _get_regime_risk_aversion(self, regime: MarketRegime) -> float:
        """Get risk aversion parameter for regime."""
        risk_aversions = {
            RegimeType.BULL: 1.0,
            RegimeType.BEAR: 3.0,
            RegimeType.SIDEWAYS: 2.0,
            RegimeType.CRISIS: 5.0,
            RegimeType.RECOVERY: 2.5
        }
        
        return risk_aversions.get(regime.regime_type, 2.0)
    
    def _get_fallback_allocations(
        self,
        positions: List[Position],
        regime: MarketRegime
    ) -> Dict[str, float]:
        """Get fallback allocations when optimization fails."""
        # Simple equal weight with regime adjustment
        n = len(positions)
        if n == 0:
            return {}
        
        base_weight = 1.0 / n
        
        # Adjust for regime
        if regime.regime_type == RegimeType.CRISIS:
            # Reduce all allocations
            base_weight *= 0.5
        
        allocations = {}
        for position in positions:
            allocations[position.symbol] = base_weight
        
        # Normalize
        total = sum(allocations.values())
        if total > 0:
            for symbol in allocations:
                allocations[symbol] /= total
        
        return allocations
    
    def save_snapshot(self, portfolio: Portfolio, market_data: pd.DataFrame):
        """Save current risk budget snapshot."""
        if not self.risk_budgets:
            return
        
        budget = self.risk_budgets[RiskBudgetType.TOTAL]
        returns = market_data.pct_change().dropna()
        
        # Calculate metrics
        portfolio_return = self._calculate_portfolio_return(portfolio, returns)
        portfolio_vol = self._calculate_portfolio_volatility(portfolio, returns)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Find largest allocation
        largest_alloc = max(
            budget.allocations.items(),
            key=lambda x: x[1].allocated_risk
        )
        
        snapshot = RiskBudgetSnapshot(
            timestamp=datetime.utcnow(),
            budget_id=budget.budget_id,
            total_risk_utilization=budget.current_utilization,
            volatility_level=portfolio_vol,
            var_95=portfolio_vol * 1.645 * portfolio.total_value / np.sqrt(252),
            cvar_95=portfolio_vol * 2.063 * portfolio.total_value / np.sqrt(252),
            n_allocations=len(budget.allocations),
            concentration_ratio=budget.get_concentration_ratio(),
            largest_allocation=(largest_alloc[0], largest_alloc[1].allocated_risk),
            market_regime=budget.regime_type,
            regime_confidence=0.0,  # Would get from regime detector
            period_return=portfolio_return,
            sharpe_ratio=sharpe,
            max_drawdown=0.0,  # Would calculate separately
            rebalancing_needed=len(self.monitoring.active_alerts) > 0
        )
        
        self.budget_history.append(snapshot)
        
        # Keep only recent history
        if len(self.budget_history) > 1000:
            self.budget_history = self.budget_history[-1000:]
    
    def _calculate_portfolio_return(
        self,
        portfolio: Portfolio,
        returns: pd.DataFrame
    ) -> float:
        """Calculate portfolio return."""
        weights = []
        symbols = []
        
        for position in portfolio.positions.values():
            if position.symbol in returns.columns:
                position_value = position.quantity * position.current_price
                weight = position_value / portfolio.total_value
                weights.append(weight)
                symbols.append(position.symbol)
        
        if not weights or len(returns) == 0:
            return 0.0
        
        # Latest return
        latest_returns = returns[symbols].iloc[-1]
        portfolio_return = (latest_returns * weights).sum()
        
        return portfolio_return * 252  # Annualized