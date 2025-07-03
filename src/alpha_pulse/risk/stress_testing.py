"""
Comprehensive stress testing framework for portfolio risk analysis.

Implements historical scenario replay, hypothetical scenarios,
and Monte Carlo-based stress testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from alpha_pulse.models.portfolio import Portfolio
from alpha_pulse.models.stress_test_results import (
    StressTestResult, ScenarioResult, PositionImpact,
    RiskMetricImpact, StressTestSummary
)
from alpha_pulse.risk.scenario_generator import ScenarioGenerator
from alpha_pulse.config.stress_test_scenarios import STRESS_TEST_SCENARIOS


logger = logging.getLogger(__name__)


class StressTestType(Enum):
    """Types of stress tests."""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"
    REVERSE = "reverse"
    SENSITIVITY = "sensitivity"


class ScenarioSeverity(Enum):
    """Severity levels for stress scenarios."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"
    TAIL = "tail"


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    scenario_types: List[StressTestType] = field(
        default_factory=lambda: [StressTestType.HISTORICAL, StressTestType.HYPOTHETICAL]
    )
    severity_levels: List[ScenarioSeverity] = field(
        default_factory=lambda: [ScenarioSeverity.MODERATE, ScenarioSeverity.SEVERE]
    )
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    monte_carlo_simulations: int = 10000
    time_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])  # days
    parallel_execution: bool = True
    max_workers: int = 4
    include_correlation_breaks: bool = True
    include_liquidity_shocks: bool = True


@dataclass
class MarketShock:
    """Represents a market shock for stress testing."""
    asset_class: str
    shock_type: str  # price, volatility, correlation
    magnitude: float
    duration_days: int
    propagation_factor: float = 1.0  # For contagion effects


class StressTester:
    """Main stress testing engine."""
    
    def __init__(
        self,
        config: StressTestConfig = None,
        scenario_generator: ScenarioGenerator = None
    ):
        """Initialize stress tester."""
        self.config = config or StressTestConfig()
        self.scenario_generator = scenario_generator or ScenarioGenerator()
        self.historical_scenarios = self._load_historical_scenarios()
        self.scenario_cache = {}
        
    def run_stress_test(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame,
        scenario_name: Optional[str] = None,
        custom_shocks: Optional[List[MarketShock]] = None
    ) -> StressTestResult:
        """Run comprehensive stress test on portfolio."""
        logger.info(f"Running stress test for portfolio with {len(portfolio.positions)} positions")
        
        # Initialize result
        result = StressTestResult(
            test_id=f"stress_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            portfolio_id=portfolio.portfolio_id,
            test_date=datetime.utcnow(),
            scenarios=[]
        )
        
        # Run different types of stress tests
        if StressTestType.HISTORICAL in self.config.scenario_types:
            historical_results = self._run_historical_scenarios(
                portfolio, market_data, scenario_name
            )
            result.scenarios.extend(historical_results)
        
        if StressTestType.HYPOTHETICAL in self.config.scenario_types:
            hypothetical_results = self._run_hypothetical_scenarios(
                portfolio, market_data, custom_shocks
            )
            result.scenarios.extend(hypothetical_results)
        
        if StressTestType.MONTE_CARLO in self.config.scenario_types:
            monte_carlo_results = self._run_monte_carlo_scenarios(
                portfolio, market_data
            )
            result.scenarios.extend(monte_carlo_results)
        
        # Calculate summary statistics
        result.summary = self._calculate_summary_statistics(result.scenarios)
        
        return result
    
    def _run_historical_scenarios(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame,
        scenario_name: Optional[str] = None
    ) -> List[ScenarioResult]:
        """Run historical stress scenarios."""
        results = []
        
        # Select scenarios to run
        if scenario_name:
            scenarios = [s for s in self.historical_scenarios 
                        if s["name"] == scenario_name]
        else:
            scenarios = self.historical_scenarios
        
        # Run scenarios in parallel if configured
        if self.config.parallel_execution:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_single_historical_scenario,
                        portfolio, market_data, scenario
                    ): scenario for scenario in scenarios
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in scenario {futures[future]['name']}: {e}")
        else:
            for scenario in scenarios:
                try:
                    result = self._run_single_historical_scenario(
                        portfolio, market_data, scenario
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in scenario {scenario['name']}: {e}")
        
        return results
    
    def _run_single_historical_scenario(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame,
        scenario: Dict[str, Any]
    ) -> ScenarioResult:
        """Run a single historical scenario."""
        logger.info(f"Running historical scenario: {scenario['name']}")
        
        # Get historical price movements
        start_date = pd.to_datetime(scenario["start_date"])
        end_date = pd.to_datetime(scenario["end_date"])
        
        # Filter market data for the scenario period
        scenario_data = market_data[
            (market_data.index >= start_date) & 
            (market_data.index <= end_date)
        ]
        
        if scenario_data.empty:
            # Use predefined shocks if historical data not available
            return self._apply_predefined_shocks(portfolio, scenario)
        
        # Calculate returns during the scenario period
        scenario_returns = scenario_data.pct_change().dropna()
        cumulative_returns = (1 + scenario_returns).cumprod() - 1
        
        # Apply to portfolio
        position_impacts = []
        total_pnl = 0
        
        for position in portfolio.positions.values():
            symbol = position.symbol
            
            if symbol in cumulative_returns.columns:
                # Direct impact
                price_change = cumulative_returns[symbol].iloc[-1]
            else:
                # Use proxy or sector average
                price_change = self._estimate_price_change(
                    position, cumulative_returns, scenario
                )
            
            # Calculate position impact
            position_value = position.quantity * position.current_price
            position_pnl = position_value * price_change
            total_pnl += position_pnl
            
            impact = PositionImpact(
                position_id=position.position_id,
                symbol=position.symbol,
                initial_value=position_value,
                stressed_value=position_value * (1 + price_change),
                pnl=position_pnl,
                price_change_pct=price_change * 100,
                var_contribution=abs(position_pnl) / portfolio.total_value
            )
            
            position_impacts.append(impact)
        
        # Calculate risk metric impacts
        risk_impacts = self._calculate_risk_metric_impacts(
            portfolio, scenario_returns, scenario
        )
        
        return ScenarioResult(
            scenario_name=scenario["name"],
            scenario_type=StressTestType.HISTORICAL,
            probability=scenario.get("probability", 0.05),
            total_pnl=total_pnl,
            pnl_percentage=(total_pnl / portfolio.total_value) * 100,
            position_impacts=position_impacts,
            risk_metric_impacts=risk_impacts,
            metadata=scenario.get("metadata", {})
        )
    
    def _run_hypothetical_scenarios(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame,
        custom_shocks: Optional[List[MarketShock]] = None
    ) -> List[ScenarioResult]:
        """Run hypothetical stress scenarios."""
        results = []
        
        # Get predefined hypothetical scenarios
        hypothetical_scenarios = STRESS_TEST_SCENARIOS["hypothetical"]
        
        # Add custom shocks if provided
        if custom_shocks:
            custom_scenario = self._create_custom_scenario(custom_shocks)
            hypothetical_scenarios.append(custom_scenario)
        
        for scenario in hypothetical_scenarios:
            try:
                result = self._run_hypothetical_scenario(
                    portfolio, market_data, scenario
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hypothetical scenario {scenario['name']}: {e}")
        
        return results
    
    def _run_hypothetical_scenario(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame,
        scenario: Dict[str, Any]
    ) -> ScenarioResult:
        """Run a single hypothetical scenario."""
        logger.info(f"Running hypothetical scenario: {scenario['name']}")
        
        # Generate shocked market data
        shocked_data = self._apply_market_shocks(
            market_data, scenario["shocks"]
        )
        
        # Calculate portfolio impact
        position_impacts = []
        total_pnl = 0
        
        for position in portfolio.positions.values():
            # Calculate position-specific shocks
            position_shock = self._calculate_position_shock(
                position, scenario["shocks"]
            )
            
            # Apply shock
            position_value = position.quantity * position.current_price
            stressed_value = position_value * (1 + position_shock)
            position_pnl = stressed_value - position_value
            total_pnl += position_pnl
            
            impact = PositionImpact(
                position_id=position.position_id,
                symbol=position.symbol,
                initial_value=position_value,
                stressed_value=stressed_value,
                pnl=position_pnl,
                price_change_pct=position_shock * 100,
                var_contribution=abs(position_pnl) / portfolio.total_value
            )
            
            position_impacts.append(impact)
        
        # Calculate risk metric impacts
        risk_impacts = self._calculate_hypothetical_risk_impacts(
            portfolio, scenario
        )
        
        return ScenarioResult(
            scenario_name=scenario["name"],
            scenario_type=StressTestType.HYPOTHETICAL,
            probability=scenario.get("probability", 0.01),
            total_pnl=total_pnl,
            pnl_percentage=(total_pnl / portfolio.total_value) * 100,
            position_impacts=position_impacts,
            risk_metric_impacts=risk_impacts,
            metadata=scenario.get("metadata", {})
        )
    
    def _run_monte_carlo_scenarios(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame
    ) -> List[ScenarioResult]:
        """Run Monte Carlo stress scenarios."""
        results = []
        
        # Generate scenarios for different confidence levels
        for confidence_level in self.config.confidence_levels:
            for time_horizon in self.config.time_horizons:
                scenario_name = f"Monte Carlo {confidence_level*100}% {time_horizon}d"
                
                logger.info(f"Running {scenario_name}")
                
                # Generate Monte Carlo paths
                simulated_returns = self.scenario_generator.generate_monte_carlo_scenarios(
                    market_data,
                    n_scenarios=self.config.monte_carlo_simulations,
                    time_horizon=time_horizon
                )
                
                # Calculate portfolio P&L distribution
                pnl_distribution = self._calculate_portfolio_pnl_distribution(
                    portfolio, simulated_returns
                )
                
                # Extract stress scenario at confidence level
                var_percentile = (1 - confidence_level) * 100
                var_pnl = np.percentile(pnl_distribution, var_percentile)
                
                # Find worst scenario near VaR
                worst_scenario_idx = np.argmin(
                    np.abs(pnl_distribution - var_pnl)
                )
                
                # Calculate position impacts for worst scenario
                position_impacts = self._calculate_monte_carlo_position_impacts(
                    portfolio, simulated_returns[worst_scenario_idx]
                )
                
                # Risk metric impacts
                risk_impacts = [
                    RiskMetricImpact(
                        metric_name="VaR",
                        initial_value=0,
                        stressed_value=var_pnl,
                        change_pct=0
                    ),
                    RiskMetricImpact(
                        metric_name="CVaR",
                        initial_value=0,
                        stressed_value=pnl_distribution[pnl_distribution <= var_pnl].mean(),
                        change_pct=0
                    )
                ]
                
                result = ScenarioResult(
                    scenario_name=scenario_name,
                    scenario_type=StressTestType.MONTE_CARLO,
                    probability=1 - confidence_level,
                    total_pnl=var_pnl,
                    pnl_percentage=(var_pnl / portfolio.total_value) * 100,
                    position_impacts=position_impacts,
                    risk_metric_impacts=risk_impacts,
                    metadata={
                        "confidence_level": confidence_level,
                        "time_horizon": time_horizon,
                        "n_simulations": self.config.monte_carlo_simulations
                    }
                )
                
                results.append(result)
        
        return results
    
    def run_reverse_stress_test(
        self,
        portfolio: Portfolio,
        target_loss: float,
        market_data: pd.DataFrame
    ) -> StressTestResult:
        """Run reverse stress test to find scenarios causing target loss."""
        logger.info(f"Running reverse stress test for {target_loss*100}% loss")
        
        # Initialize result
        result = StressTestResult(
            test_id=f"reverse_stress_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            portfolio_id=portfolio.portfolio_id,
            test_date=datetime.utcnow(),
            scenarios=[]
        )
        
        # Find scenarios that produce target loss
        scenarios = self._find_target_loss_scenarios(
            portfolio, target_loss, market_data
        )
        
        result.scenarios = scenarios
        result.summary = self._calculate_summary_statistics(scenarios)
        
        return result
    
    def run_sensitivity_analysis(
        self,
        portfolio: Portfolio,
        risk_factors: List[str],
        shock_range: Tuple[float, float] = (-0.20, 0.20),
        n_steps: int = 21
    ) -> Dict[str, pd.DataFrame]:
        """Run sensitivity analysis on specified risk factors."""
        logger.info(f"Running sensitivity analysis for {len(risk_factors)} factors")
        
        results = {}
        shocks = np.linspace(shock_range[0], shock_range[1], n_steps)
        
        for factor in risk_factors:
            sensitivities = []
            
            for shock in shocks:
                # Apply shock to factor
                shocked_portfolio_value = self._calculate_shocked_portfolio_value(
                    portfolio, factor, shock
                )
                
                pnl = shocked_portfolio_value - portfolio.total_value
                pnl_pct = (pnl / portfolio.total_value) * 100
                
                sensitivities.append({
                    "shock": shock * 100,
                    "portfolio_value": shocked_portfolio_value,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct
                })
            
            results[factor] = pd.DataFrame(sensitivities)
        
        return results
    
    def _load_historical_scenarios(self) -> List[Dict[str, Any]]:
        """Load predefined historical scenarios."""
        return STRESS_TEST_SCENARIOS["historical"]
    
    def _apply_predefined_shocks(
        self,
        portfolio: Portfolio,
        scenario: Dict[str, Any]
    ) -> ScenarioResult:
        """Apply predefined shocks when historical data not available."""
        shocks = scenario.get("fallback_shocks", {})
        position_impacts = []
        total_pnl = 0
        
        for position in portfolio.positions.values():
            # Determine asset class
            asset_class = self._get_asset_class(position)
            
            # Apply shock based on asset class
            shock = shocks.get(asset_class, shocks.get("default", -0.10))
            
            position_value = position.quantity * position.current_price
            position_pnl = position_value * shock
            total_pnl += position_pnl
            
            impact = PositionImpact(
                position_id=position.position_id,
                symbol=position.symbol,
                initial_value=position_value,
                stressed_value=position_value * (1 + shock),
                pnl=position_pnl,
                price_change_pct=shock * 100,
                var_contribution=abs(position_pnl) / portfolio.total_value
            )
            
            position_impacts.append(impact)
        
        return ScenarioResult(
            scenario_name=scenario["name"],
            scenario_type=StressTestType.HISTORICAL,
            probability=scenario.get("probability", 0.05),
            total_pnl=total_pnl,
            pnl_percentage=(total_pnl / portfolio.total_value) * 100,
            position_impacts=position_impacts,
            risk_metric_impacts=[],
            metadata={"fallback": True}
        )
    
    def _estimate_price_change(
        self,
        position: Any,
        returns: pd.DataFrame,
        scenario: Dict[str, Any]
    ) -> float:
        """Estimate price change for positions without direct data."""
        # Use sector or market average as proxy
        asset_class = self._get_asset_class(position)
        
        # Try to find similar assets
        similar_assets = self._find_similar_assets(position, returns.columns)
        
        if similar_assets:
            # Use average of similar assets
            return returns[similar_assets].mean(axis=1).iloc[-1]
        else:
            # Use market average with beta adjustment
            market_return = returns.mean(axis=1).iloc[-1]
            beta = scenario.get("betas", {}).get(asset_class, 1.0)
            return market_return * beta
    
    def _calculate_risk_metric_impacts(
        self,
        portfolio: Portfolio,
        scenario_returns: pd.DataFrame,
        scenario: Dict[str, Any]
    ) -> List[RiskMetricImpact]:
        """Calculate impacts on risk metrics."""
        impacts = []
        
        # VaR impact
        initial_var = portfolio.var_95 if hasattr(portfolio, 'var_95') else 0
        stressed_var = self._calculate_stressed_var(portfolio, scenario_returns)
        
        impacts.append(RiskMetricImpact(
            metric_name="VaR_95",
            initial_value=initial_var,
            stressed_value=stressed_var,
            change_pct=((stressed_var - initial_var) / abs(initial_var) * 100) 
                      if initial_var != 0 else 0
        ))
        
        # Volatility impact
        initial_vol = scenario_returns.std().mean() * np.sqrt(252)
        stressed_vol = initial_vol * scenario.get("volatility_multiplier", 2.0)
        
        impacts.append(RiskMetricImpact(
            metric_name="Volatility",
            initial_value=initial_vol,
            stressed_value=stressed_vol,
            change_pct=((stressed_vol - initial_vol) / initial_vol * 100)
        ))
        
        # Correlation impact
        if self.config.include_correlation_breaks:
            impacts.append(RiskMetricImpact(
                metric_name="Average_Correlation",
                initial_value=0.3,  # Placeholder
                stressed_value=0.8,  # Crisis correlation
                change_pct=166.67
            ))
        
        return impacts
    
    def _apply_market_shocks(
        self,
        market_data: pd.DataFrame,
        shocks: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply market shocks to generate stressed market data."""
        shocked_data = market_data.copy()
        
        for shock in shocks:
            asset_class = shock["asset_class"]
            shock_type = shock["type"]
            magnitude = shock["magnitude"]
            
            # Apply shock based on type
            if shock_type == "price":
                # Price shock
                affected_assets = self._get_assets_by_class(
                    shocked_data.columns, asset_class
                )
                for asset in affected_assets:
                    if asset in shocked_data.columns:
                        shocked_data[asset] = shocked_data[asset] * (1 + magnitude)
            
            elif shock_type == "volatility":
                # Volatility shock - scale returns
                affected_assets = self._get_assets_by_class(
                    shocked_data.columns, asset_class
                )
                for asset in affected_assets:
                    if asset in shocked_data.columns:
                        returns = shocked_data[asset].pct_change()
                        shocked_returns = returns * magnitude
                        shocked_data[asset] = (1 + shocked_returns).cumprod() * shocked_data[asset].iloc[0]
        
        return shocked_data
    
    def _calculate_position_shock(
        self,
        position: Any,
        shocks: List[Dict[str, Any]]
    ) -> float:
        """Calculate total shock for a position."""
        total_shock = 0
        asset_class = self._get_asset_class(position)
        
        for shock in shocks:
            if shock["asset_class"] == asset_class or shock["asset_class"] == "all":
                if shock["type"] == "price":
                    total_shock += shock["magnitude"]
                elif shock["type"] == "volatility":
                    # Simplified: use volatility shock as proxy for price impact
                    total_shock += shock["magnitude"] * 0.1  # 10% of vol shock
        
        return total_shock
    
    def _calculate_hypothetical_risk_impacts(
        self,
        portfolio: Portfolio,
        scenario: Dict[str, Any]
    ) -> List[RiskMetricImpact]:
        """Calculate risk metric impacts for hypothetical scenarios."""
        impacts = []
        
        # Extract shock parameters
        volatility_shock = next(
            (s["magnitude"] for s in scenario["shocks"] if s["type"] == "volatility"),
            1.0
        )
        
        # Sharpe ratio impact
        initial_sharpe = getattr(portfolio, 'sharpe_ratio', 1.0)
        stressed_sharpe = initial_sharpe / volatility_shock
        
        impacts.append(RiskMetricImpact(
            metric_name="Sharpe_Ratio",
            initial_value=initial_sharpe,
            stressed_value=stressed_sharpe,
            change_pct=((stressed_sharpe - initial_sharpe) / initial_sharpe * 100)
        ))
        
        # Maximum drawdown impact
        initial_dd = getattr(portfolio, 'max_drawdown', 0.10)
        stressed_dd = initial_dd * (1 + abs(scenario.get("drawdown_multiplier", 2.0)))
        
        impacts.append(RiskMetricImpact(
            metric_name="Max_Drawdown",
            initial_value=initial_dd,
            stressed_value=stressed_dd,
            change_pct=((stressed_dd - initial_dd) / initial_dd * 100)
        ))
        
        return impacts
    
    def _calculate_portfolio_pnl_distribution(
        self,
        portfolio: Portfolio,
        simulated_returns: np.ndarray
    ) -> np.ndarray:
        """Calculate P&L distribution from simulated returns."""
        n_scenarios = simulated_returns.shape[0]
        portfolio_pnls = np.zeros(n_scenarios)
        
        # Get portfolio weights
        weights = np.array([
            position.quantity * position.current_price / portfolio.total_value
            for position in portfolio.positions.values()
        ])
        
        # Calculate portfolio returns for each scenario
        for i in range(n_scenarios):
            scenario_returns = simulated_returns[i]
            portfolio_return = np.dot(weights, scenario_returns)
            portfolio_pnls[i] = portfolio.total_value * portfolio_return
        
        return portfolio_pnls
    
    def _calculate_monte_carlo_position_impacts(
        self,
        portfolio: Portfolio,
        scenario_returns: np.ndarray
    ) -> List[PositionImpact]:
        """Calculate position impacts for Monte Carlo scenario."""
        impacts = []
        
        for i, position in enumerate(portfolio.positions.values()):
            position_return = scenario_returns[i]
            position_value = position.quantity * position.current_price
            position_pnl = position_value * position_return
            
            impact = PositionImpact(
                position_id=position.position_id,
                symbol=position.symbol,
                initial_value=position_value,
                stressed_value=position_value * (1 + position_return),
                pnl=position_pnl,
                price_change_pct=position_return * 100,
                var_contribution=abs(position_pnl) / portfolio.total_value
            )
            
            impacts.append(impact)
        
        return impacts
    
    def _find_target_loss_scenarios(
        self,
        portfolio: Portfolio,
        target_loss: float,
        market_data: pd.DataFrame
    ) -> List[ScenarioResult]:
        """Find scenarios that produce target loss."""
        scenarios = []
        
        # Use optimization to find shock combinations
        from scipy.optimize import minimize
        
        def loss_function(shocks):
            # Apply shocks and calculate portfolio loss
            total_pnl = 0
            for i, position in enumerate(portfolio.positions.values()):
                position_value = position.quantity * position.current_price
                position_pnl = position_value * shocks[i]
                total_pnl += position_pnl
            
            # Return squared difference from target
            actual_loss = total_pnl / portfolio.total_value
            return (actual_loss - target_loss) ** 2
        
        # Initial guess
        n_positions = len(portfolio.positions)
        x0 = np.full(n_positions, target_loss)
        
        # Constraints: shocks between -50% and +50%
        bounds = [(-0.5, 0.5) for _ in range(n_positions)]
        
        # Optimize
        result = minimize(loss_function, x0, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            # Create scenario from optimal shocks
            position_impacts = []
            total_pnl = 0
            
            for i, (position, shock) in enumerate(
                zip(portfolio.positions.values(), result.x)
            ):
                position_value = position.quantity * position.current_price
                position_pnl = position_value * shock
                total_pnl += position_pnl
                
                impact = PositionImpact(
                    position_id=position.position_id,
                    symbol=position.symbol,
                    initial_value=position_value,
                    stressed_value=position_value * (1 + shock),
                    pnl=position_pnl,
                    price_change_pct=shock * 100,
                    var_contribution=abs(position_pnl) / portfolio.total_value
                )
                
                position_impacts.append(impact)
            
            scenario = ScenarioResult(
                scenario_name=f"Reverse Stress {target_loss*100}% Loss",
                scenario_type=StressTestType.REVERSE,
                probability=0.001,  # Rare event
                total_pnl=total_pnl,
                pnl_percentage=(total_pnl / portfolio.total_value) * 100,
                position_impacts=position_impacts,
                risk_metric_impacts=[],
                metadata={"target_loss": target_loss, "optimization_success": True}
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _calculate_shocked_portfolio_value(
        self,
        portfolio: Portfolio,
        factor: str,
        shock: float
    ) -> float:
        """Calculate portfolio value after applying factor shock."""
        shocked_value = 0
        
        for position in portfolio.positions.values():
            position_value = position.quantity * position.current_price
            
            # Apply factor-specific shock
            if factor == "market":
                # Market-wide shock
                shocked_position_value = position_value * (1 + shock)
            elif factor == "volatility":
                # Volatility shock affects value through options/vol products
                vol_sensitivity = getattr(position, 'vega', 0)
                shocked_position_value = position_value + vol_sensitivity * shock
            elif factor == "interest_rate":
                # Interest rate shock
                duration = getattr(position, 'duration', 0)
                shocked_position_value = position_value * (1 - duration * shock)
            else:
                # Asset-specific shock
                if position.symbol == factor:
                    shocked_position_value = position_value * (1 + shock)
                else:
                    shocked_position_value = position_value
            
            shocked_value += shocked_position_value
        
        return shocked_value
    
    def _calculate_stressed_var(
        self,
        portfolio: Portfolio,
        scenario_returns: pd.DataFrame
    ) -> float:
        """Calculate stressed VaR."""
        # Simple historical VaR calculation
        portfolio_returns = []
        
        weights = np.array([
            position.quantity * position.current_price / portfolio.total_value
            for position in portfolio.positions.values()
        ])
        
        # Calculate portfolio returns
        for _, returns in scenario_returns.iterrows():
            portfolio_return = np.dot(weights, returns)
            portfolio_returns.append(portfolio_return)
        
        # Calculate VaR at 95% confidence
        var_95 = np.percentile(portfolio_returns, 5) * portfolio.total_value
        
        return abs(var_95)
    
    def _get_asset_class(self, position: Any) -> str:
        """Determine asset class of a position."""
        symbol = position.symbol.upper()
        
        # Simple classification
        if any(currency in symbol for currency in ["USD", "EUR", "GBP", "JPY"]):
            return "fx"
        elif any(crypto in symbol for crypto in ["BTC", "ETH", "CRYPTO"]):
            return "crypto"
        elif hasattr(position, 'asset_class'):
            return position.asset_class
        else:
            return "equity"  # Default
    
    def _find_similar_assets(
        self,
        position: Any,
        available_assets: List[str]
    ) -> List[str]:
        """Find similar assets for proxy calculation."""
        asset_class = self._get_asset_class(position)
        similar = []
        
        for asset in available_assets:
            # Simple similarity check based on asset class
            if self._get_asset_class_from_symbol(asset) == asset_class:
                similar.append(asset)
        
        return similar[:5]  # Limit to 5 most similar
    
    def _get_asset_class_from_symbol(self, symbol: str) -> str:
        """Get asset class from symbol."""
        symbol = symbol.upper()
        
        if any(currency in symbol for currency in ["USD", "EUR", "GBP", "JPY"]):
            return "fx"
        elif any(crypto in symbol for crypto in ["BTC", "ETH", "CRYPTO"]):
            return "crypto"
        else:
            return "equity"
    
    def _get_assets_by_class(
        self,
        assets: List[str],
        asset_class: str
    ) -> List[str]:
        """Get assets belonging to specific class."""
        if asset_class == "all":
            return list(assets)
        
        return [
            asset for asset in assets
            if self._get_asset_class_from_symbol(asset) == asset_class
        ]
    
    def _create_custom_scenario(
        self,
        shocks: List[MarketShock]
    ) -> Dict[str, Any]:
        """Create custom scenario from market shocks."""
        return {
            "name": "Custom Scenario",
            "shocks": [
                {
                    "asset_class": shock.asset_class,
                    "type": shock.shock_type,
                    "magnitude": shock.magnitude
                }
                for shock in shocks
            ],
            "probability": 0.01,
            "metadata": {"custom": True}
        }
    
    def _calculate_summary_statistics(
        self,
        scenarios: List[ScenarioResult]
    ) -> StressTestSummary:
        """Calculate summary statistics across all scenarios."""
        if not scenarios:
            return StressTestSummary()
        
        pnls = [s.total_pnl for s in scenarios]
        pnl_pcts = [s.pnl_percentage for s in scenarios]
        
        # Find worst scenarios
        worst_idx = np.argmin(pnls)
        worst_scenario = scenarios[worst_idx]
        
        # Calculate probability-weighted loss
        expected_loss = sum(
            s.total_pnl * s.probability for s in scenarios
        ) / sum(s.probability for s in scenarios)
        
        return StressTestSummary(
            worst_case_scenario=worst_scenario.scenario_name,
            worst_case_pnl=worst_scenario.total_pnl,
            worst_case_pnl_pct=worst_scenario.pnl_percentage,
            expected_shortfall=np.mean([pnl for pnl in pnls if pnl < 0]),
            scenarios_passed=sum(1 for pnl in pnls if pnl > 0),
            scenarios_failed=sum(1 for pnl in pnls if pnl < 0),
            average_pnl=np.mean(pnls),
            pnl_volatility=np.std(pnls),
            risk_metrics_summary={
                "var_95": np.percentile(pnls, 5),
                "cvar_95": np.mean([pnl for pnl in pnls if pnl <= np.percentile(pnls, 5)]),
                "max_drawdown": min(pnls),
                "probability_weighted_loss": expected_loss
            }
        )