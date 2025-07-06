"""
Monte Carlo Integration Service.

This service bridges the gap between the sophisticated Monte Carlo engine
and the rest of the AlphaPulse system, making simulation results available
for risk reporting and decision making.
"""
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

from alpha_pulse.risk.monte_carlo_engine import MonteCarloEngine
from alpha_pulse.risk.scenario_generators import ScenarioGenerator, ScenarioType
from alpha_pulse.services.simulation_service import SimulationService
from alpha_pulse.risk_management.interfaces import RiskMetrics


class MonteCarloIntegrationService:
    """Service to integrate Monte Carlo simulations with risk management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Monte Carlo integration service.
        
        Args:
            config: Service configuration
        """
        self.config = config or {}
        
        # Initialize Monte Carlo engine
        self.mc_engine = MonteCarloEngine(
            n_simulations=self.config.get('n_simulations', 10000),
            time_horizon=self.config.get('time_horizon', 252),  # 1 year
            dt=1/252,  # Daily steps
            random_seed=self.config.get('random_seed', 42)
        )
        
        # Initialize scenario generator
        self.scenario_generator = ScenarioGenerator()
        
        # Initialize simulation service for advanced features
        self.simulation_service = SimulationService(config)
        
        # Cache for recent simulations
        self._simulation_cache: Dict[str, Any] = {}
        self._cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        logger.info("Initialized MonteCarloIntegrationService")
    
    async def calculate_monte_carlo_var(
        self,
        portfolio_returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99],
        n_simulations: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            portfolio_returns: Historical portfolio returns
            confidence_levels: VaR confidence levels
            n_simulations: Number of simulations (uses default if None)
            
        Returns:
            Dictionary of VaR values at different confidence levels
        """
        try:
            # Use provided simulations or default
            n_sims = n_simulations or self.mc_engine.n_simulations
            
            # Calculate return statistics
            mu = portfolio_returns.mean()
            sigma = portfolio_returns.std()
            
            # Generate Monte Carlo paths
            simulated_returns = np.random.normal(
                mu, sigma, size=(n_sims, self.mc_engine.time_horizon)
            )
            
            # Calculate terminal values
            terminal_values = np.prod(1 + simulated_returns, axis=1)
            terminal_returns = terminal_values - 1
            
            # Calculate VaR at different confidence levels
            var_results = {}
            for confidence in confidence_levels:
                var_percentile = (1 - confidence) * 100
                var_value = np.percentile(terminal_returns, var_percentile)
                var_results[f'var_{int(confidence*100)}'] = abs(var_value)
                
                # Also calculate CVaR (Expected Shortfall)
                cvar_value = terminal_returns[terminal_returns <= var_value].mean()
                var_results[f'cvar_{int(confidence*100)}'] = abs(cvar_value)
            
            logger.info(f"Calculated Monte Carlo VaR: {var_results}")
            return var_results
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return {}
    
    async def run_portfolio_simulation(
        self,
        portfolio_data: Dict[str, Any],
        scenario_type: str = "baseline"
    ) -> Dict[str, Any]:
        """
        Run portfolio simulation with specified scenario.
        
        Args:
            portfolio_data: Portfolio positions and weights
            scenario_type: Type of scenario to simulate
            
        Returns:
            Simulation results including paths and statistics
        """
        try:
            # Map scenario type to ScenarioType enum
            scenario_map = {
                "baseline": ScenarioType.BASELINE,
                "stressed": ScenarioType.STRESSED,
                "crisis": ScenarioType.CRISIS,
                "recovery": ScenarioType.RECOVERY
            }
            
            scenario = scenario_map.get(
                scenario_type.lower(),
                ScenarioType.BASELINE
            )
            
            # Generate scenario parameters
            scenario_params = self.scenario_generator.generate_scenario(scenario)
            
            # Extract portfolio positions
            positions = portfolio_data.get('positions', [])
            if not positions:
                return {"error": "No positions in portfolio"}
            
            # Simulate each position
            simulation_results = []
            for position in positions:
                symbol = position['symbol']
                initial_price = position.get('current_price', 100)
                
                # Generate price paths
                paths = self.mc_engine.simulate_gbm_paths(
                    S0=initial_price,
                    mu=scenario_params['expected_return'],
                    sigma=scenario_params['volatility']
                )
                
                # Calculate position value paths
                quantity = position.get('quantity', 0)
                value_paths = paths * quantity
                
                simulation_results.append({
                    'symbol': symbol,
                    'paths': paths,
                    'value_paths': value_paths,
                    'terminal_values': paths[:, -1],
                    'max_value': np.max(paths, axis=1),
                    'min_value': np.min(paths, axis=1)
                })
            
            # Aggregate portfolio results
            portfolio_paths = np.sum(
                [r['value_paths'] for r in simulation_results],
                axis=0
            )
            
            # Calculate portfolio statistics
            terminal_portfolio_values = portfolio_paths[:, -1]
            initial_portfolio_value = portfolio_paths[:, 0].mean()
            
            results = {
                'scenario_type': scenario_type,
                'n_simulations': self.mc_engine.n_simulations,
                'time_horizon': self.mc_engine.time_horizon,
                'initial_value': float(initial_portfolio_value),
                'statistics': {
                    'mean_terminal_value': float(np.mean(terminal_portfolio_values)),
                    'std_terminal_value': float(np.std(terminal_portfolio_values)),
                    'min_terminal_value': float(np.min(terminal_portfolio_values)),
                    'max_terminal_value': float(np.max(terminal_portfolio_values)),
                    'percentiles': {
                        '5th': float(np.percentile(terminal_portfolio_values, 5)),
                        '25th': float(np.percentile(terminal_portfolio_values, 25)),
                        '50th': float(np.percentile(terminal_portfolio_values, 50)),
                        '75th': float(np.percentile(terminal_portfolio_values, 75)),
                        '95th': float(np.percentile(terminal_portfolio_values, 95))
                    }
                },
                'risk_metrics': await self._calculate_simulation_risk_metrics(
                    portfolio_paths,
                    initial_portfolio_value
                ),
                'position_results': [
                    {
                        'symbol': r['symbol'],
                        'mean_terminal': float(np.mean(r['terminal_values'])),
                        'std_terminal': float(np.std(r['terminal_values']))
                    }
                    for r in simulation_results
                ],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache results
            cache_key = f"{scenario_type}_{datetime.utcnow().date()}"
            self._simulation_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error running portfolio simulation: {e}")
            return {"error": str(e)}
    
    async def _calculate_simulation_risk_metrics(
        self,
        portfolio_paths: np.ndarray,
        initial_value: float
    ) -> Dict[str, float]:
        """Calculate risk metrics from simulation paths."""
        # Calculate returns
        returns = np.diff(portfolio_paths, axis=1) / portfolio_paths[:, :-1]
        
        # Calculate metrics
        return {
            'simulation_var_95': float(
                abs(np.percentile(returns.flatten(), 5))
            ),
            'simulation_var_99': float(
                abs(np.percentile(returns.flatten(), 1))
            ),
            'max_drawdown': float(
                np.max(
                    np.maximum.accumulate(portfolio_paths, axis=1) - portfolio_paths,
                    axis=1
                ).mean() / initial_value
            ),
            'probability_of_loss': float(
                np.mean(portfolio_paths[:, -1] < initial_value)
            ),
            'expected_shortfall_95': float(
                abs(returns[returns < np.percentile(returns, 5)].mean())
            )
        }
    
    async def generate_stress_scenarios(
        self,
        portfolio_data: Dict[str, Any],
        custom_shocks: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate stress test scenarios for portfolio.
        
        Args:
            portfolio_data: Portfolio positions and weights
            custom_shocks: Optional custom market shocks
            
        Returns:
            List of stress scenario results
        """
        try:
            # Define stress scenarios
            scenarios = [
                {
                    'name': 'Market Crash',
                    'market_shock': -0.20,  # 20% drop
                    'volatility_multiplier': 2.0
                },
                {
                    'name': 'Flash Crash',
                    'market_shock': -0.10,  # 10% drop
                    'volatility_multiplier': 3.0
                },
                {
                    'name': 'Liquidity Crisis',
                    'market_shock': -0.15,
                    'volatility_multiplier': 2.5
                },
                {
                    'name': 'Rate Shock',
                    'market_shock': -0.05,
                    'volatility_multiplier': 1.5
                }
            ]
            
            # Add custom scenario if provided
            if custom_shocks:
                scenarios.append({
                    'name': 'Custom Scenario',
                    **custom_shocks
                })
            
            # Run each scenario
            results = []
            for scenario in scenarios:
                # Create stressed parameters
                stressed_params = {
                    'drift': scenario['market_shock'] / self.mc_engine.time_horizon,
                    'volatility': 0.2 * scenario['volatility_multiplier']
                }
                
                # Run simulation with stressed parameters
                sim_result = await self.simulation_service.run_simulation({
                    'portfolio': portfolio_data,
                    'model_params': stressed_params,
                    'n_paths': 1000,  # Fewer paths for stress tests
                    'horizon': 30  # 30 days
                })
                
                results.append({
                    'scenario_name': scenario['name'],
                    'parameters': scenario,
                    'results': sim_result,
                    'impact': self._calculate_scenario_impact(
                        sim_result,
                        portfolio_data
                    )
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating stress scenarios: {e}")
            return []
    
    def _calculate_scenario_impact(
        self,
        simulation_result: Dict[str, Any],
        portfolio_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate impact metrics for a stress scenario."""
        if 'error' in simulation_result:
            return {}
        
        initial_value = sum(
            p.get('value', p.get('quantity', 0) * p.get('current_price', 0))
            for p in portfolio_data.get('positions', [])
        )
        
        stats = simulation_result.get('statistics', {})
        mean_terminal = stats.get('mean_terminal_value', initial_value)
        
        return {
            'expected_loss': float((initial_value - mean_terminal) / initial_value),
            'worst_case_loss': float(
                (initial_value - stats.get('min_terminal_value', 0)) / initial_value
            ),
            'probability_of_10pct_loss': float(
                stats.get('percentiles', {}).get('10th', initial_value) < 
                initial_value * 0.9
            )
        }
    
    async def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary of recent simulations."""
        return {
            'cached_simulations': len(self._simulation_cache),
            'cache_keys': list(self._simulation_cache.keys()),
            'engine_config': {
                'n_simulations': self.mc_engine.n_simulations,
                'time_horizon': self.mc_engine.time_horizon,
                'available_models': [
                    'Geometric Brownian Motion',
                    'Jump Diffusion',
                    'Heston Stochastic Volatility',
                    'Variance Gamma'
                ]
            },
            'last_update': datetime.utcnow().isoformat()
        }