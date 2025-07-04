"""
Monte Carlo simulation service for portfolio risk analysis.

Orchestrates the simulation engine, scenario generation, and result analysis
to provide comprehensive risk insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import pickle
from pathlib import Path
from scipy import stats

from alpha_pulse.models.simulation_results import (
    SimulationResults, PortfolioSimulationResults, StressTestResults,
    BacktestResults, MonteCarloReport, SimulationDiagnostics,
    SensitivityResults, GreeksResults
)
from alpha_pulse.models.risk_scenarios import (
    SimulationConfig, ScenarioType, ScenarioAnalysisConfig,
    StochasticProcess, VarianceReduction, ScenarioSet
)
from alpha_pulse.risk.monte_carlo_engine import MonteCarloEngine
from alpha_pulse.risk.scenario_generators import ScenarioGenerator
from alpha_pulse.risk.path_simulation import PathSimulator
from alpha_pulse.utils.random_number_generators import (
    RandomNumberGenerator, create_generator
)
from alpha_pulse.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class SimulationService:
    """Service for running Monte Carlo simulations and risk analysis."""
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize simulation service.
        
        Args:
            config: Default simulation configuration
            metrics_collector: Metrics collection service
            cache_dir: Directory for caching results
        """
        self.config = config or SimulationConfig()
        self.metrics_collector = metrics_collector
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/simulations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_components()
        
        # Execution statistics
        self.execution_stats = {
            'simulations_run': 0,
            'scenarios_analyzed': 0,
            'total_paths_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def _setup_components(self):
        """Set up simulation components."""
        # Random number generator
        if self.config.use_quasi_random:
            self.rng = create_generator('sobol', dimension=10, seed=self.config.random_seed)
        else:
            self.rng = create_generator('mersenne', seed=self.config.random_seed)
        
        # Simulation engine
        self.engine = MonteCarloEngine(
            config=self.config,
            rng=self.rng,
            use_gpu=self.config.use_gpu,
            n_workers=self.config.n_workers
        )
        
        # Path simulator
        self.path_simulator = PathSimulator(self.rng)
        
        # Scenario generator (will be initialized with data)
        self.scenario_generator = None
        
    async def run_portfolio_simulation(
        self,
        portfolio: Dict[str, float],
        market_data: pd.DataFrame,
        correlation_matrix: Optional[np.ndarray] = None,
        scenarios: Optional[ScenarioSet] = None,
        custom_config: Optional[SimulationConfig] = None
    ) -> PortfolioSimulationResults:
        """
        Run comprehensive portfolio simulation.
        
        Args:
            portfolio: Portfolio weights by asset
            market_data: Historical market data
            correlation_matrix: Asset correlation matrix
            scenarios: Pre-generated scenarios (optional)
            custom_config: Custom configuration (overrides default)
            
        Returns:
            Portfolio simulation results
        """
        logger.info(f"Running portfolio simulation for {len(portfolio)} assets")
        
        config = custom_config or self.config
        
        # Initialize scenario generator if needed
        if self.scenario_generator is None:
            self.scenario_generator = ScenarioGenerator(
                historical_data=market_data,
                correlation_matrix=correlation_matrix
            )
        
        # Calculate current prices
        assets = list(portfolio.keys())
        current_prices = {
            asset: market_data[asset].iloc[-1] if asset in market_data.columns else 100.0
            for asset in assets
        }
        
        # Generate scenarios if not provided
        if scenarios is None:
            scenario_config = ScenarioAnalysisConfig()
            scenarios = self.scenario_generator.generate_scenarios(
                scenario_types=scenario_config.scenario_types,
                n_scenarios_per_type=100,
                time_horizon=config.time_horizon
            )
        
        # Run base simulation
        start_time = datetime.now()
        
        base_results = await self._run_base_simulation(
            portfolio, current_prices, correlation_matrix, config
        )
        
        # Run scenario simulations
        scenario_results = await self._run_scenario_simulations(
            portfolio, current_prices, scenarios, config
        )
        
        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(
            portfolio, base_results, correlation_matrix
        )
        
        # Calculate diversification metrics
        diversification_metrics = self._calculate_diversification_metrics(
            portfolio, correlation_matrix, base_results
        )
        
        simulation_time = (datetime.now() - start_time).total_seconds()
        
        # Update statistics
        self.execution_stats['simulations_run'] += 1
        self.execution_stats['scenarios_analyzed'] += len(scenarios.scenarios)
        self.execution_stats['total_paths_generated'] += config.n_scenarios
        
        # Track metrics
        if self.metrics_collector:
            self.metrics_collector.record_histogram(
                "simulation_execution_time",
                simulation_time,
                {"type": "portfolio"}
            )
        
        # Create results
        return PortfolioSimulationResults(
            asset_returns={
                asset: base_results.paths.paths[:, -1, i] / base_results.paths.paths[:, 0, i] - 1
                for i, asset in enumerate(assets)
            },
            asset_weights=portfolio,
            correlation_matrix=correlation_matrix,
            portfolio_results=base_results,
            risk_contributions=risk_contributions,
            return_contributions={
                asset: weight * base_results.risk_metrics.expected_return
                for asset, weight in portfolio.items()
            },
            diversification_ratio=diversification_metrics['diversification_ratio'],
            effective_number_of_assets=diversification_metrics['effective_assets'],
            concentration_risk=diversification_metrics['concentration_risk']
        )
    
    async def _run_base_simulation(
        self,
        portfolio: Dict[str, float],
        current_prices: Dict[str, float],
        correlation_matrix: Optional[np.ndarray],
        config: SimulationConfig
    ) -> SimulationResults:
        """Run base Monte Carlo simulation."""
        # Check cache
        cache_key = self._get_cache_key(portfolio, config)
        cached_result = self._load_from_cache(cache_key)
        
        if cached_result is not None:
            self.execution_stats['cache_hits'] += 1
            return cached_result
        
        self.execution_stats['cache_misses'] += 1
        
        # Run simulation
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.engine.run_portfolio_simulation,
            portfolio,
            current_prices,
            correlation_matrix or np.eye(len(portfolio)),
            config.n_scenarios,
            config.time_horizon,
            config.n_steps
        )
        
        # Cache result
        self._save_to_cache(cache_key, result)
        
        return result
    
    async def _run_scenario_simulations(
        self,
        portfolio: Dict[str, float],
        current_prices: Dict[str, float],
        scenarios: ScenarioSet,
        config: SimulationConfig
    ) -> List[Any]:
        """Run simulations for multiple scenarios."""
        # Prepare scenario configurations
        scenario_configs = []
        
        for scenario in scenarios.scenarios[:20]:  # Limit to 20 scenarios
            scenario_config = {
                'name': scenario.name,
                'price_shocks': getattr(scenario, 'asset_returns', {}),
                'parameter_changes': {},
                'correlations': np.eye(len(portfolio)),
                'n_paths': min(1000, config.n_scenarios // 10),
                'time_horizon': scenario.time_horizon
            }
            scenario_configs.append(scenario_config)
        
        # Run scenarios in parallel
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self.engine.parallel_scenario_simulation,
            scenario_configs,
            portfolio,
            current_prices
        )
        
        return results
    
    def _calculate_risk_contributions(
        self,
        portfolio: Dict[str, float],
        simulation_results: SimulationResults,
        correlation_matrix: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate marginal risk contributions."""
        assets = list(portfolio.keys())
        weights = np.array([portfolio[asset] for asset in assets])
        
        # Calculate portfolio volatility
        portfolio_vol = simulation_results.risk_metrics.volatility
        
        if correlation_matrix is None:
            # Equal risk contribution assumption
            return {asset: 1.0 / len(assets) for asset in assets}
        
        # Calculate marginal contributions
        # Using Euler decomposition: RC_i = w_i * ∂σ/∂w_i
        asset_vols = np.std(simulation_results.paths.paths[:, -1, :] / 
                          simulation_results.paths.paths[:, 0, :] - 1, axis=0)
        
        # Covariance matrix
        cov_matrix = correlation_matrix * np.outer(asset_vols, asset_vols)
        
        # Marginal contributions
        marginal_contrib = cov_matrix @ weights / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        # Normalize
        total_contrib = np.sum(risk_contrib)
        risk_contrib = risk_contrib / total_contrib if total_contrib > 0 else risk_contrib
        
        return dict(zip(assets, risk_contrib))
    
    def _calculate_diversification_metrics(
        self,
        portfolio: Dict[str, float],
        correlation_matrix: Optional[np.ndarray],
        simulation_results: SimulationResults
    ) -> Dict[str, float]:
        """Calculate portfolio diversification metrics."""
        weights = np.array(list(portfolio.values()))
        
        # Diversification ratio
        if correlation_matrix is not None:
            asset_vols = np.std(simulation_results.paths.paths[:, -1, :] / 
                              simulation_results.paths.paths[:, 0, :] - 1, axis=0)
            
            weighted_avg_vol = np.sum(weights * asset_vols)
            portfolio_vol = simulation_results.risk_metrics.volatility
            
            diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        else:
            diversification_ratio = 1.0
        
        # Effective number of assets (using entropy)
        # N_eff = exp(-Σ w_i * log(w_i))
        weights_positive = weights[weights > 0]
        if len(weights_positive) > 0:
            entropy = -np.sum(weights_positive * np.log(weights_positive))
            effective_assets = np.exp(entropy)
        else:
            effective_assets = 1.0
        
        # Concentration risk (Herfindahl index)
        concentration_risk = np.sum(weights ** 2)
        
        return {
            'diversification_ratio': diversification_ratio,
            'effective_assets': effective_assets,
            'concentration_risk': concentration_risk
        }
    
    async def run_stress_tests(
        self,
        portfolio: Dict[str, float],
        market_data: pd.DataFrame,
        stress_scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> StressTestResults:
        """
        Run stress tests on portfolio.
        
        Args:
            portfolio: Portfolio weights
            market_data: Historical market data
            stress_scenarios: Custom stress scenarios
            
        Returns:
            Stress test results
        """
        logger.info("Running stress tests")
        
        # Default stress scenarios if not provided
        if stress_scenarios is None:
            stress_scenarios = self._get_default_stress_scenarios()
        
        # Run baseline simulation
        baseline_results = await self.run_portfolio_simulation(
            portfolio, market_data
        )
        
        # Run stressed simulations
        stressed_results = {}
        var_impacts = {}
        return_impacts = {}
        volatility_impacts = {}
        
        for scenario in stress_scenarios:
            # Apply stress to market data
            stressed_data = self._apply_stress_scenario(market_data, scenario)
            
            # Run simulation with stressed data
            stressed_result = await self.run_portfolio_simulation(
                portfolio, stressed_data
            )
            
            scenario_name = scenario['name']
            stressed_results[scenario_name] = stressed_result.portfolio_results.risk_metrics
            
            # Calculate impacts
            var_impacts[scenario_name] = (
                stressed_result.portfolio_results.risk_metrics.var_95 -
                baseline_results.portfolio_results.risk_metrics.var_95
            )
            
            return_impacts[scenario_name] = (
                stressed_result.portfolio_results.risk_metrics.expected_return -
                baseline_results.portfolio_results.risk_metrics.expected_return
            )
            
            volatility_impacts[scenario_name] = (
                stressed_result.portfolio_results.risk_metrics.volatility -
                baseline_results.portfolio_results.risk_metrics.volatility
            )
        
        # Find worst case
        worst_case_scenario = min(return_impacts.items(), key=lambda x: x[1])[0]
        worst_case_return = return_impacts[worst_case_scenario]
        worst_case_var = min(var_impacts.values())
        
        return StressTestResults(
            baseline_metrics=baseline_results.portfolio_results.risk_metrics,
            stressed_metrics=stressed_results,
            var_impacts=var_impacts,
            return_impacts=return_impacts,
            volatility_impacts=volatility_impacts,
            worst_case_var=worst_case_var,
            worst_case_return=worst_case_return,
            worst_case_scenario=worst_case_scenario,
            recovery_times={}  # Would calculate based on mean reversion
        )
    
    def _get_default_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Get default stress test scenarios."""
        return [
            {
                'name': 'Market_Crash',
                'returns_shock': -0.20,
                'volatility_multiplier': 2.0,
                'correlation_increase': 0.3
            },
            {
                'name': 'Volatility_Spike',
                'returns_shock': -0.05,
                'volatility_multiplier': 3.0,
                'correlation_increase': 0.2
            },
            {
                'name': 'Liquidity_Crisis',
                'returns_shock': -0.15,
                'volatility_multiplier': 1.5,
                'correlation_increase': 0.5
            },
            {
                'name': 'Interest_Rate_Shock',
                'returns_shock': -0.10,
                'volatility_multiplier': 1.3,
                'correlation_increase': 0.1
            }
        ]
    
    def _apply_stress_scenario(
        self,
        market_data: pd.DataFrame,
        scenario: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply stress scenario to market data."""
        stressed_data = market_data.copy()
        
        # Apply return shock
        if 'returns_shock' in scenario:
            shock = scenario['returns_shock']
            # Apply to last portion of data
            shock_start = len(stressed_data) - 60  # Last 60 days
            for col in stressed_data.columns:
                if col not in ['Date', 'timestamp']:
                    stressed_data.loc[shock_start:, col] *= (1 + shock)
        
        # Apply volatility shock (scale returns)
        if 'volatility_multiplier' in scenario:
            multiplier = scenario['volatility_multiplier']
            returns = stressed_data.pct_change()
            scaled_returns = returns * multiplier
            
            # Reconstruct prices
            for col in stressed_data.columns:
                if col not in ['Date', 'timestamp']:
                    stressed_data[col] = (1 + scaled_returns[col]).cumprod() * stressed_data[col].iloc[0]
        
        return stressed_data
    
    async def calculate_sensitivity_analysis(
        self,
        portfolio: Dict[str, float],
        base_params: Dict[str, float],
        param_ranges: Dict[str, List[float]],
        market_data: pd.DataFrame
    ) -> SensitivityResults:
        """
        Perform sensitivity analysis on key parameters.
        
        Args:
            portfolio: Portfolio weights
            base_params: Base parameter values
            param_ranges: Parameter ranges to test
            market_data: Market data
            
        Returns:
            Sensitivity analysis results
        """
        logger.info("Running sensitivity analysis")
        
        # Base case simulation
        base_config = SimulationConfig(
            n_scenarios=5000,  # Fewer scenarios for speed
            process_parameters=base_params
        )
        
        base_result = await self.run_portfolio_simulation(
            portfolio, market_data, custom_config=base_config
        )
        
        base_value = base_result.portfolio_results.risk_metrics.expected_return
        
        # Test each parameter
        parameter_sensitivities = {}
        
        for param, values in param_ranges.items():
            param_results = {}
            
            for value in values:
                # Update parameter
                test_params = base_params.copy()
                test_params[param] = value
                
                # Run simulation
                test_config = SimulationConfig(
                    n_scenarios=5000,
                    process_parameters=test_params
                )
                
                test_result = await self.run_portfolio_simulation(
                    portfolio, market_data, custom_config=test_config
                )
                
                param_results[value] = test_result.portfolio_results.risk_metrics.expected_return
            
            parameter_sensitivities[param] = param_results
        
        return SensitivityResults(
            base_value=base_value,
            parameter_sensitivities=parameter_sensitivities
        )
    
    async def calculate_option_greeks(
        self,
        option_type: str,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        n_simulations: int = 100000
    ) -> GreeksResults:
        """
        Calculate option Greeks using Monte Carlo.
        
        Args:
            option_type: 'call' or 'put'
            spot: Current asset price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            volatility: Volatility
            n_simulations: Number of simulations
            
        Returns:
            Greeks calculation results
        """
        logger.info(f"Calculating Greeks for {option_type} option")
        
        loop = asyncio.get_event_loop()
        greeks = await loop.run_in_executor(
            None,
            self.engine.calculate_greeks,
            option_type,
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            volatility,
            n_simulations
        )
        
        return GreeksResults(**greeks)
    
    def backtest_simulation_model(
        self,
        historical_data: pd.DataFrame,
        lookback_window: int = 252,
        test_window: int = 63
    ) -> BacktestResults:
        """
        Backtest simulation model against historical data.
        
        Args:
            historical_data: Historical price data
            lookback_window: Days for model calibration
            test_window: Days for testing
            
        Returns:
            Backtest results
        """
        logger.info("Running simulation model backtest")
        
        returns = historical_data.pct_change().dropna()
        
        # Split data
        n_tests = (len(returns) - lookback_window) // test_window
        
        actual_returns_list = []
        simulated_returns_list = []
        
        for i in range(n_tests):
            # Training data
            train_start = i * test_window
            train_end = train_start + lookback_window
            train_data = returns.iloc[train_start:train_end]
            
            # Test data
            test_start = train_end
            test_end = test_start + test_window
            test_data = returns.iloc[test_start:test_end]
            
            # Calibrate model
            params = {
                'drift': train_data.mean().mean() * 252,
                'volatility': train_data.std().mean() * np.sqrt(252)
            }
            
            # Simulate
            config = SimulationConfig(
                n_scenarios=1000,
                time_horizon=test_window / 252,
                n_steps=test_window,
                process_parameters=params
            )
            
            spot_prices = historical_data.iloc[train_end - 1].values
            paths = self.engine.simulate_paths(
                spot_prices,
                StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
                config.n_scenarios,
                test_window,
                1/252,
                params
            )
            
            # Compare returns
            actual_return = (historical_data.iloc[test_end - 1] / 
                           historical_data.iloc[test_start] - 1).mean()
            simulated_returns = paths.returns.mean(axis=0).mean()
            
            actual_returns_list.append(actual_return)
            simulated_returns_list.append(simulated_returns)
        
        actual_returns = np.array(actual_returns_list)
        simulated_returns = np.array(simulated_returns_list)
        
        # Statistical tests
        ks_stat, ks_pval = stats.ks_2samp(actual_returns, simulated_returns)
        
        # Moment comparison
        mean_error = (simulated_returns.mean() - actual_returns.mean()) / actual_returns.mean()
        vol_error = (simulated_returns.std() - actual_returns.std()) / actual_returns.std()
        skew_error = stats.skew(simulated_returns) - stats.skew(actual_returns)
        kurt_error = stats.kurtosis(simulated_returns) - stats.kurtosis(actual_returns)
        
        # VaR backtesting
        var_levels = [0.95, 0.99]
        var_violations = {}
        kupiec_test = {}
        
        for level in var_levels:
            var_threshold = np.percentile(simulated_returns, (1 - level) * 100)
            violations = np.sum(actual_returns < var_threshold) / len(actual_returns)
            var_violations[level] = violations
            
            # Kupiec test
            expected_violations = 1 - level
            n = len(actual_returns)
            k = int(violations * n)
            
            if k > 0 and k < n:
                likelihood_ratio = -2 * np.log(
                    (expected_violations**k * (1-expected_violations)**(n-k)) /
                    (violations**k * (1-violations)**(n-k))
                )
                p_value = 1 - stats.chi2.cdf(likelihood_ratio, df=1)
            else:
                likelihood_ratio = np.inf
                p_value = 0.0
            
            kupiec_test[level] = (likelihood_ratio, p_value)
        
        return BacktestResults(
            actual_returns=actual_returns,
            simulated_returns=simulated_returns,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            mean_error=mean_error,
            volatility_error=vol_error,
            skewness_error=skew_error,
            kurtosis_error=kurt_error,
            var_violations=var_violations,
            kupiec_test=kupiec_test
        )
    
    def generate_simulation_report(
        self,
        portfolio_results: PortfolioSimulationResults,
        stress_results: Optional[StressTestResults] = None,
        backtest_results: Optional[BacktestResults] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MonteCarloReport:
        """
        Generate comprehensive simulation report.
        
        Args:
            portfolio_results: Main portfolio simulation results
            stress_results: Stress test results
            backtest_results: Backtest results
            metadata: Additional metadata
            
        Returns:
            Complete Monte Carlo report
        """
        # Create diagnostics
        diagnostics = SimulationDiagnostics(
            max_value=np.max(portfolio_results.portfolio_results.portfolio_values),
            min_value=np.min(portfolio_results.portfolio_results.portfolio_values),
            nan_count=np.isnan(portfolio_results.portfolio_results.portfolio_values).sum(),
            inf_count=np.isinf(portfolio_results.portfolio_results.portfolio_values).sum(),
            random_seed=self.config.random_seed or 0,
            generator_type=type(self.rng).__name__,
            uniformity_test=(0.0, 1.0),  # Would run actual test
            total_time=portfolio_results.portfolio_results.simulation_time,
            setup_time=0.0,
            simulation_time=portfolio_results.portfolio_results.simulation_time,
            analysis_time=0.0,
            peak_memory_mb=0.0,  # Would track actual memory
            average_memory_mb=0.0
        )
        
        # Create report
        report = MonteCarloReport(
            metadata=metadata or {
                'portfolio_size': len(portfolio_results.asset_weights),
                'simulation_config': self.config.__dict__,
                'execution_stats': self.execution_stats
            },
            simulation_results=portfolio_results.portfolio_results,
            portfolio_results=portfolio_results,
            stress_test_results=stress_results,
            backtest_results=backtest_results,
            diagnostics=diagnostics
        )
        
        return report
    
    def _get_cache_key(
        self,
        portfolio: Dict[str, float],
        config: SimulationConfig
    ) -> str:
        """Generate cache key for simulation."""
        # Create deterministic key from inputs
        portfolio_str = json.dumps(sorted(portfolio.items()))
        config_str = json.dumps({
            'n_scenarios': config.n_scenarios,
            'time_horizon': config.time_horizon,
            'n_steps': config.n_steps,
            'process': config.default_process.value,
            'seed': config.random_seed
        }, sort_keys=True)
        
        return f"{hash(portfolio_str)}_{hash(config_str)}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[SimulationResults]:
        """Load simulation results from cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, results: SimulationResults):
        """Save simulation results to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    async def run_live_risk_monitoring(
        self,
        portfolio: Dict[str, float],
        update_interval: int = 300,  # 5 minutes
        callback: Optional[Any] = None
    ):
        """
        Run continuous risk monitoring with periodic updates.
        
        Args:
            portfolio: Portfolio to monitor
            update_interval: Update interval in seconds
            callback: Callback function for results
        """
        logger.info("Starting live risk monitoring")
        
        while True:
            try:
                # Get latest market data
                market_data = await self._get_latest_market_data(portfolio.keys())
                
                # Run quick simulation
                config = SimulationConfig(
                    n_scenarios=1000,  # Fewer scenarios for speed
                    time_horizon=1/252,  # Daily
                    n_steps=1
                )
                
                results = await self.run_portfolio_simulation(
                    portfolio, market_data, custom_config=config
                )
                
                # Track metrics
                if self.metrics_collector:
                    self.metrics_collector.record_gauge(
                        "portfolio_var_95",
                        results.portfolio_results.risk_metrics.var_95,
                        {"monitoring": "live"}
                    )
                
                # Callback with results
                if callback:
                    await callback(results)
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in live monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _get_latest_market_data(
        self,
        assets: List[str]
    ) -> pd.DataFrame:
        """Get latest market data for assets."""
        # This would connect to real data source
        # For now, return dummy data
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        data = {}
        
        for asset in assets:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, 252)))
            data[asset] = prices
        
        return pd.DataFrame(data, index=dates)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution statistics."""
        return {
            'simulations_run': self.execution_stats['simulations_run'],
            'scenarios_analyzed': self.execution_stats['scenarios_analyzed'],
            'total_paths_generated': self.execution_stats['total_paths_generated'],
            'cache_hit_rate': (
                self.execution_stats['cache_hits'] / 
                (self.execution_stats['cache_hits'] + self.execution_stats['cache_misses'])
                if (self.execution_stats['cache_hits'] + self.execution_stats['cache_misses']) > 0
                else 0
            ),
            'avg_paths_per_simulation': (
                self.execution_stats['total_paths_generated'] / 
                self.execution_stats['simulations_run']
                if self.execution_stats['simulations_run'] > 0
                else 0
            )
        }