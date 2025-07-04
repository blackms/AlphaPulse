"""
High-performance Monte Carlo simulation engine for risk scenarios.

Implements vectorized simulation with multiple variance reduction techniques
and support for parallel processing and GPU acceleration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing as mp
from numba import jit, prange, cuda
import warnings
from scipy import stats

from alpha_pulse.models.simulation_results import (
    SimulationResults, PathResults, ConvergenceMetrics,
    ScenarioResults, RiskMetrics
)
from alpha_pulse.models.risk_scenarios import (
    SimulationConfig, StochasticProcess, VarianceReduction
)
from alpha_pulse.utils.random_number_generators import (
    RandomNumberGenerator, SobolGenerator, MersenneTwisterGenerator
)

logger = logging.getLogger(__name__)

# Check CUDA availability
CUDA_AVAILABLE = False
try:
    import cupy as cp
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    cp = None
    

class MonteCarloEngine:
    """High-performance Monte Carlo simulation engine."""
    
    def __init__(
        self,
        config: SimulationConfig,
        rng: Optional[RandomNumberGenerator] = None,
        use_gpu: bool = False,
        n_workers: int = None
    ):
        """
        Initialize Monte Carlo engine.
        
        Args:
            config: Simulation configuration
            rng: Random number generator (defaults to Mersenne Twister)
            use_gpu: Whether to use GPU acceleration if available
            n_workers: Number of worker processes (defaults to CPU count)
        """
        self.config = config
        self.rng = rng or MersenneTwisterGenerator(seed=config.random_seed)
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.n_workers = n_workers or mp.cpu_count()
        
        # Initialize variance reduction techniques
        self.variance_reduction = self._setup_variance_reduction()
        
        # Performance tracking
        self.simulation_stats = {
            'paths_generated': 0,
            'simulation_time': 0.0,
            'memory_peak': 0
        }
        
        if self.use_gpu and not CUDA_AVAILABLE:
            logger.warning("GPU requested but CUDA not available, falling back to CPU")
            self.use_gpu = False
            
    def _setup_variance_reduction(self) -> Dict[str, Any]:
        """Set up variance reduction techniques based on configuration."""
        techniques = {}
        
        if VarianceReduction.ANTITHETIC in self.config.variance_reduction:
            techniques['antithetic'] = True
            
        if VarianceReduction.CONTROL_VARIATE in self.config.variance_reduction:
            techniques['control_variate'] = {
                'enabled': True,
                'correlation_threshold': 0.7
            }
            
        if VarianceReduction.IMPORTANCE_SAMPLING in self.config.variance_reduction:
            techniques['importance_sampling'] = {
                'enabled': True,
                'tail_threshold': 0.95
            }
            
        if VarianceReduction.STRATIFIED_SAMPLING in self.config.variance_reduction:
            techniques['stratified_sampling'] = {
                'enabled': True,
                'n_strata': 10
            }
            
        return techniques
    
    def simulate_paths(
        self,
        spot_prices: Union[float, np.ndarray],
        process: StochasticProcess,
        n_paths: int,
        n_steps: int,
        dt: float,
        parameters: Dict[str, Any]
    ) -> PathResults:
        """
        Simulate asset price paths using specified stochastic process.
        
        Args:
            spot_prices: Initial asset prices
            process: Stochastic process to use
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step size
            parameters: Process-specific parameters
            
        Returns:
            PathResults object containing simulated paths
        """
        logger.info(f"Simulating {n_paths} paths with {process.value} process")
        
        start_time = datetime.now()
        
        if self.use_gpu:
            paths = self._simulate_paths_gpu(
                spot_prices, process, n_paths, n_steps, dt, parameters
            )
        else:
            paths = self._simulate_paths_cpu(
                spot_prices, process, n_paths, n_steps, dt, parameters
            )
        
        # Apply variance reduction if enabled
        if self.variance_reduction:
            paths = self._apply_variance_reduction(paths, spot_prices)
        
        simulation_time = (datetime.now() - start_time).total_seconds()
        self.simulation_stats['paths_generated'] += n_paths
        self.simulation_stats['simulation_time'] += simulation_time
        
        return PathResults(
            paths=paths,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            process=process,
            simulation_time=simulation_time
        )
    
    def _simulate_paths_cpu(
        self,
        spot_prices: Union[float, np.ndarray],
        process: StochasticProcess,
        n_paths: int,
        n_steps: int,
        dt: float,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Simulate paths on CPU with potential parallel processing."""
        # Convert scalar to array
        if isinstance(spot_prices, (int, float)):
            spot_prices = np.array([spot_prices])
        
        n_assets = len(spot_prices)
        
        # Generate random numbers
        if self.variance_reduction.get('stratified_sampling', {}).get('enabled'):
            randoms = self._generate_stratified_randoms(n_paths, n_steps, n_assets)
        else:
            randoms = self.rng.generate_standard_normal((n_paths, n_steps, n_assets))
        
        # Apply antithetic variates if enabled
        if self.variance_reduction.get('antithetic'):
            randoms = self._apply_antithetic_variates(randoms)
        
        # Simulate paths based on process type
        if process == StochasticProcess.GEOMETRIC_BROWNIAN_MOTION:
            paths = self._simulate_gbm(spot_prices, randoms, dt, parameters)
        elif process == StochasticProcess.JUMP_DIFFUSION:
            paths = self._simulate_jump_diffusion(spot_prices, randoms, dt, parameters)
        elif process == StochasticProcess.HESTON:
            paths = self._simulate_heston(spot_prices, randoms, dt, parameters)
        elif process == StochasticProcess.VARIANCE_GAMMA:
            paths = self._simulate_variance_gamma(spot_prices, randoms, dt, parameters)
        else:
            raise ValueError(f"Unsupported process: {process}")
        
        return paths
    
    def _simulate_paths_gpu(
        self,
        spot_prices: Union[float, np.ndarray],
        process: StochasticProcess,
        n_paths: int,
        n_steps: int,
        dt: float,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Simulate paths on GPU using CuPy."""
        if not self.use_gpu or cp is None:
            return self._simulate_paths_cpu(
                spot_prices, process, n_paths, n_steps, dt, parameters
            )
        
        # Transfer data to GPU
        spot_prices_gpu = cp.asarray(spot_prices)
        
        # Generate random numbers on GPU
        randoms_gpu = cp.random.standard_normal((n_paths, n_steps, len(spot_prices)))
        
        # Simulate on GPU
        if process == StochasticProcess.GEOMETRIC_BROWNIAN_MOTION:
            paths_gpu = self._simulate_gbm_gpu(
                spot_prices_gpu, randoms_gpu, dt, parameters
            )
        else:
            # Fallback to CPU for unsupported processes
            logger.warning(f"GPU simulation not implemented for {process}, using CPU")
            return self._simulate_paths_cpu(
                spot_prices, process, n_paths, n_steps, dt, parameters
            )
        
        # Transfer back to CPU
        return cp.asnumpy(paths_gpu)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _simulate_gbm(
        spot_prices: np.ndarray,
        randoms: np.ndarray,
        dt: float,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Simulate Geometric Brownian Motion paths using Numba."""
        n_paths, n_steps, n_assets = randoms.shape
        paths = np.zeros((n_paths, n_steps + 1, n_assets))
        
        # Extract parameters
        mu = parameters.get('drift', 0.05)
        sigma = parameters.get('volatility', 0.2)
        
        # Set initial prices
        for i in prange(n_paths):
            paths[i, 0, :] = spot_prices
        
        # Simulate paths
        sqrt_dt = np.sqrt(dt)
        drift = (mu - 0.5 * sigma**2) * dt
        
        for i in prange(n_paths):
            for t in range(n_steps):
                for j in range(n_assets):
                    paths[i, t + 1, j] = paths[i, t, j] * np.exp(
                        drift + sigma * sqrt_dt * randoms[i, t, j]
                    )
        
        return paths
    
    def _simulate_gbm_gpu(
        self,
        spot_prices: Any,  # cp.ndarray
        randoms: Any,  # cp.ndarray
        dt: float,
        parameters: Dict[str, Any]
    ) -> Any:  # cp.ndarray
        """Simulate GBM on GPU using CuPy."""
        n_paths, n_steps, n_assets = randoms.shape
        paths = cp.zeros((n_paths, n_steps + 1, n_assets))
        
        # Parameters
        mu = parameters.get('drift', 0.05)
        sigma = parameters.get('volatility', 0.2)
        
        # Initial prices
        paths[:, 0, :] = spot_prices
        
        # Vectorized simulation
        sqrt_dt = cp.sqrt(dt)
        drift = (mu - 0.5 * sigma**2) * dt
        
        for t in range(n_steps):
            paths[:, t + 1, :] = paths[:, t, :] * cp.exp(
                drift + sigma * sqrt_dt * randoms[:, t, :]
            )
        
        return paths
    
    def _simulate_jump_diffusion(
        self,
        spot_prices: np.ndarray,
        randoms: np.ndarray,
        dt: float,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Simulate Merton jump-diffusion process."""
        n_paths, n_steps, n_assets = randoms.shape
        paths = np.zeros((n_paths, n_steps + 1, n_assets))
        
        # Parameters
        mu = parameters.get('drift', 0.05)
        sigma = parameters.get('volatility', 0.2)
        jump_intensity = parameters.get('jump_intensity', 0.1)
        jump_mean = parameters.get('jump_mean', 0.0)
        jump_std = parameters.get('jump_std', 0.1)
        
        # Initial prices
        paths[:, 0, :] = spot_prices
        
        # Generate jump times and sizes
        n_jumps = np.random.poisson(jump_intensity * dt, (n_paths, n_steps, n_assets))
        jump_sizes = np.random.normal(jump_mean, jump_std, (n_paths, n_steps, n_assets))
        
        # Simulate paths
        sqrt_dt = np.sqrt(dt)
        k = np.exp(jump_mean + 0.5 * jump_std**2) - 1  # Compensator
        drift = (mu - 0.5 * sigma**2 - jump_intensity * k) * dt
        
        for t in range(n_steps):
            diffusion = np.exp(drift + sigma * sqrt_dt * randoms[:, t, :])
            jumps = np.exp(n_jumps[:, t, :] * jump_sizes[:, t, :])
            paths[:, t + 1, :] = paths[:, t, :] * diffusion * jumps
        
        return paths
    
    def _simulate_heston(
        self,
        spot_prices: np.ndarray,
        randoms: np.ndarray,
        dt: float,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Simulate Heston stochastic volatility model."""
        n_paths, n_steps, n_assets = randoms.shape
        
        # Need two sets of random numbers for price and volatility
        randoms_price = randoms
        randoms_vol = self.rng.generate_standard_normal((n_paths, n_steps, n_assets))
        
        # Parameters
        mu = parameters.get('drift', 0.05)
        kappa = parameters.get('mean_reversion', 2.0)
        theta = parameters.get('long_term_variance', 0.04)
        xi = parameters.get('vol_of_vol', 0.3)
        rho = parameters.get('correlation', -0.7)
        v0 = parameters.get('initial_variance', 0.04)
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1, n_assets))
        variances = np.zeros((n_paths, n_steps + 1, n_assets))
        
        paths[:, 0, :] = spot_prices
        variances[:, 0, :] = v0
        
        # Correlate random numbers
        randoms_vol = rho * randoms_price + np.sqrt(1 - rho**2) * randoms_vol
        
        # Simulate using Euler scheme with reflection
        sqrt_dt = np.sqrt(dt)
        
        for t in range(n_steps):
            # Variance process (Cox-Ingersoll-Ross)
            sqrt_v = np.sqrt(np.maximum(variances[:, t, :], 0))
            variances[:, t + 1, :] = variances[:, t, :] + kappa * (
                theta - variances[:, t, :]
            ) * dt + xi * sqrt_v * sqrt_dt * randoms_vol[:, t, :]
            
            # Ensure variance stays positive (reflection)
            variances[:, t + 1, :] = np.abs(variances[:, t + 1, :])
            
            # Price process
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(
                (mu - 0.5 * variances[:, t, :]) * dt +
                sqrt_v * sqrt_dt * randoms_price[:, t, :]
            )
        
        return paths
    
    def _simulate_variance_gamma(
        self,
        spot_prices: np.ndarray,
        randoms: np.ndarray,
        dt: float,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Simulate Variance Gamma process."""
        n_paths, n_steps, n_assets = randoms.shape
        
        # Parameters
        mu = parameters.get('drift', 0.05)
        sigma = parameters.get('volatility', 0.2)
        nu = parameters.get('variance_rate', 0.2)
        theta_vg = parameters.get('skewness', -0.1)
        
        # Generate gamma time changes
        gamma_randoms = np.random.gamma(dt / nu, nu, (n_paths, n_steps, n_assets))
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1, n_assets))
        paths[:, 0, :] = spot_prices
        
        # VG parameters
        omega = np.log(1 - theta_vg * nu - 0.5 * sigma**2 * nu) / nu
        
        # Simulate paths
        for t in range(n_steps):
            # Time-changed Brownian motion
            subordinated_bm = theta_vg * gamma_randoms[:, t, :] + \
                            sigma * np.sqrt(gamma_randoms[:, t, :]) * randoms[:, t, :]
            
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(
                (mu + omega) * dt + subordinated_bm
            )
        
        return paths
    
    def _generate_stratified_randoms(
        self,
        n_paths: int,
        n_steps: int,
        n_assets: int
    ) -> np.ndarray:
        """Generate stratified random samples for variance reduction."""
        n_strata = self.variance_reduction['stratified_sampling']['n_strata']
        paths_per_stratum = n_paths // n_strata
        
        randoms = np.zeros((n_paths, n_steps, n_assets))
        
        for i in range(n_strata):
            start_idx = i * paths_per_stratum
            end_idx = (i + 1) * paths_per_stratum if i < n_strata - 1 else n_paths
            
            # Generate uniform random numbers in stratum
            u = np.random.uniform(i / n_strata, (i + 1) / n_strata,
                                (end_idx - start_idx, n_steps, n_assets))
            
            # Convert to standard normal using inverse CDF
            randoms[start_idx:end_idx] = stats.norm.ppf(u)
        
        return randoms
    
    def _apply_antithetic_variates(self, randoms: np.ndarray) -> np.ndarray:
        """Apply antithetic variates for variance reduction."""
        n_paths = randoms.shape[0]
        n_antithetic = n_paths // 2
        
        # Create antithetic pairs
        randoms[:n_antithetic] = randoms[n_antithetic:2*n_antithetic]
        randoms[n_antithetic:2*n_antithetic] = -randoms[n_antithetic:2*n_antithetic]
        
        return randoms
    
    def _apply_variance_reduction(
        self,
        paths: np.ndarray,
        spot_prices: np.ndarray
    ) -> np.ndarray:
        """Apply variance reduction techniques to simulated paths."""
        if self.variance_reduction.get('control_variate', {}).get('enabled'):
            paths = self._apply_control_variates(paths, spot_prices)
        
        if self.variance_reduction.get('importance_sampling', {}).get('enabled'):
            paths = self._apply_importance_sampling(paths)
        
        return paths
    
    def _apply_control_variates(
        self,
        paths: np.ndarray,
        spot_prices: np.ndarray
    ) -> np.ndarray:
        """Apply control variate variance reduction."""
        # Use geometric average as control variate
        geometric_avg = np.exp(np.mean(np.log(paths[:, 1:, :]), axis=1))
        
        # Theoretical value of geometric average (simplified)
        final_prices = paths[:, -1, :]
        
        # Calculate correlation
        for i in range(paths.shape[2]):  # For each asset
            corr = np.corrcoef(final_prices[:, i], geometric_avg[:, i])[0, 1]
            
            if abs(corr) > self.variance_reduction['control_variate']['correlation_threshold']:
                # Apply control variate correction
                c = -np.cov(final_prices[:, i], geometric_avg[:, i])[0, 1] / \
                    np.var(geometric_avg[:, i])
                
                # Correct paths
                correction = c * (geometric_avg[:, i] - np.mean(geometric_avg[:, i]))
                paths[:, -1, i] += correction
        
        return paths
    
    def _apply_importance_sampling(self, paths: np.ndarray) -> np.ndarray:
        """Apply importance sampling for tail events."""
        # This is a simplified implementation
        # In practice, would need to adjust probability measure
        return paths
    
    def calculate_var_cvar(
        self,
        portfolio_returns: np.ndarray,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[float, Tuple[float, float]]:
        """
        Calculate Value at Risk and Conditional Value at Risk.
        
        Args:
            portfolio_returns: Array of portfolio returns
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary mapping confidence level to (VaR, CVaR) tuple
        """
        results = {}
        
        for confidence in confidence_levels:
            # VaR is the quantile
            var = np.percentile(portfolio_returns, (1 - confidence) * 100)
            
            # CVaR is the expected value beyond VaR
            cvar = np.mean(portfolio_returns[portfolio_returns <= var])
            
            results[confidence] = (var, cvar)
        
        return results
    
    def run_portfolio_simulation(
        self,
        portfolio: Dict[str, float],
        spot_prices: Dict[str, float],
        correlations: np.ndarray,
        n_scenarios: int,
        time_horizon: float,
        n_steps: int = 252
    ) -> SimulationResults:
        """
        Run full portfolio Monte Carlo simulation.
        
        Args:
            portfolio: Dictionary of asset weights
            spot_prices: Current prices for each asset
            correlations: Correlation matrix
            n_scenarios: Number of scenarios to simulate
            time_horizon: Time horizon in years
            n_steps: Number of time steps
            
        Returns:
            Complete simulation results
        """
        logger.info(f"Running portfolio simulation with {n_scenarios} scenarios")
        
        # Extract assets and weights
        assets = list(portfolio.keys())
        weights = np.array([portfolio[asset] for asset in assets])
        spots = np.array([spot_prices[asset] for asset in assets])
        
        # Time parameters
        dt = time_horizon / n_steps
        
        # Simulate correlated paths
        paths = self._simulate_correlated_paths(
            spots, correlations, n_scenarios, n_steps, dt
        )
        
        # Calculate portfolio values
        portfolio_values = np.zeros((n_scenarios, n_steps + 1))
        for i in range(len(assets)):
            portfolio_values += weights[i] * paths.paths[:, :, i]
        
        # Calculate returns
        portfolio_returns = portfolio_values[:, -1] / portfolio_values[:, 0] - 1
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            portfolio_returns, portfolio_values
        )
        
        # Check convergence
        convergence = self._check_convergence(portfolio_returns)
        
        return SimulationResults(
            config=self.config,
            paths=paths,
            portfolio_returns=portfolio_returns,
            portfolio_values=portfolio_values,
            risk_metrics=risk_metrics,
            convergence_metrics=convergence,
            simulation_time=self.simulation_stats['simulation_time'],
            n_scenarios=n_scenarios
        )
    
    def _simulate_correlated_paths(
        self,
        spot_prices: np.ndarray,
        correlations: np.ndarray,
        n_scenarios: int,
        n_steps: int,
        dt: float
    ) -> PathResults:
        """Simulate correlated asset paths."""
        n_assets = len(spot_prices)
        
        # Generate correlated random numbers
        randoms = self.rng.generate_standard_normal((n_scenarios, n_steps, n_assets))
        
        # Apply correlation using Cholesky decomposition
        cholesky = np.linalg.cholesky(correlations)
        
        for t in range(n_steps):
            randoms[:, t, :] = randoms[:, t, :] @ cholesky.T
        
        # Simulate paths using configured process
        paths = self._simulate_paths_cpu(
            spot_prices,
            self.config.default_process,
            n_scenarios,
            n_steps,
            dt,
            self.config.process_parameters
        )
        
        return PathResults(
            paths=paths,
            n_paths=n_scenarios,
            n_steps=n_steps,
            dt=dt,
            process=self.config.default_process,
            simulation_time=0.0
        )
    
    def _calculate_risk_metrics(
        self,
        returns: np.ndarray,
        values: np.ndarray
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics from simulation results."""
        # VaR and CVaR
        var_cvar = self.calculate_var_cvar(returns)
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(values, axis=1)
        drawdowns = (values - running_max) / running_max
        max_drawdown = np.min(drawdowns, axis=1)
        
        # Other metrics
        return RiskMetrics(
            var_95=var_cvar[0.95][0],
            cvar_95=var_cvar[0.95][1],
            var_99=var_cvar[0.99][0],
            cvar_99=var_cvar[0.99][1],
            expected_return=np.mean(returns),
            volatility=np.std(returns),
            sharpe_ratio=np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            max_drawdown=np.mean(max_drawdown),
            skewness=stats.skew(returns),
            kurtosis=stats.kurtosis(returns),
            downside_deviation=np.std(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 0
        )
    
    def _check_convergence(self, returns: np.ndarray) -> ConvergenceMetrics:
        """Check simulation convergence using various metrics."""
        n_samples = len(returns)
        
        # Calculate running statistics
        running_mean = np.zeros(n_samples)
        running_std = np.zeros(n_samples)
        running_var = np.zeros(n_samples)
        
        for i in range(1, n_samples + 1):
            sample = returns[:i]
            running_mean[i-1] = np.mean(sample)
            running_std[i-1] = np.std(sample) if i > 1 else 0
            running_var[i-1] = np.percentile(sample, 5) if i > 20 else 0
        
        # Calculate convergence metrics
        mean_convergence = np.std(running_mean[-1000:]) if n_samples > 1000 else np.inf
        std_convergence = np.std(running_std[-1000:]) if n_samples > 1000 else np.inf
        var_convergence = np.std(running_var[-1000:]) if n_samples > 1000 else np.inf
        
        # Standard error
        standard_error = np.std(returns) / np.sqrt(n_samples)
        
        # Effective sample size (accounting for autocorrelation)
        if n_samples > 10:
            autocorr = np.correlate(returns - np.mean(returns), 
                                  returns - np.mean(returns), mode='full')
            autocorr = autocorr[n_samples-1:] / autocorr[n_samples-1]
            
            # Integrated autocorrelation time
            sum_autocorr = 1 + 2 * np.sum(autocorr[1:min(100, n_samples//4)])
            effective_sample_size = n_samples / sum_autocorr
        else:
            effective_sample_size = n_samples
        
        return ConvergenceMetrics(
            mean_convergence=mean_convergence,
            std_convergence=std_convergence,
            var_convergence=var_convergence,
            standard_error=standard_error,
            effective_sample_size=int(effective_sample_size),
            is_converged=mean_convergence < 0.001 and std_convergence < 0.001
        )
    
    def parallel_scenario_simulation(
        self,
        scenarios: List[Dict[str, Any]],
        portfolio: Dict[str, float],
        base_prices: Dict[str, float]
    ) -> List[ScenarioResults]:
        """
        Run multiple scenarios in parallel.
        
        Args:
            scenarios: List of scenario configurations
            portfolio: Portfolio weights
            base_prices: Base asset prices
            
        Returns:
            List of scenario results
        """
        logger.info(f"Running {len(scenarios)} scenarios in parallel")
        
        # Create partial function for scenario simulation
        simulate_scenario = partial(
            self._simulate_single_scenario,
            portfolio=portfolio,
            base_prices=base_prices
        )
        
        # Run scenarios in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(simulate_scenario, scenarios))
        
        return results
    
    def _simulate_single_scenario(
        self,
        scenario: Dict[str, Any],
        portfolio: Dict[str, float],
        base_prices: Dict[str, float]
    ) -> ScenarioResults:
        """Simulate a single scenario."""
        # Apply scenario shocks to base prices
        shocked_prices = {}
        for asset, base_price in base_prices.items():
            shock = scenario.get('price_shocks', {}).get(asset, 0.0)
            shocked_prices[asset] = base_price * (1 + shock)
        
        # Update parameters for scenario
        scenario_params = self.config.process_parameters.copy()
        scenario_params.update(scenario.get('parameter_changes', {}))
        
        # Run simulation with shocked prices
        results = self.run_portfolio_simulation(
            portfolio=portfolio,
            spot_prices=shocked_prices,
            correlations=scenario.get('correlations', np.eye(len(portfolio))),
            n_scenarios=scenario.get('n_paths', 10000),
            time_horizon=scenario.get('time_horizon', 1.0)
        )
        
        return ScenarioResults(
            scenario_name=scenario['name'],
            scenario_type=scenario.get('type', 'custom'),
            probability=scenario.get('probability', 1.0 / len(scenarios)),
            simulation_results=results,
            impact_summary={
                'expected_loss': results.risk_metrics.expected_return,
                'var_95': results.risk_metrics.var_95,
                'max_drawdown': results.risk_metrics.max_drawdown
            }
        )
    
    def calculate_greeks(
        self,
        option_type: str,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        n_simulations: int = 100000
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using Monte Carlo simulation.
        
        Args:
            option_type: 'call' or 'put'
            spot: Current asset price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            volatility: Asset volatility
            n_simulations: Number of simulation paths
            
        Returns:
            Dictionary of Greeks
        """
        # Base case price
        base_price = self._price_option_mc(
            option_type, spot, strike, time_to_expiry,
            risk_free_rate, volatility, n_simulations
        )
        
        # Delta - price sensitivity to spot
        bump = spot * 0.01
        price_up = self._price_option_mc(
            option_type, spot + bump, strike, time_to_expiry,
            risk_free_rate, volatility, n_simulations
        )
        delta = (price_up - base_price) / bump
        
        # Gamma - delta sensitivity to spot
        price_down = self._price_option_mc(
            option_type, spot - bump, strike, time_to_expiry,
            risk_free_rate, volatility, n_simulations
        )
        gamma = (price_up - 2 * base_price + price_down) / (bump ** 2)
        
        # Vega - price sensitivity to volatility
        vol_bump = 0.01
        price_vol_up = self._price_option_mc(
            option_type, spot, strike, time_to_expiry,
            risk_free_rate, volatility + vol_bump, n_simulations
        )
        vega = (price_vol_up - base_price) / vol_bump
        
        # Theta - price sensitivity to time
        if time_to_expiry > 1/252:  # More than one day
            time_bump = 1/252  # One day
            price_time = self._price_option_mc(
                option_type, spot, strike, time_to_expiry - time_bump,
                risk_free_rate, volatility, n_simulations
            )
            theta = (price_time - base_price) / time_bump
        else:
            theta = 0.0
        
        # Rho - price sensitivity to interest rate
        rate_bump = 0.01
        price_rate_up = self._price_option_mc(
            option_type, spot, strike, time_to_expiry,
            risk_free_rate + rate_bump, volatility, n_simulations
        )
        rho = (price_rate_up - base_price) / rate_bump
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def _price_option_mc(
        self,
        option_type: str,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        n_simulations: int
    ) -> float:
        """Price option using Monte Carlo simulation."""
        # Simulate paths
        dt = time_to_expiry
        randoms = self.rng.generate_standard_normal(n_simulations)
        
        # Terminal prices
        terminal_prices = spot * np.exp(
            (risk_free_rate - 0.5 * volatility**2) * dt +
            volatility * np.sqrt(dt) * randoms
        )
        
        # Payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(terminal_prices - strike, 0)
        else:
            payoffs = np.maximum(strike - terminal_prices, 0)
        
        # Discounted expected payoff
        option_price = np.exp(-risk_free_rate * time_to_expiry) * np.mean(payoffs)
        
        return option_price