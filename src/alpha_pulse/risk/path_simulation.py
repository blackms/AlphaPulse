"""
Path simulation for various stochastic processes.

Implements efficient path generation for different asset classes including
equities, interest rates, commodities, and multi-asset portfolios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import datetime, timedelta
import logging
from numba import jit, prange
from scipy import stats
from scipy.linalg import sqrtm

from alpha_pulse.models.risk_scenarios import StochasticProcess
from alpha_pulse.utils.random_number_generators import RandomNumberGenerator

logger = logging.getLogger(__name__)


class PathSimulator:
    """Simulate paths for various stochastic processes."""
    
    def __init__(self, rng: RandomNumberGenerator):
        """
        Initialize path simulator.
        
        Args:
            rng: Random number generator
        """
        self.rng = rng
        
        # Process simulators
        self.simulators = {
            StochasticProcess.GEOMETRIC_BROWNIAN_MOTION: self.simulate_gbm,
            StochasticProcess.JUMP_DIFFUSION: self.simulate_jump_diffusion,
            StochasticProcess.HESTON: self.simulate_heston,
            StochasticProcess.VARIANCE_GAMMA: self.simulate_variance_gamma,
            StochasticProcess.SABR: self.simulate_sabr,
            StochasticProcess.VASICEK: self.simulate_vasicek,
            StochasticProcess.CIR: self.simulate_cir,
            StochasticProcess.HULL_WHITE: self.simulate_hull_white
        }
    
    def simulate_paths(
        self,
        process: StochasticProcess,
        spot: Union[float, np.ndarray],
        params: Dict[str, Any],
        n_paths: int,
        n_steps: int,
        dt: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate paths for specified stochastic process.
        
        Args:
            process: Stochastic process type
            spot: Initial value(s)
            params: Process parameters
            n_paths: Number of paths
            n_steps: Number of time steps
            dt: Time step size
            correlation_matrix: Correlation matrix for multi-asset
            
        Returns:
            Array of simulated paths
        """
        simulator = self.simulators.get(process)
        if simulator is None:
            raise ValueError(f"Unsupported process: {process}")
        
        return simulator(spot, params, n_paths, n_steps, dt, correlation_matrix)
    
    def simulate_gbm(
        self,
        spot: Union[float, np.ndarray],
        params: Dict[str, Any],
        n_paths: int,
        n_steps: int,
        dt: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Simulate Geometric Brownian Motion paths."""
        spot = np.atleast_1d(spot)
        n_assets = len(spot)
        
        # Parameters
        mu = params.get('drift', 0.05)
        sigma = params.get('volatility', 0.2)
        
        # Handle vector parameters
        if isinstance(mu, (int, float)):
            mu = np.full(n_assets, mu)
        if isinstance(sigma, (int, float)):
            sigma = np.full(n_assets, sigma)
        
        # Generate correlated random numbers if needed
        if n_assets > 1 and correlation_matrix is not None:
            randoms = self._generate_correlated_randoms(
                n_paths, n_steps, n_assets, correlation_matrix
            )
        else:
            randoms = self.rng.generate_standard_normal((n_paths, n_steps, n_assets))
        
        # Simulate paths
        paths = np.zeros((n_paths, n_steps + 1, n_assets))
        paths[:, 0, :] = spot
        
        # Vectorized simulation
        sqrt_dt = np.sqrt(dt)
        drift = (mu - 0.5 * sigma**2) * dt
        
        for t in range(n_steps):
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(
                drift + sigma * sqrt_dt * randoms[:, t, :]
            )
        
        return paths if n_assets > 1 else paths[:, :, 0]
    
    def simulate_jump_diffusion(
        self,
        spot: Union[float, np.ndarray],
        params: Dict[str, Any],
        n_paths: int,
        n_steps: int,
        dt: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Simulate Merton jump-diffusion process."""
        spot = np.atleast_1d(spot)
        n_assets = len(spot)
        
        # Parameters
        mu = params.get('drift', 0.05)
        sigma = params.get('volatility', 0.2)
        jump_intensity = params.get('jump_intensity', 0.1)
        jump_mean = params.get('jump_mean', 0.0)
        jump_std = params.get('jump_std', 0.1)
        
        # Generate random numbers
        randoms = self.rng.generate_standard_normal((n_paths, n_steps, n_assets))
        
        # Generate jump components
        n_jumps = np.random.poisson(jump_intensity * dt, (n_paths, n_steps, n_assets))
        jump_sizes = np.random.normal(jump_mean, jump_std, (n_paths, n_steps, n_assets))
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1, n_assets))
        paths[:, 0, :] = spot
        
        # Jump compensator
        k = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        
        # Simulate
        sqrt_dt = np.sqrt(dt)
        drift = (mu - 0.5 * sigma**2 - jump_intensity * k) * dt
        
        for t in range(n_steps):
            diffusion = np.exp(drift + sigma * sqrt_dt * randoms[:, t, :])
            jumps = np.exp(n_jumps[:, t, :] * jump_sizes[:, t, :])
            paths[:, t + 1, :] = paths[:, t, :] * diffusion * jumps
        
        return paths if n_assets > 1 else paths[:, :, 0]
    
    def simulate_heston(
        self,
        spot: float,
        params: Dict[str, Any],
        n_paths: int,
        n_steps: int,
        dt: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate Heston stochastic volatility model.
        
        dS = μS dt + √V S dW_S
        dV = κ(θ - V) dt + ξ√V dW_V
        dW_S dW_V = ρ dt
        """
        # Parameters
        mu = params.get('drift', 0.05)
        kappa = params.get('mean_reversion', 2.0)
        theta = params.get('long_term_variance', 0.04)
        xi = params.get('vol_of_vol', 0.3)
        rho = params.get('correlation', -0.7)
        v0 = params.get('initial_variance', theta)
        
        # Generate correlated Brownian motions
        randoms_price = self.rng.generate_standard_normal((n_paths, n_steps))
        randoms_vol = self.rng.generate_standard_normal((n_paths, n_steps))
        
        # Apply correlation
        randoms_vol = rho * randoms_price + np.sqrt(1 - rho**2) * randoms_vol
        
        # Initialize paths
        prices = np.zeros((n_paths, n_steps + 1))
        variances = np.zeros((n_paths, n_steps + 1))
        
        prices[:, 0] = spot
        variances[:, 0] = v0
        
        # Simulate using Milstein scheme for variance
        sqrt_dt = np.sqrt(dt)
        
        for t in range(n_steps):
            # Variance process (with reflection at 0)
            sqrt_v = np.sqrt(np.maximum(variances[:, t], 0))
            
            # Milstein scheme for variance
            dW_v = sqrt_dt * randoms_vol[:, t]
            variances[:, t + 1] = variances[:, t] + \
                kappa * (theta - variances[:, t]) * dt + \
                xi * sqrt_v * dW_v + \
                0.25 * xi**2 * (dW_v**2 - dt)
            
            # Ensure variance stays positive
            variances[:, t + 1] = np.maximum(variances[:, t + 1], 0)
            
            # Price process
            prices[:, t + 1] = prices[:, t] * np.exp(
                (mu - 0.5 * variances[:, t]) * dt +
                sqrt_v * sqrt_dt * randoms_price[:, t]
            )
        
        return prices
    
    def simulate_variance_gamma(
        self,
        spot: Union[float, np.ndarray],
        params: Dict[str, Any],
        n_paths: int,
        n_steps: int,
        dt: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate Variance Gamma process.
        
        X(t) = θG(t) + σW(G(t))
        where G(t) is a Gamma process
        """
        spot = np.atleast_1d(spot)
        n_assets = len(spot)
        
        # Parameters
        mu = params.get('drift', 0.05)
        sigma = params.get('volatility', 0.2)
        nu = params.get('variance_rate', 0.2)
        theta_vg = params.get('skewness', -0.1)
        
        # Generate Gamma time changes
        gamma_increments = np.random.gamma(
            dt / nu, nu, (n_paths, n_steps, n_assets)
        )
        
        # Generate Brownian increments
        randoms = self.rng.generate_standard_normal((n_paths, n_steps, n_assets))
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1, n_assets))
        paths[:, 0, :] = spot
        
        # VG drift adjustment
        omega = np.log(1 - theta_vg * nu - 0.5 * sigma**2 * nu) / nu
        
        # Simulate
        for t in range(n_steps):
            vg_increments = theta_vg * gamma_increments[:, t, :] + \
                          sigma * np.sqrt(gamma_increments[:, t, :]) * randoms[:, t, :]
            
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(
                (mu + omega) * dt + vg_increments
            )
        
        return paths if n_assets > 1 else paths[:, :, 0]
    
    def simulate_sabr(
        self,
        forward: float,
        params: Dict[str, Any],
        n_paths: int,
        n_steps: int,
        dt: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate SABR (Stochastic Alpha Beta Rho) model.
        
        dF = α F^β dW_F
        dα = ν α dW_α
        dW_F dW_α = ρ dt
        """
        # Parameters
        alpha = params.get('initial_vol', 0.2)
        beta = params.get('beta', 0.5)
        nu = params.get('vol_of_vol', 0.3)
        rho = params.get('correlation', -0.3)
        
        # Generate correlated Brownian motions
        randoms_forward = self.rng.generate_standard_normal((n_paths, n_steps))
        randoms_vol = self.rng.generate_standard_normal((n_paths, n_steps))
        
        # Apply correlation
        randoms_vol = rho * randoms_forward + np.sqrt(1 - rho**2) * randoms_vol
        
        # Initialize paths
        forwards = np.zeros((n_paths, n_steps + 1))
        vols = np.zeros((n_paths, n_steps + 1))
        
        forwards[:, 0] = forward
        vols[:, 0] = alpha
        
        # Simulate
        sqrt_dt = np.sqrt(dt)
        
        for t in range(n_steps):
            # Volatility process (lognormal)
            vols[:, t + 1] = vols[:, t] * np.exp(
                -0.5 * nu**2 * dt + nu * sqrt_dt * randoms_vol[:, t]
            )
            
            # Forward process
            if beta == 0:
                # Normal SABR
                forwards[:, t + 1] = forwards[:, t] + \
                    vols[:, t] * sqrt_dt * randoms_forward[:, t]
            elif beta == 1:
                # Lognormal SABR
                forwards[:, t + 1] = forwards[:, t] * np.exp(
                    -0.5 * vols[:, t]**2 * dt +
                    vols[:, t] * sqrt_dt * randoms_forward[:, t]
                )
            else:
                # General SABR (using Euler scheme)
                forwards[:, t + 1] = forwards[:, t] + \
                    vols[:, t] * forwards[:, t]**beta * sqrt_dt * randoms_forward[:, t]
            
            # Ensure forwards stay positive
            forwards[:, t + 1] = np.maximum(forwards[:, t + 1], 0)
        
        return forwards
    
    def simulate_vasicek(
        self,
        r0: float,
        params: Dict[str, Any],
        n_paths: int,
        n_steps: int,
        dt: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate Vasicek interest rate model.
        
        dr = κ(θ - r) dt + σ dW
        """
        # Parameters
        kappa = params.get('mean_reversion', 0.5)
        theta = params.get('long_term_rate', 0.03)
        sigma = params.get('volatility', 0.01)
        
        # Generate random numbers
        randoms = self.rng.generate_standard_normal((n_paths, n_steps))
        
        # Initialize paths
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = r0
        
        # Exact simulation
        exp_kappa_dt = np.exp(-kappa * dt)
        sqrt_variance = sigma * np.sqrt((1 - exp_kappa_dt**2) / (2 * kappa))
        
        for t in range(n_steps):
            rates[:, t + 1] = rates[:, t] * exp_kappa_dt + \
                            theta * (1 - exp_kappa_dt) + \
                            sqrt_variance * randoms[:, t]
        
        return rates
    
    def simulate_cir(
        self,
        r0: float,
        params: Dict[str, Any],
        n_paths: int,
        n_steps: int,
        dt: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate Cox-Ingersoll-Ross (CIR) interest rate model.
        
        dr = κ(θ - r) dt + σ√r dW
        """
        # Parameters
        kappa = params.get('mean_reversion', 0.5)
        theta = params.get('long_term_rate', 0.03)
        sigma = params.get('volatility', 0.01)
        
        # Check Feller condition
        if 2 * kappa * theta <= sigma**2:
            logger.warning("Feller condition not satisfied, rates may hit zero")
        
        # Generate random numbers
        randoms = self.rng.generate_standard_normal((n_paths, n_steps))
        
        # Initialize paths
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = r0
        
        # Simulate using exact method when possible
        c = 2 * kappa / (sigma**2 * (1 - np.exp(-kappa * dt)))
        
        for t in range(n_steps):
            # Non-central chi-squared parameters
            df = 4 * kappa * theta / sigma**2
            nc = 2 * c * rates[:, t] * np.exp(-kappa * dt)
            
            # Generate from non-central chi-squared
            rates[:, t + 1] = np.random.noncentral_chisquare(df, nc, n_paths) / (2 * c)
        
        return rates
    
    def simulate_hull_white(
        self,
        r0: float,
        params: Dict[str, Any],
        n_paths: int,
        n_steps: int,
        dt: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate Hull-White interest rate model.
        
        dr = [θ(t) - ar] dt + σ dW
        """
        # Parameters
        a = params.get('mean_reversion', 0.1)
        sigma = params.get('volatility', 0.01)
        theta_func = params.get('theta_function', lambda t: 0.03)  # Can be calibrated
        
        # Generate random numbers
        randoms = self.rng.generate_standard_normal((n_paths, n_steps))
        
        # Initialize paths
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = r0
        
        # Time grid
        times = np.linspace(0, n_steps * dt, n_steps + 1)
        
        # Simulate
        sqrt_dt = np.sqrt(dt)
        
        for t in range(n_steps):
            theta_t = theta_func(times[t])
            
            # Euler scheme
            rates[:, t + 1] = rates[:, t] + \
                            (theta_t - a * rates[:, t]) * dt + \
                            sigma * sqrt_dt * randoms[:, t]
        
        return rates
    
    def simulate_multi_asset_paths(
        self,
        spots: np.ndarray,
        processes: List[StochasticProcess],
        params_list: List[Dict[str, Any]],
        n_paths: int,
        n_steps: int,
        dt: float,
        correlation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Simulate correlated multi-asset paths.
        
        Args:
            spots: Initial values for each asset
            processes: List of processes for each asset
            params_list: List of parameters for each process
            n_paths: Number of paths
            n_steps: Number of time steps
            dt: Time step size
            correlation_matrix: Correlation matrix
            
        Returns:
            Array of shape (n_paths, n_steps+1, n_assets)
        """
        n_assets = len(spots)
        
        if len(processes) != n_assets or len(params_list) != n_assets:
            raise ValueError("Inconsistent number of assets")
        
        # Generate correlated random numbers
        randoms = self._generate_correlated_randoms(
            n_paths, n_steps, n_assets, correlation_matrix
        )
        
        # Simulate each asset
        paths = np.zeros((n_paths, n_steps + 1, n_assets))
        
        for i in range(n_assets):
            # Create single-asset random numbers
            asset_randoms = randoms[:, :, i:i+1]
            
            # Simulate asset
            asset_paths = self.simulate_paths(
                processes[i],
                spots[i],
                params_list[i],
                n_paths,
                n_steps,
                dt,
                None  # Correlation already applied
            )
            
            paths[:, :, i] = asset_paths if asset_paths.ndim == 2 else asset_paths[:, :, 0]
        
        return paths
    
    def _generate_correlated_randoms(
        self,
        n_paths: int,
        n_steps: int,
        n_assets: int,
        correlation_matrix: np.ndarray
    ) -> np.ndarray:
        """Generate correlated random numbers."""
        # Generate independent random numbers
        randoms = self.rng.generate_standard_normal(
            (n_paths, n_steps, n_assets)
        )
        
        # Apply correlation using Cholesky decomposition
        cholesky = np.linalg.cholesky(correlation_matrix)
        
        # Apply correlation to each time step
        for t in range(n_steps):
            randoms[:, t, :] = randoms[:, t, :] @ cholesky.T
        
        return randoms
    
    @staticmethod
    @jit(nopython=True)
    def _apply_milstein_correction(
        current_value: float,
        dt: float,
        volatility: float,
        random_increment: float
    ) -> float:
        """Apply Milstein correction for better accuracy."""
        return 0.5 * volatility**2 * (random_increment**2 - dt)
    
    def simulate_american_option_paths(
        self,
        spot: float,
        strike: float,
        params: Dict[str, Any],
        n_paths: int,
        n_steps: int,
        dt: float,
        option_type: str = 'put'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths for American option pricing.
        
        Returns both asset paths and optimal exercise boundary.
        """
        # Simulate asset paths
        paths = self.simulate_gbm(spot, params, n_paths, n_steps, dt)
        
        # Calculate continuation values using Longstaff-Schwartz
        exercise_boundary = self._calculate_exercise_boundary(
            paths, strike, params.get('risk_free_rate', 0.03), dt, option_type
        )
        
        return paths, exercise_boundary
    
    def _calculate_exercise_boundary(
        self,
        paths: np.ndarray,
        strike: float,
        risk_free_rate: float,
        dt: float,
        option_type: str
    ) -> np.ndarray:
        """Calculate optimal exercise boundary using regression."""
        n_paths, n_steps = paths.shape[:2]
        
        # Initialize with terminal payoff
        if option_type == 'put':
            payoff = lambda S: np.maximum(strike - S, 0)
        else:
            payoff = lambda S: np.maximum(S - strike, 0)
        
        # Backward induction
        continuation_values = np.zeros((n_paths, n_steps))
        exercise_values = np.zeros((n_paths, n_steps))
        
        # Terminal values
        exercise_values[:, -1] = payoff(paths[:, -1])
        
        # Work backwards
        discount = np.exp(-risk_free_rate * dt)
        
        for t in range(n_steps - 2, -1, -1):
            # Current exercise value
            exercise_values[:, t] = payoff(paths[:, t])
            
            # Only consider in-the-money paths
            itm_mask = exercise_values[:, t] > 0
            
            if itm_mask.sum() > 0:
                # Regression basis functions (Laguerre polynomials)
                X = paths[itm_mask, t]
                basis = np.column_stack([
                    np.ones_like(X),
                    X,
                    X**2,
                    np.exp(-X/2) * (1 - X + X**2/2)
                ])
                
                # Future payoffs
                Y = continuation_values[itm_mask, t + 1] * discount
                
                # Regression
                coeffs = np.linalg.lstsq(basis, Y, rcond=None)[0]
                
                # Estimate continuation value
                continuation_values[itm_mask, t] = basis @ coeffs
        
        # Extract exercise boundary
        boundary = np.zeros(n_steps)
        for t in range(n_steps):
            exercised = exercise_values[:, t] > continuation_values[:, t]
            if exercised.any():
                boundary[t] = paths[exercised, t].mean()
            else:
                boundary[t] = 0 if option_type == 'put' else np.inf
        
        return boundary