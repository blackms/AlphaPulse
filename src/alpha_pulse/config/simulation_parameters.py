"""
Monte Carlo simulation configuration parameters.

Defines default settings for simulation engine, scenario generation,
and risk calculations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from alpha_pulse.models.risk_scenarios import (
    StochasticProcess, VarianceReduction, ScenarioType
)


@dataclass
class SimulationDefaults:
    """Default parameters for Monte Carlo simulations."""
    # Simulation size
    n_scenarios_quick: int = 1000      # For quick analysis
    n_scenarios_standard: int = 10000  # Standard simulation
    n_scenarios_full: int = 100000     # Full simulation
    
    # Time parameters
    time_horizon_daily: float = 1/252  # One day
    time_horizon_weekly: float = 1/52  # One week
    time_horizon_monthly: float = 1/12 # One month
    time_horizon_annual: float = 1.0   # One year
    
    # Time steps
    daily_steps: int = 1
    weekly_steps: int = 5
    monthly_steps: int = 21
    annual_steps: int = 252
    
    # Convergence
    convergence_check_interval: int = 1000
    convergence_tolerance: float = 0.001
    max_convergence_iterations: int = 1000000


@dataclass
class ProcessParameters:
    """Default parameters for stochastic processes."""
    # Geometric Brownian Motion
    gbm_drift: float = 0.05           # 5% annual drift
    gbm_volatility: float = 0.20      # 20% annual volatility
    
    # Jump Diffusion
    jump_intensity: float = 0.1       # 0.1 jumps per year
    jump_mean: float = -0.02          # -2% average jump
    jump_std: float = 0.05            # 5% jump volatility
    
    # Heston Model
    heston_mean_reversion: float = 2.0
    heston_long_term_variance: float = 0.04  # (20% vol)^2
    heston_vol_of_vol: float = 0.3
    heston_correlation: float = -0.7
    
    # SABR Model
    sabr_alpha: float = 0.2           # Initial volatility
    sabr_beta: float = 0.5            # CEV exponent
    sabr_rho: float = -0.3            # Correlation
    sabr_nu: float = 0.3              # Vol of vol
    
    # Interest Rate Models
    vasicek_mean_reversion: float = 0.5
    vasicek_long_term_rate: float = 0.03  # 3%
    vasicek_volatility: float = 0.01
    
    cir_mean_reversion: float = 0.5
    cir_long_term_rate: float = 0.03
    cir_volatility: float = 0.01
    
    def get_default_params(self, process: StochasticProcess) -> Dict[str, float]:
        """Get default parameters for a process."""
        if process == StochasticProcess.GEOMETRIC_BROWNIAN_MOTION:
            return {
                'drift': self.gbm_drift,
                'volatility': self.gbm_volatility
            }
        elif process == StochasticProcess.JUMP_DIFFUSION:
            return {
                'drift': self.gbm_drift,
                'volatility': self.gbm_volatility,
                'jump_intensity': self.jump_intensity,
                'jump_mean': self.jump_mean,
                'jump_std': self.jump_std
            }
        elif process == StochasticProcess.HESTON:
            return {
                'drift': self.gbm_drift,
                'mean_reversion': self.heston_mean_reversion,
                'long_term_variance': self.heston_long_term_variance,
                'vol_of_vol': self.heston_vol_of_vol,
                'correlation': self.heston_correlation
            }
        elif process == StochasticProcess.SABR:
            return {
                'alpha': self.sabr_alpha,
                'beta': self.sabr_beta,
                'rho': self.sabr_rho,
                'nu': self.sabr_nu
            }
        elif process == StochasticProcess.VASICEK:
            return {
                'mean_reversion': self.vasicek_mean_reversion,
                'long_term_rate': self.vasicek_long_term_rate,
                'volatility': self.vasicek_volatility
            }
        elif process == StochasticProcess.CIR:
            return {
                'mean_reversion': self.cir_mean_reversion,
                'long_term_rate': self.cir_long_term_rate,
                'volatility': self.cir_volatility
            }
        else:
            return {}


@dataclass
class ScenarioParameters:
    """Parameters for scenario generation."""
    # Historical scenarios
    historical_lookback_years: int = 10
    historical_block_size_days: int = 60
    historical_overlap_allowed: bool = True
    
    # Stress scenarios
    stress_return_shocks: List[float] = field(
        default_factory=lambda: [-0.10, -0.20, -0.30, -0.40]
    )
    stress_volatility_multipliers: List[float] = field(
        default_factory=lambda: [1.5, 2.0, 3.0, 5.0]
    )
    stress_correlation_increase: float = 0.3
    
    # Tail risk scenarios
    tail_percentiles: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.1, 0.05]
    )
    use_extreme_value_theory: bool = True
    evt_threshold_percentile: float = 10.0  # Use bottom 10% for EVT
    
    # Macro scenarios
    gdp_shocks: List[float] = field(
        default_factory=lambda: [-0.05, -0.03, -0.01, 0.01, 0.03]
    )
    inflation_shocks: List[float] = field(
        default_factory=lambda: [-0.02, 0.0, 0.02, 0.04, 0.06]
    )
    rate_shocks: List[float] = field(
        default_factory=lambda: [-0.02, -0.01, 0.0, 0.01, 0.02]
    )
    
    # Scenario weights
    scenario_type_weights: Dict[ScenarioType, float] = field(
        default_factory=lambda: {
            ScenarioType.HISTORICAL: 0.40,
            ScenarioType.MONTE_CARLO: 0.30,
            ScenarioType.STRESS: 0.20,
            ScenarioType.TAIL_RISK: 0.10
        }
    )


@dataclass
class VarianceReductionSettings:
    """Settings for variance reduction techniques."""
    # Antithetic variates
    use_antithetic: bool = True
    antithetic_pairing: str = "consecutive"  # or "split"
    
    # Control variates
    use_control_variate: bool = True
    control_variate_asset: Optional[str] = None  # Use geometric average if None
    control_correlation_threshold: float = 0.7
    
    # Importance sampling
    use_importance_sampling: bool = False
    importance_shift_factor: float = 0.5
    importance_tail_focus: bool = True
    
    # Stratified sampling
    use_stratified: bool = True
    n_strata: int = 10
    stratification_dimension: str = "returns"  # or "volatility"
    
    # Moment matching
    use_moment_matching: bool = True
    match_moments: List[int] = field(default_factory=lambda: [1, 2])  # Mean, variance
    
    # Quasi-random
    use_quasi_random: bool = False
    quasi_random_type: str = "sobol"  # or "halton"
    quasi_random_scramble: bool = True


@dataclass
class PerformanceSettings:
    """Performance and optimization settings."""
    # Parallelization
    use_parallel: bool = True
    n_workers: int = 4
    chunk_size: int = 1000
    
    # GPU acceleration
    use_gpu: bool = False
    gpu_device_id: int = 0
    gpu_batch_size: int = 10000
    
    # Memory management
    max_memory_gb: float = 8.0
    streaming_mode: bool = False
    cache_results: bool = True
    cache_ttl_hours: float = 24.0
    
    # Batch processing
    batch_mode: bool = True
    max_batch_size: int = 100000
    
    # Progress tracking
    show_progress: bool = True
    progress_update_interval: int = 1000


@dataclass
class RiskCalculationSettings:
    """Settings for risk metric calculations."""
    # VaR/CVaR settings
    var_confidence_levels: List[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.99]
    )
    use_cornish_fisher_var: bool = False  # Adjust for skewness/kurtosis
    
    # Drawdown settings
    calculate_drawdowns: bool = True
    drawdown_lookback_days: int = 252
    
    # Greeks calculation
    greek_bump_size: float = 0.01  # 1% bump
    calculate_cross_gammas: bool = False
    calculate_higher_order: bool = False
    
    # Sensitivity analysis
    sensitivity_parameters: List[str] = field(
        default_factory=lambda: ['volatility', 'drift', 'correlation']
    )
    sensitivity_range: float = 0.20  # +/- 20%
    sensitivity_n_points: int = 5
    
    # Portfolio metrics
    calculate_risk_contributions: bool = True
    calculate_component_var: bool = True
    calculate_diversification_ratio: bool = True


@dataclass
class ValidationSettings:
    """Settings for simulation validation and quality checks."""
    # Convergence checking
    check_convergence: bool = True
    convergence_metrics: List[str] = field(
        default_factory=lambda: ['mean', 'std', 'var_95']
    )
    
    # Statistical tests
    run_normality_tests: bool = True
    run_independence_tests: bool = True
    run_stationarity_tests: bool = False
    
    # Model validation
    backtest_window_days: int = 252
    backtest_confidence_level: float = 0.95
    min_backtest_observations: int = 100
    
    # Quality thresholds
    max_nan_ratio: float = 0.001  # 0.1% NaN tolerance
    max_inf_ratio: float = 0.0001  # 0.01% inf tolerance
    min_effective_sample_ratio: float = 0.5  # ESS/N ratio


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    # Component configurations
    defaults: SimulationDefaults = field(default_factory=SimulationDefaults)
    process_params: ProcessParameters = field(default_factory=ProcessParameters)
    scenario_params: ScenarioParameters = field(default_factory=ScenarioParameters)
    variance_reduction: VarianceReductionSettings = field(
        default_factory=VarianceReductionSettings
    )
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    risk_calculation: RiskCalculationSettings = field(
        default_factory=RiskCalculationSettings
    )
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    
    # Global settings
    random_seed: Optional[int] = 42
    reproducible: bool = True
    debug_mode: bool = False
    
    def get_quick_config(self) -> Dict[str, Any]:
        """Get configuration for quick simulations."""
        return {
            'n_scenarios': self.defaults.n_scenarios_quick,
            'time_horizon': self.defaults.time_horizon_daily,
            'n_steps': self.defaults.daily_steps,
            'use_gpu': False,
            'variance_reduction': [VarianceReduction.ANTITHETIC],
            'convergence_check_interval': 100
        }
    
    def get_standard_config(self) -> Dict[str, Any]:
        """Get configuration for standard simulations."""
        return {
            'n_scenarios': self.defaults.n_scenarios_standard,
            'time_horizon': self.defaults.time_horizon_monthly,
            'n_steps': self.defaults.monthly_steps,
            'use_gpu': self.performance.use_gpu,
            'variance_reduction': [
                VarianceReduction.ANTITHETIC,
                VarianceReduction.CONTROL_VARIATE
            ],
            'convergence_check_interval': self.defaults.convergence_check_interval
        }
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get configuration for comprehensive simulations."""
        variance_reduction = []
        if self.variance_reduction.use_antithetic:
            variance_reduction.append(VarianceReduction.ANTITHETIC)
        if self.variance_reduction.use_control_variate:
            variance_reduction.append(VarianceReduction.CONTROL_VARIATE)
        if self.variance_reduction.use_importance_sampling:
            variance_reduction.append(VarianceReduction.IMPORTANCE_SAMPLING)
        if self.variance_reduction.use_stratified:
            variance_reduction.append(VarianceReduction.STRATIFIED_SAMPLING)
        
        return {
            'n_scenarios': self.defaults.n_scenarios_full,
            'time_horizon': self.defaults.time_horizon_annual,
            'n_steps': self.defaults.annual_steps,
            'use_gpu': self.performance.use_gpu,
            'variance_reduction': variance_reduction,
            'convergence_check_interval': self.defaults.convergence_check_interval,
            'n_workers': self.performance.n_workers
        }


# Pre-configured simulation profiles
QUICK_SIMULATION = SimulationConfig(
    defaults=SimulationDefaults(
        n_scenarios_standard=1000,
        convergence_check_interval=100
    ),
    performance=PerformanceSettings(
        use_gpu=False,
        n_workers=2
    ),
    variance_reduction=VarianceReductionSettings(
        use_control_variate=False,
        use_importance_sampling=False
    )
)

PRODUCTION_SIMULATION = SimulationConfig(
    defaults=SimulationDefaults(
        n_scenarios_standard=10000,
        convergence_tolerance=0.0001
    ),
    performance=PerformanceSettings(
        use_gpu=True,
        n_workers=8,
        cache_results=True
    ),
    validation=ValidationSettings(
        check_convergence=True,
        run_normality_tests=True
    )
)

RESEARCH_SIMULATION = SimulationConfig(
    defaults=SimulationDefaults(
        n_scenarios_standard=100000,
        convergence_tolerance=0.00001,
        max_convergence_iterations=10000000
    ),
    variance_reduction=VarianceReductionSettings(
        use_antithetic=True,
        use_control_variate=True,
        use_importance_sampling=True,
        use_stratified=True,
        use_quasi_random=True
    ),
    validation=ValidationSettings(
        check_convergence=True,
        run_normality_tests=True,
        run_independence_tests=True,
        run_stationarity_tests=True
    )
)


def get_simulation_config(profile: str = "standard") -> SimulationConfig:
    """
    Get pre-configured simulation settings.
    
    Args:
        profile: Configuration profile ('quick', 'production', 'research')
        
    Returns:
        Simulation configuration
    """
    profiles = {
        'quick': QUICK_SIMULATION,
        'production': PRODUCTION_SIMULATION,
        'research': RESEARCH_SIMULATION,
        'standard': SimulationConfig()
    }
    
    return profiles.get(profile, SimulationConfig())