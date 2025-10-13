"""
Data models for risk scenarios and simulation configuration.

Defines structures for various types of scenarios including market crashes,
macro shocks, stress tests, and tail events.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from enum import Enum


class ScenarioType(Enum):
    """Types of risk scenarios."""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"
    STRESS = "stress"
    REVERSE_STRESS = "reverse_stress"
    TAIL_RISK = "tail_risk"
    FACTOR = "factor"
    MACRO = "macro"
    REGULATORY = "regulatory"


class StochasticProcess(Enum):
    """Supported stochastic processes."""
    GEOMETRIC_BROWNIAN_MOTION = "gbm"
    JUMP_DIFFUSION = "jump_diffusion"
    HESTON = "heston"
    VARIANCE_GAMMA = "variance_gamma"
    SABR = "sabr"
    VASICEK = "vasicek"
    CIR = "cir"
    HULL_WHITE = "hull_white"
    MULTI_FACTOR = "multi_factor"


class VarianceReduction(Enum):
    """Variance reduction techniques."""
    NONE = "none"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    IMPORTANCE_SAMPLING = "importance_sampling"
    STRATIFIED_SAMPLING = "stratified_sampling"
    QUASI_RANDOM = "quasi_random"
    MOMENT_MATCHING = "moment_matching"


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    # Basic settings
    n_scenarios: int = 10000
    time_horizon: float = 1.0  # Years
    n_steps: int = 252  # Time steps
    
    # Random number generation
    random_seed: Optional[int] = None
    use_quasi_random: bool = False
    
    # Process settings
    default_process: StochasticProcess = StochasticProcess.GEOMETRIC_BROWNIAN_MOTION
    process_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Variance reduction
    variance_reduction: List[VarianceReduction] = field(
        default_factory=lambda: [VarianceReduction.NONE]
    )
    
    # Performance settings
    use_gpu: bool = False
    n_workers: int = 4
    batch_size: int = 1000
    
    # Convergence criteria
    convergence_check_interval: int = 1000
    convergence_tolerance: float = 0.001
    max_iterations: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.n_scenarios <= 0:
            errors.append("n_scenarios must be positive")
        
        if self.time_horizon <= 0:
            errors.append("time_horizon must be positive")
        
        if self.n_steps <= 0:
            errors.append("n_steps must be positive")
        
        if self.n_workers <= 0:
            errors.append("n_workers must be positive")
        
        return errors


@dataclass
class Scenario:
    """Base class for all scenario types."""
    name: str
    scenario_type: ScenarioType
    probability: float
    time_horizon: float  # Years
    description: Optional[str] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get scenario summary."""
        return {
            'name': self.name,
            'type': self.scenario_type.value,
            'probability': self.probability,
            'time_horizon': self.time_horizon,
            'description': self.description
        }


@dataclass
class MarketScenario(Scenario):
    """Market-specific scenario."""
    # All fields need defaults since parent class has defaults
    asset_returns: Dict[str, float] = field(default_factory=dict)
    volatility_shocks: Dict[str, float] = field(default_factory=dict)
    correlation_changes: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Optional fields with defaults
    vix_level: Optional[float] = None
    market_regime: Optional[str] = None
    liquidity_conditions: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    historical_reference: Optional[str] = None
    factor_exposures: Optional[Dict[str, float]] = None
    
    def apply_to_portfolio(
        self,
        portfolio_weights: Dict[str, float]
    ) -> float:
        """Apply scenario to portfolio and get return."""
        portfolio_return = sum(
            portfolio_weights.get(asset, 0) * return_val
            for asset, return_val in self.asset_returns.items()
        )
        return portfolio_return


@dataclass
class MacroeconomicScenario(Scenario):
    """Macroeconomic scenario."""
    # All fields need defaults since parent class has optional field
    # Asset returns
    asset_returns: Dict[str, float] = field(default_factory=dict)

    # Macro variables
    gdp_shock: float = 0.0
    inflation_shock: float = 0.0
    interest_rate_shock: float = 0.0
    unemployment_change: float = 0.0

    # Financial conditions
    credit_spread_change: float = 0.0  # In basis points
    term_spread_change: Optional[float] = None
    
    # Currency impacts
    fx_changes: Optional[Dict[str, float]] = None  # Currency -> change
    
    # Sector impacts
    sector_impacts: Optional[Dict[str, float]] = None
    
    def get_macro_summary(self) -> Dict[str, float]:
        """Get summary of macro shocks."""
        return {
            'gdp_impact': self.gdp_shock,
            'inflation_impact': self.inflation_shock,
            'rate_impact': self.interest_rate_shock,
            'unemployment_impact': self.unemployment_change,
            'credit_spread_impact': self.credit_spread_change
        }


@dataclass
class StressScenario(Scenario):
    """Stress test scenario."""
    # All fields need defaults since parent class has optional field
    # Asset returns
    asset_returns: Dict[str, float] = field(default_factory=dict)

    # Systemic risk indicators
    systemic_risk_indicators: Dict[str, Any] = field(default_factory=dict)

    # Contagion effects
    contagion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    # Severity metrics
    severity_score: float = 0.0  # 0-1 scale
    recovery_time: Optional[float] = None  # Expected recovery in years
    
    # Regulatory mapping
    regulatory_scenario: Optional[str] = None  # e.g., "CCAR Severely Adverse"
    
    def calculate_total_impact(
        self,
        portfolio_weights: Dict[str, float],
        include_contagion: bool = True
    ) -> float:
        """Calculate total impact including contagion effects."""
        # Direct impact
        direct_impact = sum(
            portfolio_weights.get(asset, 0) * return_val
            for asset, return_val in self.asset_returns.items()
        )
        
        if not include_contagion:
            return direct_impact
        
        # Add contagion effects
        assets = list(self.asset_returns.keys())
        weights_vector = np.array([portfolio_weights.get(a, 0) for a in assets])
        returns_vector = np.array([self.asset_returns[a] for a in assets])
        
        # Apply contagion
        contagion_impact = weights_vector @ self.contagion_matrix @ returns_vector
        
        return direct_impact + contagion_impact * 0.1  # Scale contagion effect


@dataclass
class TailRiskScenario(Scenario):
    """Tail risk scenario using extreme value theory."""
    # All fields need defaults since parent class has optional field
    # Asset returns (extreme events)
    asset_returns: Dict[str, float] = field(default_factory=dict)

    # Tail characteristics
    tail_probability: float = 0.01  # e.g., 1%, 0.1%
    extreme_value_parameters: Dict[str, Any] = field(default_factory=dict)  # EVT parameters

    # Risk metrics
    expected_shortfall: float = 0.0
    tail_dependence: Optional[np.ndarray] = None  # Tail correlation
    
    # Jump parameters (if applicable)
    jump_sizes: Optional[Dict[str, float]] = None
    jump_probabilities: Optional[Dict[str, float]] = None
    
    def is_black_swan(self) -> bool:
        """Check if scenario qualifies as black swan event."""
        return (
            self.tail_probability < 0.001 and  # Less than 0.1%
            abs(self.expected_shortfall) > 0.30  # More than 30% loss
        )


@dataclass
class ReverseStressScenario(Scenario):
    """Reverse stress test scenario."""
    # All fields need defaults since parent class has optional field
    # Target outcome
    target_loss: float = 0.0  # Target portfolio loss
    target_metric: str = "portfolio_value"  # e.g., "portfolio_value", "var_breach"

    # Derived shocks that achieve target
    asset_shocks: Dict[str, float] = field(default_factory=dict)
    parameter_shocks: Dict[str, float] = field(default_factory=dict)

    # Plausibility assessment
    plausibility_score: float = 0.0  # 0-1 scale
    most_likely_path: str = ""  # Description of most likely path

    # Contributing factors
    key_risk_drivers: List[str] = field(default_factory=list)
    vulnerability_points: List[str] = field(default_factory=list)


@dataclass
class ScenarioSet:
    """Collection of scenarios for comprehensive analysis."""
    scenarios: List[Scenario]
    generation_date: datetime
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    
    # Generation parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    total_probability: Optional[float] = None
    is_complete: bool = False
    
    def __post_init__(self):
        """Calculate total probability."""
        self.total_probability = sum(s.probability for s in self.scenarios)
    
    def get_scenarios_by_type(
        self,
        scenario_type: ScenarioType
    ) -> List[Scenario]:
        """Get scenarios of specific type."""
        return [s for s in self.scenarios if s.scenario_type == scenario_type]
    
    def get_worst_scenarios(
        self,
        n: int = 10,
        metric: str = 'expected_return'
    ) -> List[Scenario]:
        """Get worst n scenarios by specified metric."""
        if metric == 'expected_return':
            # For market scenarios with asset returns
            market_scenarios = [
                s for s in self.scenarios
                if isinstance(s, (MarketScenario, MacroeconomicScenario))
            ]
            
            # Calculate average return for each scenario
            scenario_returns = []
            for s in market_scenarios:
                avg_return = np.mean(list(s.asset_returns.values()))
                scenario_returns.append((s, avg_return))
            
            # Sort by return (ascending for worst)
            scenario_returns.sort(key=lambda x: x[1])
            
            return [s for s, _ in scenario_returns[:n]]
        
        return self.scenarios[:n]
    
    def normalize_probabilities(self):
        """Normalize scenario probabilities to sum to 1."""
        if self.total_probability > 0:
            for scenario in self.scenarios:
                scenario.probability /= self.total_probability
            self.total_probability = 1.0


@dataclass
class ScenarioImpact:
    """Impact of a scenario on portfolio."""
    scenario: Scenario
    portfolio_return: float
    portfolio_value_change: float
    
    # Risk metric changes
    var_change: Optional[float] = None
    volatility_change: Optional[float] = None
    sharpe_change: Optional[float] = None
    
    # Position-level impacts
    position_impacts: Optional[Dict[str, float]] = None
    
    # Attribution
    factor_attribution: Optional[Dict[str, float]] = None
    
    def get_severity_classification(self) -> str:
        """Classify impact severity."""
        loss = abs(self.portfolio_return)
        
        if loss > 0.20:
            return "severe"
        elif loss > 0.10:
            return "high"
        elif loss > 0.05:
            return "moderate"
        else:
            return "low"


@dataclass
class ScenarioAnalysisConfig:
    """Configuration for scenario analysis."""
    # Scenario generation
    scenario_types: List[ScenarioType] = field(
        default_factory=lambda: [ScenarioType.HISTORICAL, ScenarioType.STRESS]
    )
    n_scenarios_per_type: int = 100
    
    # Historical calibration
    historical_lookback_years: int = 10
    use_overlapping_periods: bool = True
    block_size_days: int = 60
    
    # Stress testing
    stress_multipliers: List[float] = field(
        default_factory=lambda: [1.5, 2.0, 3.0]
    )
    correlation_stress: float = 0.3
    
    # Tail risk
    tail_percentiles: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.1]
    )
    use_evt: bool = True
    
    # Reverse stress testing
    target_losses: List[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.30]
    )
    
    # Performance
    parallel_scenarios: bool = True
    cache_scenarios: bool = True


@dataclass
class StochasticProcessParameters:
    """Parameters for stochastic processes."""
    # Common parameters
    drift: float = 0.05  # Annual drift/mean
    volatility: float = 0.20  # Annual volatility
    
    # Jump diffusion
    jump_intensity: float = 0.1  # Jumps per year
    jump_mean: float = 0.0
    jump_std: float = 0.1
    
    # Stochastic volatility (Heston)
    mean_reversion: float = 2.0
    long_term_variance: float = 0.04
    vol_of_vol: float = 0.3
    correlation: float = -0.7
    initial_variance: Optional[float] = None
    
    # SABR
    alpha: float = 0.2  # Initial volatility
    beta: float = 0.5  # CEV exponent
    rho_sabr: float = -0.3  # Correlation
    nu: float = 0.3  # Vol of vol
    
    # Interest rate models
    theta: float = 0.03  # Long-term rate
    
    def get_process_params(
        self,
        process: StochasticProcess
    ) -> Dict[str, Any]:
        """Get parameters for specific process."""
        if process == StochasticProcess.GEOMETRIC_BROWNIAN_MOTION:
            return {
                'drift': self.drift,
                'volatility': self.volatility
            }
        elif process == StochasticProcess.JUMP_DIFFUSION:
            return {
                'drift': self.drift,
                'volatility': self.volatility,
                'jump_intensity': self.jump_intensity,
                'jump_mean': self.jump_mean,
                'jump_std': self.jump_std
            }
        elif process == StochasticProcess.HESTON:
            return {
                'drift': self.drift,
                'mean_reversion': self.mean_reversion,
                'long_term_variance': self.long_term_variance,
                'vol_of_vol': self.vol_of_vol,
                'correlation': self.correlation,
                'initial_variance': self.initial_variance or self.long_term_variance
            }
        elif process == StochasticProcess.SABR:
            return {
                'alpha': self.alpha,
                'beta': self.beta,
                'nu': self.nu,
                'rho': self.rho_sabr
            }
        else:
            return {}


@dataclass
class RandomNumberConfig:
    """Configuration for random number generation."""
    generator_type: str = "mersenne_twister"  # or "sobol", "halton"
    seed: Optional[int] = None
    
    # Quasi-random settings
    skip_initial: int = 1000  # Skip first n numbers
    leap_size: int = 1  # Leap frogging for parallel
    
    # Quality settings
    use_box_muller: bool = True  # For normal transformation
    use_inverse_cdf: bool = False  # For general distributions
    
    # Antithetic variates
    use_antithetic: bool = False
    antithetic_pairs: bool = True  # Pair consecutive paths
    
    def get_generator_params(self) -> Dict[str, Any]:
        """Get parameters for random number generator."""
        return {
            'seed': self.seed,
            'skip': self.skip_initial,
            'leap': self.leap_size,
            'box_muller': self.use_box_muller
        }