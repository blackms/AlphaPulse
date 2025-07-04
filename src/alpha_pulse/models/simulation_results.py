"""
Data models for Monte Carlo simulation results.

Defines structures for storing and analyzing simulation outputs including
paths, risk metrics, convergence diagnostics, and scenario results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from enum import Enum

from alpha_pulse.models.risk_scenarios import (
    SimulationConfig, StochasticProcess, Scenario
)


@dataclass
class PathResults:
    """Results from path simulation."""
    paths: np.ndarray  # Shape: (n_paths, n_steps+1, n_assets)
    n_paths: int
    n_steps: int
    dt: float
    process: StochasticProcess
    simulation_time: float
    
    @property
    def final_values(self) -> np.ndarray:
        """Get final values from all paths."""
        return self.paths[:, -1, :]
    
    @property
    def returns(self) -> np.ndarray:
        """Calculate returns from paths."""
        return self.paths[:, -1, :] / self.paths[:, 0, :] - 1
    
    def percentile(self, q: float) -> np.ndarray:
        """Get percentile of final values."""
        return np.percentile(self.final_values, q, axis=0)
    
    def get_path_statistics(self) -> Dict[str, np.ndarray]:
        """Calculate statistics across paths."""
        return {
            'mean': np.mean(self.paths, axis=0),
            'std': np.std(self.paths, axis=0),
            'min': np.min(self.paths, axis=0),
            'max': np.max(self.paths, axis=0),
            'median': np.median(self.paths, axis=0)
        }


@dataclass
class ConvergenceMetrics:
    """Metrics for assessing simulation convergence."""
    mean_convergence: float  # Std dev of running mean
    std_convergence: float   # Std dev of running std
    var_convergence: float   # Std dev of running VaR
    standard_error: float    # Standard error of mean
    effective_sample_size: int
    is_converged: bool
    
    def get_summary(self) -> Dict[str, Any]:
        """Get convergence summary."""
        return {
            'converged': self.is_converged,
            'mean_stability': self.mean_convergence < 0.001,
            'std_stability': self.std_convergence < 0.001,
            'var_stability': self.var_convergence < 0.01,
            'effective_samples': self.effective_sample_size,
            'standard_error': self.standard_error
        }


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics from simulation."""
    # Value at Risk metrics
    var_95: float
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    var_99: float
    cvar_99: float
    
    # Return statistics
    expected_return: float
    volatility: float
    sharpe_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: Optional[float] = None
    drawdown_duration: Optional[float] = None
    
    # Higher moments
    skewness: float = 0.0
    kurtosis: float = 3.0
    
    # Downside risk
    downside_deviation: float = 0.0
    sortino_ratio: Optional[float] = None
    
    # Tail risk
    expected_tail_loss: Optional[float] = None
    tail_risk_ratio: Optional[float] = None
    
    def get_risk_summary(self) -> Dict[str, float]:
        """Get summary of key risk metrics."""
        return {
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'downside_risk': self.downside_deviation
        }


@dataclass
class ScenarioResults:
    """Results from a specific scenario simulation."""
    scenario_name: str
    scenario_type: str
    probability: float
    simulation_results: 'SimulationResults'
    impact_summary: Dict[str, float]
    
    @property
    def expected_loss(self) -> float:
        """Expected loss from scenario."""
        return self.simulation_results.risk_metrics.expected_return
    
    @property
    def tail_loss(self) -> float:
        """95% tail loss from scenario."""
        return self.simulation_results.risk_metrics.var_95
    
    def get_weighted_impact(self) -> Dict[str, float]:
        """Get probability-weighted impact."""
        return {
            metric: value * self.probability
            for metric, value in self.impact_summary.items()
        }


@dataclass
class SimulationResults:
    """Complete results from Monte Carlo simulation."""
    config: SimulationConfig
    paths: PathResults
    portfolio_returns: np.ndarray
    portfolio_values: np.ndarray
    risk_metrics: RiskMetrics
    convergence_metrics: ConvergenceMetrics
    simulation_time: float
    n_scenarios: int
    
    # Optional components
    scenario_results: Optional[List[ScenarioResults]] = None
    sensitivity_analysis: Optional[Dict[str, Any]] = None
    stress_test_results: Optional[Dict[str, Any]] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if simulation results are valid."""
        return (
            self.convergence_metrics.is_converged and
            not np.isnan(self.portfolio_returns).any() and
            self.risk_metrics.volatility > 0
        )
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """Get summary statistics of simulation."""
        return {
            'expected_return': self.risk_metrics.expected_return,
            'volatility': self.risk_metrics.volatility,
            'var_95': self.risk_metrics.var_95,
            'cvar_95': self.risk_metrics.cvar_95,
            'max_drawdown': self.risk_metrics.max_drawdown,
            'sharpe_ratio': self.risk_metrics.sharpe_ratio,
            'scenarios_run': self.n_scenarios,
            'simulation_time': self.simulation_time,
            'converged': self.convergence_metrics.is_converged
        }
    
    def get_percentile_returns(
        self,
        percentiles: List[float] = [5, 25, 50, 75, 95]
    ) -> Dict[float, float]:
        """Get return percentiles."""
        return {
            p: np.percentile(self.portfolio_returns, p)
            for p in percentiles
        }
    
    def get_scenario_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of scenario analysis if available."""
        if not self.scenario_results:
            return None
        
        return {
            'n_scenarios': len(self.scenario_results),
            'expected_scenario_loss': sum(
                s.expected_loss * s.probability
                for s in self.scenario_results
            ),
            'worst_scenario': max(
                self.scenario_results,
                key=lambda s: abs(s.expected_loss)
            ).scenario_name,
            'scenario_var_95': np.percentile(
                [s.tail_loss for s in self.scenario_results],
                95
            )
        }


@dataclass
class GreeksResults:
    """Option Greeks calculation results."""
    price: float
    delta: float  # Price sensitivity to underlying
    gamma: float  # Delta sensitivity to underlying
    vega: float   # Price sensitivity to volatility
    theta: float  # Price sensitivity to time
    rho: float    # Price sensitivity to interest rate
    
    # Optional higher-order Greeks
    vanna: Optional[float] = None  # Cross-derivative of delta and vega
    volga: Optional[float] = None  # Second derivative w.r.t. volatility
    charm: Optional[float] = None  # Cross-derivative of delta and time
    
    def get_risk_decomposition(self, bumps: Dict[str, float]) -> Dict[str, float]:
        """Decompose risk based on sensitivities."""
        return {
            'delta_risk': self.delta * bumps.get('spot', 0),
            'gamma_risk': 0.5 * self.gamma * bumps.get('spot', 0) ** 2,
            'vega_risk': self.vega * bumps.get('volatility', 0),
            'theta_risk': self.theta * bumps.get('time', 0),
            'rho_risk': self.rho * bumps.get('rate', 0)
        }


@dataclass
class SensitivityResults:
    """Results from sensitivity analysis."""
    base_value: float
    parameter_sensitivities: Dict[str, Dict[str, float]]  # param -> {value: result}
    
    def get_elasticities(self) -> Dict[str, float]:
        """Calculate elasticities for each parameter."""
        elasticities = {}
        
        for param, sensitivities in self.parameter_sensitivities.items():
            values = sorted(sensitivities.keys())
            if len(values) >= 2:
                # Calculate average elasticity
                base_idx = len(values) // 2
                base_param = values[base_idx]
                
                elasticity_sum = 0
                count = 0
                
                for i in range(len(values)):
                    if i != base_idx:
                        param_change = (values[i] - base_param) / base_param
                        value_change = (sensitivities[values[i]] - self.base_value) / self.base_value
                        
                        if param_change != 0:
                            elasticity_sum += value_change / param_change
                            count += 1
                
                elasticities[param] = elasticity_sum / count if count > 0 else 0
        
        return elasticities


@dataclass
class BacktestResults:
    """Results from historical backtesting of simulation model."""
    actual_returns: np.ndarray
    simulated_returns: np.ndarray
    
    # Statistical tests
    ks_statistic: float  # Kolmogorov-Smirnov test
    ks_pvalue: float
    
    # Moment comparison
    mean_error: float
    volatility_error: float
    skewness_error: float
    kurtosis_error: float
    
    # VaR backtesting
    var_violations: Dict[float, float]  # confidence level -> violation rate
    kupiec_test: Dict[float, Tuple[float, float]]  # level -> (statistic, p-value)
    
    @property
    def is_model_valid(self) -> bool:
        """Check if model passes validation tests."""
        return (
            self.ks_pvalue > 0.05 and  # Distribution test
            abs(self.mean_error) < 0.01 and  # 1% mean error
            abs(self.volatility_error) < 0.1 and  # 10% vol error
            all(p > 0.05 for _, p in self.kupiec_test.values())  # VaR tests
        )


@dataclass
class SimulationDiagnostics:
    """Diagnostics for simulation quality."""
    # Numerical stability
    max_value: float
    min_value: float
    nan_count: int
    inf_count: int
    
    # Random number quality
    random_seed: int
    generator_type: str
    uniformity_test: Tuple[float, float]  # statistic, p-value
    
    # Performance metrics
    total_time: float
    setup_time: float
    simulation_time: float
    analysis_time: float
    
    # Memory usage
    peak_memory_mb: float
    average_memory_mb: float
    
    def get_warnings(self) -> List[str]:
        """Get list of potential issues."""
        warnings = []
        
        if self.nan_count > 0:
            warnings.append(f"Found {self.nan_count} NaN values")
        
        if self.inf_count > 0:
            warnings.append(f"Found {self.inf_count} infinite values")
        
        if self.uniformity_test[1] < 0.05:
            warnings.append("Random number generator may have uniformity issues")
        
        if self.peak_memory_mb > 1000:
            warnings.append(f"High memory usage: {self.peak_memory_mb:.1f} MB")
        
        return warnings


@dataclass
class PortfolioSimulationResults:
    """Results specific to portfolio simulation."""
    asset_returns: Dict[str, np.ndarray]  # Asset-level returns
    asset_weights: Dict[str, float]
    correlation_matrix: np.ndarray
    
    # Portfolio metrics
    portfolio_results: SimulationResults
    
    # Asset contribution
    risk_contributions: Dict[str, float]  # Marginal VaR contributions
    return_contributions: Dict[str, float]
    
    # Diversification metrics
    diversification_ratio: float
    effective_number_of_assets: float
    concentration_risk: float
    
    def get_asset_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each asset."""
        summary = {}
        
        for asset, returns in self.asset_returns.items():
            summary[asset] = {
                'weight': self.asset_weights[asset],
                'expected_return': np.mean(returns),
                'volatility': np.std(returns),
                'var_95': np.percentile(returns, 5),
                'risk_contribution': self.risk_contributions.get(asset, 0),
                'return_contribution': self.return_contributions.get(asset, 0)
            }
        
        return summary


@dataclass
class StressTestResults:
    """Results from stress testing scenarios."""
    baseline_metrics: RiskMetrics
    stressed_metrics: Dict[str, RiskMetrics]  # scenario -> metrics
    
    # Impact analysis
    var_impacts: Dict[str, float]  # scenario -> VaR change
    return_impacts: Dict[str, float]
    volatility_impacts: Dict[str, float]
    
    # Worst case analysis
    worst_case_var: float
    worst_case_return: float
    worst_case_scenario: str
    
    # Recovery analysis
    recovery_times: Dict[str, float]  # scenario -> recovery time
    
    def get_stress_summary(self) -> Dict[str, Any]:
        """Get stress test summary."""
        return {
            'n_scenarios': len(self.stressed_metrics),
            'worst_case_scenario': self.worst_case_scenario,
            'worst_case_loss': self.worst_case_return,
            'average_var_increase': np.mean(list(self.var_impacts.values())),
            'scenarios_breaching_limit': sum(
                1 for impact in self.var_impacts.values()
                if abs(impact) > 0.10  # 10% VaR increase
            )
        }


@dataclass
class MonteCarloReport:
    """Complete Monte Carlo simulation report."""
    metadata: Dict[str, Any]
    simulation_results: SimulationResults
    portfolio_results: Optional[PortfolioSimulationResults]
    stress_test_results: Optional[StressTestResults]
    backtest_results: Optional[BacktestResults]
    diagnostics: SimulationDiagnostics
    
    generated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            'metadata': self.metadata,
            'summary': self.simulation_results.get_summary_statistics(),
            'risk_metrics': self.simulation_results.risk_metrics.get_risk_summary(),
            'convergence': self.simulation_results.convergence_metrics.get_summary(),
            'portfolio': self.portfolio_results.get_asset_summary() if self.portfolio_results else None,
            'stress_tests': self.stress_test_results.get_stress_summary() if self.stress_test_results else None,
            'diagnostics': {
                'warnings': self.diagnostics.get_warnings(),
                'performance': {
                    'total_time': self.diagnostics.total_time,
                    'memory_usage': self.diagnostics.peak_memory_mb
                }
            },
            'generated_at': self.generated_at.isoformat(),
            'version': self.version
        }