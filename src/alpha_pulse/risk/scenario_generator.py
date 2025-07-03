"""
Scenario generation for stress testing and risk analysis.

Generates various market scenarios including historical replays,
hypothetical shocks, and Monte Carlo simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.stats import multivariate_normal, t as t_dist
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of scenarios that can be generated."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    STRESS = "stress"
    FACTOR_BASED = "factor_based"


class DistributionType(Enum):
    """Statistical distributions for scenario generation."""
    NORMAL = "normal"
    STUDENT_T = "student_t"
    MIXTURE = "mixture"
    EMPIRICAL = "empirical"
    GARCH = "garch"


@dataclass
class ScenarioConfig:
    """Configuration for scenario generation."""
    n_scenarios: int = 1000
    time_horizon: int = 20  # days
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    distribution_type: DistributionType = DistributionType.NORMAL
    use_fat_tails: bool = True
    correlation_stress: float = 1.5  # Multiplier for correlations in stress
    volatility_stress: float = 2.0   # Multiplier for volatility in stress
    include_regime_shifts: bool = True
    random_seed: Optional[int] = None


@dataclass
class MarketScenario:
    """Represents a generated market scenario."""
    scenario_id: str
    scenario_type: ScenarioType
    asset_returns: pd.DataFrame
    factor_returns: Optional[pd.DataFrame] = None
    probability: float = 1.0
    severity: str = "moderate"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScenarioGenerator:
    """Advanced scenario generation for risk analysis."""
    
    def __init__(self, config: ScenarioConfig = None):
        """Initialize scenario generator."""
        self.config = config or ScenarioConfig()
        if self.config.random_seed:
            np.random.seed(self.config.random_seed)
        
        self.factor_model = None
        self.regime_model = None
        
    def generate_scenarios(
        self,
        historical_data: pd.DataFrame,
        scenario_type: ScenarioType,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> List[MarketScenario]:
        """Generate scenarios based on specified type."""
        if scenario_type == ScenarioType.HISTORICAL:
            return self.generate_historical_scenarios(historical_data, custom_params)
        elif scenario_type == ScenarioType.MONTE_CARLO:
            return self.generate_monte_carlo_scenarios(
                historical_data,
                self.config.n_scenarios,
                self.config.time_horizon
            )
        elif scenario_type == ScenarioType.STRESS:
            return self.generate_stress_scenarios(historical_data, custom_params)
        elif scenario_type == ScenarioType.FACTOR_BASED:
            return self.generate_factor_scenarios(historical_data, custom_params)
        else:
            raise ValueError(f"Unsupported scenario type: {scenario_type}")
    
    def generate_historical_scenarios(
        self,
        historical_data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> List[MarketScenario]:
        """Generate scenarios based on historical episodes."""
        scenarios = []
        
        # Define historical stress periods
        stress_periods = [
            {
                "name": "2008 Financial Crisis",
                "start": "2008-09-01",
                "end": "2009-03-01",
                "severity": "extreme"
            },
            {
                "name": "COVID-19 Crash",
                "start": "2020-02-20",
                "end": "2020-03-23",
                "severity": "severe"
            },
            {
                "name": "2022 Rate Hike Cycle",
                "start": "2022-01-01",
                "end": "2022-06-30",
                "severity": "moderate"
            },
            {
                "name": "Flash Crash 2010",
                "start": "2010-05-06",
                "end": "2010-05-06",
                "severity": "severe"
            },
            {
                "name": "Taper Tantrum 2013",
                "start": "2013-05-22",
                "end": "2013-06-24",
                "severity": "moderate"
            }
        ]
        
        for period in stress_periods:
            try:
                # Extract period data
                mask = (historical_data.index >= period["start"]) & \
                       (historical_data.index <= period["end"])
                period_data = historical_data[mask]
                
                if len(period_data) > 0:
                    # Calculate returns
                    period_returns = period_data.pct_change().dropna()
                    
                    scenario = MarketScenario(
                        scenario_id=f"hist_{period['name'].replace(' ', '_')}",
                        scenario_type=ScenarioType.HISTORICAL,
                        asset_returns=period_returns,
                        probability=self._estimate_historical_probability(period["severity"]),
                        severity=period["severity"],
                        metadata={
                            "period_name": period["name"],
                            "start_date": period["start"],
                            "end_date": period["end"],
                            "duration_days": len(period_returns)
                        }
                    )
                    
                    scenarios.append(scenario)
                    
            except Exception as e:
                logger.warning(f"Failed to generate scenario for {period['name']}: {e}")
        
        return scenarios
    
    def generate_monte_carlo_scenarios(
        self,
        historical_data: pd.DataFrame,
        n_scenarios: int,
        time_horizon: int
    ) -> np.ndarray:
        """Generate Monte Carlo scenarios for portfolio returns."""
        # Calculate historical statistics
        returns = historical_data.pct_change().dropna()
        
        # Fit distribution
        if self.config.distribution_type == DistributionType.NORMAL:
            scenarios = self._generate_normal_scenarios(
                returns, n_scenarios, time_horizon
            )
        elif self.config.distribution_type == DistributionType.STUDENT_T:
            scenarios = self._generate_student_t_scenarios(
                returns, n_scenarios, time_horizon
            )
        elif self.config.distribution_type == DistributionType.MIXTURE:
            scenarios = self._generate_mixture_scenarios(
                returns, n_scenarios, time_horizon
            )
        else:
            scenarios = self._generate_empirical_scenarios(
                returns, n_scenarios, time_horizon
            )
        
        return scenarios
    
    def generate_stress_scenarios(
        self,
        historical_data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> List[MarketScenario]:
        """Generate stress test scenarios."""
        scenarios = []
        returns = historical_data.pct_change().dropna()
        
        # Market crash scenario
        crash_scenario = self._generate_market_crash_scenario(returns)
        scenarios.append(crash_scenario)
        
        # Interest rate shock scenario
        rate_shock = self._generate_rate_shock_scenario(returns)
        scenarios.append(rate_shock)
        
        # Correlation breakdown scenario
        corr_breakdown = self._generate_correlation_breakdown_scenario(returns)
        scenarios.append(corr_breakdown)
        
        # Liquidity crisis scenario
        liquidity_crisis = self._generate_liquidity_crisis_scenario(returns)
        scenarios.append(liquidity_crisis)
        
        # Sector rotation scenario
        sector_rotation = self._generate_sector_rotation_scenario(returns)
        scenarios.append(sector_rotation)
        
        return scenarios
    
    def generate_factor_scenarios(
        self,
        historical_data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> List[MarketScenario]:
        """Generate factor-based scenarios."""
        returns = historical_data.pct_change().dropna()
        
        # Extract factors using PCA
        if self.factor_model is None:
            self.factor_model = self._build_factor_model(returns)
        
        scenarios = []
        
        # Generate factor shocks
        factor_shocks = [
            {"name": "Market Factor Shock", "factors": [0], "magnitude": -3},
            {"name": "Size Factor Shock", "factors": [1], "magnitude": 2},
            {"name": "Value Factor Shock", "factors": [2], "magnitude": -2},
            {"name": "Multi-Factor Shock", "factors": [0, 1], "magnitude": -2}
        ]
        
        for shock in factor_shocks:
            scenario = self._apply_factor_shock(returns, shock)
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_normal_scenarios(
        self,
        returns: pd.DataFrame,
        n_scenarios: int,
        time_horizon: int
    ) -> np.ndarray:
        """Generate scenarios using multivariate normal distribution."""
        # Calculate parameters
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Generate daily scenarios
        daily_scenarios = []
        for _ in range(time_horizon):
            daily_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, n_scenarios
            )
            daily_scenarios.append(daily_returns)
        
        # Compound returns over time horizon
        scenarios = np.zeros((n_scenarios, len(mean_returns)))
        for t, daily in enumerate(daily_scenarios):
            scenarios = scenarios * (1 + daily) - 1 if t > 0 else daily
        
        return scenarios
    
    def _generate_student_t_scenarios(
        self,
        returns: pd.DataFrame,
        n_scenarios: int,
        time_horizon: int
    ) -> np.ndarray:
        """Generate scenarios using multivariate Student-t distribution."""
        # Fit Student-t distribution
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Estimate degrees of freedom
        df = self._estimate_t_degrees_of_freedom(returns)
        
        # Generate scenarios
        n_assets = len(mean_returns)
        scenarios = np.zeros((n_scenarios, n_assets))
        
        for i in range(n_scenarios):
            # Generate multivariate t-distributed returns
            chi2 = np.random.chisquare(df, 1)[0]
            z = np.random.multivariate_normal(
                np.zeros(n_assets), cov_matrix, 1
            )[0]
            
            # Scale by t-distribution
            t_returns = mean_returns + z * np.sqrt(df / chi2)
            
            # Compound over time horizon
            compound_return = np.prod(1 + np.tile(t_returns, (time_horizon, 1)), axis=0) - 1
            scenarios[i] = compound_return
        
        return scenarios
    
    def _generate_mixture_scenarios(
        self,
        returns: pd.DataFrame,
        n_scenarios: int,
        time_horizon: int
    ) -> np.ndarray:
        """Generate scenarios using mixture of distributions."""
        # Define mixture components
        # Normal regime (80% probability)
        normal_mean = returns.mean().values
        normal_cov = returns.cov().values * 0.8
        
        # Crisis regime (20% probability)
        crisis_mean = returns.mean().values * 0.5  # Lower returns
        crisis_cov = returns.cov().values * 3.0    # Higher volatility
        
        # Generate scenarios
        n_crisis = int(n_scenarios * 0.2)
        n_normal = n_scenarios - n_crisis
        
        normal_scenarios = self._generate_regime_scenarios(
            normal_mean, normal_cov, n_normal, time_horizon
        )
        
        crisis_scenarios = self._generate_regime_scenarios(
            crisis_mean, crisis_cov, n_crisis, time_horizon
        )
        
        # Combine scenarios
        scenarios = np.vstack([normal_scenarios, crisis_scenarios])
        np.random.shuffle(scenarios)
        
        return scenarios
    
    def _generate_empirical_scenarios(
        self,
        returns: pd.DataFrame,
        n_scenarios: int,
        time_horizon: int
    ) -> np.ndarray:
        """Generate scenarios using empirical distribution (bootstrap)."""
        n_periods = len(returns)
        n_assets = len(returns.columns)
        
        scenarios = np.zeros((n_scenarios, n_assets))
        
        for i in range(n_scenarios):
            # Bootstrap time_horizon periods
            indices = np.random.choice(n_periods, time_horizon, replace=True)
            sampled_returns = returns.iloc[indices].values
            
            # Compound returns
            compound_return = np.prod(1 + sampled_returns, axis=0) - 1
            scenarios[i] = compound_return
        
        return scenarios
    
    def _generate_market_crash_scenario(
        self, returns: pd.DataFrame
    ) -> MarketScenario:
        """Generate market crash scenario."""
        # Calculate crash magnitudes based on historical extremes
        worst_returns = returns.quantile(0.01)
        
        # Amplify for stress testing
        crash_returns = pd.DataFrame(
            [worst_returns * self.config.volatility_stress],
            columns=returns.columns
        )
        
        # Multi-day crash
        multi_day_crash = pd.concat([crash_returns] * 5, ignore_index=True)
        multi_day_crash.index = pd.date_range(
            start=datetime.now(), periods=5, freq='D'
        )
        
        return MarketScenario(
            scenario_id="stress_market_crash",
            scenario_type=ScenarioType.STRESS,
            asset_returns=multi_day_crash,
            probability=0.001,
            severity="extreme",
            metadata={
                "type": "market_crash",
                "magnitude": "5-day severe decline",
                "avg_daily_loss": float(worst_returns.mean() * self.config.volatility_stress)
            }
        )
    
    def _generate_rate_shock_scenario(
        self, returns: pd.DataFrame
    ) -> MarketScenario:
        """Generate interest rate shock scenario."""
        # Simulate impact of rate shock on different assets
        n_days = 10
        n_assets = len(returns.columns)
        
        # Generate heterogeneous impacts
        rate_betas = np.random.uniform(-2, 0.5, n_assets)  # Most assets negatively affected
        
        # Create shock pattern
        shock_magnitude = 0.03  # 300 bps shock
        shocked_returns = pd.DataFrame(
            index=pd.date_range(start=datetime.now(), periods=n_days, freq='D'),
            columns=returns.columns
        )
        
        for i, col in enumerate(returns.columns):
            # Rate-sensitive assets more affected
            daily_impact = rate_betas[i] * shock_magnitude / n_days
            shocked_returns[col] = daily_impact + np.random.normal(0, 0.01, n_days)
        
        return MarketScenario(
            scenario_id="stress_rate_shock",
            scenario_type=ScenarioType.STRESS,
            asset_returns=shocked_returns,
            probability=0.05,
            severity="severe",
            metadata={
                "type": "interest_rate_shock",
                "shock_size": "300bps",
                "duration_days": n_days
            }
        )
    
    def _generate_correlation_breakdown_scenario(
        self, returns: pd.DataFrame
    ) -> MarketScenario:
        """Generate correlation breakdown scenario."""
        # In crisis, correlations tend to 1
        mean_returns = returns.mean()
        crisis_vol = returns.std() * self.config.volatility_stress
        
        # High correlation matrix
        n_assets = len(returns.columns)
        high_corr = np.full((n_assets, n_assets), 0.9)
        np.fill_diagonal(high_corr, 1.0)
        
        # Generate correlated crisis returns
        cov_matrix = np.outer(crisis_vol, crisis_vol) * high_corr
        
        crisis_returns = pd.DataFrame(
            np.random.multivariate_normal(
                mean_returns * -2,  # Negative returns in crisis
                cov_matrix,
                20  # 20 days of crisis
            ),
            columns=returns.columns,
            index=pd.date_range(start=datetime.now(), periods=20, freq='D')
        )
        
        return MarketScenario(
            scenario_id="stress_correlation_breakdown",
            scenario_type=ScenarioType.STRESS,
            asset_returns=crisis_returns,
            probability=0.01,
            severity="severe",
            metadata={
                "type": "correlation_breakdown",
                "avg_correlation": 0.9,
                "normal_correlation": returns.corr().values[np.triu_indices(n_assets, 1)].mean()
            }
        )
    
    def _generate_liquidity_crisis_scenario(
        self, returns: pd.DataFrame
    ) -> MarketScenario:
        """Generate liquidity crisis scenario."""
        # Liquidity crisis: high volatility, negative returns, gaps
        n_days = 15
        
        crisis_returns = pd.DataFrame(
            index=pd.date_range(start=datetime.now(), periods=n_days, freq='D'),
            columns=returns.columns
        )
        
        for col in returns.columns:
            # Base negative drift
            drift = -0.02
            
            # High volatility
            vol = returns[col].std() * 3
            
            # Add jump risk (gaps)
            jumps = np.random.binomial(1, 0.2, n_days)  # 20% chance of jump each day
            jump_sizes = np.random.normal(-0.05, 0.02, n_days) * jumps
            
            # Generate returns
            normal_returns = np.random.normal(drift, vol, n_days)
            crisis_returns[col] = normal_returns + jump_sizes
        
        return MarketScenario(
            scenario_id="stress_liquidity_crisis",
            scenario_type=ScenarioType.STRESS,
            asset_returns=crisis_returns,
            probability=0.005,
            severity="extreme",
            metadata={
                "type": "liquidity_crisis",
                "includes_gaps": True,
                "avg_daily_volume_reduction": 0.7
            }
        )
    
    def _generate_sector_rotation_scenario(
        self, returns: pd.DataFrame
    ) -> MarketScenario:
        """Generate sector rotation scenario."""
        n_days = 30
        n_assets = len(returns.columns)
        
        # Divide assets into winners and losers
        n_winners = n_assets // 3
        winners_idx = np.random.choice(n_assets, n_winners, replace=False)
        
        rotation_returns = pd.DataFrame(
            index=pd.date_range(start=datetime.now(), periods=n_days, freq='D'),
            columns=returns.columns
        )
        
        for i, col in enumerate(returns.columns):
            if i in winners_idx:
                # Winners: positive returns
                trend = 0.002  # 0.2% daily
                vol = returns[col].std() * 0.8
            else:
                # Losers: negative returns
                trend = -0.001  # -0.1% daily
                vol = returns[col].std() * 1.2
            
            rotation_returns[col] = np.random.normal(trend, vol, n_days)
        
        return MarketScenario(
            scenario_id="stress_sector_rotation",
            scenario_type=ScenarioType.STRESS,
            asset_returns=rotation_returns,
            probability=0.10,
            severity="moderate",
            metadata={
                "type": "sector_rotation",
                "winners_pct": n_winners / n_assets,
                "rotation_period_days": n_days
            }
        )
    
    def _build_factor_model(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Build factor model using PCA."""
        # Standardize returns
        standardized = (returns - returns.mean()) / returns.std()
        
        # Apply PCA
        n_factors = min(5, len(returns.columns) - 1)
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(standardized)
        
        # Store model
        model = {
            "pca": pca,
            "loadings": pca.components_,
            "explained_variance": pca.explained_variance_ratio_,
            "mean_returns": returns.mean().values,
            "std_returns": returns.std().values,
            "factors": pd.DataFrame(
                factors,
                index=returns.index,
                columns=[f"Factor_{i+1}" for i in range(n_factors)]
            )
        }
        
        return model
    
    def _apply_factor_shock(
        self,
        returns: pd.DataFrame,
        shock: Dict[str, Any]
    ) -> MarketScenario:
        """Apply shock to specific factors."""
        model = self.factor_model
        n_days = 10
        
        # Create factor shock
        factor_returns = np.zeros((n_days, len(model["factors"].columns)))
        for factor_idx in shock["factors"]:
            factor_returns[:, factor_idx] = shock["magnitude"] * \
                                           model["explained_variance"][factor_idx]
        
        # Transform back to asset returns
        asset_returns = np.dot(factor_returns, model["loadings"])
        
        # Add idiosyncratic risk
        idio_risk = np.random.normal(
            0, 
            model["std_returns"] * 0.3,
            (n_days, len(returns.columns))
        )
        
        total_returns = asset_returns * model["std_returns"] + \
                       model["mean_returns"] + idio_risk
        
        scenario_returns = pd.DataFrame(
            total_returns,
            columns=returns.columns,
            index=pd.date_range(start=datetime.now(), periods=n_days, freq='D')
        )
        
        return MarketScenario(
            scenario_id=f"factor_{shock['name'].replace(' ', '_')}",
            scenario_type=ScenarioType.FACTOR_BASED,
            asset_returns=scenario_returns,
            factor_returns=pd.DataFrame(
                factor_returns,
                columns=model["factors"].columns,
                index=scenario_returns.index
            ),
            probability=0.05,
            severity="moderate",
            metadata={
                "shocked_factors": shock["factors"],
                "shock_magnitude": shock["magnitude"],
                "explained_variance": sum(
                    model["explained_variance"][i] for i in shock["factors"]
                )
            }
        )
    
    def _estimate_historical_probability(self, severity: str) -> float:
        """Estimate probability based on severity."""
        probabilities = {
            "mild": 0.20,
            "moderate": 0.10,
            "severe": 0.05,
            "extreme": 0.01
        }
        return probabilities.get(severity, 0.05)
    
    def _estimate_t_degrees_of_freedom(self, returns: pd.DataFrame) -> float:
        """Estimate degrees of freedom for Student-t distribution."""
        # Use method of moments
        kurtosis = returns.kurtosis().mean()
        
        # For Student-t: excess kurtosis = 6/(df-4) for df > 4
        if kurtosis > 3:
            df = 6 / (kurtosis - 3) + 4
            df = max(4.1, min(df, 30))  # Bound between 4.1 and 30
        else:
            df = 30  # High df approximates normal
        
        return df
    
    def _generate_regime_scenarios(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        n_scenarios: int,
        time_horizon: int
    ) -> np.ndarray:
        """Generate scenarios for a specific regime."""
        # Generate daily returns
        daily_returns = np.random.multivariate_normal(
            mean, cov, (n_scenarios, time_horizon)
        )
        
        # Compound returns
        scenarios = np.prod(1 + daily_returns, axis=1) - 1
        
        return scenarios
    
    def calculate_scenario_statistics(
        self,
        scenarios: List[MarketScenario]
    ) -> Dict[str, Any]:
        """Calculate statistics across scenarios."""
        if not scenarios:
            return {}
        
        # Aggregate returns across scenarios
        all_returns = []
        probabilities = []
        
        for scenario in scenarios:
            # Calculate total return for each scenario
            total_return = (1 + scenario.asset_returns).prod() - 1
            all_returns.append(total_return.mean())  # Average across assets
            probabilities.append(scenario.probability)
        
        all_returns = np.array(all_returns)
        probabilities = np.array(probabilities)
        
        # Normalize probabilities
        probabilities = probabilities / probabilities.sum()
        
        # Calculate statistics
        stats = {
            "expected_return": np.average(all_returns, weights=probabilities),
            "worst_case_return": np.min(all_returns),
            "best_case_return": np.max(all_returns),
            "scenario_volatility": np.std(all_returns),
            "var_95": np.percentile(all_returns, 5),
            "cvar_95": all_returns[all_returns <= np.percentile(all_returns, 5)].mean(),
            "n_scenarios": len(scenarios),
            "severity_distribution": {
                "extreme": sum(1 for s in scenarios if s.severity == "extreme"),
                "severe": sum(1 for s in scenarios if s.severity == "severe"),
                "moderate": sum(1 for s in scenarios if s.severity == "moderate"),
                "mild": sum(1 for s in scenarios if s.severity == "mild")
            }
        }
        
        return stats
    
    def generate_conditional_scenarios(
        self,
        historical_data: pd.DataFrame,
        condition: Callable[[pd.DataFrame], bool],
        n_scenarios: int = 100
    ) -> List[MarketScenario]:
        """Generate scenarios conditional on specific market conditions."""
        # Filter historical data based on condition
        returns = historical_data.pct_change().dropna()
        mask = condition(returns)
        conditional_returns = returns[mask]
        
        if len(conditional_returns) < 30:
            logger.warning("Insufficient conditional data, using full dataset")
            conditional_returns = returns
        
        # Generate scenarios based on conditional distribution
        scenarios = []
        
        for i in range(n_scenarios):
            # Bootstrap from conditional returns
            sampled_days = np.random.choice(
                len(conditional_returns),
                self.config.time_horizon,
                replace=True
            )
            
            scenario_returns = conditional_returns.iloc[sampled_days].reset_index(drop=True)
            
            scenario = MarketScenario(
                scenario_id=f"conditional_{i}",
                scenario_type=ScenarioType.MONTE_CARLO,
                asset_returns=scenario_returns,
                probability=1.0 / n_scenarios,
                severity="moderate",
                metadata={"condition": str(condition)}
            )
            
            scenarios.append(scenario)
        
        return scenarios