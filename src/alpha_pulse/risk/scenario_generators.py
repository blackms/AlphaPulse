"""
Scenario generation framework for Monte Carlo simulations.

Provides various methods for generating market scenarios including historical,
hypothetical, and stress scenarios with different distributions and correlations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import warnings

from alpha_pulse.models.risk_scenarios import (
    Scenario, ScenarioType, MarketScenario, MacroeconomicScenario,
    StressScenario, TailRiskScenario, ScenarioSet
)

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """Generate scenarios for Monte Carlo simulation."""
    
    def __init__(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        n_factors: int = 5
    ):
        """
        Initialize scenario generator.
        
        Args:
            historical_data: Historical market data for calibration
            correlation_matrix: Asset correlation matrix
            n_factors: Number of factors for PCA decomposition
        """
        self.historical_data = historical_data
        self.correlation_matrix = correlation_matrix
        self.n_factors = n_factors
        
        # Calibrate if historical data provided
        if historical_data is not None:
            self._calibrate_from_historical()
        
        # Initialize generators
        self._init_generators()
        
    def _calibrate_from_historical(self):
        """Calibrate scenario parameters from historical data."""
        if self.historical_data is None:
            return
        
        # Calculate returns
        returns = self.historical_data.pct_change().dropna()
        
        # Basic statistics
        self.mean_returns = returns.mean()
        self.volatilities = returns.std()
        self.skewness = returns.skew()
        self.kurtosis = returns.kurtosis()
        
        # Correlation matrix if not provided
        if self.correlation_matrix is None:
            self.correlation_matrix = returns.corr().values
        
        # PCA decomposition
        pca = PCA(n_components=min(self.n_factors, len(returns.columns)))
        self.factors = pca.fit_transform(returns)
        self.factor_loadings = pca.components_
        self.explained_variance = pca.explained_variance_ratio_
        
        # Fit mixture model for regime detection
        n_regimes = 3  # Bull, Bear, Normal
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.gmm.fit(returns)
        
        # Calculate regime statistics
        self.regime_means = []
        self.regime_covs = []
        for i in range(n_regimes):
            regime_mask = self.gmm.predict(returns) == i
            regime_returns = returns[regime_mask]
            self.regime_means.append(regime_returns.mean())
            self.regime_covs.append(regime_returns.cov())
    
    def _init_generators(self):
        """Initialize specific scenario generators."""
        self.market_generator = MarketScenarioGenerator(self)
        self.macro_generator = MacroeconomicScenarioGenerator(self)
        self.stress_generator = StressScenarioGenerator(self)
        self.tail_generator = TailRiskScenarioGenerator(self)
    
    def generate_scenarios(
        self,
        scenario_types: List[ScenarioType],
        n_scenarios_per_type: int = 100,
        time_horizon: float = 1.0,
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> ScenarioSet:
        """
        Generate a comprehensive set of scenarios.
        
        Args:
            scenario_types: Types of scenarios to generate
            n_scenarios_per_type: Number of scenarios per type
            time_horizon: Time horizon in years
            custom_parameters: Custom parameters for generators
            
        Returns:
            Set of generated scenarios
        """
        all_scenarios = []
        
        for scenario_type in scenario_types:
            logger.info(f"Generating {n_scenarios_per_type} {scenario_type.value} scenarios")
            
            if scenario_type == ScenarioType.HISTORICAL:
                scenarios = self.generate_historical_scenarios(
                    n_scenarios_per_type, time_horizon
                )
            elif scenario_type == ScenarioType.MONTE_CARLO:
                scenarios = self.generate_monte_carlo_scenarios(
                    n_scenarios_per_type, time_horizon
                )
            elif scenario_type == ScenarioType.STRESS:
                scenarios = self.stress_generator.generate_stress_scenarios(
                    n_scenarios_per_type
                )
            elif scenario_type == ScenarioType.TAIL_RISK:
                scenarios = self.tail_generator.generate_tail_scenarios(
                    n_scenarios_per_type
                )
            else:
                logger.warning(f"Unknown scenario type: {scenario_type}")
                continue
            
            all_scenarios.extend(scenarios)
        
        # Assign probabilities
        self._assign_probabilities(all_scenarios)
        
        return ScenarioSet(
            scenarios=all_scenarios,
            generation_date=datetime.now(),
            parameters=custom_parameters or {}
        )
    
    def generate_historical_scenarios(
        self,
        n_scenarios: int,
        time_horizon: float,
        block_size: Optional[int] = None
    ) -> List[Scenario]:
        """Generate scenarios by sampling from historical data."""
        if self.historical_data is None:
            raise ValueError("Historical data required for historical scenarios")
        
        scenarios = []
        returns = self.historical_data.pct_change().dropna()
        
        # Default block size is time horizon in days
        if block_size is None:
            block_size = int(time_horizon * 252)
        
        # Bootstrap historical periods
        n_periods = len(returns) - block_size + 1
        if n_periods <= 0:
            raise ValueError("Insufficient historical data for block size")
        
        for i in range(n_scenarios):
            # Random starting point
            start_idx = np.random.randint(0, n_periods)
            historical_block = returns.iloc[start_idx:start_idx + block_size]
            
            # Calculate scenario returns
            cumulative_returns = (1 + historical_block).prod() - 1
            
            # Create scenario
            scenario = MarketScenario(
                name=f"Historical_{i+1}",
                scenario_type=ScenarioType.HISTORICAL,
                probability=1.0 / n_scenarios,
                time_horizon=time_horizon,
                asset_returns=cumulative_returns.to_dict(),
                volatility_shocks={},
                correlation_changes=np.zeros_like(self.correlation_matrix),
                start_date=historical_block.index[0],
                end_date=historical_block.index[-1]
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_monte_carlo_scenarios(
        self,
        n_scenarios: int,
        time_horizon: float,
        distribution: str = 'normal'
    ) -> List[Scenario]:
        """Generate scenarios using Monte Carlo simulation."""
        scenarios = []
        n_assets = len(self.mean_returns) if hasattr(self, 'mean_returns') else 10
        
        for i in range(n_scenarios):
            # Generate returns based on distribution
            if distribution == 'normal':
                returns = self._generate_normal_returns(n_assets, time_horizon)
            elif distribution == 't':
                returns = self._generate_t_returns(n_assets, time_horizon)
            elif distribution == 'mixture':
                returns = self._generate_mixture_returns(n_assets, time_horizon)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            
            # Create scenario
            asset_names = self.historical_data.columns.tolist() if self.historical_data is not None else [f"Asset_{j}" for j in range(n_assets)]
            
            scenario = MarketScenario(
                name=f"MonteCarlo_{i+1}",
                scenario_type=ScenarioType.MONTE_CARLO,
                probability=1.0 / n_scenarios,
                time_horizon=time_horizon,
                asset_returns=dict(zip(asset_names, returns)),
                volatility_shocks={},
                correlation_changes=np.zeros((n_assets, n_assets))
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_normal_returns(
        self,
        n_assets: int,
        time_horizon: float
    ) -> np.ndarray:
        """Generate returns from multivariate normal distribution."""
        # Annual parameters
        if hasattr(self, 'mean_returns'):
            annual_means = self.mean_returns.values * 252
            annual_vols = self.volatilities.values * np.sqrt(252)
            cov_matrix = self.correlation_matrix * np.outer(annual_vols, annual_vols)
        else:
            annual_means = np.full(n_assets, 0.05)  # 5% annual return
            annual_vols = np.full(n_assets, 0.15)   # 15% annual volatility
            cov_matrix = np.eye(n_assets) * (0.15 ** 2)
        
        # Scale to time horizon
        scaled_means = annual_means * time_horizon
        scaled_cov = cov_matrix * time_horizon
        
        # Generate returns
        returns = np.random.multivariate_normal(scaled_means, scaled_cov)
        
        return returns
    
    def _generate_t_returns(
        self,
        n_assets: int,
        time_horizon: float,
        df: float = 5
    ) -> np.ndarray:
        """Generate returns from multivariate t-distribution."""
        # Generate from standard multivariate t
        if hasattr(self, 'correlation_matrix'):
            corr = self.correlation_matrix
        else:
            corr = np.eye(n_assets)
        
        # Simulate multivariate t
        chi2 = np.random.chisquare(df)
        z = np.random.multivariate_normal(np.zeros(n_assets), corr)
        t_samples = z * np.sqrt(df / chi2)
        
        # Scale to appropriate mean and volatility
        if hasattr(self, 'mean_returns'):
            means = self.mean_returns.values * 252 * time_horizon
            vols = self.volatilities.values * np.sqrt(252 * time_horizon)
        else:
            means = np.full(n_assets, 0.05 * time_horizon)
            vols = np.full(n_assets, 0.15 * np.sqrt(time_horizon))
        
        # Adjust for t-distribution variance
        t_adjustment = np.sqrt((df - 2) / df) if df > 2 else 1
        
        returns = means + t_samples * vols * t_adjustment
        
        return returns
    
    def _generate_mixture_returns(
        self,
        n_assets: int,
        time_horizon: float
    ) -> np.ndarray:
        """Generate returns from Gaussian mixture model."""
        if hasattr(self, 'gmm'):
            # Use calibrated mixture model
            sample, regime = self.gmm.sample(1)
            returns = sample[0] * np.sqrt(252 * time_horizon)
        else:
            # Default mixture: 80% normal, 20% crisis
            if np.random.rand() < 0.8:
                # Normal regime
                returns = self._generate_normal_returns(n_assets, time_horizon)
            else:
                # Crisis regime: higher volatility, negative skew
                means = np.full(n_assets, -0.20 * time_horizon)  # -20% annual
                vols = np.full(n_assets, 0.40 * np.sqrt(time_horizon))  # 40% vol
                returns = np.random.normal(means, vols)
        
        return returns
    
    def _assign_probabilities(self, scenarios: List[Scenario]):
        """Assign probabilities to scenarios based on type and characteristics."""
        # Group by type
        type_groups = {}
        for scenario in scenarios:
            scenario_type = scenario.scenario_type
            if scenario_type not in type_groups:
                type_groups[scenario_type] = []
            type_groups[scenario_type].append(scenario)
        
        # Assign probabilities by type
        type_weights = {
            ScenarioType.HISTORICAL: 0.4,
            ScenarioType.MONTE_CARLO: 0.3,
            ScenarioType.STRESS: 0.2,
            ScenarioType.TAIL_RISK: 0.1
        }
        
        for scenario_type, group_scenarios in type_groups.items():
            group_weight = type_weights.get(scenario_type, 0.25)
            scenario_prob = group_weight / len(group_scenarios)
            
            for scenario in group_scenarios:
                scenario.probability = scenario_prob
    
    def generate_factor_scenarios(
        self,
        n_scenarios: int,
        factor_shocks: Dict[str, Tuple[float, float]],
        time_horizon: float = 1.0
    ) -> List[Scenario]:
        """
        Generate scenarios based on factor model.
        
        Args:
            n_scenarios: Number of scenarios
            factor_shocks: Dict of factor name to (mean, std) shock
            time_horizon: Time horizon
            
        Returns:
            List of factor-based scenarios
        """
        if not hasattr(self, 'factor_loadings'):
            raise ValueError("Factor model not calibrated")
        
        scenarios = []
        n_factors = self.factor_loadings.shape[0]
        
        for i in range(n_scenarios):
            # Generate factor shocks
            factor_returns = np.zeros(n_factors)
            
            for j, (factor_mean, factor_std) in enumerate(factor_shocks.values()):
                if j >= n_factors:
                    break
                factor_returns[j] = np.random.normal(
                    factor_mean * time_horizon,
                    factor_std * np.sqrt(time_horizon)
                )
            
            # Convert to asset returns
            asset_returns = self.factor_loadings.T @ factor_returns
            
            # Add idiosyncratic risk
            idio_vol = 0.05 * np.sqrt(time_horizon)  # 5% annual idiosyncratic vol
            asset_returns += np.random.normal(0, idio_vol, len(asset_returns))
            
            # Create scenario
            asset_names = self.historical_data.columns.tolist()
            
            scenario = MarketScenario(
                name=f"Factor_{i+1}",
                scenario_type=ScenarioType.FACTOR,
                probability=1.0 / n_scenarios,
                time_horizon=time_horizon,
                asset_returns=dict(zip(asset_names, asset_returns)),
                factor_exposures=dict(zip(factor_shocks.keys(), factor_returns))
            )
            
            scenarios.append(scenario)
        
        return scenarios


class MarketScenarioGenerator:
    """Generate market-specific scenarios."""
    
    def __init__(self, parent_generator: ScenarioGenerator):
        """Initialize with parent generator."""
        self.parent = parent_generator
    
    def generate_crash_scenarios(
        self,
        crash_magnitude: float = -0.20,
        recovery_periods: List[int] = [30, 60, 90],
        correlation_spike: float = 0.3
    ) -> List[MarketScenario]:
        """Generate market crash scenarios."""
        scenarios = []
        
        for recovery_period in recovery_periods:
            # Initial crash
            crash_returns = {}
            n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
            
            for i in range(n_assets):
                # Crashes affect different assets differently
                beta = np.random.uniform(0.8, 1.2)  # Market beta
                asset_crash = crash_magnitude * beta
                
                # Add some randomness
                asset_crash += np.random.normal(0, 0.05)
                
                asset_name = f"Asset_{i}"
                if self.parent.historical_data is not None:
                    asset_name = self.parent.historical_data.columns[i]
                
                crash_returns[asset_name] = asset_crash
            
            # Correlation changes (increase in crash)
            corr_changes = np.ones((n_assets, n_assets)) * correlation_spike
            np.fill_diagonal(corr_changes, 0)
            
            scenario = MarketScenario(
                name=f"MarketCrash_Recovery{recovery_period}d",
                scenario_type=ScenarioType.STRESS,
                probability=0.05,  # 5% probability
                time_horizon=recovery_period / 252,
                asset_returns=crash_returns,
                volatility_shocks={asset: 2.0 for asset in crash_returns.keys()},
                correlation_changes=corr_changes,
                vix_level=40.0,  # High VIX
                market_regime='crisis'
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_volatility_scenarios(
        self,
        vol_multipliers: List[float] = [1.5, 2.0, 3.0],
        duration_days: int = 30
    ) -> List[MarketScenario]:
        """Generate volatility spike scenarios."""
        scenarios = []
        
        for multiplier in vol_multipliers:
            n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
            
            # Generate returns with higher volatility
            returns = {}
            vol_shocks = {}
            
            for i in range(n_assets):
                # Higher volatility typically means negative returns
                mean_return = -0.05 * (multiplier - 1) * duration_days / 252
                vol = 0.15 * multiplier * np.sqrt(duration_days / 252)
                
                asset_return = np.random.normal(mean_return, vol)
                
                asset_name = f"Asset_{i}"
                if self.parent.historical_data is not None:
                    asset_name = self.parent.historical_data.columns[i]
                
                returns[asset_name] = asset_return
                vol_shocks[asset_name] = multiplier
            
            scenario = MarketScenario(
                name=f"VolSpike_{int(multiplier*100)}pct",
                scenario_type=ScenarioType.STRESS,
                probability=0.10 / len(vol_multipliers),
                time_horizon=duration_days / 252,
                asset_returns=returns,
                volatility_shocks=vol_shocks,
                correlation_changes=np.zeros((n_assets, n_assets)),
                vix_level=20 * multiplier
            )
            
            scenarios.append(scenario)
        
        return scenarios


class MacroeconomicScenarioGenerator:
    """Generate macroeconomic scenarios."""
    
    def __init__(self, parent_generator: ScenarioGenerator):
        """Initialize with parent generator."""
        self.parent = parent_generator
    
    def generate_recession_scenarios(
        self,
        severities: List[str] = ['mild', 'moderate', 'severe'],
        durations: List[int] = [6, 12, 18]  # months
    ) -> List[MacroeconomicScenario]:
        """Generate recession scenarios."""
        scenarios = []
        
        severity_params = {
            'mild': {'gdp_impact': -0.02, 'unemployment_rise': 2.0},
            'moderate': {'gdp_impact': -0.05, 'unemployment_rise': 4.0},
            'severe': {'gdp_impact': -0.10, 'unemployment_rise': 6.0}
        }
        
        for severity in severities:
            for duration in durations:
                params = severity_params[severity]
                
                # Asset returns based on recession severity
                n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
                returns = {}
                
                # Sector-specific impacts
                sector_betas = {
                    'defensive': 0.5,   # Consumer staples, utilities
                    'cyclical': 1.5,    # Industrials, materials
                    'financial': 1.2,   # Banks, insurance
                    'technology': 1.0   # Tech companies
                }
                
                for i in range(n_assets):
                    # Assign random sector
                    sector = np.random.choice(list(sector_betas.keys()))
                    beta = sector_betas[sector]
                    
                    # Calculate return based on GDP impact
                    base_impact = params['gdp_impact'] * 2  # Equity markets overreact
                    asset_return = base_impact * beta * (duration / 12)
                    
                    # Add randomness
                    asset_return += np.random.normal(0, 0.05)
                    
                    asset_name = f"Asset_{i}"
                    if self.parent.historical_data is not None:
                        asset_name = self.parent.historical_data.columns[i]
                    
                    returns[asset_name] = asset_return
                
                scenario = MacroeconomicScenario(
                    name=f"Recession_{severity}_{duration}m",
                    scenario_type=ScenarioType.MACRO,
                    probability=0.15 / (len(severities) * len(durations)),
                    time_horizon=duration / 12,
                    asset_returns=returns,
                    gdp_shock=params['gdp_impact'],
                    inflation_shock=np.random.uniform(-0.02, 0.01),  # Deflation risk
                    interest_rate_shock=np.random.uniform(-0.02, -0.01),  # Rate cuts
                    unemployment_change=params['unemployment_rise'],
                    credit_spread_change=np.random.uniform(100, 300) / 10000  # bps
                )
                
                scenarios.append(scenario)
        
        return scenarios
    
    def generate_inflation_scenarios(
        self,
        inflation_levels: List[float] = [0.04, 0.06, 0.08],
        duration_months: int = 12
    ) -> List[MacroeconomicScenario]:
        """Generate inflation shock scenarios."""
        scenarios = []
        
        for inflation in inflation_levels:
            n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
            returns = {}
            
            # Asset class responses to inflation
            asset_responses = {
                'commodities': 1.5,      # Positive correlation
                'real_estate': 0.8,      # Moderate positive
                'equities': -0.5,        # Negative short-term
                'bonds': -2.0,           # Strong negative
                'tips': 0.5              # Inflation protected
            }
            
            for i in range(n_assets):
                # Assign random asset class
                asset_class = np.random.choice(list(asset_responses.keys()))
                response = asset_responses[asset_class]
                
                # Calculate return based on inflation surprise
                inflation_surprise = inflation - 0.02  # Assume 2% expected
                asset_return = response * inflation_surprise * (duration_months / 12)
                
                # Add randomness
                asset_return += np.random.normal(0, 0.03)
                
                asset_name = f"Asset_{i}"
                if self.parent.historical_data is not None:
                    asset_name = self.parent.historical_data.columns[i]
                
                returns[asset_name] = asset_return
            
            # Interest rate response to inflation
            taylor_rule_response = 1.5 * (inflation - 0.02)
            
            scenario = MacroeconomicScenario(
                name=f"Inflation_{int(inflation*100)}pct",
                scenario_type=ScenarioType.MACRO,
                probability=0.10 / len(inflation_levels),
                time_horizon=duration_months / 12,
                asset_returns=returns,
                gdp_shock=np.random.uniform(-0.01, 0.01),
                inflation_shock=inflation - 0.02,
                interest_rate_shock=taylor_rule_response,
                unemployment_change=np.random.uniform(-1, 0),  # Phillips curve
                credit_spread_change=np.random.uniform(0, 50) / 10000
            )
            
            scenarios.append(scenario)
        
        return scenarios


class StressScenarioGenerator:
    """Generate stress test scenarios."""
    
    def __init__(self, parent_generator: ScenarioGenerator):
        """Initialize with parent generator."""
        self.parent = parent_generator
    
    def generate_stress_scenarios(self, n_scenarios: int) -> List[StressScenario]:
        """Generate comprehensive stress scenarios."""
        scenarios = []
        
        # Predefined stress scenarios
        stress_templates = [
            self._generate_2008_crisis_scenario(),
            self._generate_covid_scenario(),
            self._generate_tech_bubble_scenario(),
            self._generate_emerging_market_crisis(),
            self._generate_liquidity_crisis()
        ]
        
        # Add custom stress scenarios
        for i in range(n_scenarios - len(stress_templates)):
            scenarios.append(self._generate_custom_stress_scenario(i))
        
        scenarios.extend(stress_templates[:n_scenarios])
        
        return scenarios
    
    def _generate_2008_crisis_scenario(self) -> StressScenario:
        """Generate 2008 financial crisis scenario."""
        n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
        
        # Asset-specific shocks
        returns = {}
        for i in range(n_assets):
            # Financial stocks hit hardest
            if i % 5 == 0:  # Every 5th as financial
                shock = np.random.uniform(-0.50, -0.40)
            else:
                shock = np.random.uniform(-0.35, -0.25)
            
            asset_name = f"Asset_{i}"
            if self.parent.historical_data is not None:
                asset_name = self.parent.historical_data.columns[i]
            
            returns[asset_name] = shock
        
        return StressScenario(
            name="2008_Financial_Crisis",
            scenario_type=ScenarioType.STRESS,
            probability=0.02,
            time_horizon=0.5,  # 6 months
            asset_returns=returns,
            systemic_risk_indicators={
                'counterparty_risk': 0.9,
                'funding_stress': 0.95,
                'market_liquidity': 0.2,
                'credit_freeze': True
            },
            contagion_matrix=np.ones((n_assets, n_assets)) * 0.8,
            severity_score=0.95
        )
    
    def _generate_covid_scenario(self) -> StressScenario:
        """Generate COVID-19 pandemic scenario."""
        n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
        
        returns = {}
        # Sector-specific impacts
        for i in range(n_assets):
            if i % 6 == 0:  # Travel/hospitality
                shock = np.random.uniform(-0.60, -0.50)
            elif i % 6 == 1:  # Technology
                shock = np.random.uniform(-0.10, 0.10)
            elif i % 6 == 2:  # Healthcare
                shock = np.random.uniform(0.00, 0.20)
            else:
                shock = np.random.uniform(-0.30, -0.20)
            
            asset_name = f"Asset_{i}"
            if self.parent.historical_data is not None:
                asset_name = self.parent.historical_data.columns[i]
            
            returns[asset_name] = shock
        
        return StressScenario(
            name="COVID_19_Pandemic",
            scenario_type=ScenarioType.STRESS,
            probability=0.01,
            time_horizon=0.25,  # 3 months
            asset_returns=returns,
            systemic_risk_indicators={
                'economic_shutdown': True,
                'supply_chain_disruption': 0.8,
                'demand_shock': 0.9,
                'policy_response': 0.95
            },
            contagion_matrix=np.ones((n_assets, n_assets)) * 0.6,
            severity_score=0.85
        )
    
    def _generate_tech_bubble_scenario(self) -> StressScenario:
        """Generate tech bubble burst scenario."""
        n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
        
        returns = {}
        for i in range(n_assets):
            if i % 4 == 0:  # Tech stocks
                shock = np.random.uniform(-0.70, -0.50)
            else:
                shock = np.random.uniform(-0.20, -0.10)
            
            asset_name = f"Asset_{i}"
            if self.parent.historical_data is not None:
                asset_name = self.parent.historical_data.columns[i]
            
            returns[asset_name] = shock
        
        return StressScenario(
            name="Tech_Bubble_Burst",
            scenario_type=ScenarioType.STRESS,
            probability=0.03,
            time_horizon=1.0,
            asset_returns=returns,
            systemic_risk_indicators={
                'valuation_correction': 0.9,
                'sentiment_reversal': 0.95,
                'leverage_unwind': 0.7
            },
            contagion_matrix=np.ones((n_assets, n_assets)) * 0.5,
            severity_score=0.75
        )
    
    def _generate_emerging_market_crisis(self) -> StressScenario:
        """Generate emerging market crisis scenario."""
        n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
        
        returns = {}
        for i in range(n_assets):
            if i % 3 == 0:  # EM assets
                shock = np.random.uniform(-0.40, -0.30)
            else:  # Developed market spillover
                shock = np.random.uniform(-0.15, -0.05)
            
            asset_name = f"Asset_{i}"
            if self.parent.historical_data is not None:
                asset_name = self.parent.historical_data.columns[i]
            
            returns[asset_name] = shock
        
        return StressScenario(
            name="Emerging_Market_Crisis",
            scenario_type=ScenarioType.STRESS,
            probability=0.05,
            time_horizon=0.5,
            asset_returns=returns,
            systemic_risk_indicators={
                'currency_crisis': 0.8,
                'capital_flight': 0.85,
                'sovereign_stress': 0.7
            },
            contagion_matrix=np.ones((n_assets, n_assets)) * 0.4,
            severity_score=0.65
        )
    
    def _generate_liquidity_crisis(self) -> StressScenario:
        """Generate liquidity crisis scenario."""
        n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
        
        returns = {}
        for i in range(n_assets):
            # Liquidity-driven selling
            shock = np.random.uniform(-0.25, -0.15)
            
            asset_name = f"Asset_{i}"
            if self.parent.historical_data is not None:
                asset_name = self.parent.historical_data.columns[i]
            
            returns[asset_name] = shock
        
        return StressScenario(
            name="Liquidity_Crisis",
            scenario_type=ScenarioType.STRESS,
            probability=0.04,
            time_horizon=0.1,  # 1 month
            asset_returns=returns,
            systemic_risk_indicators={
                'market_liquidity': 0.1,
                'funding_stress': 0.8,
                'margin_calls': 0.9,
                'forced_selling': 0.85
            },
            contagion_matrix=np.ones((n_assets, n_assets)) * 0.7,
            severity_score=0.80
        )
    
    def _generate_custom_stress_scenario(self, index: int) -> StressScenario:
        """Generate custom stress scenario."""
        n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
        
        # Random severe shock
        base_shock = np.random.uniform(-0.30, -0.15)
        
        returns = {}
        for i in range(n_assets):
            asset_shock = base_shock * np.random.uniform(0.8, 1.2)
            
            asset_name = f"Asset_{i}"
            if self.parent.historical_data is not None:
                asset_name = self.parent.historical_data.columns[i]
            
            returns[asset_name] = asset_shock
        
        # Random contagion pattern
        contagion = np.random.uniform(0.3, 0.8, (n_assets, n_assets))
        np.fill_diagonal(contagion, 1.0)
        
        return StressScenario(
            name=f"Custom_Stress_{index+1}",
            scenario_type=ScenarioType.STRESS,
            probability=0.02,
            time_horizon=np.random.uniform(0.1, 0.5),
            asset_returns=returns,
            systemic_risk_indicators={
                'stress_level': np.random.uniform(0.6, 0.9)
            },
            contagion_matrix=contagion,
            severity_score=np.random.uniform(0.5, 0.9)
        )


class TailRiskScenarioGenerator:
    """Generate tail risk scenarios."""
    
    def __init__(self, parent_generator: ScenarioGenerator):
        """Initialize with parent generator."""
        self.parent = parent_generator
    
    def generate_tail_scenarios(
        self,
        n_scenarios: int,
        tail_threshold: float = 0.05
    ) -> List[TailRiskScenario]:
        """Generate tail risk scenarios using extreme value theory."""
        scenarios = []
        
        # Use different methods for tail generation
        methods = [
            self._generate_evt_scenarios,
            self._generate_black_swan_scenarios,
            self._generate_fat_tail_scenarios
        ]
        
        scenarios_per_method = n_scenarios // len(methods)
        
        for method in methods:
            scenarios.extend(method(scenarios_per_method, tail_threshold))
        
        return scenarios
    
    def _generate_evt_scenarios(
        self,
        n_scenarios: int,
        tail_threshold: float
    ) -> List[TailRiskScenario]:
        """Generate scenarios using Extreme Value Theory."""
        scenarios = []
        
        if self.parent.historical_data is None:
            return scenarios
        
        returns = self.parent.historical_data.pct_change().dropna()
        
        for i in range(n_scenarios):
            tail_returns = {}
            
            for asset in returns.columns:
                asset_returns = returns[asset]
                
                # Fit GPD to tail
                threshold = np.percentile(asset_returns, tail_threshold * 100)
                tail_data = asset_returns[asset_returns < threshold]
                
                if len(tail_data) > 10:
                    # Fit Generalized Pareto Distribution
                    shape, loc, scale = stats.genpareto.fit(-tail_data + threshold)
                    
                    # Generate extreme event
                    tail_draw = stats.genpareto.rvs(shape, loc, scale)
                    extreme_return = -(tail_draw + threshold)
                else:
                    # Fallback to empirical
                    extreme_return = np.random.choice(tail_data)
                
                tail_returns[asset] = extreme_return
            
            scenario = TailRiskScenario(
                name=f"EVT_Tail_{i+1}",
                scenario_type=ScenarioType.TAIL_RISK,
                probability=tail_threshold / n_scenarios,
                time_horizon=1/252,  # Daily
                asset_returns=tail_returns,
                tail_probability=tail_threshold,
                extreme_value_parameters={'method': 'gpd'},
                expected_shortfall=np.mean(list(tail_returns.values()))
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_black_swan_scenarios(
        self,
        n_scenarios: int,
        tail_threshold: float
    ) -> List[TailRiskScenario]:
        """Generate black swan scenarios."""
        scenarios = []
        
        # Black swan events
        black_swans = [
            {
                'name': 'Cyber_Attack_Financial_System',
                'impact': -0.30,
                'duration': 0.1,
                'contagion': 0.9
            },
            {
                'name': 'Major_Geopolitical_Conflict',
                'impact': -0.40,
                'duration': 0.5,
                'contagion': 0.8
            },
            {
                'name': 'Climate_Catastrophe',
                'impact': -0.25,
                'duration': 1.0,
                'contagion': 0.6
            },
            {
                'name': 'Pandemic_Worse_Than_COVID',
                'impact': -0.50,
                'duration': 0.5,
                'contagion': 0.95
            },
            {
                'name': 'AI_Market_Manipulation',
                'impact': -0.35,
                'duration': 0.05,
                'contagion': 0.7
            }
        ]
        
        n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
        
        for i in range(min(n_scenarios, len(black_swans))):
            swan = black_swans[i]
            
            returns = {}
            for j in range(n_assets):
                # Random variation around base impact
                asset_impact = swan['impact'] * np.random.uniform(0.8, 1.2)
                
                asset_name = f"Asset_{j}"
                if self.parent.historical_data is not None:
                    asset_name = self.parent.historical_data.columns[j]
                
                returns[asset_name] = asset_impact
            
            scenario = TailRiskScenario(
                name=swan['name'],
                scenario_type=ScenarioType.TAIL_RISK,
                probability=0.001,  # Very low probability
                time_horizon=swan['duration'],
                asset_returns=returns,
                tail_probability=0.001,
                extreme_value_parameters={
                    'type': 'black_swan',
                    'contagion': swan['contagion']
                },
                expected_shortfall=swan['impact'] * 1.5
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_fat_tail_scenarios(
        self,
        n_scenarios: int,
        tail_threshold: float
    ) -> List[TailRiskScenario]:
        """Generate scenarios from fat-tailed distributions."""
        scenarios = []
        n_assets = len(self.parent.mean_returns) if hasattr(self.parent, 'mean_returns') else 10
        
        for i in range(n_scenarios):
            returns = {}
            
            # Use stable distribution for fat tails
            alpha = 1.5  # Tail parameter (lower = fatter tails)
            
            for j in range(n_assets):
                # Generate from stable distribution
                tail_return = stats.levy_stable.rvs(
                    alpha=alpha,
                    beta=0,  # Symmetric
                    scale=0.02,  # 2% scale
                    loc=-0.05  # Negative bias for tail events
                )
                
                # Ensure it's a tail event
                if tail_return > -tail_threshold:
                    tail_return = -tail_threshold * np.random.uniform(1, 3)
                
                asset_name = f"Asset_{j}"
                if self.parent.historical_data is not None:
                    asset_name = self.parent.historical_data.columns[j]
                
                returns[asset_name] = tail_return
            
            scenario = TailRiskScenario(
                name=f"FatTail_{i+1}",
                scenario_type=ScenarioType.TAIL_RISK,
                probability=tail_threshold / n_scenarios,
                time_horizon=1/252,
                asset_returns=returns,
                tail_probability=tail_threshold,
                extreme_value_parameters={
                    'distribution': 'levy_stable',
                    'alpha': alpha
                },
                expected_shortfall=np.mean(list(returns.values()))
            )
            
            scenarios.append(scenario)
        
        return scenarios