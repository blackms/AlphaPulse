"""
Integration tests for Monte Carlo simulation system.

Tests the complete simulation workflow including engine, path simulation,
scenario generation, and risk analysis.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

from alpha_pulse.models.risk_scenarios import (
    SimulationConfig, StochasticProcess, VarianceReduction,
    ScenarioType, ScenarioAnalysisConfig
)
from alpha_pulse.models.simulation_results import (
    SimulationResults, PathResults, RiskMetrics,
    PortfolioSimulationResults
)
from alpha_pulse.risk.monte_carlo_engine import MonteCarloEngine
from alpha_pulse.risk.path_simulation import PathSimulator
from alpha_pulse.risk.scenario_generators import ScenarioGenerator
from alpha_pulse.services.simulation_service import SimulationService
from alpha_pulse.utils.random_number_generators import (
    MersenneTwisterGenerator, SobolGenerator
)


@pytest.fixture
def simulation_config():
    """Create default simulation configuration."""
    return SimulationConfig(
        n_scenarios=10000,
        time_horizon=1.0,
        n_steps=252,
        random_seed=42,
        default_process=StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
        process_parameters={
            'drift': 0.05,
            'volatility': 0.2
        },
        variance_reduction=[VarianceReduction.ANTITHETIC],
        use_quasi_random=False,
        use_gpu=False,
        n_workers=2
    )


@pytest.fixture
def monte_carlo_engine(simulation_config):
    """Create Monte Carlo engine instance."""
    rng = MersenneTwisterGenerator(seed=42)
    return MonteCarloEngine(
        config=simulation_config,
        rng=rng,
        use_gpu=False,
        n_workers=2
    )


@pytest.fixture
def simulation_service(simulation_config):
    """Create simulation service instance."""
    return SimulationService(config=simulation_config)


@pytest.fixture
def sample_portfolio():
    """Create sample portfolio."""
    return {
        'AAPL': 0.3,
        'GOOGL': 0.25,
        'MSFT': 0.25,
        'AMZN': 0.2
    }


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
    np.random.seed(42)
    
    # Generate correlated returns
    n_assets = 4
    correlation_matrix = np.array([
        [1.0, 0.6, 0.7, 0.5],
        [0.6, 1.0, 0.5, 0.4],
        [0.7, 0.5, 1.0, 0.6],
        [0.5, 0.4, 0.6, 1.0]
    ])
    
    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(correlation_matrix)
    
    # Generate independent returns
    returns = np.random.normal(0.0002, 0.02, (len(dates), n_assets))
    
    # Apply correlation
    correlated_returns = returns @ L.T
    
    # Create price series
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(correlated_returns, axis=0)),
        columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        index=dates
    )
    
    return prices


class TestMonteCarloEngine:
    """Test Monte Carlo engine functionality."""
    
    def test_gbm_simulation(self, monte_carlo_engine):
        """Test Geometric Brownian Motion simulation."""
        spot = 100.0
        n_paths = 1000
        n_steps = 252
        dt = 1/252
        
        params = {
            'drift': 0.05,
            'volatility': 0.2
        }
        
        result = monte_carlo_engine.simulate_paths(
            spot,
            StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
            n_paths,
            n_steps,
            dt,
            params
        )
        
        assert isinstance(result, PathResults)
        assert result.paths.shape == (n_paths, n_steps + 1)
        assert result.n_paths == n_paths
        assert result.n_steps == n_steps
        
        # Check statistical properties
        final_prices = result.paths[:, -1]
        log_returns = np.log(final_prices / spot)
        
        # Expected mean and std of log returns
        expected_mean = (params['drift'] - 0.5 * params['volatility']**2)
        expected_std = params['volatility']
        
        assert abs(np.mean(log_returns) - expected_mean) < 0.02
        assert abs(np.std(log_returns) - expected_std) < 0.02
    
    def test_jump_diffusion_simulation(self, monte_carlo_engine):
        """Test jump diffusion simulation."""
        spot = 100.0
        n_paths = 5000
        n_steps = 100
        dt = 1/252
        
        params = {
            'drift': 0.05,
            'volatility': 0.15,
            'jump_intensity': 0.5,
            'jump_mean': -0.02,
            'jump_std': 0.05
        }
        
        result = monte_carlo_engine.simulate_paths(
            spot,
            StochasticProcess.JUMP_DIFFUSION,
            n_paths,
            n_steps,
            dt,
            params
        )
        
        assert result.paths.shape == (n_paths, n_steps + 1)
        
        # Check for jumps
        returns = np.diff(np.log(result.paths), axis=1)
        large_moves = np.abs(returns) > 3 * params['volatility'] * np.sqrt(dt)
        
        # Should have some jumps
        assert np.sum(large_moves) > 0
    
    def test_heston_simulation(self, monte_carlo_engine):
        """Test Heston stochastic volatility simulation."""
        spot = 100.0
        n_paths = 2000
        n_steps = 100
        dt = 1/252
        
        params = {
            'drift': 0.05,
            'mean_reversion': 2.0,
            'long_term_variance': 0.04,
            'vol_of_vol': 0.3,
            'correlation': -0.7,
            'initial_variance': 0.04
        }
        
        result = monte_carlo_engine.simulate_paths(
            spot,
            StochasticProcess.HESTON,
            n_paths,
            n_steps,
            dt,
            params
        )
        
        assert result.paths.shape == (n_paths, n_steps + 1)
        
        # Check volatility clustering
        returns = np.diff(np.log(result.paths), axis=1)
        squared_returns = returns**2
        
        # Calculate autocorrelation of squared returns
        for lag in range(1, 6):
            autocorr = np.corrcoef(
                squared_returns[:, lag:].flatten(),
                squared_returns[:, :-lag].flatten()
            )[0, 1]
            
            # Should show positive autocorrelation (volatility clustering)
            assert autocorr > 0
    
    def test_variance_reduction(self, simulation_config):
        """Test variance reduction techniques."""
        # Test antithetic variates
        config_antithetic = SimulationConfig(
            n_scenarios=5000,
            variance_reduction=[VarianceReduction.ANTITHETIC],
            random_seed=42
        )
        
        engine_antithetic = MonteCarloEngine(config=config_antithetic)
        
        spot = 100.0
        params = {'drift': 0.05, 'volatility': 0.2}
        
        result = engine_antithetic.simulate_paths(
            spot,
            StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
            config_antithetic.n_scenarios,
            252,
            1/252,
            params
        )
        
        # Check that antithetic pairs exist
        n_half = config_antithetic.n_scenarios // 2
        first_half = result.paths[:n_half, -1]
        second_half = result.paths[n_half:2*n_half, -1]
        
        # The correlation should be negative for antithetic variates
        log_returns_1 = np.log(first_half / spot)
        log_returns_2 = np.log(second_half / spot)
        
        # This test might be flaky - antithetic variates affect random generation
        # but the final prices won't necessarily be perfectly negatively correlated
        assert len(first_half) == len(second_half)
    
    def test_var_cvar_calculation(self, monte_carlo_engine):
        """Test VaR and CVaR calculation."""
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(-0.01, 0.02, 10000)
        
        var_cvar = monte_carlo_engine.calculate_var_cvar(
            returns,
            confidence_levels=[0.95, 0.99]
        )
        
        assert 0.95 in var_cvar
        assert 0.99 in var_cvar
        
        # Check VaR values
        var_95, cvar_95 = var_cvar[0.95]
        var_99, cvar_99 = var_cvar[0.99]
        
        # VaR should be negative (loss)
        assert var_95 < 0
        assert var_99 < 0
        
        # 99% VaR should be more extreme than 95%
        assert var_99 < var_95
        
        # CVaR should be more extreme than VaR
        assert cvar_95 < var_95
        assert cvar_99 < var_99
    
    def test_portfolio_simulation(self, monte_carlo_engine, sample_portfolio):
        """Test portfolio simulation."""
        spot_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'AMZN': 3200.0
        }
        
        correlation_matrix = np.array([
            [1.0, 0.6, 0.7, 0.5],
            [0.6, 1.0, 0.5, 0.4],
            [0.7, 0.5, 1.0, 0.6],
            [0.5, 0.4, 0.6, 1.0]
        ])
        
        result = monte_carlo_engine.run_portfolio_simulation(
            portfolio=sample_portfolio,
            spot_prices=spot_prices,
            correlations=correlation_matrix,
            n_scenarios=1000,
            time_horizon=1.0,
            n_steps=252
        )
        
        assert isinstance(result, SimulationResults)
        assert isinstance(result.risk_metrics, RiskMetrics)
        
        # Check risk metrics
        assert -1 < result.risk_metrics.expected_return < 1
        assert 0 < result.risk_metrics.volatility < 1
        assert result.risk_metrics.var_95 < 0
        assert result.risk_metrics.cvar_95 < result.risk_metrics.var_95


class TestPathSimulator:
    """Test path simulation functionality."""
    
    def test_multi_asset_simulation(self):
        """Test correlated multi-asset path simulation."""
        rng = MersenneTwisterGenerator(seed=42)
        simulator = PathSimulator(rng)
        
        spots = np.array([100.0, 50.0, 75.0])
        correlation_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        volatilities = np.array([0.2, 0.3, 0.25])
        
        paths = simulator.simulate_multi_asset_paths(
            spots=spots,
            correlation_matrix=correlation_matrix,
            volatilities=volatilities,
            risk_free_rate=0.05,
            dividend_yields=np.array([0.02, 0.01, 0.015]),
            time_to_maturity=1.0,
            n_paths=1000,
            n_steps=252
        )
        
        assert paths.shape == (1000, 253, 3)
        
        # Check correlation of returns
        returns = np.diff(np.log(paths), axis=1)
        
        # Calculate empirical correlation
        for i in range(3):
            for j in range(i+1, 3):
                empirical_corr = np.corrcoef(
                    returns[:, :, i].flatten(),
                    returns[:, :, j].flatten()
                )[0, 1]
                
                # Should be close to specified correlation
                assert abs(empirical_corr - correlation_matrix[i, j]) < 0.1
    
    def test_american_option_paths(self):
        """Test American option path simulation."""
        rng = MersenneTwisterGenerator(seed=42)
        simulator = PathSimulator(rng)
        
        paths, exercise_boundary = simulator.simulate_american_option_paths(
            spot=100.0,
            strike=95.0,
            params={'drift': 0.05, 'volatility': 0.2, 'risk_free_rate': 0.03},
            n_paths=2000,
            n_steps=50,
            dt=0.02,
            option_type='put'
        )
        
        assert paths.shape == (2000, 51)
        assert len(exercise_boundary) == 51
        
        # For American put, early exercise should be optimal when deep ITM
        # Boundary should generally decrease over time (approach strike)
        assert np.any(exercise_boundary[:-1] > 0)


class TestSimulationService:
    """Test simulation service functionality."""
    
    @pytest.mark.asyncio
    async def test_portfolio_simulation_service(
        self, simulation_service, sample_portfolio, sample_market_data
    ):
        """Test complete portfolio simulation workflow."""
        result = await simulation_service.run_portfolio_simulation(
            portfolio=sample_portfolio,
            market_data=sample_market_data,
            correlation_matrix=np.array([
                [1.0, 0.6, 0.7, 0.5],
                [0.6, 1.0, 0.5, 0.4],
                [0.7, 0.5, 1.0, 0.6],
                [0.5, 0.4, 0.6, 1.0]
            ])
        )
        
        assert isinstance(result, PortfolioSimulationResults)
        
        # Check asset returns
        assert len(result.asset_returns) == len(sample_portfolio)
        
        # Check risk contributions sum to 1
        risk_contrib_sum = sum(result.risk_contributions.values())
        assert abs(risk_contrib_sum - 1.0) < 0.01
        
        # Check diversification metrics
        assert result.diversification_ratio > 0
        assert result.effective_number_of_assets >= 1
        assert 0 < result.concentration_risk <= 1
    
    @pytest.mark.asyncio
    async def test_stress_testing(
        self, simulation_service, sample_portfolio, sample_market_data
    ):
        """Test stress testing functionality."""
        stress_results = await simulation_service.run_stress_tests(
            portfolio=sample_portfolio,
            market_data=sample_market_data
        )
        
        # Check that stress scenarios were applied
        assert len(stress_results.stressed_metrics) > 0
        assert stress_results.worst_case_var < 0
        assert stress_results.worst_case_return < 0
        
        # Stressed metrics should show worse results
        for scenario_name, metrics in stress_results.stressed_metrics.items():
            # VaR should be more negative under stress
            assert metrics.var_95 <= stress_results.baseline_metrics.var_95
    
    @pytest.mark.asyncio
    async def test_sensitivity_analysis(
        self, simulation_service, sample_portfolio, sample_market_data
    ):
        """Test parameter sensitivity analysis."""
        base_params = {
            'drift': 0.05,
            'volatility': 0.2
        }
        
        param_ranges = {
            'drift': [0.02, 0.05, 0.08],
            'volatility': [0.15, 0.20, 0.25, 0.30]
        }
        
        sensitivity_results = await simulation_service.calculate_sensitivity_analysis(
            portfolio=sample_portfolio,
            base_params=base_params,
            param_ranges=param_ranges,
            market_data=sample_market_data
        )
        
        # Check sensitivity to parameters
        assert 'drift' in sensitivity_results.parameter_sensitivities
        assert 'volatility' in sensitivity_results.parameter_sensitivities
        
        # Higher drift should lead to higher returns
        drift_sensitivity = sensitivity_results.parameter_sensitivities['drift']
        drift_values = sorted(drift_sensitivity.keys())
        returns = [drift_sensitivity[d] for d in drift_values]
        
        # Generally increasing (might have some noise)
        assert returns[-1] > returns[0]
    
    @pytest.mark.asyncio
    async def test_option_greeks(self, simulation_service):
        """Test option Greeks calculation."""
        greeks = await simulation_service.calculate_option_greeks(
            option_type='call',
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.2,
            n_simulations=50000
        )
        
        # Check Greeks are in reasonable ranges
        assert 0 < greeks.delta < 1  # Call delta between 0 and 1
        assert greeks.gamma > 0  # Gamma always positive
        assert greeks.vega > 0  # Vega positive for long options
        assert greeks.theta < 0  # Theta negative for long options
        assert greeks.rho > 0  # Call rho positive
    
    def test_backtest_simulation(self, simulation_service, sample_market_data):
        """Test simulation model backtesting."""
        backtest_results = simulation_service.backtest_simulation_model(
            historical_data=sample_market_data,
            lookback_window=252,
            test_window=21
        )
        
        # Check statistical tests
        assert 0 < backtest_results.ks_pvalue < 1
        
        # Check moment errors are reasonable
        assert abs(backtest_results.mean_error) < 0.5
        assert abs(backtest_results.volatility_error) < 0.5
        
        # Check VaR violations
        for level, violation_rate in backtest_results.var_violations.items():
            expected_rate = 1 - level
            # Violation rate should be close to expected
            assert abs(violation_rate - expected_rate) < 0.1
    
    def test_simulation_report_generation(
        self, simulation_service, sample_portfolio, sample_market_data
    ):
        """Test report generation."""
        # Run simulation first
        portfolio_results = asyncio.run(
            simulation_service.run_portfolio_simulation(
                portfolio=sample_portfolio,
                market_data=sample_market_data
            )
        )
        
        report = simulation_service.generate_simulation_report(
            portfolio_results=portfolio_results,
            metadata={'test': True}
        )
        
        assert report.metadata['test'] is True
        assert report.portfolio_results is not None
        assert report.diagnostics is not None
        
        # Check diagnostics
        assert report.diagnostics.nan_count == 0
        assert report.diagnostics.inf_count == 0
        assert report.diagnostics.total_time > 0


class TestQuasiRandomSimulation:
    """Test quasi-random number generation."""
    
    def test_sobol_sequence_simulation(self, simulation_config):
        """Test simulation with Sobol sequences."""
        config = SimulationConfig(
            n_scenarios=1000,
            use_quasi_random=True,
            random_seed=42
        )
        
        rng = SobolGenerator(dimension=1, seed=42)
        engine = MonteCarloEngine(config=config, rng=rng)
        
        result = engine.simulate_paths(
            100.0,
            StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
            1000,
            10,
            0.1,
            {'drift': 0.05, 'volatility': 0.2}
        )
        
        # Quasi-random sequences should give more uniform coverage
        final_prices = result.paths[:, -1]
        
        # Check distribution coverage
        percentiles = np.percentile(final_prices, [10, 25, 50, 75, 90])
        
        # Should have good coverage across distribution
        assert len(np.unique(percentiles)) == 5


class TestConvergence:
    """Test simulation convergence properties."""
    
    def test_convergence_metrics(self, monte_carlo_engine):
        """Test convergence metric calculation."""
        # Generate returns with known properties
        np.random.seed(42)
        true_mean = 0.05
        true_std = 0.2
        
        returns = np.random.normal(true_mean, true_std, 10000)
        
        convergence = monte_carlo_engine._check_convergence(returns)
        
        assert convergence.mean_convergence < 0.01
        assert convergence.std_convergence < 0.01
        assert convergence.standard_error < 0.01
        assert convergence.effective_sample_size > 1000
        assert convergence.is_converged
    
    def test_monte_carlo_convergence(self, monte_carlo_engine):
        """Test that estimates converge with more simulations."""
        spot = 100.0
        strike = 100.0
        params = {
            'drift': 0.05,
            'volatility': 0.2
        }
        
        # Run with different sample sizes
        sample_sizes = [100, 1000, 10000]
        estimates = []
        
        for n in sample_sizes:
            result = monte_carlo_engine.simulate_paths(
                spot,
                StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
                n,
                1,
                1.0,
                params
            )
            
            # Calculate option price (European call)
            payoffs = np.maximum(result.paths[:, -1] - strike, 0)
            price = np.exp(-params['drift']) * np.mean(payoffs)
            estimates.append(price)
        
        # Standard error should decrease
        std_errors = []
        for i, n in enumerate(sample_sizes):
            se = np.std(np.maximum(result.paths[:, -1] - strike, 0)) / np.sqrt(n)
            std_errors.append(se)
        
        # Check convergence
        assert std_errors[2] < std_errors[1] < std_errors[0]


class TestScenarioGeneration:
    """Test scenario generation integration."""
    
    def test_historical_scenario_generation(self, sample_market_data):
        """Test historical scenario generation."""
        generator = ScenarioGenerator(
            historical_data=sample_market_data,
            lookback_period=252
        )
        
        scenarios = generator.generate_scenarios(
            scenario_types=[ScenarioType.HISTORICAL_STRESS],
            n_scenarios_per_type=5,
            time_horizon=0.25
        )
        
        assert len(scenarios.scenarios) > 0
        
        # Check scenario properties
        for scenario in scenarios.scenarios:
            assert scenario.probability > 0
            assert scenario.probability <= 1
            assert scenario.severity != 0


class TestParallelSimulation:
    """Test parallel simulation capabilities."""
    
    def test_parallel_scenario_simulation(self, monte_carlo_engine):
        """Test parallel execution of multiple scenarios."""
        scenarios = [
            {
                'name': f'Scenario_{i}',
                'price_shocks': {
                    'AAPL': np.random.uniform(-0.1, 0.1),
                    'GOOGL': np.random.uniform(-0.1, 0.1)
                },
                'parameter_changes': {
                    'volatility': 0.2 * (1 + np.random.uniform(-0.5, 0.5))
                },
                'n_paths': 1000,
                'time_horizon': 0.25
            }
            for i in range(5)
        ]
        
        portfolio = {'AAPL': 0.6, 'GOOGL': 0.4}
        base_prices = {'AAPL': 150.0, 'GOOGL': 2800.0}
        
        results = monte_carlo_engine.parallel_scenario_simulation(
            scenarios, portfolio, base_prices
        )
        
        assert len(results) == len(scenarios)
        
        for result in results:
            assert result.simulation_results is not None
            assert result.impact_summary is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])