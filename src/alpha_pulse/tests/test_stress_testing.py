"""
Tests for stress testing functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from alpha_pulse.risk.stress_testing import (
    StressTester,
    StressTestConfig,
    StressTestType,
    ScenarioSeverity,
    MarketShock
)
from alpha_pulse.risk.scenario_generator import (
    ScenarioGenerator,
    ScenarioType,
    ScenarioConfig,
    MarketScenario
)
from alpha_pulse.models.stress_test_results import (
    StressTestResult,
    ScenarioResult,
    PositionImpact,
    RiskMetricImpact,
    StressTestSummary
)
from alpha_pulse.models.portfolio import Portfolio, Position


class TestStressTester:
    """Test stress testing functionality."""
    
    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio."""
        portfolio = Mock(spec=Portfolio)
        portfolio.portfolio_id = "test_portfolio"
        portfolio.total_value = 1000000.0
        
        # Create mock positions
        positions = {
            "pos1": Mock(
                position_id="pos1",
                symbol="AAPL",
                quantity=100,
                current_price=150.0,
                asset_class="equity"
            ),
            "pos2": Mock(
                position_id="pos2",
                symbol="GOOGL",
                quantity=50,
                current_price=2000.0,
                asset_class="equity"
            ),
            "pos3": Mock(
                position_id="pos3",
                symbol="TLT",
                quantity=200,
                current_price=100.0,
                asset_class="bond"
            )
        }
        
        portfolio.positions = positions
        return portfolio
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        # Generate correlated price data
        n_days = len(dates)
        
        # Base returns
        aapl_returns = np.random.normal(0.0005, 0.02, n_days)
        googl_returns = 0.7 * aapl_returns + 0.3 * np.random.normal(0.0003, 0.025, n_days)
        tlt_returns = -0.3 * aapl_returns + np.random.normal(0.0001, 0.01, n_days)
        
        # Convert to prices
        aapl_prices = 150 * np.cumprod(1 + aapl_returns)
        googl_prices = 2000 * np.cumprod(1 + googl_returns)
        tlt_prices = 100 * np.cumprod(1 + tlt_returns)
        
        return pd.DataFrame({
            'AAPL': aapl_prices,
            'GOOGL': googl_prices,
            'TLT': tlt_prices
        }, index=dates)
    
    @pytest.fixture
    def stress_tester(self):
        """Create stress tester instance."""
        config = StressTestConfig(
            scenario_types=[StressTestType.HISTORICAL, StressTestType.HYPOTHETICAL],
            severity_levels=[ScenarioSeverity.MODERATE, ScenarioSeverity.SEVERE],
            monte_carlo_simulations=1000,
            parallel_execution=False  # Disable for testing
        )
        
        return StressTester(config)
    
    def test_run_stress_test_basic(self, stress_tester, mock_portfolio, sample_market_data):
        """Test basic stress test execution."""
        result = stress_tester.run_stress_test(
            portfolio=mock_portfolio,
            market_data=sample_market_data
        )
        
        assert isinstance(result, StressTestResult)
        assert result.portfolio_id == "test_portfolio"
        assert len(result.scenarios) > 0
        assert result.summary is not None
    
    def test_historical_scenario_execution(self, stress_tester, mock_portfolio, sample_market_data):
        """Test historical scenario execution."""
        # Mock historical scenarios
        with patch.object(stress_tester, 'historical_scenarios', [
            {
                "name": "Test Crisis",
                "start_date": "2022-03-01",
                "end_date": "2022-03-31",
                "probability": 0.05,
                "severity": "severe",
                "fallback_shocks": {
                    "equity": -0.20,
                    "bond": -0.05,
                    "default": -0.15
                }
            }
        ]):
            results = stress_tester._run_historical_scenarios(
                mock_portfolio,
                sample_market_data,
                scenario_name="Test Crisis"
            )
            
            assert len(results) == 1
            scenario_result = results[0]
            
            assert isinstance(scenario_result, ScenarioResult)
            assert scenario_result.scenario_name == "Test Crisis"
            assert scenario_result.scenario_type == StressTestType.HISTORICAL
            assert len(scenario_result.position_impacts) == 3
    
    def test_hypothetical_scenario_execution(self, stress_tester, mock_portfolio, sample_market_data):
        """Test hypothetical scenario execution."""
        custom_shocks = [
            MarketShock(
                asset_class="equity",
                shock_type="price",
                magnitude=-0.25,
                duration_days=10
            ),
            MarketShock(
                asset_class="bond",
                shock_type="price",
                magnitude=-0.10,
                duration_days=10
            )
        ]
        
        results = stress_tester._run_hypothetical_scenarios(
            mock_portfolio,
            sample_market_data,
            custom_shocks=custom_shocks
        )
        
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, ScenarioResult)
            assert result.scenario_type == StressTestType.HYPOTHETICAL
            assert result.total_pnl < 0  # Expect losses from negative shocks
    
    def test_monte_carlo_scenarios(self, stress_tester, mock_portfolio, sample_market_data):
        """Test Monte Carlo scenario generation."""
        # Use smaller number of simulations for test
        stress_tester.config.monte_carlo_simulations = 100
        stress_tester.config.confidence_levels = [0.95]
        stress_tester.config.time_horizons = [10]
        
        with patch.object(stress_tester.scenario_generator, 'generate_monte_carlo_scenarios') as mock_generate:
            # Mock the scenario generation
            n_assets = 3
            mock_scenarios = np.random.normal(-0.01, 0.02, (100, n_assets))
            mock_generate.return_value = mock_scenarios
            
            results = stress_tester._run_monte_carlo_scenarios(
                mock_portfolio,
                sample_market_data
            )
            
            assert len(results) == 1  # One confidence level × one time horizon
            result = results[0]
            
            assert result.scenario_type == StressTestType.MONTE_CARLO
            assert "Monte Carlo" in result.scenario_name
            assert result.probability == 0.05  # 1 - confidence level
    
    def test_reverse_stress_test(self, stress_tester, mock_portfolio, sample_market_data):
        """Test reverse stress testing."""
        target_loss = -0.20  # 20% loss
        
        result = stress_tester.run_reverse_stress_test(
            portfolio=mock_portfolio,
            target_loss=target_loss,
            market_data=sample_market_data
        )
        
        assert isinstance(result, StressTestResult)
        if result.scenarios:
            # Check that scenarios achieve approximately the target loss
            for scenario in result.scenarios:
                assert abs(scenario.pnl_percentage - (target_loss * 100)) < 5
    
    def test_sensitivity_analysis(self, stress_tester, mock_portfolio):
        """Test sensitivity analysis."""
        risk_factors = ["market", "volatility", "interest_rate"]
        
        results = stress_tester.run_sensitivity_analysis(
            portfolio=mock_portfolio,
            risk_factors=risk_factors,
            shock_range=(-0.10, 0.10),
            n_steps=11
        )
        
        assert len(results) == len(risk_factors)
        
        for factor in risk_factors:
            assert factor in results
            df = results[factor]
            assert len(df) == 11
            assert "shock" in df.columns
            assert "pnl" in df.columns
            
            # Check monotonicity (generally true for single factor shocks)
            shocks = df["shock"].values
            assert np.all(np.diff(shocks) > 0)  # Shocks are increasing
    
    def test_position_impact_calculation(self, stress_tester, mock_portfolio):
        """Test position impact calculation."""
        # Create a simple scenario
        scenario_data = pd.DataFrame({
            'AAPL': [150, 135],  # 10% drop
            'GOOGL': [2000, 1900],  # 5% drop
            'TLT': [100, 102]  # 2% gain
        })
        
        scenario_returns = scenario_data.pct_change().iloc[1:]
        cumulative_returns = (1 + scenario_returns).cumprod() - 1
        
        # Calculate impacts manually
        position_impacts = []
        total_pnl = 0
        
        for position in mock_portfolio.positions.values():
            symbol = position.symbol
            price_change = cumulative_returns[symbol].iloc[-1]
            position_value = position.quantity * position.current_price
            position_pnl = position_value * price_change
            total_pnl += position_pnl
            
            impact = PositionImpact(
                position_id=position.position_id,
                symbol=position.symbol,
                initial_value=position_value,
                stressed_value=position_value * (1 + price_change),
                pnl=position_pnl,
                price_change_pct=price_change * 100,
                var_contribution=abs(position_pnl) / mock_portfolio.total_value
            )
            
            position_impacts.append(impact)
        
        # Verify calculations
        assert position_impacts[0].pnl < 0  # AAPL loss
        assert position_impacts[1].pnl < 0  # GOOGL loss
        assert position_impacts[2].pnl > 0  # TLT gain
        
        assert abs(total_pnl - sum(p.pnl for p in position_impacts)) < 1e-6
    
    def test_stress_test_summary_calculation(self, stress_tester):
        """Test summary statistics calculation."""
        # Create mock scenario results
        scenarios = [
            ScenarioResult(
                scenario_name="Scenario 1",
                scenario_type=StressTestType.HISTORICAL,
                probability=0.05,
                total_pnl=-50000,
                pnl_percentage=-5.0,
                position_impacts=[],
                risk_metric_impacts=[]
            ),
            ScenarioResult(
                scenario_name="Scenario 2",
                scenario_type=StressTestType.HYPOTHETICAL,
                probability=0.02,
                total_pnl=-100000,
                pnl_percentage=-10.0,
                position_impacts=[],
                risk_metric_impacts=[]
            ),
            ScenarioResult(
                scenario_name="Scenario 3",
                scenario_type=StressTestType.MONTE_CARLO,
                probability=0.10,
                total_pnl=20000,
                pnl_percentage=2.0,
                position_impacts=[],
                risk_metric_impacts=[]
            )
        ]
        
        summary = stress_tester._calculate_summary_statistics(scenarios)
        
        assert isinstance(summary, StressTestSummary)
        assert summary.worst_case_scenario == "Scenario 2"
        assert summary.worst_case_pnl == -100000
        assert summary.worst_case_pnl_pct == -10.0
        assert summary.scenarios_passed == 1
        assert summary.scenarios_failed == 2
        assert summary.average_pnl == pytest.approx(-43333.33, rel=1e-2)
    
    def test_parallel_execution(self):
        """Test parallel execution of scenarios."""
        config = StressTestConfig(
            parallel_execution=True,
            max_workers=2
        )
        
        stress_tester = StressTester(config)
        
        # Mock portfolio and market data
        mock_portfolio = Mock()
        mock_portfolio.positions = {}
        mock_portfolio.total_value = 1000000
        
        market_data = pd.DataFrame({
            'Asset1': np.random.randn(100).cumsum() + 100
        })
        
        # Mock scenarios
        stress_tester.historical_scenarios = [
            {"name": f"Scenario_{i}", "start_date": "2022-01-01", 
             "end_date": "2022-01-31", "severity": "moderate"}
            for i in range(5)
        ]
        
        with patch.object(stress_tester, '_run_single_historical_scenario') as mock_run:
            mock_run.return_value = ScenarioResult(
                scenario_name="Test",
                scenario_type=StressTestType.HISTORICAL,
                probability=0.05,
                total_pnl=-10000,
                pnl_percentage=-1.0,
                position_impacts=[],
                risk_metric_impacts=[]
            )
            
            results = stress_tester._run_historical_scenarios(
                mock_portfolio,
                market_data
            )
            
            assert len(results) == 5
            assert mock_run.call_count == 5


class TestScenarioGenerator:
    """Test scenario generation functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create scenario generator instance."""
        config = ScenarioConfig(
            n_scenarios=100,
            time_horizon=20,
            random_seed=42
        )
        return ScenarioGenerator(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample historical data."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n_assets = 3
        
        # Generate some realistic looking data
        data = pd.DataFrame(
            index=dates,
            columns=[f'Asset{i+1}' for i in range(n_assets)]
        )
        
        for i in range(n_assets):
            returns = np.random.normal(0.0002, 0.01 + i*0.005, len(dates))
            data[f'Asset{i+1}'] = 100 * np.cumprod(1 + returns)
        
        return data
    
    def test_generate_historical_scenarios(self, generator, sample_data):
        """Test historical scenario generation."""
        scenarios = generator.generate_scenarios(
            sample_data,
            ScenarioType.HISTORICAL
        )
        
        assert len(scenarios) > 0
        
        for scenario in scenarios:
            assert isinstance(scenario, MarketScenario)
            assert scenario.scenario_type == ScenarioType.HISTORICAL
            assert scenario.probability > 0
            assert scenario.severity in ["mild", "moderate", "severe", "extreme"]
    
    def test_generate_monte_carlo_normal(self, generator, sample_data):
        """Test Monte Carlo scenario generation with normal distribution."""
        generator.config.distribution_type = "normal"
        
        scenarios = generator.generate_monte_carlo_scenarios(
            sample_data,
            n_scenarios=100,
            time_horizon=10
        )
        
        assert scenarios.shape == (100, 3)  # n_scenarios x n_assets
        
        # Check that scenarios have reasonable properties
        mean_scenario = np.mean(scenarios, axis=0)
        std_scenario = np.std(scenarios, axis=0)
        
        # Mean should be close to historical
        returns = sample_data.pct_change().dropna()
        historical_mean = returns.mean().values * 10  # 10-day horizon
        
        assert np.allclose(mean_scenario, historical_mean, atol=0.05)
    
    def test_generate_stress_scenarios(self, generator, sample_data):
        """Test stress scenario generation."""
        scenarios = generator.generate_scenarios(
            sample_data,
            ScenarioType.STRESS
        )
        
        assert len(scenarios) >= 5  # At least 5 different stress types
        
        scenario_names = [s.scenario_id for s in scenarios]
        assert "stress_market_crash" in scenario_names
        assert "stress_rate_shock" in scenario_names
        assert "stress_correlation_breakdown" in scenario_names
        assert "stress_liquidity_crisis" in scenario_names
        assert "stress_sector_rotation" in scenario_names
        
        # Check market crash scenario
        crash_scenario = next(s for s in scenarios if s.scenario_id == "stress_market_crash")
        assert crash_scenario.severity == "extreme"
        assert crash_scenario.probability < 0.01
        assert crash_scenario.asset_returns.mean().mean() < -0.05  # Significant negative returns
    
    def test_factor_based_scenarios(self, generator, sample_data):
        """Test factor-based scenario generation."""
        scenarios = generator.generate_scenarios(
            sample_data,
            ScenarioType.FACTOR_BASED
        )
        
        assert len(scenarios) > 0
        
        for scenario in scenarios:
            assert scenario.scenario_type == ScenarioType.FACTOR_BASED
            assert scenario.factor_returns is not None
            assert "shocked_factors" in scenario.metadata
            assert "explained_variance" in scenario.metadata
    
    def test_student_t_scenarios(self, generator, sample_data):
        """Test Student-t distribution scenarios."""
        generator.config.distribution_type = "student_t"
        
        scenarios = generator.generate_monte_carlo_scenarios(
            sample_data,
            n_scenarios=1000,
            time_horizon=20
        )
        
        # Student-t should have fatter tails than normal
        extreme_scenarios = np.sum(np.abs(scenarios) > 0.10, axis=0) / 1000
        
        # Should have more extreme events than normal distribution
        assert np.any(extreme_scenarios > 0.05)
    
    def test_scenario_statistics(self, generator):
        """Test scenario statistics calculation."""
        # Create mock scenarios
        scenarios = [
            MarketScenario(
                scenario_id="test1",
                scenario_type=ScenarioType.STRESS,
                asset_returns=pd.DataFrame({"A": [-0.05, -0.03], "B": [-0.02, -0.01]}),
                probability=0.1,
                severity="moderate"
            ),
            MarketScenario(
                scenario_id="test2",
                scenario_type=ScenarioType.STRESS,
                asset_returns=pd.DataFrame({"A": [-0.10, -0.08], "B": [-0.05, -0.04]}),
                probability=0.05,
                severity="severe"
            )
        ]
        
        stats = generator.calculate_scenario_statistics(scenarios)
        
        assert "expected_return" in stats
        assert "worst_case_return" in stats
        assert "scenario_volatility" in stats
        assert stats["n_scenarios"] == 2
        assert stats["severity_distribution"]["moderate"] == 1
        assert stats["severity_distribution"]["severe"] == 1