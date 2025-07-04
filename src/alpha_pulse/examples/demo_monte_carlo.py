"""
Demo script for Monte Carlo simulation capabilities.

Shows how to use the Monte Carlo engine for portfolio risk analysis,
option pricing, and scenario analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import asyncio
import logging

from alpha_pulse.models.risk_scenarios import (
    SimulationConfig, StochasticProcess, VarianceReduction,
    ScenarioType, ScenarioAnalysisConfig
)
from alpha_pulse.risk.monte_carlo_engine import MonteCarloEngine
from alpha_pulse.services.simulation_service import SimulationService
from alpha_pulse.utils.random_number_generators import MersenneTwisterGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_basic_simulation():
    """Demonstrate basic Monte Carlo simulation."""
    print("\n=== Basic Monte Carlo Simulation ===")
    
    # Configuration
    config = SimulationConfig(
        n_scenarios=10000,
        time_horizon=1.0,  # 1 year
        n_steps=252,       # Daily steps
        random_seed=42,
        default_process=StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
        process_parameters={
            'drift': 0.08,      # 8% annual return
            'volatility': 0.20  # 20% annual volatility
        }
    )
    
    # Create engine
    rng = MersenneTwisterGenerator(seed=42)
    engine = MonteCarloEngine(config=config, rng=rng)
    
    # Simulate paths
    spot_price = 100.0
    result = engine.simulate_paths(
        spot_price,
        config.default_process,
        config.n_scenarios,
        config.n_steps,
        config.time_horizon / config.n_steps,
        config.process_parameters
    )
    
    print(f"Simulated {result.n_paths} paths with {result.n_steps} steps each")
    print(f"Simulation time: {result.simulation_time:.2f} seconds")
    
    # Calculate statistics
    final_prices = result.paths[:, -1]
    returns = (final_prices / spot_price) - 1
    
    print(f"\nFinal Price Statistics:")
    print(f"Mean return: {np.mean(returns):.2%}")
    print(f"Std deviation: {np.std(returns):.2%}")
    print(f"5th percentile: {np.percentile(returns, 5):.2%}")
    print(f"95th percentile: {np.percentile(returns, 95):.2%}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Annual Return')
    plt.ylabel('Frequency')
    plt.title('Distribution of Annual Returns (GBM Simulation)')
    plt.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2%}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    return result


def demo_variance_reduction():
    """Demonstrate variance reduction techniques."""
    print("\n=== Variance Reduction Techniques ===")
    
    spot_price = 100.0
    strike = 105.0
    time_to_expiry = 1.0
    risk_free_rate = 0.05
    volatility = 0.2
    
    # Run simulations with different variance reduction techniques
    techniques = [
        ([], "No Variance Reduction"),
        ([VarianceReduction.ANTITHETIC], "Antithetic Variates"),
        ([VarianceReduction.CONTROL_VARIATE], "Control Variates"),
        ([VarianceReduction.ANTITHETIC, VarianceReduction.CONTROL_VARIATE], "Combined")
    ]
    
    results = {}
    
    for vr_techniques, name in techniques:
        config = SimulationConfig(
            n_scenarios=5000,
            variance_reduction=vr_techniques,
            random_seed=42,
            process_parameters={
                'drift': risk_free_rate,
                'volatility': volatility
            }
        )
        
        engine = MonteCarloEngine(config=config)
        
        # Simulate
        sim_result = engine.simulate_paths(
            spot_price,
            StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
            config.n_scenarios,
            1,  # Single step for European option
            time_to_expiry,
            config.process_parameters
        )
        
        # Calculate option price
        payoffs = np.maximum(sim_result.paths[:, -1] - strike, 0)
        option_price = np.exp(-risk_free_rate * time_to_expiry) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(config.n_scenarios)
        
        results[name] = {
            'price': option_price,
            'std_error': std_error,
            'time': sim_result.simulation_time
        }
        
        print(f"\n{name}:")
        print(f"  Option price: ${option_price:.4f}")
        print(f"  Standard error: ${std_error:.4f}")
        print(f"  Simulation time: {sim_result.simulation_time:.3f}s")
    
    # Compare efficiency
    base_variance = results["No Variance Reduction"]['std_error']**2
    for name, result in results.items():
        if name != "No Variance Reduction":
            variance_reduction = 1 - (result['std_error']**2 / base_variance)
            print(f"\n{name} variance reduction: {variance_reduction:.1%}")


def demo_exotic_options():
    """Demonstrate exotic option simulation."""
    print("\n=== Exotic Options Simulation ===")
    
    # Asian option example
    from alpha_pulse.risk.path_simulation import PathSimulator
    
    rng = MersenneTwisterGenerator(seed=42)
    simulator = PathSimulator(rng)
    
    # Parameters
    spot = 100.0
    volatility = 0.25
    risk_free_rate = 0.05
    dividend_yield = 0.02
    time_to_maturity = 0.5
    strike = 100.0
    
    # Simulate Asian option paths
    paths, average_prices = simulator.simulate_asian_option_paths(
        spot=spot,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        time_to_maturity=time_to_maturity,
        n_paths=10000,
        n_steps=126,  # Daily monitoring for 6 months
        averaging_type="arithmetic"
    )
    
    # Calculate Asian option payoffs
    asian_payoffs = np.maximum(average_prices - strike, 0)
    asian_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(asian_payoffs)
    
    print(f"Asian Call Option Price: ${asian_price:.2f}")
    
    # Compare with European option
    final_prices = paths[:, -1]
    european_payoffs = np.maximum(final_prices - strike, 0)
    european_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(european_payoffs)
    
    print(f"European Call Option Price: ${european_price:.2f}")
    print(f"Asian discount: {(european_price - asian_price) / european_price:.1%}")
    
    # Barrier option example
    barrier = 120.0  # Up-and-out barrier
    paths_barrier, barrier_crossed = simulator.simulate_barrier_option_paths(
        spot=spot,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        time_to_maturity=time_to_maturity,
        barrier=barrier,
        barrier_type="up-out",
        n_paths=10000,
        n_steps=126,
        monitoring_frequency="continuous"
    )
    
    # Calculate barrier option price
    # Only count payoffs where barrier wasn't crossed
    barrier_payoffs = np.where(
        ~barrier_crossed,
        np.maximum(paths_barrier[:, -1] - strike, 0),
        0
    )
    barrier_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(barrier_payoffs)
    
    print(f"\nUp-and-Out Barrier Call Option Price: ${barrier_price:.2f}")
    print(f"Barrier hit probability: {np.mean(barrier_crossed):.1%}")


async def demo_portfolio_simulation():
    """Demonstrate portfolio risk simulation."""
    print("\n=== Portfolio Risk Simulation ===")
    
    # Create simulation service
    config = SimulationConfig(
        n_scenarios=10000,
        time_horizon=1.0,
        n_steps=252,
        variance_reduction=[VarianceReduction.ANTITHETIC]
    )
    
    service = SimulationService(config=config)
    
    # Define portfolio
    portfolio = {
        'AAPL': 0.25,
        'MSFT': 0.25,
        'GOOGL': 0.20,
        'AMZN': 0.15,
        'JPM': 0.15
    }
    
    # Generate synthetic market data
    dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
    np.random.seed(42)
    
    # Correlation matrix
    correlation_matrix = np.array([
        [1.00, 0.65, 0.70, 0.60, 0.40],  # AAPL
        [0.65, 1.00, 0.75, 0.55, 0.35],  # MSFT
        [0.70, 0.75, 1.00, 0.65, 0.30],  # GOOGL
        [0.60, 0.55, 0.65, 1.00, 0.25],  # AMZN
        [0.40, 0.35, 0.30, 0.25, 1.00]   # JPM
    ])
    
    # Generate correlated returns
    L = np.linalg.cholesky(correlation_matrix)
    returns = np.random.normal(0.0003, 0.015, (len(dates), 5))
    correlated_returns = returns @ L.T
    
    # Create price series
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(correlated_returns, axis=0)),
        columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM'],
        index=dates
    )
    
    # Run portfolio simulation
    result = await service.run_portfolio_simulation(
        portfolio=portfolio,
        market_data=prices,
        correlation_matrix=correlation_matrix
    )
    
    print(f"\nPortfolio Risk Metrics:")
    print(f"Expected Return: {result.portfolio_results.risk_metrics.expected_return:.2%}")
    print(f"Volatility: {result.portfolio_results.risk_metrics.volatility:.2%}")
    print(f"Sharpe Ratio: {result.portfolio_results.risk_metrics.sharpe_ratio:.2f}")
    print(f"95% VaR: {result.portfolio_results.risk_metrics.var_95:.2%}")
    print(f"95% CVaR: {result.portfolio_results.risk_metrics.cvar_95:.2%}")
    print(f"Max Drawdown: {result.portfolio_results.risk_metrics.max_drawdown:.2%}")
    
    print(f"\nDiversification Metrics:")
    print(f"Diversification Ratio: {result.diversification_ratio:.2f}")
    print(f"Effective Number of Assets: {result.effective_number_of_assets:.1f}")
    print(f"Concentration Risk (HHI): {result.concentration_risk:.3f}")
    
    print(f"\nRisk Contributions:")
    for asset, contribution in result.risk_contributions.items():
        print(f"  {asset}: {contribution:.1%}")
    
    # Plot portfolio returns distribution
    portfolio_returns = result.portfolio_results.portfolio_returns
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(portfolio_returns, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(result.portfolio_results.risk_metrics.var_95, 
                color='red', linestyle='--', label=f'95% VaR: {result.portfolio_results.risk_metrics.var_95:.2%}')
    plt.axvline(result.portfolio_results.risk_metrics.expected_return, 
                color='green', linestyle='--', label=f'Mean: {result.portfolio_results.risk_metrics.expected_return:.2%}')
    plt.xlabel('Portfolio Return')
    plt.ylabel('Frequency')
    plt.title('Portfolio Return Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot sample paths
    plt.subplot(1, 2, 2)
    sample_paths = result.portfolio_results.portfolio_values[:100, :]  # First 100 paths
    for i in range(min(100, len(sample_paths))):
        plt.plot(sample_paths[i, :], alpha=0.1, color='blue')
    
    # Plot percentiles
    percentiles = np.percentile(result.portfolio_results.portfolio_values, [5, 50, 95], axis=0)
    plt.plot(percentiles[1, :], color='black', linewidth=2, label='Median')
    plt.plot(percentiles[0, :], color='red', linestyle='--', label='5th percentile')
    plt.plot(percentiles[2, :], color='green', linestyle='--', label='95th percentile')
    
    plt.xlabel('Time (days)')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Evolution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


async def demo_stress_testing():
    """Demonstrate stress testing capabilities."""
    print("\n=== Stress Testing ===")
    
    # Create simulation service
    service = SimulationService()
    
    # Define portfolio
    portfolio = {
        'SPY': 0.40,   # S&P 500
        'AGG': 0.30,   # Bonds
        'GLD': 0.20,   # Gold
        'VNQ': 0.10    # Real Estate
    }
    
    # Generate synthetic market data
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
    
    # Different asset characteristics
    asset_params = {
        'SPY': {'return': 0.08, 'vol': 0.16},
        'AGG': {'return': 0.03, 'vol': 0.04},
        'GLD': {'return': 0.05, 'vol': 0.18},
        'VNQ': {'return': 0.06, 'vol': 0.22}
    }
    
    prices = pd.DataFrame()
    for asset, params in asset_params.items():
        daily_return = params['return'] / 252
        daily_vol = params['vol'] / np.sqrt(252)
        returns = np.random.normal(daily_return, daily_vol, len(dates))
        prices[asset] = 100 * np.exp(np.cumsum(returns))
    
    prices.index = dates
    
    # Define custom stress scenarios
    stress_scenarios = [
        {
            'name': 'COVID-19_Style_Crash',
            'returns_shock': -0.35,
            'volatility_multiplier': 4.0,
            'correlation_increase': 0.4
        },
        {
            'name': '2008_Financial_Crisis',
            'returns_shock': -0.45,
            'volatility_multiplier': 3.5,
            'correlation_increase': 0.5
        },
        {
            'name': 'Inflation_Shock',
            'returns_shock': -0.15,
            'volatility_multiplier': 2.0,
            'correlation_increase': 0.2
        },
        {
            'name': 'Tech_Bubble_Burst',
            'returns_shock': -0.30,
            'volatility_multiplier': 2.5,
            'correlation_increase': 0.3
        }
    ]
    
    # Run stress tests
    stress_results = await service.run_stress_tests(
        portfolio=portfolio,
        market_data=prices,
        stress_scenarios=stress_scenarios
    )
    
    print(f"\nBaseline Risk Metrics:")
    print(f"Expected Return: {stress_results.baseline_metrics.expected_return:.2%}")
    print(f"Volatility: {stress_results.baseline_metrics.volatility:.2%}")
    print(f"95% VaR: {stress_results.baseline_metrics.var_95:.2%}")
    
    print(f"\nStress Test Results:")
    print(f"{'Scenario':<25} {'Return Impact':<15} {'VaR Impact':<15} {'Vol Impact':<15}")
    print("-" * 70)
    
    for scenario_name in stress_results.return_impacts:
        return_impact = stress_results.return_impacts[scenario_name]
        var_impact = stress_results.var_impacts[scenario_name]
        vol_impact = stress_results.volatility_impacts[scenario_name]
        
        print(f"{scenario_name:<25} {return_impact:>14.2%} {var_impact:>14.2%} {vol_impact:>14.2%}")
    
    print(f"\nWorst Case Scenario: {stress_results.worst_case_scenario}")
    print(f"Worst Case Return: {stress_results.worst_case_return:.2%}")
    print(f"Worst Case VaR: {stress_results.worst_case_var:.2%}")


def demo_option_greeks():
    """Demonstrate option Greeks calculation using Monte Carlo."""
    print("\n=== Option Greeks (Monte Carlo) ===")
    
    # Create engine
    config = SimulationConfig(n_scenarios=100000, random_seed=42)
    engine = MonteCarloEngine(config=config)
    
    # Option parameters
    spot = 100.0
    strike = 100.0
    time_to_expiry = 0.25  # 3 months
    risk_free_rate = 0.05
    volatility = 0.2
    
    # Calculate Greeks
    greeks = engine.calculate_greeks(
        option_type='call',
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        n_simulations=100000
    )
    
    print(f"\nCall Option Greeks (Monte Carlo):")
    print(f"Price: ${greeks['price']:.4f}")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Vega: {greeks['vega']:.4f}")
    print(f"Theta: {greeks['theta']:.4f}")
    print(f"Rho: {greeks['rho']:.4f}")
    
    # Compare with Black-Scholes (for validation)
    from scipy.stats import norm
    
    d1 = (np.log(spot/strike) + (risk_free_rate + 0.5*volatility**2)*time_to_expiry) / (volatility*np.sqrt(time_to_expiry))
    d2 = d1 - volatility*np.sqrt(time_to_expiry)
    
    bs_price = spot*norm.cdf(d1) - strike*np.exp(-risk_free_rate*time_to_expiry)*norm.cdf(d2)
    bs_delta = norm.cdf(d1)
    
    print(f"\nBlack-Scholes Comparison:")
    print(f"BS Price: ${bs_price:.4f} (Error: {abs(greeks['price']-bs_price)/bs_price:.2%})")
    print(f"BS Delta: {bs_delta:.4f} (Error: {abs(greeks['delta']-bs_delta)/bs_delta:.2%})")


def demo_convergence_analysis():
    """Demonstrate convergence properties of Monte Carlo simulation."""
    print("\n=== Convergence Analysis ===")
    
    # Parameters
    spot = 100.0
    strike = 105.0
    time_to_expiry = 1.0
    risk_free_rate = 0.05
    volatility = 0.2
    
    # Different sample sizes
    sample_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    prices = []
    std_errors = []
    
    for n in sample_sizes:
        config = SimulationConfig(
            n_scenarios=n,
            random_seed=42,
            process_parameters={'drift': risk_free_rate, 'volatility': volatility}
        )
        
        engine = MonteCarloEngine(config=config)
        
        # Simulate
        result = engine.simulate_paths(
            spot,
            StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
            n,
            1,
            time_to_expiry,
            config.process_parameters
        )
        
        # Calculate option price
        payoffs = np.maximum(result.paths[:, -1] - strike, 0)
        price = np.exp(-risk_free_rate * time_to_expiry) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(n) * np.exp(-risk_free_rate * time_to_expiry)
        
        prices.append(price)
        std_errors.append(std_error)
        
        print(f"n={n:>6}: Price=${price:.4f}, Std Error=${std_error:.4f}")
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(sample_sizes, prices, yerr=1.96*np.array(std_errors), 
                 fmt='o-', capsize=5, label='MC Estimate ± 95% CI')
    plt.xscale('log')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Option Price ($)')
    plt.title('Monte Carlo Price Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.loglog(sample_sizes, std_errors, 'o-', label='Actual Std Error')
    theoretical_stderr = std_errors[0] * np.sqrt(sample_sizes[0] / np.array(sample_sizes))
    plt.loglog(sample_sizes, theoretical_stderr, '--', label='Theoretical (1/√n)')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Standard Error ($)')
    plt.title('Standard Error Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Monte Carlo Simulation Demo")
    print("==========================")
    
    # Run demos
    demo_basic_simulation()
    demo_variance_reduction()
    demo_exotic_options()
    
    # Run async demos
    asyncio.run(demo_portfolio_simulation())
    asyncio.run(demo_stress_testing())
    
    demo_option_greeks()
    demo_convergence_analysis()
    
    print("\nDemo completed!")