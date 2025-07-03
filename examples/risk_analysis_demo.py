"""
Demonstration of correlation analysis and stress testing capabilities.

This example shows how to:
1. Analyze correlations in a portfolio
2. Detect correlation regimes
3. Run comprehensive stress tests
4. Generate risk reports
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# AlphaPulse imports
from alpha_pulse.risk.correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationAnalysisConfig,
    CorrelationMethod
)
from alpha_pulse.risk.stress_testing import (
    StressTester,
    StressTestConfig,
    StressTestType,
    ScenarioSeverity
)
from alpha_pulse.models.portfolio import Portfolio, Position


def generate_sample_market_data(n_days=1000):
    """Generate sample market data for demonstration."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Asset universe
    assets = ['SPY', 'TLT', 'GLD', 'VXX', 'AAPL', 'GOOGL', 'JPM', 'XOM']
    
    # Generate correlated returns
    n_assets = len(assets)
    
    # Define correlation structure
    base_corr = np.array([
        [1.00, -0.30, 0.10, -0.60, 0.85, 0.80, 0.75, 0.65],  # SPY
        [-0.30, 1.00, 0.20, 0.30, -0.25, -0.20, -0.35, -0.15],  # TLT
        [0.10, 0.20, 1.00, 0.00, 0.05, 0.10, 0.00, 0.30],  # GLD
        [-0.60, 0.30, 0.00, 1.00, -0.55, -0.50, -0.45, -0.40],  # VXX
        [0.85, -0.25, 0.05, -0.55, 1.00, 0.70, 0.65, 0.55],  # AAPL
        [0.80, -0.20, 0.10, -0.50, 0.70, 1.00, 0.60, 0.50],  # GOOGL
        [0.75, -0.35, 0.00, -0.45, 0.65, 0.60, 1.00, 0.45],  # JPM
        [0.65, -0.15, 0.30, -0.40, 0.55, 0.50, 0.45, 1.00],  # XOM
    ])
    
    # Generate returns with regime changes
    returns_data = []
    
    # Normal regime (first 60%)
    normal_days = int(n_days * 0.6)
    mean_returns = np.array([0.0005, 0.0002, 0.0001, -0.001, 0.0008, 0.0007, 0.0004, 0.0003])
    volatilities = np.array([0.01, 0.005, 0.008, 0.03, 0.015, 0.016, 0.012, 0.014])
    
    cov_matrix = np.outer(volatilities, volatilities) * base_corr
    normal_returns = np.random.multivariate_normal(mean_returns, cov_matrix, normal_days)
    returns_data.append(normal_returns)
    
    # Stress regime (next 20%)
    stress_days = int(n_days * 0.2)
    stress_mean = mean_returns * np.array([-2, 1, 0.5, 5, -2.5, -2.5, -3, -2])
    stress_vol = volatilities * 2.5
    
    # Increase correlations during stress
    stress_corr = base_corr.copy()
    stress_corr[stress_corr > 0] = stress_corr[stress_corr > 0] * 1.3
    stress_corr = np.clip(stress_corr, -1, 1)
    np.fill_diagonal(stress_corr, 1)
    
    stress_cov = np.outer(stress_vol, stress_vol) * stress_corr
    stress_returns = np.random.multivariate_normal(stress_mean, stress_cov, stress_days)
    returns_data.append(stress_returns)
    
    # Recovery regime (last 20%)
    recovery_days = n_days - normal_days - stress_days
    recovery_mean = mean_returns * 1.5
    recovery_vol = volatilities * 1.2
    
    recovery_cov = np.outer(recovery_vol, recovery_vol) * base_corr
    recovery_returns = np.random.multivariate_normal(recovery_mean, recovery_cov, recovery_days)
    returns_data.append(recovery_returns)
    
    # Combine all regimes
    all_returns = np.vstack(returns_data)
    
    # Convert to prices
    prices = pd.DataFrame(
        100 * np.cumprod(1 + all_returns, axis=0),
        index=dates,
        columns=assets
    )
    
    return prices


def create_sample_portfolio():
    """Create a sample portfolio for testing."""
    positions = [
        Position(
            position_id="pos_1",
            symbol="SPY",
            quantity=1000,
            entry_price=400.0,
            current_price=420.0,
            position_type="long"
        ),
        Position(
            position_id="pos_2",
            symbol="TLT",
            quantity=500,
            entry_price=95.0,
            current_price=92.0,
            position_type="long"
        ),
        Position(
            position_id="pos_3",
            symbol="GLD",
            quantity=300,
            entry_price=180.0,
            current_price=185.0,
            position_type="long"
        ),
        Position(
            position_id="pos_4",
            symbol="AAPL",
            quantity=200,
            entry_price=150.0,
            current_price=175.0,
            position_type="long"
        ),
        Position(
            position_id="pos_5",
            symbol="GOOGL",
            quantity=100,
            entry_price=2500.0,
            current_price=2800.0,
            position_type="long"
        ),
        Position(
            position_id="pos_6",
            symbol="JPM",
            quantity=400,
            entry_price=140.0,
            current_price=145.0,
            position_type="long"
        )
    ]
    
    portfolio = Portfolio(portfolio_id="demo_portfolio")
    for position in positions:
        portfolio.add_position(position)
    
    return portfolio


def run_correlation_analysis(market_data):
    """Demonstrate correlation analysis features."""
    print("=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Calculate returns
    returns = market_data.pct_change().dropna()
    
    # Initialize analyzer
    config = CorrelationAnalysisConfig(
        lookback_period=252,
        rolling_window=63,
        detect_regimes=True,
        calculate_tail_dependencies=True
    )
    analyzer = CorrelationAnalyzer(config)
    
    # 1. Calculate correlation matrices
    print("\n1. Correlation Matrices:")
    print("-" * 40)
    
    pearson_corr = analyzer.calculate_correlation_matrix(returns, CorrelationMethod.PEARSON)
    spearman_corr = analyzer.calculate_correlation_matrix(returns, CorrelationMethod.SPEARMAN)
    
    print(f"Average Pearson correlation: {pearson_corr.get_average_correlation():.3f}")
    print(f"Average Spearman correlation: {spearman_corr.get_average_correlation():.3f}")
    
    # Find extreme correlations
    max_assets, max_corr = pearson_corr.get_max_correlation()[:2], pearson_corr.get_max_correlation()[2]
    min_assets, min_corr = pearson_corr.get_min_correlation()[:2], pearson_corr.get_min_correlation()[2]
    
    print(f"\nHighest correlation: {max_assets[0]}-{max_assets[1]} = {max_corr:.3f}")
    print(f"Lowest correlation: {min_assets[0]}-{min_assets[1]} = {min_corr:.3f}")
    
    # 2. Detect correlation regimes
    print("\n2. Correlation Regimes:")
    print("-" * 40)
    
    regimes = analyzer.detect_correlation_regimes(returns, n_regimes=3)
    
    for regime in regimes:
        print(f"\nRegime: {regime.regime_id}")
        print(f"  Period: {regime.start_date.date()} to {regime.end_date.date()}")
        print(f"  Duration: {regime.duration_days} days")
        print(f"  Type: {regime.regime_type}")
        print(f"  Average correlation: {regime.average_correlation:.3f}")
        print(f"  Volatility regime: {regime.volatility_regime}")
    
    # 3. Calculate tail dependencies
    print("\n3. Tail Dependencies:")
    print("-" * 40)
    
    # Focus on key pairs
    key_returns = returns[['SPY', 'TLT', 'GLD', 'VXX']]
    tail_deps = analyzer.calculate_tail_dependencies(key_returns, threshold=0.95)
    
    print("\nTail dependency analysis (95% threshold):")
    for td in tail_deps[:5]:  # Show first 5
        print(f"\n{td.asset1}-{td.asset2}:")
        print(f"  Lower tail: {td.lower_tail:.3f}")
        print(f"  Upper tail: {td.upper_tail:.3f}")
        print(f"  Asymmetry: {td.asymmetry:.3f}")
        print(f"  CI: [{td.confidence_interval[0]:.3f}, {td.confidence_interval[1]:.3f}]")
    
    # 4. Conditional correlations
    print("\n4. Conditional Correlations:")
    print("-" * 40)
    
    # Use VIX as proxy for market stress
    vix_proxy = returns['VXX'].rolling(20).std()
    conditional_corrs = analyzer.calculate_conditional_correlations(
        returns[['SPY', 'TLT', 'GLD']],
        conditioning_variable=vix_proxy
    )
    
    for condition, corr_matrix in conditional_corrs.items():
        if corr_matrix:
            print(f"\n{condition.capitalize()} volatility regime:")
            print(f"  Average correlation: {corr_matrix.get_average_correlation():.3f}")
            print(f"  SPY-TLT correlation: {corr_matrix.get_correlation('SPY', 'TLT'):.3f}")
    
    # 5. Correlation summary
    print("\n5. Correlation Summary:")
    print("-" * 40)
    
    summary = analyzer.get_correlation_summary(returns)
    
    print(f"\nCorrelation stability:")
    print(f"  Average std: {summary['correlation_stability']['average_std']:.3f}")
    print(f"  Stability score: {summary['correlation_stability']['stability_score']:.3f}")
    
    print(f"\nHigh correlation pairs (>0.8):")
    for pair in summary['high_correlation_pairs'][:3]:
        print(f"  {pair['asset1']}-{pair['asset2']}: {pair['correlation']:.3f}")
    
    return analyzer, regimes


def run_stress_testing(portfolio, market_data):
    """Demonstrate stress testing capabilities."""
    print("\n" + "=" * 80)
    print("STRESS TESTING")
    print("=" * 80)
    
    # Configure stress tester
    config = StressTestConfig(
        scenario_types=[
            StressTestType.HISTORICAL,
            StressTestType.HYPOTHETICAL,
            StressTestType.MONTE_CARLO
        ],
        severity_levels=[
            ScenarioSeverity.MODERATE,
            ScenarioSeverity.SEVERE,
            ScenarioSeverity.EXTREME
        ],
        monte_carlo_simulations=1000,
        time_horizons=[1, 5, 20],
        confidence_levels=[0.95, 0.99]
    )
    
    stress_tester = StressTester(config)
    
    # Run comprehensive stress test
    print("\nRunning comprehensive stress test...")
    result = stress_tester.run_stress_test(
        portfolio=portfolio,
        market_data=market_data
    )
    
    # Display results
    print(f"\nStress Test Results")
    print("-" * 40)
    print(f"Test ID: {result.test_id}")
    print(f"Portfolio: {result.portfolio_id}")
    print(f"Scenarios tested: {len(result.scenarios)}")
    print(f"Execution time: {result.execution_time_seconds:.2f} seconds")
    
    if result.summary:
        print(f"\nSummary Statistics:")
        print(f"  Worst case scenario: {result.summary.worst_case_scenario}")
        print(f"  Worst case P&L: ${result.summary.worst_case_pnl:,.2f}")
        print(f"  Worst case %: {result.summary.worst_case_pnl_pct:.2f}%")
        print(f"  Expected shortfall: ${result.summary.expected_shortfall:,.2f}")
        print(f"  Scenarios passed: {result.summary.scenarios_passed}")
        print(f"  Scenarios failed: {result.summary.scenarios_failed}")
        print(f"  Pass rate: {result.summary.pass_rate:.1%}")
    
    # Show top 5 worst scenarios
    print(f"\nTop 5 Worst Scenarios:")
    print("-" * 40)
    
    worst_scenarios = sorted(result.scenarios, key=lambda x: x.total_pnl)[:5]
    
    for i, scenario in enumerate(worst_scenarios, 1):
        print(f"\n{i}. {scenario.scenario_name}")
        print(f"   Type: {scenario.scenario_type}")
        print(f"   P&L: ${scenario.total_pnl:,.2f} ({scenario.pnl_percentage:.2f}%)")
        print(f"   Probability: {scenario.probability:.2%}")
        
        # Show worst position
        worst_pos = scenario.get_worst_position()
        if worst_pos:
            print(f"   Worst position: {worst_pos.symbol} (${worst_pos.pnl:,.2f})")
    
    # Run sensitivity analysis
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    print("\nRunning sensitivity analysis...")
    sensitivity_results = stress_tester.run_sensitivity_analysis(
        portfolio=portfolio,
        risk_factors=["market", "volatility"],
        shock_range=(-0.20, 0.20),
        n_steps=21
    )
    
    for factor, results_df in sensitivity_results.items():
        print(f"\n{factor.capitalize()} sensitivity:")
        print(f"  Range: {results_df['shock'].min():.1f}% to {results_df['shock'].max():.1f}%")
        print(f"  P&L range: ${results_df['pnl'].min():,.2f} to ${results_df['pnl'].max():,.2f}")
        
        # Find breakeven point
        breakeven_idx = results_df['pnl'].abs().idxmin()
        breakeven_shock = results_df.loc[breakeven_idx, 'shock']
        print(f"  Breakeven at: {breakeven_shock:.1f}% shock")
    
    # Run reverse stress test
    print("\n" + "=" * 80)
    print("REVERSE STRESS TESTING")
    print("=" * 80)
    
    target_losses = [-0.10, -0.20, -0.30]  # 10%, 20%, 30% losses
    
    for target_loss in target_losses:
        print(f"\nFinding scenarios for {target_loss*100:.0f}% loss...")
        
        reverse_result = stress_tester.run_reverse_stress_test(
            portfolio=portfolio,
            target_loss=target_loss,
            market_data=market_data
        )
        
        if reverse_result.scenarios:
            scenario = reverse_result.scenarios[0]
            print(f"  Found scenario achieving {scenario.pnl_percentage:.1f}% loss")
            print(f"  Required shocks:")
            
            for position_impact in scenario.position_impacts[:3]:
                print(f"    {position_impact.symbol}: {position_impact.price_change_pct:.1f}%")
    
    return result


def visualize_results(market_data, correlation_analyzer, stress_test_result):
    """Create visualizations of the analysis results."""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlation heatmap
    returns = market_data.pct_change().dropna()
    corr_matrix = returns.corr()
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Asset Correlation Matrix', fontsize=14)
    
    # 2. Rolling correlation plot
    spy_tlt_corr = returns['SPY'].rolling(63).corr(returns['TLT'])
    spy_gld_corr = returns['SPY'].rolling(63).corr(returns['GLD'])
    
    axes[0, 1].plot(spy_tlt_corr.index, spy_tlt_corr, label='SPY-TLT', linewidth=2)
    axes[0, 1].plot(spy_gld_corr.index, spy_gld_corr, label='SPY-GLD', linewidth=2)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_title('Rolling 63-Day Correlations', fontsize=14)
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Stress test P&L distribution
    pnls = [s.pnl_percentage for s in stress_test_result.scenarios]
    
    axes[1, 0].hist(pnls, bins=30, alpha=0.7, color='darkred', edgecolor='black')
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=np.percentile(pnls, 5), color='orange', 
                       linestyle='--', label='5% VaR', linewidth=2)
    axes[1, 0].set_title('Stress Test P&L Distribution', fontsize=14)
    axes[1, 0].set_xlabel('P&L (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scenario severity vs probability
    severities = []
    probabilities = []
    names = []
    
    for scenario in stress_test_result.scenarios[:20]:  # Top 20
        severities.append(scenario.pnl_percentage)
        probabilities.append(scenario.probability * 100)
        names.append(scenario.scenario_name.split('_')[0])
    
    scatter = axes[1, 1].scatter(severities, probabilities, 
                                s=100, alpha=0.6, c=severities, 
                                cmap='RdYlGn_r', edgecolors='black')
    
    axes[1, 1].set_title('Scenario Severity vs Probability', fontsize=14)
    axes[1, 1].set_xlabel('P&L Impact (%)')
    axes[1, 1].set_ylabel('Probability (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('P&L Impact (%)')
    
    plt.tight_layout()
    plt.savefig('risk_analysis_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'risk_analysis_results.png'")
    
    # Create correlation regime plot
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative returns colored by regime
    cumulative_returns = (1 + returns['SPY']).cumprod()
    ax.plot(cumulative_returns.index, cumulative_returns, color='black', alpha=0.7)
    
    # Color background by regime
    colors = {'high_correlation': 'red', 'normal_correlation': 'yellow', 
              'low_correlation': 'green'}
    
    # Note: This assumes regimes were calculated earlier
    # In practice, you'd use the actual regime data
    
    ax.set_title('Market Performance with Correlation Regimes', fontsize=14)
    ax.set_ylabel('Cumulative Return')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_regimes.png', dpi=300, bbox_inches='tight')
    print("Regime plot saved as 'correlation_regimes.png'")


def main():
    """Run the complete risk analysis demonstration."""
    print("AlphaPulse Risk Analysis Demonstration")
    print("=" * 80)
    
    # Generate sample data
    print("\nGenerating sample market data...")
    market_data = generate_sample_market_data(n_days=1000)
    print(f"Generated {len(market_data)} days of data for {len(market_data.columns)} assets")
    
    # Create sample portfolio
    print("\nCreating sample portfolio...")
    portfolio = create_sample_portfolio()
    print(f"Portfolio value: ${portfolio.total_value:,.2f}")
    print(f"Number of positions: {len(portfolio.positions)}")
    
    # Run correlation analysis
    analyzer, regimes = run_correlation_analysis(market_data)
    
    # Run stress testing
    stress_result = run_stress_testing(portfolio, market_data)
    
    # Generate visualizations
    visualize_results(market_data, analyzer, stress_result)
    
    print("\n" + "=" * 80)
    print("Risk analysis demonstration completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()