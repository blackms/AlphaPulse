#!/usr/bin/env python3
"""
Test script to demonstrate risk reporting with correlation analysis.

This script shows how correlation analysis has been integrated into the
risk management reports.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from alpha_pulse.risk_management.manager import RiskManager, RiskConfig
from alpha_pulse.risk.correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationAnalysisConfig
)

def generate_sample_returns(n_assets=5, n_days=252):
    """Generate sample return data for testing."""
    np.random.seed(42)
    
    # Create asset names
    assets = [f"ASSET_{i+1}" for i in range(n_assets)]
    
    # Generate base correlation structure
    base_corr = np.random.rand(n_assets, n_assets)
    base_corr = (base_corr + base_corr.T) / 2  # Make symmetric
    np.fill_diagonal(base_corr, 1)  # Set diagonal to 1
    
    # Ensure positive definite
    eigenvalues, eigenvectors = np.linalg.eig(base_corr)
    eigenvalues = np.maximum(eigenvalues, 0.1)  # Ensure positive eigenvalues
    base_corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Normalize to correlation matrix
    D = np.sqrt(np.diag(base_corr))
    base_corr = base_corr / np.outer(D, D)
    
    # Generate returns
    mean_returns = np.random.normal(0.0005, 0.0002, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)
    
    cov_matrix = np.outer(volatilities, volatilities) * base_corr
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    # Create DataFrame
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    return pd.DataFrame(returns, index=dates, columns=assets)

def create_mock_exchange():
    """Create a mock exchange for testing."""
    class MockExchange:
        async def get_portfolio_value(self):
            return 100000.0
        
        async def get_balances(self):
            return {}
        
        async def get_ticker_price(self, symbol):
            return 100.0
    
    return MockExchange()

async def test_risk_report_with_correlation():
    """Test the risk report generation with correlation analysis."""
    
    print("Risk Report with Correlation Analysis Test")
    print("=" * 50)
    
    # Create mock exchange
    exchange = create_mock_exchange()
    
    # Initialize risk manager
    risk_config = RiskConfig(
        max_position_size=0.2,
        max_portfolio_leverage=1.5,
        max_drawdown=0.25,
        target_volatility=0.15
    )
    
    risk_manager = RiskManager(
        exchange=exchange,
        config=risk_config
    )
    
    # Generate sample data
    n_assets = 5
    n_days = 252  # 1 year of data
    
    print(f"\nGenerating sample data: {n_assets} assets, {n_days} days")
    asset_returns = generate_sample_returns(n_assets, n_days)
    
    # Calculate portfolio returns (equal weighted for simplicity)
    portfolio_returns = asset_returns.mean(axis=1)
    
    # Update risk metrics with the data
    print("\nUpdating risk metrics...")
    risk_manager.update_risk_metrics(
        portfolio_returns=portfolio_returns,
        asset_returns={col: asset_returns[col] for col in asset_returns.columns}
    )
    
    # Generate risk report
    print("\nGenerating comprehensive risk report...")
    risk_report = risk_manager.get_risk_report()
    
    # Display results
    print("\n" + "=" * 50)
    print("RISK REPORT SUMMARY")
    print("=" * 50)
    
    # Basic metrics
    print(f"\nPortfolio Value: ${risk_report['portfolio_value']:,.2f}")
    print(f"Current Leverage: {risk_report['current_leverage']:.2f}x")
    
    # Risk metrics
    print("\nRisk Metrics:")
    for metric, value in risk_report['risk_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Correlation analysis
    if 'correlation_analysis' in risk_report:
        corr_analysis = risk_report['correlation_analysis']
        
        print("\n" + "-" * 50)
        print("CORRELATION ANALYSIS")
        print("-" * 50)
        
        # Summary statistics
        if 'summary' in corr_analysis:
            summary = corr_analysis['summary']
            
            print("\nAverage Correlations:")
            avg_corr = summary['average_correlation']
            print(f"  Pearson: {avg_corr['pearson']:.4f}")
            print(f"  Spearman: {avg_corr['spearman']:.4f}")
            print(f"  Difference: {avg_corr['difference']:.4f}")
            
            print("\nCorrelation Distribution:")
            dist = summary['correlation_distribution']
            print(f"  Min: {dist['min']:.4f}")
            print(f"  25%: {dist['25%']:.4f}")
            print(f"  Median: {dist['median']:.4f}")
            print(f"  75%: {dist['75%']:.4f}")
            print(f"  Max: {dist['max']:.4f}")
            print(f"  Std Dev: {dist['std']:.4f}")
            
            print("\nHigh Correlation Pairs:")
            for pair in summary.get('high_correlation_pairs', [])[:3]:
                print(f"  {pair['asset1']} - {pair['asset2']}: {pair['correlation']:.4f}")
            
            print("\nCorrelation Stability:")
            stability = summary['correlation_stability']
            print(f"  Average Std: {stability['average_std']:.4f}")
            print(f"  Stability Score: {stability['stability_score']:.4f}")
        
        # Correlation matrix
        if 'correlation_matrix' in corr_analysis:
            matrix_data = corr_analysis['correlation_matrix']
            print(f"\nCorrelation Matrix Average: {matrix_data['average_correlation']:.4f}")
            print(f"Assets: {', '.join(matrix_data['assets'])}")
        
        # Tail dependencies
        if corr_analysis.get('tail_dependencies'):
            print("\nTail Dependencies (sample):")
            for td in corr_analysis['tail_dependencies'][:3]:
                print(f"  {td['pair']}:")
                print(f"    Lower tail: {td['lower_tail']:.4f}")
                print(f"    Upper tail: {td['upper_tail']:.4f}")
                print(f"    Asymmetry: {td['asymmetry']:.4f}")
        
        # Correlation regimes
        if corr_analysis.get('correlation_regimes'):
            print("\nCorrelation Regimes:")
            for regime in corr_analysis['correlation_regimes']:
                print(f"  {regime['regime_id']}: {regime['type']}")
                print(f"    Period: {regime['start_date']} to {regime['end_date']}")
                print(f"    Avg Correlation: {regime['avg_correlation']:.4f}")
        
        # Recent rolling correlations
        if 'rolling_correlations' in corr_analysis:
            print("\nRecent Rolling Correlations (30-day):")
            for pair, corr in list(corr_analysis['rolling_correlations'].items())[:5]:
                print(f"  {pair}: {corr:.4f}")
    
    print("\n" + "=" * 50)
    print("Risk report generated successfully!")
    
    # Save report to file
    output_file = "risk_report_with_correlation.json"
    with open(output_file, 'w') as f:
        json.dump(risk_report, f, indent=2, default=str)
    print(f"\nFull report saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(test_risk_report_with_correlation())