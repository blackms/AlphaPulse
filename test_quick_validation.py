#!/usr/bin/env python3
"""
Quick test script for empirical validation without complex dependencies.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_quick_test():
    """Run a quick test of the empirical validation system"""
    
    try:
        # Import our modules
        from alpha_pulse.research.empirical_validation.simulation.market_simulator import MarketDataSimulator
        from alpha_pulse.research.empirical_validation.agents.lstm_agent import LSTMForecastingAgent, LSTMConfig
        from alpha_pulse.research.empirical_validation.agents.hrl_agents import HierarchicalTradingSystem
        from alpha_pulse.research.empirical_validation.experiments.baselines import BaselineComparison
        
        logger.info("🚀 Starting Quick Empirical Validation Test")
        
        # Step 1: Generate synthetic market data
        logger.info("📊 Step 1: Generating synthetic market data...")
        simulator = MarketDataSimulator(random_seed=42)
        data = simulator.generate_dataset(n_days=5)  # Very small dataset for testing
        logger.info(f"Generated {len(data)} data points")
        
        # Step 2: Test baseline strategies
        logger.info("📈 Step 2: Testing baseline strategies...")
        baseline_comparison = BaselineComparison(initial_cash=10000)  # Smaller amount for testing
        baseline_results = baseline_comparison.run_all_backtests(data.copy())
        
        logger.info("Baseline Results:")
        for strategy, metrics in baseline_results.items():
            if isinstance(metrics, dict) and 'error' not in metrics:
                logger.info(f"  {strategy}: {metrics.get('total_return', 0):.2%} return")
            else:
                logger.info(f"  {strategy}: Error - {metrics}")
        
        # Step 3: Test LSTM agent (simplified - no training for quick test)
        logger.info("🧠 Step 3: Testing LSTM agent (mock signals)...")
        
        # Use mock LSTM signals instead of training
        mock_lstm_signals = {
            'price_signal': 0.3,
            'volatility_signal': 1.2, 
            'confidence': 0.7
        }
        
        # Step 4: Test HRL system
        logger.info("🏗️ Step 4: Testing HRL system...")
        hrl_system = HierarchicalTradingSystem(initial_cash=10000)
        
        hrl_results = []
        for i in range(20, len(data)):  # Skip warmup
            window_data = data.iloc[:i+1]
            step_result = hrl_system.step(window_data, mock_lstm_signals)
            hrl_results.append(step_result)
        
        hrl_performance = hrl_system.get_performance_metrics()
        logger.info(f"HRL System: {hrl_performance.get('total_return', 0):.2%} return")
        
        # Step 5: Simple comparison
        logger.info("📊 Step 5: Performance Summary")
        print("\n" + "="*60)
        print("EMPIRICAL VALIDATION QUICK TEST RESULTS")
        print("="*60)
        
        all_results = [
            ("HRL System", hrl_performance.get('total_return', 0))
        ]
        
        for strategy, metrics in baseline_results.items():
            if isinstance(metrics, dict) and 'error' not in metrics:
                all_results.append((strategy, metrics.get('total_return', 0)))
        
        # Sort by performance
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Strategy':<25} {'Total Return':<12} {'Rank'}")
        print("-" * 45)
        
        for rank, (strategy, return_val) in enumerate(all_results, 1):
            marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{strategy:<25} {return_val:<11.1%} #{rank} {marker}")
        
        # Check if HRL outperformed
        hrl_rank = next(i for i, (name, _) in enumerate(all_results, 1) if "HRL" in name)
        
        print(f"\n📊 Results Summary:")
        print(f"   • Total strategies tested: {len(all_results)}")
        print(f"   • HRL system rank: #{hrl_rank} out of {len(all_results)}")
        
        if hrl_rank == 1:
            print(f"   ✅ HRL system OUTPERFORMED all baselines!")
        elif hrl_rank <= 3:
            print(f"   📈 HRL system performed well (top 3)")
        else:
            print(f"   📉 HRL system needs improvement")
        
        print(f"   • Data period: 5 days ({len(data)} hourly points)")
        print(f"   • Market regimes: {data['regime'].value_counts().to_dict()}")
        
        print("="*60)
        logger.info("✅ Quick validation test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)