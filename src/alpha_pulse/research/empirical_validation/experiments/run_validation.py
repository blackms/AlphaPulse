"""
Main experiment runner for empirical validation of the HRL trading system.

This script orchestrates the complete validation pipeline:
1. Generate synthetic market data
2. Train LSTM forecasting agent
3. Run HRL system
4. Run baseline strategies
5. Compare performance metrics
6. Generate results and plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import json
import os

# Import our modules
from ..simulation.market_simulator import MarketDataSimulator, MarketRegime
from ..agents.lstm_agent import LSTMForecastingAgent, LSTMConfig
from ..agents.hrl_agents import HierarchicalTradingSystem
from .baselines import BaselineComparison

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmpiricalValidationExperiment:
    """
    Main experiment class for empirical validation.
    
    Coordinates all components to run comprehensive experiments
    comparing HRL system against baseline strategies.
    """
    
    def __init__(self, 
                 initial_cash: float = 100000.0,
                 random_seed: int = 42,
                 results_dir: str = "results"):
        """
        Initialize experiment.
        
        Args:
            initial_cash: Starting capital for all strategies
            random_seed: Random seed for reproducibility
            results_dir: Directory to save results
        """
        self.initial_cash = initial_cash
        self.random_seed = random_seed
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.market_simulator = MarketDataSimulator(random_seed=random_seed)
        self.baseline_comparison = BaselineComparison(initial_cash)
        
        # Results storage
        self.results = {}
        self.market_data = None
        
        logger.info(f"Initialized validation experiment with seed {random_seed}")
    
    def generate_market_scenarios(self, scenarios: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Generate different market scenarios for testing.
        
        Args:
            scenarios: Dictionary defining different market scenarios
            
        Returns:
            Dictionary of market data for each scenario
        """
        logger.info("Generating market scenarios...")
        
        market_datasets = {}
        
        for scenario_name, params in scenarios.items():
            logger.info(f"Generating scenario: {scenario_name}")
            
            # Set simulator parameters
            if 'initial_price' in params:
                self.market_simulator.initial_price = params['initial_price']
            
            # Generate data
            data = self.market_simulator.generate_dataset(
                n_days=params.get('n_days', 30),
                include_microstructure=params.get('include_microstructure', True),
                include_news=params.get('include_news', True)
            )
            
            market_datasets[scenario_name] = data
            logger.info(f"Generated {len(data)} data points for {scenario_name}")
        
        return market_datasets
    
    def train_lstm_agent(self, train_data: pd.DataFrame, config: LSTMConfig = None) -> LSTMForecastingAgent:
        """
        Train LSTM forecasting agent on training data.
        
        Args:
            train_data: Training dataset
            config: LSTM configuration
            
        Returns:
            Trained LSTM agent
        """
        logger.info("Training LSTM forecasting agent...")
        
        if config is None:
            config = LSTMConfig(
                sequence_length=24,
                hidden_size=64,
                num_layers=2,
                epochs=50,
                batch_size=32
            )
        
        lstm_agent = LSTMForecastingAgent(config)
        
        # Train the agent
        training_history = lstm_agent.train(train_data, validation_split=0.2)
        
        logger.info(f"LSTM training completed. Final validation loss: {training_history['final_val_loss']:.6f}")
        
        return lstm_agent, training_history
    
    def run_hrl_experiment(self, 
                          test_data: pd.DataFrame, 
                          lstm_agent: LSTMForecastingAgent) -> Dict:
        """
        Run HRL trading system experiment.
        
        Args:
            test_data: Testing dataset
            lstm_agent: Trained LSTM agent
            
        Returns:
            HRL experiment results
        """
        logger.info("Running HRL system experiment...")
        
        # Initialize HRL system
        hrl_system = HierarchicalTradingSystem(self.initial_cash)
        
        # Run simulation
        hrl_results = []
        lstm_signals_history = []
        
        warmup_period = 50  # Skip first 50 points for warmup
        
        for i in range(warmup_period, len(test_data)):
            # Get window of data up to current point
            window_data = test_data.iloc[:i+1]
            
            # Get LSTM signals
            try:
                lstm_signals = lstm_agent.get_signals(window_data)
                lstm_signals_history.append(lstm_signals)
            except Exception as e:
                logger.warning(f"LSTM signal generation failed at step {i}: {e}")
                lstm_signals = {'price_signal': 0.0, 'volatility_signal': 1.0, 'confidence': 0.5}
                lstm_signals_history.append(lstm_signals)
            
            # Execute HRL step
            step_result = hrl_system.step(window_data, lstm_signals)
            hrl_results.append(step_result)
            
            # Log progress
            if i % 100 == 0:
                logger.info(f"HRL step {i}: Portfolio Value = ${step_result['portfolio_value']:.2f}")
        
        # Calculate performance metrics
        performance_metrics = hrl_system.get_performance_metrics()
        
        logger.info(f"HRL experiment completed. Total Return: {performance_metrics.get('total_return', 0):.2%}")
        
        return {
            'strategy': 'HRL System',
            'results': hrl_results,
            'lstm_signals': lstm_signals_history,
            'performance': performance_metrics,
            'system': hrl_system
        }
    
    def run_baseline_experiments(self, test_data: pd.DataFrame) -> Dict:
        """
        Run all baseline strategy experiments.
        
        Args:
            test_data: Testing dataset
            
        Returns:
            Baseline experiment results
        """
        logger.info("Running baseline strategy experiments...")
        
        # Run all baseline strategies
        baseline_results = self.baseline_comparison.run_all_backtests(test_data.copy())
        
        logger.info(f"Completed {len(baseline_results)} baseline strategies")
        
        return baseline_results
    
    def run_complete_experiment(self, 
                              scenario_name: str = "mixed_regime",
                              n_days: int = 60,
                              train_test_split: float = 0.7) -> Dict:
        """
        Run complete empirical validation experiment.
        
        Args:
            scenario_name: Name for this experiment scenario
            n_days: Total days of data to generate
            train_test_split: Fraction of data for training
            
        Returns:
            Complete experiment results
        """
        logger.info(f"Starting complete validation experiment: {scenario_name}")
        
        # Step 1: Generate market data
        logger.info("Step 1: Generating market data...")
        scenarios = {
            scenario_name: {
                'n_days': n_days,
                'initial_price': 50000.0,
                'include_microstructure': True,
                'include_news': True
            }
        }
        
        market_datasets = self.generate_market_scenarios(scenarios)
        test_data = market_datasets[scenario_name]
        self.market_data = test_data
        
        # Split data
        split_idx = int(len(test_data) * train_test_split)
        train_data = test_data.iloc[:split_idx]
        test_data_subset = test_data.iloc[split_idx:]
        
        logger.info(f"Training data: {len(train_data)} points, Testing data: {len(test_data_subset)} points")
        
        # Step 2: Train LSTM agent
        logger.info("Step 2: Training LSTM agent...")
        lstm_agent, lstm_training_history = self.train_lstm_agent(train_data)
        
        # Step 3: Run HRL experiment
        logger.info("Step 3: Running HRL experiment...")
        hrl_results = self.run_hrl_experiment(test_data_subset, lstm_agent)
        
        # Step 4: Run baseline experiments
        logger.info("Step 4: Running baseline experiments...")
        baseline_results = self.run_baseline_experiments(test_data_subset)
        
        # Step 5: Combine and analyze results
        logger.info("Step 5: Analyzing results...")
        
        # Combine all results
        all_results = {
            'HRL System': hrl_results['performance'],
            **baseline_results
        }
        
        # Create comparison
        comparison_metrics = self._create_detailed_comparison(all_results, hrl_results, baseline_results)
        
        # Store complete results
        experiment_results = {
            'scenario_name': scenario_name,
            'experiment_date': datetime.now().isoformat(),
            'parameters': {
                'n_days': n_days,
                'train_test_split': train_test_split,
                'initial_cash': self.initial_cash,
                'random_seed': self.random_seed
            },
            'market_data_summary': {
                'total_points': len(test_data),
                'regime_distribution': test_data['regime'].value_counts().to_dict(),
                'price_range': [float(test_data['close'].min()), float(test_data['close'].max())],
                'volatility_stats': {
                    'mean': float(test_data['close'].pct_change().std()),
                    'max': float(test_data['close'].pct_change().rolling(24).std().max())
                }
            },
            'lstm_training': lstm_training_history,
            'hrl_results': hrl_results,
            'baseline_results': baseline_results,
            'comparison': comparison_metrics,
            'all_performance': all_results
        }
        
        # Save results
        self.results[scenario_name] = experiment_results
        self._save_results(scenario_name, experiment_results)
        
        logger.info(f"Experiment {scenario_name} completed successfully!")
        
        return experiment_results
    
    def _create_detailed_comparison(self, 
                                  all_results: Dict, 
                                  hrl_results: Dict, 
                                  baseline_results: Dict) -> Dict:
        """Create detailed comparison analysis"""
        
        # Extract performance metrics
        performance_data = []
        for strategy_name, metrics in all_results.items():
            if isinstance(metrics, dict) and 'error' not in metrics:
                performance_data.append({
                    'Strategy': strategy_name,
                    'Total_Return': metrics.get('total_return', 0),
                    'Sharpe_Ratio': metrics.get('sharpe_ratio', 0),
                    'Max_Drawdown': metrics.get('max_drawdown', 0),
                    'Total_Trades': metrics.get('total_trades', 0),
                    'Win_Rate': metrics.get('win_rate', 0),
                    'Final_Value': metrics.get('final_value', 0)
                })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Calculate rankings
        rankings = {}
        for metric in ['Total_Return', 'Sharpe_Ratio', 'Win_Rate']:
            rankings[f'{metric}_rank'] = performance_df.groupby('Strategy')[metric].rank(ascending=False)
        
        for metric in ['Max_Drawdown']:
            rankings[f'{metric}_rank'] = performance_df.groupby('Strategy')[metric].rank(ascending=True)  # Lower is better
        
        # HRL vs Baseline comparison
        hrl_performance = all_results.get('HRL System', {})
        best_baseline = None
        best_baseline_return = -float('inf')
        
        for name, metrics in baseline_results.items():
            if isinstance(metrics, dict) and 'error' not in metrics:
                return_val = metrics.get('total_return', -float('inf'))
                if return_val > best_baseline_return:
                    best_baseline_return = return_val
                    best_baseline = name
        
        comparison = {
            'performance_summary': performance_df.to_dict('records'),
            'rankings': rankings,
            'hrl_vs_best_baseline': {
                'hrl_return': hrl_performance.get('total_return', 0),
                'best_baseline': best_baseline,
                'best_baseline_return': best_baseline_return,
                'outperformance': hrl_performance.get('total_return', 0) - best_baseline_return,
                'hrl_sharpe': hrl_performance.get('sharpe_ratio', 0),
                'hrl_max_drawdown': hrl_performance.get('max_drawdown', 0)
            }
        }
        
        return comparison
    
    def _save_results(self, scenario_name: str, results: Dict):
        """Save experiment results to files"""
        
        # Save JSON results
        results_file = os.path.join(self.results_dir, f"{scenario_name}_results.json")
        
        # Convert numpy types to native Python types for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save CSV comparison
        if 'comparison' in results and 'performance_summary' in results['comparison']:
            comparison_df = pd.DataFrame(results['comparison']['performance_summary'])
            csv_file = os.path.join(self.results_dir, f"{scenario_name}_comparison.csv")
            comparison_df.to_csv(csv_file, index=False)
            logger.info(f"Comparison table saved to {csv_file}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def generate_plots(self, scenario_name: str):
        """Generate visualization plots for the experiment"""
        if scenario_name not in self.results:
            logger.error(f"No results found for scenario {scenario_name}")
            return
        
        results = self.results[scenario_name]
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Empirical Validation Results: {scenario_name}', fontsize=16)
        
        # Plot 1: Performance Comparison
        ax1 = axes[0, 0]
        comparison_data = results['comparison']['performance_summary']
        comparison_df = pd.DataFrame(comparison_data)
        
        strategies = comparison_df['Strategy']
        returns = [x * 100 for x in comparison_df['Total_Return']]  # Convert to percentage
        
        bars = ax1.bar(range(len(strategies)), returns)
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Total Return (%)')
        ax1.set_title('Total Return Comparison')
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        
        # Highlight HRL system
        hrl_idx = next((i for i, s in enumerate(strategies) if 'HRL' in s), None)
        if hrl_idx is not None:
            bars[hrl_idx].set_color('red')
        
        # Plot 2: Risk-Return Scatter
        ax2 = axes[0, 1]
        sharpe_ratios = comparison_df['Sharpe_Ratio']
        max_drawdowns = [x * 100 for x in comparison_df['Max_Drawdown']]
        
        scatter = ax2.scatter(max_drawdowns, returns, s=100, alpha=0.7)
        ax2.set_xlabel('Max Drawdown (%)')
        ax2.set_ylabel('Total Return (%)')
        ax2.set_title('Risk-Return Profile')
        
        # Annotate points
        for i, strategy in enumerate(strategies):
            ax2.annotate(strategy, (max_drawdowns[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 3: LSTM Training Loss
        ax3 = axes[1, 0]
        lstm_history = results['lstm_training']
        epochs = range(1, len(lstm_history['train_losses']) + 1)
        
        ax3.plot(epochs, lstm_history['train_losses'], label='Training Loss', alpha=0.8)
        ax3.plot(epochs, lstm_history['val_losses'], label='Validation Loss', alpha=0.8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('LSTM Training Progress')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Portfolio Value Over Time (HRL vs Best Baseline)
        ax4 = axes[1, 1]
        
        # Get HRL portfolio values
        hrl_results = results['hrl_results']['results']
        hrl_times = range(len(hrl_results))
        hrl_values = [r['portfolio_value'] for r in hrl_results]
        
        ax4.plot(hrl_times, hrl_values, label='HRL System', linewidth=2, color='red')
        
        # Find best baseline and plot
        best_baseline_name = results['comparison']['hrl_vs_best_baseline']['best_baseline']
        if best_baseline_name:
            # This would require storing baseline portfolio values over time
            # For now, just show the HRL system
            pass
        
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Portfolio Value ($)')
        ax4.set_title('Portfolio Value Over Time')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.results_dir, f"{scenario_name}_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Analysis plots saved to {plot_file}")
        
        plt.show()
    
    def print_summary(self, scenario_name: str):
        """Print a summary of experiment results"""
        if scenario_name not in self.results:
            logger.error(f"No results found for scenario {scenario_name}")
            return
        
        results = self.results[scenario_name]
        
        print(f"\n{'='*60}")
        print(f"EMPIRICAL VALIDATION RESULTS: {scenario_name.upper()}")
        print(f"{'='*60}")
        
        # Experiment parameters
        params = results['parameters']
        print(f"\nExperiment Parameters:")
        print(f"  • Data period: {params['n_days']} days")
        print(f"  • Train/test split: {params['train_test_split']:.1%}")
        print(f"  • Initial cash: ${params['initial_cash']:,.2f}")
        print(f"  • Random seed: {params['random_seed']}")
        
        # Market data summary
        market_summary = results['market_data_summary']
        print(f"\nMarket Data Summary:")
        print(f"  • Total data points: {market_summary['total_points']:,}")
        print(f"  • Price range: ${market_summary['price_range'][0]:,.0f} - ${market_summary['price_range'][1]:,.0f}")
        print(f"  • Average volatility: {market_summary['volatility_stats']['mean']:.2%}")
        
        regime_dist = market_summary['regime_distribution']
        print(f"  • Regime distribution: {', '.join([f'{k}: {v}' for k, v in regime_dist.items()])}")
        
        # Performance comparison
        print(f"\nPerformance Comparison:")
        print(f"{'Strategy':<25} {'Total Return':<12} {'Sharpe Ratio':<12} {'Max Drawdown':<12} {'Trades':<8}")
        print("-" * 70)
        
        for strategy_data in results['comparison']['performance_summary']:
            print(f"{strategy_data['Strategy']:<25} "
                  f"{strategy_data['Total_Return']:<11.1%} "
                  f"{strategy_data['Sharpe_Ratio']:<11.2f} "
                  f"{strategy_data['Max_Drawdown']:<11.1%} "
                  f"{strategy_data['Total_Trades']:<8.0f}")
        
        # HRL vs Best Baseline
        hrl_vs_best = results['comparison']['hrl_vs_best_baseline']
        print(f"\nHRL System vs Best Baseline:")
        print(f"  • Best baseline: {hrl_vs_best['best_baseline']}")
        print(f"  • HRL return: {hrl_vs_best['hrl_return']:.1%}")
        print(f"  • Best baseline return: {hrl_vs_best['best_baseline_return']:.1%}")
        print(f"  • Outperformance: {hrl_vs_best['outperformance']:+.1%}")
        
        if hrl_vs_best['outperformance'] > 0:
            print(f"  ✅ HRL system OUTPERFORMED the best baseline!")
        else:
            print(f"  ❌ HRL system UNDERPERFORMED the best baseline.")
        
        print(f"\n{'='*60}")


def main():
    """Main function to run empirical validation"""
    
    # Initialize experiment
    experiment = EmpiricalValidationExperiment(
        initial_cash=100000.0,
        random_seed=42,
        results_dir="validation_results"
    )
    
    # Run complete experiment
    results = experiment.run_complete_experiment(
        scenario_name="mixed_regime_validation",
        n_days=90,
        train_test_split=0.7
    )
    
    # Print summary
    experiment.print_summary("mixed_regime_validation")
    
    # Generate plots
    experiment.generate_plots("mixed_regime_validation")
    
    return results


if __name__ == "__main__":
    main()