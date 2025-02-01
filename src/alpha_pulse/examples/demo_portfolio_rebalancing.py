"""
Demo script showing how to use the portfolio rebalancing functionality.
"""
import asyncio
import argparse
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from alpha_pulse.exchanges import ExchangeType, ExchangeFactory, credentials_manager
from alpha_pulse.portfolio.analyzer import PortfolioAnalyzer
from alpha_pulse.portfolio.mpt_strategy import MPTStrategy
from alpha_pulse.portfolio.hrp_strategy import HRPStrategy


def plot_allocations(current: dict, target: dict, title: str) -> None:
    """Plot current vs target allocations.
    
    Args:
        current: Current portfolio weights
        target: Target portfolio weights
        title: Plot title
    """
    try:
        # Prepare data
        assets = sorted(set(current.keys()) | set(target.keys()))
        current_weights = [float(current.get(asset, 0)) for asset in assets]
        target_weights = [float(target.get(asset, 0)) for asset in assets]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        x = range(len(assets))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], current_weights, width, label='Current', alpha=0.8)
        plt.bar([i + width/2 for i in x], target_weights, width, label='Target', alpha=0.8)
        
        plt.xlabel('Assets')
        plt.ylabel('Weight')
        plt.title(title)
        plt.xticks(x, assets, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig('portfolio_allocation.png')
        plt.close()
        logger.info("Portfolio allocation plot saved to portfolio_allocation.png")
        
    except Exception as e:
        logger.error(f"Error plotting allocations: {e}")


async def analyze_with_strategy(analyzer: PortfolioAnalyzer, strategy_class, name: str) -> None:
    """Analyze portfolio using specified strategy.
    
    Args:
        analyzer: Portfolio analyzer instance
        strategy_class: Strategy class to use
        name: Strategy name for logging
    """
    logger.info(f"\nAnalyzing portfolio with {name}...")
    
    try:
        # Get current allocation
        current_weights = await analyzer.get_current_allocation()
        if not current_weights:
            logger.error("No assets found in portfolio")
            return
            
        logger.info("\nCurrent portfolio weights:")
        for asset, weight in current_weights.items():
            logger.info(f"{asset}: {weight:.2%}")
        
        # Calculate optimal allocation
        result = await analyzer.analyze_portfolio(
            strategy_class=strategy_class,
            constraints={
                'min_weight': 0.05,  # Minimum 5% per asset
                'max_weight': 0.40   # Maximum 40% per asset
            }
        )
        
        logger.info(f"\n{name} Results:")
        logger.info(f"Expected Return: {result.expected_return:.2%}")
        logger.info(f"Expected Risk: {result.expected_risk:.2%}")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Rebalance Score: {result.rebalance_score:.2%}")
        
        logger.info("\nTarget portfolio weights:")
        for asset, weight in result.weights.items():
            logger.info(f"{asset}: {weight:.2%}")
        
        # Plot allocations
        plot_allocations(
            current_weights,
            result.weights,
            f"Portfolio Allocation - {name}"
        )
        
        # Calculate rebalancing trades
        try:
            balances = await analyzer.exchange.get_balances()
            if not balances:
                logger.error("No balance information available")
                return
                
            total_value = sum(
                balance.in_base_currency
                for balance in balances.values()
            )
            
            trades = analyzer.get_rebalancing_trades(
                current_weights,
                result.weights,
                total_value,
                min_trade_value=Decimal("10.0")
            )
            
            if trades:
                logger.info("\nSuggested rebalancing trades:")
                for trade in trades:
                    logger.info(
                        f"{trade['side'].upper()} {trade['asset']}: "
                        f"${float(trade['value']):.2f}"
                    )
            else:
                logger.info("\nNo rebalancing trades needed")
                
        except Exception as e:
            logger.error(f"Error calculating trades: {e}")
            
    except ValueError as e:
        logger.error(f"Error in portfolio analysis: {e}")
    except Exception as e:
        logger.error(f"Unexpected error analyzing with {name}: {e}")


async def main() -> None:
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Portfolio rebalancing demo")
    parser.add_argument("--api-key", help="Exchange API key")
    parser.add_argument("--api-secret", help="Exchange API secret")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    args = parser.parse_args()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Portfolio rebalancing demo")
    parser.add_argument("--exchange", choices=["binance", "bybit"], default="binance", help="Exchange to use")
    parser.add_argument("--api-key", help="Exchange API key")
    parser.add_argument("--api-secret", help="Exchange API secret")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    args = parser.parse_args()
    
    exchange = None
    try:
        # Save credentials if provided
        if args.api_key and args.api_secret:
            credentials_manager.save_credentials(
                args.exchange,
                args.api_key,
                args.api_secret,
                args.testnet
            )
        
        # Create exchange instance
        exchange_type = ExchangeType(args.exchange)
        exchange = await ExchangeFactory.create_exchange(
            exchange_type,
            testnet=args.testnet
        )
        
        # Validate API keys
        if not await exchange.validate_api_keys():
            logger.error("Invalid API keys")
            return
        
        # Initialize portfolio analyzer
        analyzer = PortfolioAnalyzer(exchange)
        
        # Analyze with different strategies
        await analyze_with_strategy(analyzer, MPTStrategy, "Modern Portfolio Theory")
        await analyze_with_strategy(analyzer, HRPStrategy, "Hierarchical Risk Parity")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Clean up
        if exchange:
            await exchange.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())