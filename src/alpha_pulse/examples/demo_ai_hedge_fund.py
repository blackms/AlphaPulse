"""
Demo script showcasing the complete AI Hedge Fund flow.
Demonstrates the interaction between agents, risk management, and portfolio management.
"""
import asyncio
import os
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import yaml
from loguru import logger
import sys
from pathlib import Path
from collections import defaultdict

from alpha_pulse.agents.interfaces import MarketData

from alpha_pulse.agents.manager import AgentManager
from alpha_pulse.risk_management.manager import RiskManager, RiskConfig
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.execution.paper_broker import PaperBroker
from alpha_pulse.data_pipeline.manager import DataManager
from alpha_pulse.monitoring.metrics import MetricsCollector
from alpha_pulse.portfolio.html_report import generate_portfolio_report

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "<level>{message}</level>",
    level="DEBUG"
)
logger.add(
    "logs/ai_hedge_fund_demo_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
           "{name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="500 MB"
)

async def initialize_components():
    """Initialize all system components."""
    logger.info("Initializing AI Hedge Fund components...")

    # Load configurations
    with open("config/ai_hedge_fund_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    with open("config/data_pipeline_config.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    # Initialize components
    agent_manager = AgentManager(config.get("agents", {}))
    await agent_manager.initialize()
    logger.info("Agent Manager initialized")

    # Extract top-level risk management parameters
    risk_config = {
        k: v for k, v in config.get("risk_management", {}).items()
        if k in [
            "max_position_size",
            "max_portfolio_leverage",
            "max_drawdown",
            "stop_loss",
            "var_confidence",
            "risk_free_rate",
            "target_volatility",
            "rebalance_threshold",
            "initial_portfolio_value"
        ]
    }

    risk_manager = RiskManager(
        exchange=None,  # Will be set later
        config=RiskConfig(**risk_config)
    )
    logger.info("Risk Manager initialized")

    portfolio_manager = PortfolioManager(
        config_path="config/portfolio_config.yaml"
    )
    logger.info("Portfolio Manager initialized")

    data_manager = DataManager(config=data_config)
    await data_manager.initialize()
    logger.info("Data Manager initialized")

    # Initialize paper broker with initial balance
    paper_broker = PaperBroker(
        initial_balance=float(config["risk_management"]["initial_portfolio_value"])
    )
    logger.info("Paper Broker initialized")

    # Initialize market prices for trading pairs
    symbols = config["trading"]["symbols"]
    for symbol in symbols:
        # Initialize with a default price of 1.0
        # Real prices will be updated when we fetch market data
        paper_broker.update_market_data(symbol, 1.0)

    metrics_collector = MetricsCollector()
    logger.info("Metrics Collector initialized")

    return {
        "agent_manager": agent_manager,
        "risk_manager": risk_manager,
        "portfolio_manager": portfolio_manager,
        "data_manager": data_manager,
        "broker": paper_broker,
        "metrics": metrics_collector,
        "config": config
    }

async def fetch_market_data(data_manager, symbols, lookback_days=180):
    """Fetch market data for analysis."""
    logger.info(f"Fetching market data for {len(symbols)} symbols...")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
    market_data = await data_manager.get_market_data(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        interval="1d"
    )
    
    logger.info(f"Fetched {sum(len(data) for data in market_data.values())} data points")
    return market_data

async def generate_and_process_signals(components, market_data):
    """Generate and process trading signals."""
    logger.info("Generating trading signals...")
    
    # Update broker with latest market prices
    for symbol, data in market_data.items():
        if data:  # If we have data for this symbol
            latest_price = float(data[-1].close)
            components["broker"].update_market_data(symbol, latest_price)
    
    # Convert market data to DataFrames for agents
    data_by_symbol = defaultdict(list)
    timestamps = set()

    # First pass: collect all timestamps and data
    for symbol, data_list in market_data.items():
        if not data_list:  # Skip empty data
            continue
        for data in data_list:
            timestamps.add(data.timestamp)
            data_by_symbol[symbol].append({
                'timestamp': data.timestamp,
                'price': float(data.close),
                'volume': float(data.volume)
            })

    if not timestamps:  # No data available
        logger.warning("No market data available")
        return []

    # Sort timestamps
    sorted_timestamps = sorted(timestamps)

    # Create price and volume DataFrames
    prices_data = {}
    volumes_data = {}

    for symbol, data_list in data_by_symbol.items():
        # Create temporary DataFrame for this symbol
        symbol_df = pd.DataFrame(data_list)
        symbol_df.set_index('timestamp', inplace=True)
        symbol_df.sort_index(inplace=True)

        # Reindex to include all timestamps
        symbol_df = symbol_df.reindex(sorted_timestamps)
        
        # Forward fill missing values
        symbol_df = symbol_df.ffill()
        
        # Extract price and volume series
        prices_data[symbol] = symbol_df['price']
        volumes_data[symbol] = symbol_df['volume']

    # Create final DataFrames
    prices_df = pd.DataFrame(prices_data, index=sorted_timestamps)
    volumes_df = pd.DataFrame(volumes_data, index=sorted_timestamps)

    # Create MarketData object with DataFrames
    market_data_obj = MarketData(
        prices=prices_df,
        volumes=volumes_df,
        fundamentals={},  # Add if available
        sentiment={},     # Add if available
        technical_indicators={},  # Add if available
        timestamp=datetime.now(),
        data_by_symbol=market_data  # Store raw data
    )

    # Generate signals from all agents
    signals = await components["agent_manager"].generate_signals(market_data_obj)
    logger.info(f"Generated {len(signals)} initial signals from agent manager")
    for signal in signals:
        logger.debug(f"Signal: {signal.symbol} {signal.direction.value} (confidence: {signal.confidence:.2f})")
    
    # Filter signals through risk management
    valid_signals = []
    portfolio_value = await components["broker"].get_portfolio_value()
    current_positions = {
        symbol: {
            'quantity': pos['quantity'],
            'current_price': pos['current_price']
        }
        for symbol, pos in (await components["broker"].get_positions()).items()
    }
    
    logger.debug(f"Portfolio value: ${float(portfolio_value):,.2f}")
    logger.debug(f"Current positions: {current_positions}")
    
    for signal in signals:
        current_price = market_data[signal.symbol][-1].close
        logger.debug(f"Evaluating {signal.symbol} {signal.direction.value} signal (price: ${float(current_price):,.2f})")
        
        if await components["risk_manager"].evaluate_trade(
            symbol=signal.symbol,
            side=signal.direction.value,
            quantity=0,  # Will be determined by position sizer
            current_price=current_price,
            portfolio_value=portfolio_value,
            current_positions=current_positions
        ):
            logger.debug(f"Signal passed risk evaluation: {signal.symbol} {signal.direction.value}")
            valid_signals.append(signal)
        else:
            logger.debug(f"Signal failed risk evaluation: {signal.symbol} {signal.direction.value}")
    
    logger.info(f"{len(valid_signals)} signals passed risk evaluation")
    return valid_signals

async def execute_portfolio_decisions(components, valid_signals, market_data):
    """Execute portfolio decisions based on signals."""
    logger.info("Executing portfolio decisions...")
    
    # Get current portfolio state
    portfolio_data = await components["portfolio_manager"].get_portfolio_data(
        components["broker"]
    )
    
    # Check if rebalancing is needed
    needs_rebalance = await components["portfolio_manager"].needs_rebalancing(
        components["broker"],
        portfolio_data.asset_allocation  # Use asset_allocation instead of current_weights
    )
    
    if needs_rebalance:
        logger.info("Portfolio rebalancing required")
        # Execute rebalancing
        rebalance_result = await components["portfolio_manager"].rebalance_portfolio(
            components["broker"],
            market_data
        )
        logger.info(f"Rebalancing completed: {rebalance_result['status']}")
    
    return portfolio_data

async def monitor_and_report(components, portfolio_data):
    """Monitor system performance and generate reports."""
    logger.info("Generating performance reports...")
    
    # Collect metrics
    metrics = components["metrics"].collect_metrics(portfolio_data)
    
    # Generate HTML report
    report_path = Path("reports") / f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path.parent.mkdir(exist_ok=True)
    
    generate_portfolio_report(
        portfolio_data=portfolio_data,
        metrics=metrics,
        output_path=str(report_path)
    )
    
    logger.info(f"Report generated: {report_path}")
    return metrics

async def main():
    """Main execution flow."""
    components = {}
    try:
        # Initialize all components
        components = await initialize_components()
        
        # Get configuration
        symbols = components["config"]["trading"]["symbols"]
        
        # Fetch market data
        market_data = await fetch_market_data(
            components["data_manager"],
            symbols
        )
        
        # Generate and process signals
        valid_signals = await generate_and_process_signals(
            components,
            market_data
        )
        
        # Execute portfolio decisions
        portfolio_data = await execute_portfolio_decisions(
            components,
            valid_signals,
            market_data
        )
        
        # Monitor and report
        metrics = await monitor_and_report(
            components,
            portfolio_data
        )
        
        logger.info("Demo completed successfully")
        logger.info(f"Portfolio Value: ${portfolio_data.total_value:,.2f}")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
    except Exception as e:
        logger.exception(f"Error in main execution: {str(e)}")
        raise
    finally:
        # Cleanup
        if "data_manager" in components:
            await components["data_manager"].__aexit__(None, None, None)
        logger.info("Cleanup completed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.exception(f"Process terminated with error: {str(e)}")