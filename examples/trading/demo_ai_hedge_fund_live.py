"""
Demo script showcasing the complete AI Hedge Fund flow with live data streaming.
This version continuously generates data and sends it to the API for real-time dashboard updates.
"""
import asyncio
import argparse
import os
import random
import time
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import yaml
from loguru import logger
import sys
import json
import aiohttp
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
    "logs/ai_hedge_fund_live_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
           "{name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="500 MB"
)

# API configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_USERNAME = "admin"
API_PASSWORD = "password"

async def get_auth_token():
    """Get authentication token from API."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/token",
            data={"username": API_USERNAME, "password": API_PASSWORD}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data["access_token"]
            else:
                logger.error(f"Failed to get auth token: {response.status}")
                return None

async def send_data_to_api(endpoint, data, token):
    """Send data to API endpoint."""
    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_BASE_URL}/{endpoint}",
            json=data,
            headers=headers
        ) as response:
            if response.status != 200:
                logger.error(f"Failed to send data to {endpoint}: {response.status}")
                return False
            return True

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
        
        # Calculate position size
        position_result = components["risk_manager"].calculate_position_size(
            symbol=signal.symbol,
            current_price=current_price,
            signal_strength=signal.confidence,
            historical_returns=prices_df[signal.symbol].pct_change().dropna()
        )
        
        position_value = float(position_result.size) * float(current_price)
        quantity = position_value / float(current_price)
        logger.debug(f"Calculated position size: {quantity:.4f} {signal.symbol} (${position_value:,.2f})")
        
        if await components["risk_manager"].evaluate_trade(
            symbol=signal.symbol,
            side=signal.direction.value,
            quantity=quantity,
            current_price=current_price,
            portfolio_value=portfolio_value,
            current_positions=current_positions
        ):
            signal.metadata["quantity"] = quantity
            signal.metadata["position_value"] = position_value
            signal.metadata["position_size"] = quantity
            signal.metadata["position_confidence"] = float(position_result.confidence)
            logger.debug(f"Signal passed risk evaluation: {signal.symbol} {signal.direction.value} ({quantity:.4f} units)")
            valid_signals.append(signal)
        else:
            logger.debug(f"Signal failed risk evaluation: {signal.symbol} {signal.direction.value} ({quantity:.4f} units)")
    
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
        portfolio_data.asset_allocation
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

async def send_metrics_to_api(metrics, portfolio_data, token):
    """Send metrics to API for dashboard display."""
    # Format metrics for API
    api_metrics = {
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": float(portfolio_data.total_value),
        "cash_balance": float(portfolio_data.cash_balance),
        "asset_value": float(portfolio_data.total_value - portfolio_data.cash_balance),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0)),
        "volatility": float(metrics.get("volatility", 0)),
        "positions": [
            {
                "symbol": symbol,
                "quantity": float(position.quantity),
                "value": float(position.value),
                "allocation": float(position.allocation),
                "price": float(position.current_price),
                "pnl": float(position.unrealized_pnl),
                "pnl_percent": float(position.unrealized_pnl_percent)
            }
            for symbol, position in portfolio_data.positions.items()
        ]
    }
    
    # Send to API
    await send_data_to_api("metrics/update", api_metrics, token)
    logger.info("Metrics sent to API")

async def send_alerts_to_api(signals, token):
    """Send alerts to API for dashboard display."""
    # Format alerts for API
    alerts = []
    for signal in signals:
        severity = "info"
        if signal.confidence > 0.7:
            severity = "critical"
        elif signal.confidence > 0.5:
            severity = "warning"
            
        alerts.append({
            "timestamp": datetime.now().isoformat(),
            "title": f"{signal.symbol} {signal.direction.value} Signal",
            "message": f"Generated {signal.direction.value} signal for {signal.symbol} with confidence {signal.confidence:.2f}",
            "severity": severity,
            "source": signal.source,
            "metadata": {
                "symbol": signal.symbol,
                "direction": signal.direction.value,
                "confidence": float(signal.confidence),
                "quantity": float(signal.metadata.get("quantity", 0))
            }
        })
    
    if alerts:
        # Send to API
        await send_data_to_api("alerts/create", {"alerts": alerts}, token)
        logger.info(f"Sent {len(alerts)} alerts to API")

async def send_trades_to_api(portfolio_data, token):
    """Send trade data to API for dashboard display."""
    # Generate some simulated trades based on portfolio
    trades = []
    for symbol, position in portfolio_data.positions.items():
        # Simulate a recent trade for each position
        trade_time = datetime.now() - timedelta(minutes=random.randint(1, 60))
        
        # Randomly decide if it's a buy or sell
        side = "buy" if random.random() > 0.3 else "sell"
        quantity = float(position.quantity) * random.uniform(0.05, 0.2)
        price = float(position.current_price) * random.uniform(0.98, 1.02)
        
        trades.append({
            "timestamp": trade_time.isoformat(),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "value": quantity * price,
            "status": "executed",
            "source": random.choice(["technical_agent", "fundamental_agent", "sentiment_agent", "value_agent"]),
            "metadata": {
                "confidence": random.uniform(0.5, 0.9),
                "execution_time_ms": random.randint(50, 500)
            }
        })
    
    if trades:
        # Send to API
        await send_data_to_api("trades/record", {"trades": trades}, token)
        logger.info(f"Sent {len(trades)} trades to API")

async def update_system_status(token):
    """Send system status to API for dashboard display."""
    # Generate system status data
    status_data = {
        "timestamp": datetime.now().isoformat(),
        "components": {
            "data_pipeline": {
                "status": "healthy",
                "latency_ms": random.randint(10, 100),
                "throughput": random.randint(100, 1000)
            },
            "agent_manager": {
                "status": "healthy",
                "active_agents": 5,
                "signals_generated": random.randint(10, 50)
            },
            "risk_management": {
                "status": "healthy",
                "checks_performed": random.randint(50, 200),
                "trades_approved": random.randint(5, 20)
            },
            "portfolio_manager": {
                "status": "healthy",
                "rebalances": random.randint(0, 3),
                "optimization_runs": random.randint(1, 5)
            },
            "execution": {
                "status": "healthy",
                "orders_processed": random.randint(10, 30),
                "execution_latency_ms": random.randint(50, 200)
            }
        },
        "resources": {
            "cpu_usage": random.uniform(10, 80),
            "memory_usage": random.uniform(20, 70),
            "disk_usage": random.uniform(30, 60),
            "network_throughput": random.uniform(1, 10)
        }
    }
    
    # Send to API
    await send_data_to_api("system/status", status_data, token)
    logger.info("System status sent to API")

async def live_data_loop(components):
    """Run continuous loop to generate and send live data to API."""
    logger.info("Starting live data loop...")
    
    # Get auth token
    token = await get_auth_token()
    if not token:
        logger.error("Failed to get authentication token. Exiting live data loop.")
        return
    
    # Get configuration
    symbols = components["config"]["trading"]["symbols"]
    
    # Initial market data fetch
    market_data = await fetch_market_data(
        components["data_manager"],
        symbols,
        lookback_days=180
    )
    
    # Main loop
    iteration = 0
    while True:
        try:
            logger.info(f"Live data iteration {iteration}")
            
            # Update market data with some random changes
            for symbol, data_list in market_data.items():
                if data_list:
                    last_data = data_list[-1]
                    # Create a new data point with a small random change
                    price_change = random.uniform(-0.02, 0.02)
                    new_price = float(last_data.close) * (1 + price_change)
                    volume = float(last_data.volume) * random.uniform(0.8, 1.2)
                    
                    # Update the last data point
                    last_data.close = Decimal(str(new_price))
                    last_data.volume = Decimal(str(volume))
            
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
            
            # Send data to API
            await send_metrics_to_api(metrics, portfolio_data, token)
            await send_alerts_to_api(valid_signals, token)
            await send_trades_to_api(portfolio_data, token)
            await update_system_status(token)
            
            # Log current status
            logger.info(f"Iteration {iteration} completed")
            logger.info(f"Portfolio Value: ${portfolio_data.total_value:,.2f}")
            logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            
            # Increment iteration counter
            iteration += 1
            
            # Wait before next iteration
            await asyncio.sleep(10)  # Update every 10 seconds
            
            # Refresh token every 10 iterations
            if iteration % 10 == 0:
                token = await get_auth_token()
                if not token:
                    logger.error("Failed to refresh authentication token.")
                    break
            
        except Exception as e:
            logger.exception(f"Error in live data loop: {str(e)}")
            await asyncio.sleep(5)  # Wait a bit before retrying

async def main():
    """Main execution flow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Hedge Fund Demo")
    parser.add_argument("--live-data", action="store_true", help="Run in live data mode")
    args = parser.parse_args()
    
    components = {}
    try:
        # Initialize all components
        components = await initialize_components()
        
        if args.live_data:
            # Run in live data mode
            logger.info("Running in live data mode")
            await live_data_loop(components)
        else:
            # Run in single execution mode
            logger.info("Running in single execution mode")
            
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