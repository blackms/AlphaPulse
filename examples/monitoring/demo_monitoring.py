#!/usr/bin/env python
"""
Demonstration of the enhanced monitoring system.

This example shows how to:
1. Initialize the metrics collector
2. Collect and store metrics
3. Query historical metrics
4. Set up real-time monitoring
"""
import asyncio
import logging
import random
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.alpha_pulse.monitoring.collector import EnhancedMetricsCollector
from src.alpha_pulse.monitoring.config import MonitoringConfig
from src.alpha_pulse.portfolio.data_models import PortfolioData, Position


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monitoring_demo")


async def generate_sample_portfolio(days=30, volatility=0.02):
    """
    Generate a sample portfolio with random price movements.
    
    Args:
        days: Number of days of history
        volatility: Daily price volatility
        
    Returns:
        List of PortfolioData objects
    """
    # Initial portfolio
    initial_cash = 100000.0
    symbols = ["BTC", "ETH", "SOL", "ADA", "DOT"]
    initial_prices = {
        "BTC": 50000.0,
        "ETH": 3000.0,
        "SOL": 150.0,
        "ADA": 2.0,
        "DOT": 30.0
    }
    
    # Initial allocation
    allocation = {
        "BTC": 0.4,  # 40% in BTC
        "ETH": 0.3,  # 30% in ETH
        "SOL": 0.15, # 15% in SOL
        "ADA": 0.1,  # 10% in ADA
        "DOT": 0.05  # 5% in DOT
    }
    
    # Calculate initial quantities
    quantities = {}
    for symbol, alloc in allocation.items():
        quantities[symbol] = (initial_cash * alloc) / initial_prices[symbol]
    
    # Generate price movements
    price_history = {}
    for symbol in symbols:
        # Generate log returns
        returns = np.random.normal(0.0005, volatility, days)  # Slight positive drift
        
        # Convert to price series
        prices = [initial_prices[symbol]]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_history[symbol] = prices
    
    # Generate portfolio snapshots
    portfolio_history = []
    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    for day in range(days + 1):
        current_date = start_date + timedelta(days=day)
        positions = []
        
        for symbol in symbols:
            current_price = price_history[symbol][day]
            positions.append(
                Position(
                    symbol=symbol,
                    quantity=quantities[symbol],
                    current_price=current_price,
                    cost_basis=initial_prices[symbol]
                )
            )
        
        # Calculate remaining cash (assume no additional trades)
        remaining_cash = initial_cash * 0.0  # All invested
        
        # Create portfolio data
        portfolio = PortfolioData(
            positions=positions,
            cash=remaining_cash,
            timestamp=current_date,
            initial_value=initial_cash,
            start_date=start_date
        )
        
        portfolio_history.append(portfolio)
    
    return portfolio_history


async def generate_sample_trades(days=30):
    """
    Generate sample trade data.
    
    Args:
        days: Number of days of history
        
    Returns:
        List of trade data dictionaries
    """
    symbols = ["BTC", "ETH", "SOL", "ADA", "DOT"]
    sides = ["buy", "sell"]
    order_types = ["market", "limit"]
    
    trades = []
    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Generate random trades
    for day in range(days):
        # 1-3 trades per day
        num_trades = random.randint(1, 3)
        
        for _ in range(num_trades):
            # Random trade time during the day
            hours = random.randint(0, 23)
            minutes = random.randint(0, 59)
            seconds = random.randint(0, 59)
            
            trade_time = start_date + timedelta(days=day, hours=hours, minutes=minutes, seconds=seconds)
            
            # Random trade details
            symbol = random.choice(symbols)
            side = random.choice(sides)
            order_type = random.choice(order_types)
            
            # Price based on symbol
            base_price = {
                "BTC": 50000.0,
                "ETH": 3000.0,
                "SOL": 150.0,
                "ADA": 2.0,
                "DOT": 30.0
            }[symbol]
            
            # Add some randomness to price
            price_noise = random.uniform(-0.02, 0.02)  # ±2%
            price = base_price * (1 + price_noise)
            
            # Expected price might be slightly different
            expected_price = price * (1 + random.uniform(-0.005, 0.005))  # ±0.5%
            
            # Random quantity
            quantity = random.uniform(0.1, 2.0) if symbol in ["BTC", "ETH"] else random.uniform(10, 100)
            
            # Sometimes have partial fills
            requested_quantity = quantity
            if random.random() < 0.2:  # 20% chance of partial fill
                quantity = quantity * random.uniform(0.7, 0.95)
            
            # Random commission
            commission = price * quantity * random.uniform(0.0005, 0.002)  # 0.05% to 0.2%
            
            # Create trade data
            trade = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "requested_quantity": requested_quantity,
                "price": price,
                "expected_price": expected_price,
                "order_type": order_type,
                "execution_time": (datetime.now(timezone.utc) - trade_time).total_seconds() * 1000,  # ms
                "commission": commission,
                "timestamp": trade_time
            }
            
            trades.append(trade)
    
    # Sort by timestamp
    trades.sort(key=lambda x: x["timestamp"])
    
    return trades


async def generate_sample_agent_data(days=30):
    """
    Generate sample agent performance data.
    
    Args:
        days: Number of days of history
        
    Returns:
        List of agent data dictionaries
    """
    agent_types = ["technical", "fundamental", "sentiment", "value", "activist"]
    
    agent_data_history = []
    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Generate random agent data
    for day in range(days):
        # Random time during the day
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        
        signal_time = start_date + timedelta(days=day, hours=hours, minutes=minutes)
        
        # Create agent data for this timestamp
        agent_data = {}
        
        for agent_type in agent_types:
            # Random direction (-1, 0, 1)
            direction = random.choice([-1, 0, 1])
            
            # Random confidence
            confidence = random.uniform(0.5, 0.95) if direction != 0 else random.uniform(0.1, 0.4)
            
            # Random prediction (boolean)
            prediction = direction > 0
            
            # Random outcome (boolean)
            # Higher chance of being correct if confidence is high
            if random.random() < confidence:
                actual_outcome = prediction
            else:
                actual_outcome = not prediction
            
            # Create agent signal
            agent_data[agent_type] = {
                "direction": direction,
                "confidence": confidence,
                "prediction": prediction,
                "actual_outcome": actual_outcome,
                "timestamp": signal_time
            }
        
        agent_data_history.append((signal_time, agent_data))
    
    # Sort by timestamp
    agent_data_history.sort(key=lambda x: x[0])
    
    return agent_data_history


async def plot_metrics(metrics_history, metric_type, fields, title):
    """
    Plot metrics from history.
    
    Args:
        metrics_history: List of metric data points
        metric_type: Type of metrics
        fields: List of fields to plot
        title: Plot title
    """
    if not metrics_history:
        logger.warning(f"No {metric_type} metrics to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_history)
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set timestamp as index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # Plot specified fields
    plt.figure(figsize=(12, 6))
    for field in fields:
        if field in df.columns:
            plt.plot(df.index, df[field], label=field)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Save plot
    filename = f"{metric_type}_{'-'.join(fields)}.png"
    plt.savefig(plots_dir / filename)
    logger.info(f"Saved plot to plots/{filename}")
    
    plt.close()


async def main():
    """Main demonstration function."""
    logger.info("Starting monitoring system demonstration")
    
    # Create a monitoring configuration
    config = MonitoringConfig(
        storage=MonitoringConfig.StorageConfig(
            type="memory",  # Use in-memory storage for demo
            memory_max_points=10000
        ),
        collection_interval=5,  # 5 seconds between collections
        enable_realtime=True
    )
    
    # Initialize the metrics collector
    collector = EnhancedMetricsCollector(config=config)
    
    # Start the collector
    await collector.start()
    logger.info("Metrics collector started")
    
    try:
        # Generate sample data
        logger.info("Generating sample portfolio data...")
        portfolio_history = await generate_sample_portfolio(days=30)
        
        logger.info("Generating sample trade data...")
        trades = await generate_sample_trades(days=30)
        
        logger.info("Generating sample agent data...")
        agent_data_history = await generate_sample_agent_data(days=30)
        
        # Process historical data
        logger.info("Processing historical data...")
        for i, portfolio in enumerate(portfolio_history):
            # Find trades and agent data for this day
            day_trades = [t for t in trades if abs((t["timestamp"] - portfolio.timestamp).total_seconds()) < 86400]
            day_agent_data = [a[1] for a in agent_data_history if abs((a[0] - portfolio.timestamp).total_seconds()) < 86400]
            
            # Process each trade
            for trade in day_trades:
                await collector.collect_and_store(
                    portfolio_data=portfolio,
                    trade_data=trade,
                    system_data=False
                )
            
            # Process each agent data point
            for agent_data in day_agent_data:
                await collector.collect_and_store(
                    portfolio_data=portfolio,
                    agent_data=agent_data,
                    system_data=False
                )
            
            # Log progress
            if i % 5 == 0:
                logger.info(f"Processed {i+1}/{len(portfolio_history)} days of data")
        
        # Query historical metrics
        logger.info("Querying historical performance metrics...")
        performance_metrics = await collector.get_metrics_history("performance")
        
        logger.info("Querying historical risk metrics...")
        risk_metrics = await collector.get_metrics_history("risk")
        
        logger.info("Querying historical trade metrics...")
        trade_metrics = await collector.get_metrics_history("trade")
        
        # Plot metrics
        logger.info("Plotting metrics...")
        
        # Performance metrics
        await plot_metrics(
            performance_metrics,
            "performance",
            ["sharpe_ratio", "sortino_ratio", "max_drawdown"],
            "Performance Metrics"
        )
        
        # Risk metrics
        await plot_metrics(
            risk_metrics,
            "risk",
            ["leverage", "concentration_hhi", "portfolio_value"],
            "Risk Metrics"
        )
        
        # Run real-time monitoring for a short period
        logger.info("Running real-time monitoring for 30 seconds...")
        for i in range(6):
            # Generate some random portfolio data
            portfolio = portfolio_history[-1]  # Use the latest portfolio
            
            # Generate a random trade
            trade = random.choice(trades)
            
            # Generate random agent data
            _, agent_data = random.choice(agent_data_history)
            
            # Collect and store metrics
            await collector.collect_and_store(
                portfolio_data=portfolio,
                trade_data=trade,
                agent_data=agent_data,
                system_data=True
            )
            
            logger.info(f"Collected real-time metrics ({i+1}/6)")
            await asyncio.sleep(5)
        
        # Get the latest metrics
        logger.info("Getting latest metrics...")
        latest_performance = await collector.get_latest_metrics("performance")
        latest_system = await collector.get_latest_metrics("system")
        
        if latest_performance:
            logger.info(f"Latest Sharpe Ratio: {latest_performance[0].get('sharpe_ratio', 'N/A')}")
        
        if latest_system:
            logger.info(f"Latest CPU Usage: {latest_system[0].get('cpu_usage_percent', 'N/A')}%")
            logger.info(f"Latest Memory Usage: {latest_system[0].get('memory_usage_percent', 'N/A')}%")
        
    finally:
        # Stop the collector
        await collector.stop()
        logger.info("Metrics collector stopped")


if __name__ == "__main__":
    asyncio.run(main())