#!/usr/bin/env python
"""
Demo script for AlphaPulse database infrastructure.

This script demonstrates how to use the database infrastructure
to perform common operations.
"""
import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.alpha_pulse.data_pipeline.database import (
    connection_manager,
    UserRepository,
    PortfolioRepository,
    PositionRepository,
    TradeRepository,
    AlertRepository,
    MetricRepository,
    Metric
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demo_database')


async def demo_user_operations():
    """Demonstrate user operations."""
    logger.info("Demonstrating user operations...")
    
    user_repo = UserRepository()
    
    # Find user by username
    admin_user = await user_repo.find_by_username('admin')
    if admin_user:
        logger.info(f"Found admin user: {admin_user['username']}")
    else:
        logger.warning("Admin user not found")
    
    # Create a new user
    new_user = await user_repo.create({
        'username': 'demo_user',
        'password_hash': '$2a$12$1InE4/AkbV4/Ye7nKrLPOOYILsLPjpqiLjHXgH6iBp2o7QEzW.ZpG',  # 'admin123'
        'email': 'demo@alphapulse.com',
        'role': 'user'
    })
    logger.info(f"Created new user: {new_user['username']}")
    
    # Update user
    updated_user = await user_repo.update(new_user['id'], {
        'role': 'operator'
    })
    logger.info(f"Updated user role to: {updated_user['role']}")
    
    # Find all users
    all_users = await user_repo.find_all()
    logger.info(f"Found {len(all_users)} users")
    
    # Delete user
    deleted = await user_repo.delete(new_user['id'])
    logger.info(f"Deleted user: {deleted}")


async def demo_portfolio_operations():
    """Demonstrate portfolio operations."""
    logger.info("Demonstrating portfolio operations...")
    
    portfolio_repo = PortfolioRepository()
    position_repo = PositionRepository()
    
    # Find all portfolios
    portfolios = await portfolio_repo.find_all()
    if not portfolios:
        logger.warning("No portfolios found")
        return
    
    logger.info(f"Found {len(portfolios)} portfolios")
    
    # Get the first portfolio
    portfolio = portfolios[0]
    logger.info(f"Using portfolio: {portfolio['name']}")
    
    # Get portfolio with positions
    portfolio_with_positions = await portfolio_repo.find_with_positions(portfolio['id'])
    logger.info(f"Portfolio has {len(portfolio_with_positions['positions'])} positions")
    
    # Create a new position
    new_position = await position_repo.create({
        'portfolio_id': portfolio['id'],
        'symbol': 'DEMO-USD',
        'quantity': 100.0,
        'entry_price': 10.0,
        'current_price': 12.0
    })
    logger.info(f"Created new position: {new_position['symbol']}")
    
    # Update position price
    updated_position = await position_repo.update_current_price(new_position['id'], 15.0)
    logger.info(f"Updated position price to: {updated_position['current_price']}")
    
    # Delete position
    deleted = await position_repo.delete(new_position['id'])
    logger.info(f"Deleted position: {deleted}")


async def demo_trade_operations():
    """Demonstrate trade operations."""
    logger.info("Demonstrating trade operations...")
    
    trade_repo = TradeRepository()
    portfolio_repo = PortfolioRepository()
    
    # Find all portfolios
    portfolios = await portfolio_repo.find_all()
    if not portfolios:
        logger.warning("No portfolios found")
        return
    
    # Get the first portfolio
    portfolio = portfolios[0]
    
    # Create a new trade
    new_trade = await trade_repo.create({
        'portfolio_id': portfolio['id'],
        'symbol': 'DEMO-USD',
        'side': 'buy',
        'quantity': 100.0,
        'price': 10.0,
        'fees': 1.0,
        'order_type': 'market',
        'status': 'filled',
        'executed_at': datetime.now()
    })
    logger.info(f"Created new trade: {new_trade['symbol']} {new_trade['side']}")
    
    # Find trades by portfolio
    portfolio_trades = await trade_repo.find_by_portfolio(portfolio['id'])
    logger.info(f"Found {len(portfolio_trades)} trades for portfolio")
    
    # Find trades by symbol
    symbol_trades = await trade_repo.find_by_symbol('DEMO-USD')
    logger.info(f"Found {len(symbol_trades)} trades for symbol DEMO-USD")
    
    # Find trades by time range
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now()
    time_trades = await trade_repo.find_by_time_range(start_time, end_time)
    logger.info(f"Found {len(time_trades)} trades in the last 24 hours")


async def demo_alert_operations():
    """Demonstrate alert operations."""
    logger.info("Demonstrating alert operations...")
    
    alert_repo = AlertRepository()
    
    # Create a new alert
    new_alert = await alert_repo.create({
        'title': 'Demo Alert',
        'message': 'This is a demo alert for testing purposes',
        'severity': 'info',
        'source': 'demo_script',
        'tags': ['demo', 'test']
    })
    logger.info(f"Created new alert: {new_alert['title']}")
    
    # Find unacknowledged alerts
    unack_alerts = await alert_repo.find_unacknowledged()
    logger.info(f"Found {len(unack_alerts)} unacknowledged alerts")
    
    # Acknowledge alert
    ack_alert = await alert_repo.acknowledge(new_alert['id'], 'demo_user')
    if ack_alert:
        logger.info(f"Acknowledged alert: {ack_alert['title']}")
    else:
        logger.warning("Failed to acknowledge alert")


async def demo_metric_operations():
    """Demonstrate metric operations."""
    logger.info("Demonstrating metric operations...")
    
    metric_repo = MetricRepository()
    
    # Create a new metric
    metric = Metric(
        metric_name='demo_metric',
        value=42.0,
        labels={'source': 'demo_script', 'type': 'test'},
        timestamp=datetime.now()
    )
    await metric_repo.insert(metric)
    logger.info(f"Inserted new metric: {metric.metric_name}")
    
    # Create batch of metrics
    metrics = []
    for i in range(10):
        metrics.append(Metric(
            metric_name='demo_batch_metric',
            value=i * 10.0,
            labels={'source': 'demo_script', 'type': 'batch', 'index': str(i)},
            timestamp=datetime.now() - timedelta(minutes=i)
        ))
    await metric_repo.insert_batch(metrics)
    logger.info(f"Inserted {len(metrics)} metrics in batch")
    
    # Find metrics by name
    start_time = datetime.now() - timedelta(hours=1)
    end_time = datetime.now()
    found_metrics = await metric_repo.find_by_name('demo_batch_metric', start_time, end_time)
    logger.info(f"Found {len(found_metrics)} metrics for name demo_batch_metric")
    
    # Find latest metric
    latest = await metric_repo.find_latest_by_name('demo_metric')
    if latest:
        logger.info(f"Latest demo_metric value: {latest['value']}")
    else:
        logger.warning("No demo_metric found")
    
    # Aggregate metrics
    aggregated = await metric_repo.aggregate_by_time(
        'demo_batch_metric',
        start_time,
        end_time,
        '5 minutes',
        'avg'
    )
    logger.info(f"Aggregated {len(aggregated)} time buckets for demo_batch_metric")


async def main():
    """Main function to run the demo."""
    logger.info("Starting database demo...")
    
    try:
        # Initialize connection manager
        await connection_manager.initialize()
        
        # Run demos
        await demo_user_operations()
        await demo_portfolio_operations()
        await demo_trade_operations()
        await demo_alert_operations()
        await demo_metric_operations()
        
        # Close connections
        await connection_manager.close()
        
        logger.info("Database demo completed successfully!")
    except Exception as e:
        logger.error(f"Error in database demo: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)