#!/usr/bin/env python
"""
Exchange Synchronization Example

This example demonstrates how to use the new exchange synchronization module
for both one-time synchronization and as a scheduled service.
"""
import asyncio
import logging
import sys
import os
import time
from datetime import datetime

# Ensure the AlphaPulse package is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.alpha_pulse.exchange_sync.config import configure_logging
from src.alpha_pulse.exchange_sync.portfolio_service import PortfolioService
from src.alpha_pulse.exchange_sync.scheduler import ExchangeSyncScheduler
from src.alpha_pulse.exchange_sync.repository import PortfolioRepository


async def initialize_database():
    """Initialize database tables for the example."""
    print("Initializing database tables...")
    repo = PortfolioRepository()
    await repo.initialize_tables()
    print("Database tables initialized.")


async def run_one_time_sync():
    """Run a one-time synchronization for all configured exchanges."""
    print("\n=== Running One-Time Synchronization ===")
    
    # Get exchange configuration from environment variables
    # For example, set EXCHANGE_SYNC_EXCHANGES=bybit,binance
    # and set BYBIT_API_KEY and BYBIT_API_SECRET environment variables
    
    # Initialize the database
    await initialize_database()
    
    # Start time for measuring duration
    start_time = datetime.now()
    
    # Run synchronization for all configured exchanges
    results = await ExchangeSyncScheduler.run_once()
    
    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()
    
    # Print results
    print(f"\nSynchronization completed in {duration:.2f} seconds")
    print(f"Results:")
    
    for exchange_id, result in results.items():
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  {exchange_id}: {status}")
        print(f"    Processed: {result.items_processed}")
        print(f"    Synced: {result.items_synced}")
        if not result.success:
            print(f"    Errors: {', '.join(result.errors)}")
    
    # Fetch and display the portfolio data
    for exchange_id in results.keys():
        service = PortfolioService(exchange_id)
        portfolio = await service.get_portfolio()
        
        print(f"\nPortfolio for {exchange_id}:")
        if not portfolio:
            print("  No portfolio items found")
            continue
            
        for item in portfolio:
            value_str = f"${item.value:.2f}" if item.value is not None else "Unknown"
            print(f"  {item.asset}: {item.quantity} (Value: {value_str})")
            if item.avg_entry_price is not None and item.current_price is not None:
                pnl_pct = item.profit_loss_percentage
                pnl_str = f"{pnl_pct:.2f}%" if pnl_pct is not None else "Unknown"
                print(f"    Entry: ${item.avg_entry_price:.2f}, Current: ${item.current_price:.2f}, P/L: {pnl_str}")


async def run_scheduler_example():
    """Run the scheduler for a short period as a demonstration."""
    print("\n=== Running Scheduler Example ===")
    print("This will run the scheduler for 2 minutes, syncing at 1-minute intervals")
    print("Press Ctrl+C to stop earlier")
    
    # Initialize the database
    await initialize_database()
    
    # Create a scheduler with a short interval for demonstration
    scheduler = ExchangeSyncScheduler(interval_minutes=1)
    
    # Set a task to stop the scheduler after 2 minutes
    stop_after = 120  # seconds
    
    try:
        # Start the scheduler
        scheduler_task = asyncio.create_task(scheduler.start())
        
        # Wait for the specified time
        for i in range(stop_after):
            await asyncio.sleep(1)
            if i % 10 == 0:
                print(f"Running for {i} seconds...")
        
        # Stop the scheduler
        print("Stopping scheduler...")
        await scheduler.stop()
        
        # Wait for the scheduler to stop
        await scheduler_task
        
    except asyncio.CancelledError:
        print("Scheduler cancelled")
    except KeyboardInterrupt:
        print("User interrupted")
    finally:
        if scheduler.running:
            await scheduler.stop()
    
    print("Scheduler example completed")


async def main():
    """Main entry point for the example."""
    # Configure logging
    configure_logging(log_level="INFO")
    
    print("Exchange Synchronization Example")
    print("--------------------------------")
    print("This example demonstrates using the exchange synchronization module")
    print("Make sure your environment variables are properly set:")
    print("  EXCHANGE_SYNC_EXCHANGES=bybit,binance")
    print("  BYBIT_API_KEY=your_api_key")
    print("  BYBIT_API_SECRET=your_api_secret")
    print("  DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME for database\n")
    
    while True:
        print("\nChoose an example to run:")
        print("1. One-time synchronization")
        print("2. Scheduler example (runs for 2 minutes)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            await run_one_time_sync()
        elif choice == '2':
            await run_scheduler_example()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")