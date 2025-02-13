"""
Example script demonstrating the supervised multi-agent system with real data.
"""
import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yaml
import pandas as pd
from loguru import logger
import sys
import warnings
import numpy as np

# Suppress specific numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.lib._polynomial_impl')

from alpha_pulse.data_pipeline.manager import DataManager
from alpha_pulse.agents.supervisor import (
    SupervisorAgent,
    AgentFactory,
    AgentState
)
from alpha_pulse.agents.interfaces import MarketData


# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "<level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/supervised_agents_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
           "{name}:{function}:{line} | "
           "{message}",
    level="DEBUG",
    rotation="500 MB"
)


async def create_market_data(data_manager: DataManager) -> MarketData:
    """Create MarketData object from real data."""
    # Test symbols
    crypto_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
    
    # Time range (180 days for better analysis)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=180)
    
    # Fetch market data
    market_data_dict = await data_manager.get_market_data(
        symbols=crypto_symbols,
        start_time=start_time,
        end_time=end_time,
        interval="1d"
    )
    
    # Convert to DataFrame format expected by agents
    prices_data = {}
    volumes_data = {}
    
    for symbol, data in market_data_dict.items():
        dates = [entry.timestamp for entry in data]
        prices_data[symbol] = pd.Series(
            [entry.close for entry in data],
            index=dates
        )
        volumes_data[symbol] = pd.Series(
            [entry.volume for entry in data],
            index=dates
        )
        
    prices_df = pd.DataFrame(prices_data)
    volumes_df = pd.DataFrame(volumes_data)
    
    # Calculate technical indicators
    technical_data = data_manager.get_technical_indicators(market_data_dict)
    
    # Create MarketData object
    return MarketData(
        prices=prices_df,
        volumes=volumes_df,
        technical_indicators=technical_data
    )


async def main():
    """Main execution function."""
    manager = None
    supervisor = None
    
    try:
        # Load environment variables and config
        load_dotenv()
        with open("config/data_pipeline_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        # Initialize data manager
        manager = DataManager(config=config)
        await manager.initialize()
        
        # Get market data
        logger.info("Fetching market data...")
        market_data = await create_market_data(manager)
        logger.info(f"Fetched data for {len(market_data.prices.columns)} symbols")
        
        # Initialize supervisor
        supervisor = SupervisorAgent.instance()
        await supervisor.start()
        logger.info("Supervisor agent started")
        
        # Create agents with different configurations
        agents = []
        
        # Technical agent focusing on trend following
        tech_config = {
            "type": "technical",
            "optimization_threshold": 0.7,
            "timeframes": {
                "short": 20,
                "medium": 50,
                "long": 200
            }
        }
        tech_agent = await supervisor.register_agent("tech_trend", tech_config)
        agents.append(tech_agent)
        
        # Technical agent focusing on mean reversion
        tech_config_2 = {
            "type": "technical",
            "optimization_threshold": 0.6,
            "timeframes": {
                "short": 10,
                "medium": 30,
                "long": 90
            }
        }
        tech_agent_2 = await supervisor.register_agent("tech_reversion", tech_config_2)
        agents.append(tech_agent_2)
        
        # Sentiment agent
        sentiment_config = {
            "type": "sentiment",
            "optimization_threshold": 0.65,
            "sentiment_sources": {
                "news": 0.4,
                "social_media": 0.3,
                "market_data": 0.3
            }
        }
        sentiment_agent = await supervisor.register_agent("sentiment_1", sentiment_config)
        agents.append(sentiment_agent)
        
        # Run simulation
        logger.info("Starting simulation...")
        
        for i in range(5):  # Simulate 5 trading periods
            logger.info(f"\nTrading Period {i+1}")
            
            # Generate signals from all agents
            all_signals = []
            for agent in agents:
                signals = await agent.generate_signals(market_data)
                all_signals.extend(signals)
                logger.info(f"Agent {agent.agent_id} generated {len(signals)} signals")
                
            # Get system status
            system_status = await supervisor.get_system_status()
            logger.info("\nSystem Status:")
            logger.info(f"Active Agents: {system_status['active_agents']}")
            logger.info(f"Total Agents: {system_status['total_agents']}")
            
            # Get performance analytics
            for agent in agents:
                status = await supervisor.get_agent_status(agent.agent_id)
                logger.info(f"\nAgent {agent.agent_id} Status:")
                logger.info(f"State: {status.state}")
                logger.info(f"Error Count: {status.error_count}")
                logger.info(f"CPU Usage: {status.cpu_usage:.2f}%")
                logger.info(f"Memory Usage: {status.memory_usage:.2f} MB")
                logger.info(f"Performance Metrics: {status.metrics}")
                
            # Wait before next iteration
            await asyncio.sleep(5)
            
        logger.info("\nSimulation completed")
        
    except Exception as e:
        logger.exception(f"Error in main execution: {str(e)}")
        raise
    finally:
        # Cleanup
        if supervisor:
            await supervisor.stop()
        if manager:
            await manager.__aexit__(None, None, None)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.exception(f"Process terminated with error: {str(e)}")