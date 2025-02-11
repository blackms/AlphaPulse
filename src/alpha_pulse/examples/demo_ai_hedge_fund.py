"""
Example script demonstrating the AI Hedge Fund system with paper trading.
"""
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any
import yaml
import logging
from pathlib import Path

from alpha_pulse.agents.manager import AgentManager
from alpha_pulse.agents.interfaces import MarketData
from alpha_pulse.risk_management.manager import RiskManager, RiskConfig
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.data_pipeline.managers.mock_data import MockDataManager
from alpha_pulse.execution.paper_actuator import PaperActuator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def load_market_data(symbols: list, lookback_days: int = 365) -> MarketData:
    """Load market data for demonstration."""
    data_manager = MockDataManager()  # Using mock data for demo
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Fetch historical data
    prices = await data_manager.get_historical_prices(symbols, start_date, end_date)
    volumes = await data_manager.get_historical_volumes(symbols, start_date, end_date)
    
    # Fetch fundamental data
    fundamentals = {}
    for symbol in symbols:
        fundamental_data = await data_manager.get_fundamental_data(symbol)
        if fundamental_data:
            fundamentals[symbol] = fundamental_data
            
    # Fetch sentiment data
    sentiment = {}
    for symbol in symbols:
        sentiment_data = await data_manager.get_sentiment_data(symbol)
        if sentiment_data:
            sentiment[symbol] = sentiment_data
            
    return MarketData(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        sentiment=sentiment,
        timestamp=datetime.now()
    )


def ensure_config_exists():
    """Ensure the configuration file exists."""
    config_path = Path("config/ai_hedge_fund_config.yaml")
    if not config_path.exists():
        config = {
            "agents": {
                "agent_weights": {
                    "activist": 0.15,
                    "value": 0.20,
                    "fundamental": 0.20,
                    "sentiment": 0.15,
                    "technical": 0.15,
                    "valuation": 0.15
                }
            },
            "risk": {
                "max_position_size": 0.20,
                "max_portfolio_leverage": 1.5,
                "max_drawdown": 0.25,
                "stop_loss": 0.10,
                "var_confidence": 0.95,
                "risk_free_rate": 0.00,
                "target_volatility": 0.15,
                "rebalance_threshold": 0.10
            },
            "execution": {
                "mode": "paper",
                "initial_balance": 1000000,
                "slippage": 0.001,
                "fee_rate": 0.001
            }
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


async def run_paper_trading_demo(config: Dict[str, Any], symbols: list):
    """Run paper trading demonstration."""
    logger.info("Initializing paper trading demo...")
    
    # Initialize paper actuator
    paper_actuator = PaperActuator(config.get("execution", {}))
    
    # Initialize components
    agent_manager = AgentManager(config.get("agents"))
    await agent_manager.initialize()
    
    risk_manager = RiskManager(
        exchange=paper_actuator,  # Use paper actuator as exchange
        config=RiskConfig(**config.get("risk", {}))
    )
    
    portfolio_manager = PortfolioManager(
        config_path="config/portfolio_config.yaml"
    )
    
    logger.info(f"Analyzing {len(symbols)} symbols...")
    
    try:
        # Load market data
        market_data = await load_market_data(symbols)
        
        logger.info("Generating trading signals...")
        # Generate signals from all agents
        signals = await agent_manager.generate_signals(market_data)
        
        logger.info(f"Received {len(signals)} signals from agents:")
        for signal in signals:
            logger.info(f"\nSymbol: {signal.symbol}")
            logger.info(f"Direction: {signal.direction}")
            logger.info(f"Confidence: {signal.confidence:.2f}")
            if signal.target_price:
                logger.info(f"Target Price: {signal.target_price:.2f}")
            logger.info(f"Contributing Agents: {len(signal.metadata.get('agent_signals', []))}")
            
        logger.info("\nValidating signals with risk management...")
        # Get current portfolio value
        portfolio_value = await paper_actuator.get_portfolio_value()
        current_positions = await paper_actuator.get_positions()
        
        # Process signals
        executed_trades = []
        for signal in signals:
            # Calculate position size (non-async)
            position_size = risk_manager.calculate_position_size(
                symbol=signal.symbol,
                current_price=market_data.prices[signal.symbol].iloc[-1],
                signal_strength=signal.confidence
            )
            
            # Calculate trade size in units
            current_price = market_data.prices[signal.symbol].iloc[-1]
            trade_size = position_size.size / current_price if current_price > 0 else 0
            
            # Validate trade with risk management
            if risk_manager.evaluate_trade(
                symbol=signal.symbol,
                side=signal.direction.value,
                quantity=trade_size,
                current_price=current_price,
                portfolio_value=portfolio_value,
                current_positions=current_positions
            ):
                # Execute paper trade
                trade_result = await paper_actuator.execute_trade(
                    symbol=signal.symbol,
                    side=signal.direction.value,
                    quantity=trade_size,
                    price=current_price
                )
                executed_trades.append(trade_result)
                
        logger.info(f"\nExecuted {len(executed_trades)} paper trades:")
        for trade in executed_trades:
            logger.info(f"\nSymbol: {trade['symbol']}")
            logger.info(f"Side: {trade['side']}")
            logger.info(f"Quantity: {trade['quantity']:.2f}")
            logger.info(f"Price: ${trade['price']:.2f}")
            logger.info(f"Value: ${trade['value']:.2f}")
            
        # Get updated portfolio status
        final_portfolio_value = await paper_actuator.get_portfolio_value()
        final_positions = await paper_actuator.get_positions()
        
        logger.info("\nFinal Portfolio Status:")
        logger.info(f"Portfolio Value: ${final_portfolio_value:,.2f}")
        logger.info("\nCurrent Positions:")
        for symbol, position in final_positions.items():
            logger.info(f"\n{symbol}:")
            logger.info(f"Quantity: {position['quantity']:.2f}")
            logger.info(f"Average Entry: ${position['avg_entry']:.2f}")
            logger.info(f"Current Value: ${position['current_value']:.2f}")
            logger.info(f"Unrealized P&L: ${position['unrealized_pnl']:.2f} ({position['unrealized_pnl_pct']:.2%})")
            
        # Get agent performance metrics
        logger.info("\nAgent Performance Metrics:")
        performance = agent_manager.get_agent_performance()
        for agent_type, metrics in performance.items():
            logger.info(f"\n{agent_type.capitalize()} Agent:")
            if metrics:
                logger.info(f"Signal Accuracy: {metrics.signal_accuracy:.2%}")
                logger.info(f"Profit Factor: {metrics.profit_factor:.2f}")
                logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
                logger.info(f"Max Drawdown: {metrics.max_drawdown:.2%}")
            
        # Get risk report
        logger.info("\nRisk Management Report:")
        risk_report = risk_manager.get_risk_report()
        logger.info(f"Portfolio Value: ${risk_report.get('portfolio_value', 0):,.2f}")
        logger.info(f"Current Leverage: {risk_report.get('current_leverage', 0):.2f}x")
        
        if 'risk_metrics' in risk_report:
            rm = risk_report['risk_metrics']
            logger.info(f"Portfolio Volatility: {rm.get('volatility', 0):.2%}")
            logger.info(f"Value at Risk (95%): ${rm.get('var_95', 0):,.2f}")
            logger.info(f"Expected Shortfall: ${rm.get('cvar_95', 0):,.2f}")
            logger.info(f"Sharpe Ratio: {rm.get('sharpe_ratio', 0):.2f}")
            
    except Exception as e:
        logger.error(f"Error during paper trading demo: {str(e)}")
        raise
        
    logger.info("\nPaper trading demonstration completed.")


async def main():
    """Main entry point."""
    # Ensure configuration exists
    ensure_config_exists()
    
    # Load configuration
    with open("config/ai_hedge_fund_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Define universe of symbols
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "JPM", "V", "JNJ"
    ]
    
    # Run paper trading demo
    await run_paper_trading_demo(config, symbols)


if __name__ == "__main__":
    asyncio.run(main())