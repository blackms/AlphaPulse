"""
Replay mode for supervised multi-agent system with historical data analysis.
"""
import asyncio
import json
import yaml
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger
import sys

# Configure loguru logging
logger.remove()  # Remove default handler

# Add console handler with original format but filtered messages
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    filter=lambda record: (
        "detailed" not in record["extra"] and
        not str(record["message"]).startswith("Agent <") and  # Filter object representations
        "object at 0x" not in str(record["message"])  # Filter memory addresses
    )
)

# Add detailed file handler
log_path = Path("logs")
log_path.mkdir(exist_ok=True)
logger.add(
    log_path / "supervised_agents_{time}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    rotation="1 day",
    retention="30 days",
    compression="zip"
)

from alpha_pulse.data_pipeline.manager import DataManager
from alpha_pulse.agents.supervisor.supervisor import SupervisorAgent
from alpha_pulse.agents.supervisor.factory import create_self_supervised_agent
from alpha_pulse.agents.supervisor.distributed.coordinator import ClusterCoordinator
from alpha_pulse.agents.interfaces import MarketData, TradeSignal
from alpha_pulse.exchanges.factories import ExchangeRegistry


class SignalAnalyzer:
    """Analyzes and stores trading signals with detailed metrics."""
    
    def __init__(self, output_dir: str = "reports/signals"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.signals: List[Dict[str, Any]] = []
        self.metrics: Dict[str, List[float]] = {
            "regime_confidence": [],
            "signal_confidence": [],
            "performance_score": []
        }
        
    def add_signal(self, signal: TradeSignal, market_data: MarketData):
        """Add a signal with enriched market context."""
        signal_data = {
            "timestamp": signal.timestamp.isoformat(),
            "agent_id": signal.agent_id,
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "confidence": signal.confidence,
            "target_price": signal.target_price,
            "stop_loss": signal.stop_loss,
            "metadata": signal.metadata,
            "market_context": {
                "price": float(market_data.prices[signal.symbol].iloc[-1]),
                "volume": float(market_data.volumes[signal.symbol].iloc[-1]) if market_data.volumes is not None else None,
                "regime": signal.metadata.get("market_regime", "unknown"),
                "regime_confidence": signal.metadata.get("regime_confidence", 0.0)
            }
        }
        self.signals.append(signal_data)
        
        # Update metrics
        self.metrics["regime_confidence"].append(signal.metadata.get("regime_confidence", 0.0))
        self.metrics["signal_confidence"].append(signal.confidence)
        self.metrics["performance_score"].append(signal.metadata.get("performance_score", 1.0))
        
        # Log detailed signal info to file only
        logger.bind(detailed=True).debug(f"New signal: {json.dumps(signal_data, indent=2)}")
        
    def save_results(self):
        """Save signals and analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed signals
        signals_file = self.output_dir / f"signals_{timestamp}.json"
        with open(signals_file, "w") as f:
            json.dump({"signals": self.signals}, f, indent=2)
            
        # Calculate and save metrics
        metrics = {
            "total_signals": len(self.signals),
            "signals_by_agent": {},
            "signals_by_regime": {},
            "average_metrics": {
                name: sum(values) / len(values) if values else 0
                for name, values in self.metrics.items()
            },
            "confidence_distribution": {
                "low": len([s for s in self.signals if s["confidence"] < 0.4]),
                "medium": len([s for s in self.signals if 0.4 <= s["confidence"] < 0.7]),
                "high": len([s for s in self.signals if s["confidence"] >= 0.7])
            }
        }
        
        # Count signals by agent
        for signal in self.signals:
            agent_id = signal["agent_id"]
            regime = signal["metadata"].get("market_regime", "unknown")
            metrics["signals_by_agent"][agent_id] = metrics["signals_by_agent"].get(agent_id, 0) + 1
            metrics["signals_by_regime"][regime] = metrics["signals_by_regime"].get(regime, 0) + 1
            
        # Save metrics
        metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
        # Log summary to console
        logger.info(f"Analysis complete - {len(self.signals)} signals generated")
        
        # Log detailed metrics to file
        logger.bind(detailed=True).debug(f"Final metrics: {json.dumps(metrics, indent=2)}")
        
        return metrics


async def run_replay(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    interval: str = "1h"
) -> Dict[str, Any]:
    """
    Run the multi-agent system in replay mode with historical data.
    
    Args:
        symbols: List of trading symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        interval: Data interval (e.g., "1h", "4h", "1d")
        
    Returns:
        Analysis metrics from the replay run
    """
    # Load data pipeline configuration
    config_path = Path("config/data_pipeline_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path) as f:
        import yaml
        config = yaml.safe_load(f)
        
    # Initialize components
    data_manager = DataManager(config)
    await data_manager.initialize()
    
    # Download historical data
    logger.info("Fetching market data...")
    raw_data = await data_manager.get_market_data(
        symbols=symbols,
        start_time=start_date,
        end_time=end_date,
        interval=interval
    )
    
    # Convert to MarketData format
    prices = pd.DataFrame()
    volumes = pd.DataFrame()
    
    for symbol, data_list in raw_data.items():
        if not data_list:
            continue
            
        # Extract timestamps, prices and volumes
        symbol_data = pd.DataFrame([
            {
                'timestamp': d.timestamp,
                'close': float(d.close),
                'volume': float(d.volume)
            }
            for d in data_list
        ])
        symbol_data.set_index('timestamp', inplace=True)
        
        # Add to main DataFrames
        prices[symbol] = symbol_data['close']
        volumes[symbol] = symbol_data['volume']
        
    market_data = MarketData(
        prices=prices,
        volumes=volumes,
        sentiment={}  # No sentiment data in replay mode
    )
    
    logger.bind(detailed=True).debug(f"Market data loaded: {len(prices)} timestamps per symbol")
    
    # Initialize agents
    coordinator = ClusterCoordinator()
    supervisor = SupervisorAgent()
    await supervisor.start()
    
    # Create agent configurations
    agent_configs = {
        "tech_trend": {
            "type": "technical",
            "timeframes": {
                "short": 14,
                "medium": 50,
                "long": 180
            }
        },
        "tech_reversion": {
            "type": "technical",
            "timeframes": {
                "short": 14,
                "medium": 50,
                "long": 180
            }
        },
        "sentiment_1": {
            "type": "sentiment",
            "sentiment_sources": {
                "news_weight": 0.3,
                "social_media_weight": 0.25,
                "market_data_weight": 0.25,
                "analyst_weight": 0.2
            }
        }
    }
    
    # Create and register agents
    agents = [
        await create_self_supervised_agent("technical", "tech_trend", agent_configs["tech_trend"]),
        await create_self_supervised_agent("technical", "tech_reversion", agent_configs["tech_reversion"]),
        await create_self_supervised_agent("sentiment", "sentiment_1", agent_configs["sentiment_1"])
    ]
    
    for agent in agents:
        await agent.initialize(agent_configs[agent.agent_id])
        await supervisor.register_agent(agent, agent_configs[agent.agent_id])
        logger.bind(detailed=True).debug(f"Initialized agent: {agent.agent_id}")
        
    # Initialize signal analyzer
    analyzer = SignalAnalyzer()
    
    # Run simulation
    logger.info("Starting replay simulation...")
    timestamps = market_data.prices.index
    for i in range(len(timestamps)):
        current_data = MarketData(
            prices=market_data.prices.iloc[:i+1],
            volumes=market_data.volumes.iloc[:i+1] if market_data.volumes is not None else None,
            sentiment=market_data.sentiment
        )
        
        # Generate signals from each agent
        for agent in agents:
            signals = await agent.generate_signals(current_data)
            for signal in signals:
                analyzer.add_signal(signal, current_data)
                
        # Log progress less frequently (every 1000 timestamps)
        if i % 1000 == 0:
            progress = (i/len(timestamps)*100)
            logger.info(f"Simulation progress: {progress:.1f}%")
            
    # Stop agents
    for agent in agents:
        await agent.stop()
    await supervisor.stop()
    
    # Save and return results
    return analyzer.save_results()


if __name__ == "__main__":
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Last year
    
    logger.info(f"Starting replay analysis for {len(symbols)} symbols")
    logger.bind(detailed=True).debug(f"Symbols: {symbols}")
    logger.bind(detailed=True).debug(f"Period: {start_date} to {end_date}")
    
    # Run replay
    metrics = asyncio.run(run_replay(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval="1h"
    ))
    
    # Print concise summary
    print("\n=== Replay Analysis ===")
    print(f"Total Signals: {metrics['total_signals']}")
    
    # Only show key metrics
    print("\nPerformance:")
    for name, value in metrics['average_metrics'].items():
        if name in ['performance_score', 'signal_confidence']:
            print(f"  {name}: {value:.3f}")
            
    print("\nDistribution:")
    for level, count in metrics['confidence_distribution'].items():
        pct = (count / metrics['total_signals']) * 100
        print(f"  {level}: {count} ({pct:.1f}%)")