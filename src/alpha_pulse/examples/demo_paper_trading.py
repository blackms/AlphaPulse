"""
Demo script showing how to use the paper trading system with trained models.
"""
import asyncio
import logging
import argparse
from datetime import datetime, timedelta, UTC
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from functools import wraps

from alpha_pulse.exchanges import ExchangeType, ExchangeFactory
from alpha_pulse.data_pipeline import (
    SQLAlchemyStorage,
    ExchangeFetcher,
    HistoricalDataManager,
    StorageConfig,
    DataFetchConfig
)
from alpha_pulse.features.feature_engineering import calculate_technical_indicators
from alpha_pulse.execution import PaperBroker, OrderSide, OrderType, Order
from alpha_pulse.config.settings import TRAINED_MODELS_DIR, settings
from alpha_pulse.config.exchanges import get_exchange_config
from alpha_pulse.monitoring.metrics import (
    start_metrics_server,
    MetricsTracker,
    track_latency,
    MODEL_LATENCY,
    API_LATENCY
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperTradingSystem:
    """Paper trading system that combines data fetching, model predictions, and execution."""

    def __init__(
        self,
        symbols: list[str],
        model_path: Path,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        use_testnet: bool = False,
        initial_balance: float = 100000.0,
        update_interval: int = 60,  # seconds
    ):
        self.symbols = symbols
        self.update_interval = update_interval
        
        # Start metrics server
        start_metrics_server()
        self.metrics = MetricsTracker()
        
        # Configure exchange
        exchange_config = get_exchange_config(f"{exchange_id}_testnet" if use_testnet else exchange_id)
        if not exchange_config:
            self.metrics.record_error("exchange_config")
            raise ValueError(f"Unsupported exchange: {exchange_id}")
        
        exchange_config.api_key = api_key
        exchange_config.api_secret = api_secret
        settings.exchange = exchange_config
        
        # Initialize components
        storage = SQLAlchemyStorage(config=StorageConfig())
        fetcher = ExchangeFetcher(config=DataFetchConfig())
        self.data_manager = HistoricalDataManager(storage, fetcher)
        
        # Initialize paper broker with initial balance
        self.broker = PaperBroker(initial_balance=initial_balance)
        
        # Load trained model and feature names
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Using exchange: {exchange_config.name} ({'testnet' if use_testnet else 'mainnet'})")
        
        # Track last update time per symbol
        self.last_updates = {symbol: None for symbol in symbols}
        
        # Performance tracking
        self.initial_portfolio_value = initial_balance
        self.portfolio_values = []
        self.trade_history = []

    async def run(self):
        """Main trading loop."""
        logger.info("Starting paper trading system...")
        
        try:
            while True:
                current_time = datetime.now(UTC)
                
                for symbol in self.symbols:
                    # Skip if not enough time has passed since last update
                    if (self.last_updates[symbol] and 
                        (current_time - self.last_updates[symbol]).seconds < self.update_interval):
                        continue
                    
                    # Ensure data is available
                    end_time = current_time
                    start_time = end_time - timedelta(hours=2)  # Get 2 hours of data for features
                    
                    await self.data_manager.ensure_data_available(
                        exchange_type=ExchangeType.BINANCE,
                        symbol=symbol,
                        timeframe="1m",
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    # Get historical data
                    ohlcv_data = self.data_manager.get_historical_data(
                        exchange_type=ExchangeType.BINANCE,
                        symbol=symbol,
                        timeframe="1m",
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if not ohlcv_data:
                        logger.warning(f"No data received for {symbol}")
                        continue
                    
                    # Convert OHLCV data to DataFrame
                    df = pd.DataFrame([{
                        'timestamp': candle.timestamp,
                        'open': candle.open,
                        'high': candle.high,
                        'low': candle.low,
                        'close': candle.close,
                        'volume': candle.volume
                    } for candle in ohlcv_data])
                    df.set_index('timestamp', inplace=True)
                    
                    # Process data and make trading decisions
                    await self._process_symbol(symbol, df)
                    self.last_updates[symbol] = current_time
                
                # Log performance metrics
                self._log_performance()
                
                # Wait before next update
                await asyncio.sleep(1)  # Check every second for new data needs
                
        except KeyboardInterrupt:
            logger.info("Shutting down paper trading system...")
            self._log_final_results()

    @track_latency(MODEL_LATENCY.labels(model_name='crypto_prediction'))
    async def _process_symbol(self, symbol: str, df: pd.DataFrame):
        """Process data for a single symbol and execute trades if needed."""
        # Calculate features
        features_df = calculate_technical_indicators(df)
        if features_df.empty:
            self.metrics.record_error("empty_features")
            return
        
        try:
            # Ensure features are in the same order as training
            features_df = features_df.reindex(columns=self.feature_names)
            
            # Get latest feature values
            latest_features = features_df.iloc[-1].values.reshape(1, -1)
            
            # Get model prediction
            prediction = self.model.predict_proba(latest_features)[0]
            signal = prediction[1]  # Probability of price increase
            logger.info(f"Signal for {symbol}: {signal:.3f}")
            
            # Record prediction
            self.metrics.record_prediction('crypto_prediction', signal)
            
            # Log feature values
            logger.info(f"Features for {symbol}:")
            for fname, fvalue in zip(self.feature_names, latest_features[0]):
                logger.info(f"  {fname}: {fvalue:.4f}")
            
            # Get current position
            current_position = self.broker.get_position(symbol)
            current_price = df['close'].iloc[-1]
            
            # Update broker's market data
            self.broker.update_market_data(symbol, current_price)
            
            # Trading logic with adjusted thresholds
            if signal > 0.75 and (current_position is None or current_position.quantity <= 0):
                # Very strong buy signal
                logger.info(f"Strong buy signal detected for {symbol} (signal: {signal:.3f})")
                self._place_trade(symbol, OrderSide.BUY, current_price)
            elif signal < 0.25 and current_position is not None and current_position.quantity > 0:
                # Very strong sell signal
                logger.info(f"Strong sell signal detected for {symbol} (signal: {signal:.3f})")
                self._place_trade(symbol, OrderSide.SELL, current_price)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return

    def _place_trade(self, symbol: str, side: OrderSide, current_price: float):
        """Place a trade with position sizing."""
        # Calculate position size (2% of account per trade)
        account_value = self.broker.get_portfolio_value()
        position_value = account_value * 0.02
        quantity = position_value / current_price
        
        # Create and place order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
        
        try:
            executed_order = self.broker.place_order(order)
            if executed_order.status.value == "filled":
                trade_value = quantity * current_price
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': side.value,
                    'quantity': quantity,
                    'price': current_price,
                    'value': trade_value
                })
                # Record trade metrics
                self.metrics.record_trade(symbol, side.value)
                self.metrics.update_position(symbol, quantity if side == OrderSide.BUY else -quantity)
                logger.info(f"Executed {side.value} order for {symbol}: {quantity:.6f} @ {current_price:.2f}")
        except Exception as e:
            self.metrics.record_error("order_execution")
            logger.error(f"Error placing order: {e}")

    def _log_performance(self):
        """Log current performance metrics."""
        portfolio_value = float(self.broker.get_portfolio_value())  # Convert Decimal to float
        self.portfolio_values.append({
            'timestamp': datetime.now(),
            'value': portfolio_value
        })
        
        # Calculate metrics
        pnl = portfolio_value - self.initial_portfolio_value
        pnl_pct = (pnl / self.initial_portfolio_value) * 100
        
        # Update metrics
        for symbol in self.symbols:
            self.metrics.update_pnl(symbol, pnl)
            # Calculate win rate
            if len(self.trade_history) > 0:
                profitable_trades = sum(1 for trade in self.trade_history
                                     if trade['symbol'] == symbol and
                                     (trade['side'] == 'BUY' and pnl > 0 or
                                      trade['side'] == 'SELL' and pnl < 0))
                win_rate = profitable_trades / len(self.trade_history) * 100
                self.metrics.update_win_rate(symbol, "1m", win_rate)
        
        # Log current state
        logger.info(f"Portfolio Value: ${portfolio_value:.2f} (PnL: ${pnl:.2f} / {pnl_pct:.2f}%)")
        
        # Log positions
        positions = self.broker.get_positions()
        for symbol, pos in positions.items():
            self.metrics.update_position(symbol, pos.quantity)
            logger.info(f"Position - {symbol}: {pos.quantity:.6f} @ {pos.avg_entry_price:.2f} "
                       f"(PnL: ${pos.unrealized_pnl:.2f})")

    def _log_final_results(self):
        """Log final trading session results."""
        final_value = float(self.broker.get_portfolio_value())  # Convert Decimal to float
        total_pnl = final_value - self.initial_portfolio_value
        total_pnl_pct = (total_pnl / self.initial_portfolio_value) * 100
        
        logger.info("\n=== Trading Session Results ===")
        logger.info(f"Initial Portfolio Value: ${self.initial_portfolio_value:.2f}")
        logger.info(f"Final Portfolio Value: ${final_value:.2f}")
        logger.info(f"Total PnL: ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")
        logger.info(f"Total Trades: {len(self.trade_history)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Paper trading system")
    parser.add_argument("--exchange", default="binance", help="Exchange ID (default: binance)")
    parser.add_argument("--testnet", action="store_true", help="Use exchange testnet")
    parser.add_argument("--api-key", default="", help="Exchange API key")
    parser.add_argument("--api-secret", default="", help="Exchange API secret")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"], help="Trading symbols")
    parser.add_argument("--balance", type=float, default=100000.0, help="Initial balance")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds")
    return parser.parse_args()


async def main():
    # Parse command line arguments
    args = parse_args()
    
    # Check for model
    model_path = Path(TRAINED_MODELS_DIR) / 'crypto_prediction_model.joblib'
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    # Initialize and run paper trading system
    trading_system = PaperTradingSystem(
        symbols=args.symbols,
        model_path=model_path,
        exchange_id=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret,
        use_testnet=args.testnet,
        initial_balance=args.balance,
        update_interval=args.interval
    )
    
    await trading_system.run()


if __name__ == "__main__":
    asyncio.run(main())