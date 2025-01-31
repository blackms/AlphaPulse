"""
Demo script showing how to use the paper trading system with trained models.
"""
import asyncio
import logging
from datetime import datetime, timedelta, UTC
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.data_pipeline.interfaces import IExchangeFactory, IDataStorage, IExchange
from alpha_pulse.data_pipeline.models import OHLCV
from alpha_pulse.features.feature_engineering import calculate_technical_indicators
from alpha_pulse.execution import PaperBroker, OrderSide, OrderType, Order, RiskLimits
from alpha_pulse.config.settings import TRAINED_MODELS_DIR


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockExchange(IExchange):
    """Mock exchange implementation for testing."""
    
    def __init__(self):
        self.base_prices = {
            "BTC/USD": 50000,
            "ETH/USD": 2000
        }
        # Initialize with random phase for each symbol
        self.time_offsets = {
            "BTC/USD": np.random.uniform(0, 2*np.pi),
            "ETH/USD": np.random.uniform(0, 2*np.pi)
        }
        self.start_time = datetime.now(UTC)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[List]:
        """Generate mock OHLCV data with trends and patterns."""
        current_time = datetime.now(UTC)
        data = []
        
        # Generate some mock price data with trends and cycles
        base_price = self.base_prices.get(symbol, 1000)
        time_offset = self.time_offsets.get(symbol, 0)
        
        for i in range(limit or 100):
            timestamp = int((current_time - timedelta(minutes=i)).timestamp() * 1000)
            
            # Calculate time in hours since start for smooth patterns
            hours_elapsed = (current_time - self.start_time).total_seconds() / 3600
            t = hours_elapsed - i/60  # Convert minute offset to hours
            
            # Generate more volatile price movements
            trend = 0.002 * t  # Stronger upward trend
            cycles = (
                0.05 * np.sin(t + time_offset) +  # Main cycle (stronger)
                0.03 * np.sin(3 * t) +            # Faster cycle (stronger)
                0.02 * np.sin(0.5 * t)            # Slower cycle (stronger)
            )
            
            # Add more randomness
            noise = np.random.normal(0, 0.003)  # More volatility
            
            # Add volatility clustering
            volatility = 0.002 * (1 + 0.5 * np.abs(np.sin(t/2)))
            
            # Combine components with volatility clustering
            price_factor = np.exp(trend + cycles + noise * volatility)
            price = base_price * price_factor
            
            # Create OHLCV candle
            candle = [
                timestamp,
                price * (1 - 0.001),  # Open
                price * (1 + 0.001),  # High
                price * (1 - 0.001),  # Low
                price,                # Close
                np.random.normal(100, 10) * (1 + abs(noise))  # Volume
            ]
            data.insert(0, candle)
        
        return data

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict:
        """Mock historical data fetch."""
        data = self.fetch_ohlcv(symbol, timeframe, limit=100)
        return {
            'timestamp': [d[0] for d in data],
            'open': [d[1] for d in data],
            'high': [d[2] for d in data],
            'low': [d[3] for d in data],
            'close': [d[4] for d in data],
            'volume': [d[5] for d in data]
        }

    def get_current_price(self, symbol: str) -> float:
        """Get current mock price."""
        data = self.fetch_ohlcv(symbol, "1m", limit=1)
        return data[0][4]  # Return close price

    def get_available_pairs(self) -> List[str]:
        """Get list of supported pairs."""
        return list(self.base_prices.keys())


class MockExchangeFactory(IExchangeFactory):
    """Mock exchange factory implementation."""
    
    def create_exchange(self, exchange_id: str) -> IExchange:
        """Create a mock exchange instance."""
        return MockExchange()


class MockStorage(IDataStorage):
    """Mock storage implementation."""
    
    def __init__(self):
        self.latest_data = {}  # Store latest timestamp per symbol

    def save_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        data: Dict
    ) -> None:
        """Mock historical data save."""
        pass

    def get_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict:
        """Mock historical data retrieval."""
        return {}

    def save_ohlcv(self, data: List[OHLCV]) -> None:
        """Mock OHLCV data save."""
        if data:
            key = (data[0].exchange, data[0].symbol)
            self.latest_data[key] = data[-1].timestamp

    def get_latest_ohlcv(
        self,
        exchange_id: str,
        symbol: str
    ) -> Optional[datetime]:
        """Get timestamp of latest OHLCV record."""
        return self.latest_data.get((exchange_id, symbol))


class PaperTradingSystem:
    """Paper trading system that combines data fetching, model predictions, and execution."""

    def __init__(
        self,
        symbols: list[str],
        model_path: Path,
        initial_balance: float = 100000.0,
        update_interval: int = 60,  # seconds
    ):
        self.symbols = symbols
        self.update_interval = update_interval
        
        # Initialize components
        self.data_fetcher = DataFetcher(
            exchange_factory=MockExchangeFactory(),
            storage=MockStorage()
        )
        self.broker = PaperBroker(
            initial_balance=initial_balance,
            risk_limits=RiskLimits(
                max_position_size=initial_balance * 0.2,  # 20% max per position
                max_portfolio_size=initial_balance * 1.5,  # 150% max portfolio value
                max_drawdown_pct=0.25,  # 25% max drawdown
                stop_loss_pct=0.10,  # 10% stop loss per position
            )
        )
        
        # Load trained model and feature names
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        logger.info(f"Loaded model from {model_path}")
        
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
                current_time = datetime.now()
                
                for symbol in self.symbols:
                    # Skip if not enough time has passed since last update
                    if (self.last_updates[symbol] and 
                        (current_time - self.last_updates[symbol]).seconds < self.update_interval):
                        continue
                    
                    # Fetch latest data
                    ohlcv_data = self.data_fetcher.fetch_ohlcv(
                        exchange_id="mock",
                        symbol=symbol,
                        timeframe="1m",
                        limit=100  # Enough data for feature calculation
                    )
                    
                    if not ohlcv_data:
                        logger.warning(f"No data received for {symbol}")
                        continue
                    
                    # Convert OHLCV data to DataFrame
                    df = pd.DataFrame([
                        {
                            'timestamp': candle.timestamp,
                            'open': candle.open,
                            'high': candle.high,
                            'low': candle.low,
                            'close': candle.close,
                            'volume': candle.volume
                        }
                        for candle in ohlcv_data
                    ])
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

    async def _process_symbol(self, symbol: str, df: pd.DataFrame):
        """Process data for a single symbol and execute trades if needed."""
        # Calculate features
        features_df = calculate_technical_indicators(df)
        if features_df.empty:
            return
        
        # Get latest feature values and ensure correct order
        latest_features = features_df.iloc[-1][self.feature_names]
        
        # Get model prediction
        try:
            prediction = self.model.predict_proba([latest_features])[0]
            signal = prediction[1]  # Probability of price increase
            logger.info(f"Signal for {symbol}: {signal:.3f}")
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {e}")
            return
        
        # Get current position
        current_position = self.broker.get_position(symbol)
        current_price = df['close'].iloc[-1]
        
        # Update broker's market data
        self.broker.update_market_data(symbol, current_price)
        
        # Log feature values
        logger.info(f"Features for {symbol}:")
        for fname, fvalue in zip(self.feature_names, latest_features):
            logger.info(f"  {fname}: {fvalue:.4f}")
        
        # Trading logic with adjusted thresholds
        if signal > 0.75 and (current_position is None or current_position.quantity <= 0):
            # Very strong buy signal
            logger.info(f"Strong buy signal detected for {symbol} (signal: {signal:.3f})")
            self._place_trade(symbol, OrderSide.BUY, current_price)
        elif signal < 0.25 and current_position is not None and current_position.quantity > 0:
            # Very strong sell signal
            logger.info(f"Strong sell signal detected for {symbol} (signal: {signal:.3f})")
            self._place_trade(symbol, OrderSide.SELL, current_price)

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
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': side.value,
                    'quantity': quantity,
                    'price': current_price,
                    'value': quantity * current_price
                })
                logger.info(f"Executed {side.value} order for {symbol}: {quantity:.6f} @ {current_price:.2f}")
        except Exception as e:
            logger.error(f"Error placing order: {e}")

    def _log_performance(self):
        """Log current performance metrics."""
        portfolio_value = self.broker.get_portfolio_value()
        self.portfolio_values.append({
            'timestamp': datetime.now(),
            'value': portfolio_value
        })
        
        # Calculate metrics
        pnl = portfolio_value - self.initial_portfolio_value
        pnl_pct = (pnl / self.initial_portfolio_value) * 100
        
        # Log current state
        logger.info(f"Portfolio Value: ${portfolio_value:.2f} (PnL: ${pnl:.2f} / {pnl_pct:.2f}%)")
        
        # Log positions
        positions = self.broker.get_positions()
        for symbol, pos in positions.items():
            logger.info(f"Position - {symbol}: {pos.quantity:.6f} @ {pos.avg_entry_price:.2f} "
                       f"(PnL: ${pos.unrealized_pnl:.2f})")

    def _log_final_results(self):
        """Log final trading session results."""
        final_value = self.broker.get_portfolio_value()
        total_pnl = final_value - self.initial_portfolio_value
        total_pnl_pct = (total_pnl / self.initial_portfolio_value) * 100
        
        logger.info("\n=== Trading Session Results ===")
        logger.info(f"Initial Portfolio Value: ${self.initial_portfolio_value:.2f}")
        logger.info(f"Final Portfolio Value: ${final_value:.2f}")
        logger.info(f"Total PnL: ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")
        logger.info(f"Total Trades: {len(self.trade_history)}")


async def main():
    # Example usage
    symbols = ['BTC/USD', 'ETH/USD']
    model_path = Path(TRAINED_MODELS_DIR) / 'crypto_prediction_model.joblib'
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    # Initialize and run paper trading system
    trading_system = PaperTradingSystem(
        symbols=symbols,
        model_path=model_path,
        initial_balance=100000.0,
        update_interval=60  # Update every minute
    )
    
    await trading_system.run()


if __name__ == "__main__":
    asyncio.run(main())