"""
Demo script showing multi-asset trading with risk management.
"""
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.data_pipeline.exchange import CCXTExchangeFactory
from alpha_pulse.data_pipeline.storage import SQLAlchemyStorage
from alpha_pulse.features.feature_engineering import calculate_technical_indicators
from alpha_pulse.risk_management import (
    RiskManager,
    RiskConfig,
    AdaptivePositionSizer,
    RiskAnalyzer,
    AdaptivePortfolioOptimizer,
)
from alpha_pulse.config.settings import TRAINED_MODELS_DIR, settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAssetTradingSystem:
    """Multi-asset trading system with risk management."""

    def __init__(
        self,
        symbols: List[str],
        model_path: Path,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        initial_balance: float = 100000.0,
        update_interval: int = 60,  # seconds
    ):
        self.symbols = symbols
        self.update_interval = update_interval
        
        # Configure exchange
        settings.exchange.id = exchange_id
        settings.exchange.api_key = api_key
        settings.exchange.api_secret = api_secret
        
        # Initialize components
        self.data_fetcher = DataFetcher(
            exchange_factory=CCXTExchangeFactory(),
            storage=SQLAlchemyStorage()
        )
        
        # Initialize risk management system
        self.risk_manager = RiskManager(
            config=RiskConfig(
                max_position_size=0.2,
                max_portfolio_leverage=1.5,
                max_drawdown=0.25,
                stop_loss=0.1,
                target_volatility=0.15,
            ),
            position_sizer=AdaptivePositionSizer(),
            risk_analyzer=RiskAnalyzer(),
            portfolio_optimizer=AdaptivePortfolioOptimizer(),
        )
        
        # Load trained model
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        logger.info(f"Loaded model from {model_path}")
        
        # Initialize state
        self.portfolio_value = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.returns_history: Dict[str, List[float]] = {
            symbol: [] for symbol in symbols
        }
        self.portfolio_returns: List[float] = []
        
        logger.info(f"Initialized multi-asset trading system with {len(symbols)} symbols")

    async def run(self):
        """Main trading loop."""
        logger.info("Starting multi-asset trading system...")
        
        try:
            while True:
                await self._update_market_data()
                await self._update_portfolio()
                await self._execute_trades()
                await self._log_performance()
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Shutting down trading system...")
            self._log_final_results()

    async def _update_market_data(self):
        """Update market data for all symbols."""
        for symbol in self.symbols:
            # Fetch OHLCV data
            ohlcv_data = self.data_fetcher.fetch_ohlcv(
                exchange_id=settings.exchange.id,
                symbol=symbol,
                timeframe="1m",
                limit=100
            )
            
            if not ohlcv_data:
                logger.warning(f"No data received for {symbol}")
                continue
            
            # Convert to DataFrame
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
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                self.returns_history[symbol].append(returns.iloc[-1])
            
            # Update position data
            if symbol in self.positions:
                current_price = df['close'].iloc[-1]
                self.positions[symbol].update({
                    'current_price': current_price,
                    'unrealized_pnl': (
                        current_price - self.positions[symbol]['avg_entry_price']
                    ) * self.positions[symbol]['quantity']
                })

    async def _update_portfolio(self):
        """Update portfolio state and risk metrics."""
        # Calculate portfolio returns
        if self.positions:
            total_pnl = sum(
                pos['unrealized_pnl']
                for pos in self.positions.values()
            )
            portfolio_return = total_pnl / self.portfolio_value
            self.portfolio_returns.append(portfolio_return)
        
        # Update risk metrics
        if len(self.portfolio_returns) > 0:
            portfolio_returns_series = pd.Series(self.portfolio_returns)
            asset_returns = {
                symbol: pd.Series(returns)
                for symbol, returns in self.returns_history.items()
                if returns
            }
            
            self.risk_manager.update_risk_metrics(
                portfolio_returns_series,
                asset_returns
            )

    async def _execute_trades(self):
        """Execute trades based on signals and risk management."""
        for symbol in self.symbols:
            # Get latest data
            ohlcv_data = self.data_fetcher.fetch_ohlcv(
                exchange_id=settings.exchange.id,
                symbol=symbol,
                timeframe="1m",
                limit=100
            )
            
            if not ohlcv_data:
                continue
                
            # Calculate features
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
            
            features_df = calculate_technical_indicators(df)
            if features_df.empty:
                continue
            
            # Get model prediction
            features = features_df[self.feature_names].iloc[-1].values.reshape(1, -1)
            signal = self.model.predict_proba(features)[0][1]
            
            # Calculate position size
            current_price = df['close'].iloc[-1]
            historical_returns = pd.Series(self.returns_history[symbol])
            
            size_result = self.risk_manager.calculate_position_size(
                symbol=symbol,
                current_price=current_price,
                signal_strength=signal,
                historical_returns=historical_returns,
            )
            
            # Evaluate trade
            if signal > 0.75:  # Strong buy signal
                quantity = size_result.size / current_price
                if self.risk_manager.evaluate_trade(
                    symbol=symbol,
                    side="buy",
                    quantity=quantity,
                    current_price=current_price,
                    portfolio_value=self.portfolio_value,
                    current_positions=self.positions
                ):
                    self._execute_trade(symbol, "buy", quantity, current_price)
            
            elif signal < 0.25:  # Strong sell signal
                if symbol in self.positions:
                    quantity = self.positions[symbol]['quantity']
                    if self.risk_manager.evaluate_trade(
                        symbol=symbol,
                        side="sell",
                        quantity=quantity,
                        current_price=current_price,
                        portfolio_value=self.portfolio_value,
                        current_positions=self.positions
                    ):
                        self._execute_trade(symbol, "sell", quantity, current_price)

    def _execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ):
        """Execute a trade and update portfolio state."""
        if side == "buy":
            cost = quantity * price
            if cost > self.portfolio_value:
                logger.warning(f"Insufficient funds for {symbol} buy order")
                return
                
            self.portfolio_value -= cost
            self.positions[symbol] = {
                'quantity': quantity,
                'avg_entry_price': price,
                'current_price': price,
                'unrealized_pnl': 0.0
            }
            logger.info(f"Bought {quantity:.6f} {symbol} @ {price:.2f}")
            
        else:  # sell
            if symbol not in self.positions:
                return
                
            proceeds = quantity * price
            self.portfolio_value += proceeds
            realized_pnl = (
                price - self.positions[symbol]['avg_entry_price']
            ) * quantity
            logger.info(
                f"Sold {quantity:.6f} {symbol} @ {price:.2f} "
                f"(PnL: ${realized_pnl:.2f})"
            )
            del self.positions[symbol]

    async def _log_performance(self):
        """Log current performance metrics."""
        # Get risk report
        risk_report = self.risk_manager.get_risk_report()
        
        # Log portfolio state
        logger.info("\n=== Portfolio Update ===")
        logger.info(f"Portfolio Value: ${self.portfolio_value:.2f}")
        logger.info(f"Current Leverage: {risk_report.get('current_leverage', 0):.2f}x")
        
        # Log risk metrics
        if 'risk_metrics' in risk_report:
            metrics = risk_report['risk_metrics']
            logger.info("\nRisk Metrics:")
            logger.info(f"Volatility: {metrics['volatility']:.2%}")
            logger.info(f"VaR (95%): {metrics['var_95']:.2%}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        # Log positions
        if self.positions:
            logger.info("\nCurrent Positions:")
            for symbol, pos in self.positions.items():
                logger.info(
                    f"{symbol}: {pos['quantity']:.6f} @ {pos['avg_entry_price']:.2f} "
                    f"(PnL: ${pos['unrealized_pnl']:.2f})"
                )

    def _log_final_results(self):
        """Log final trading session results."""
        total_pnl = self.portfolio_value - 100000.0  # Initial balance
        logger.info("\n=== Trading Session Results ===")
        logger.info(f"Final Portfolio Value: ${self.portfolio_value:.2f}")
        logger.info(f"Total PnL: ${total_pnl:.2f} ({total_pnl/100000.0:.2%})")
        
        if self.positions:
            logger.info("\nFinal Positions:")
            for symbol, pos in self.positions.items():
                logger.info(
                    f"{symbol}: {pos['quantity']:.6f} @ {pos['avg_entry_price']:.2f} "
                    f"(PnL: ${pos['unrealized_pnl']:.2f})"
                )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-asset trading system")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        help="Trading symbols"
    )
    parser.add_argument(
        "--exchange",
        default="binance",
        help="Exchange ID"
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Exchange API key"
    )
    parser.add_argument(
        "--api-secret",
        default="",
        help="Exchange API secret"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Update interval in seconds"
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Check for model
    model_path = Path(TRAINED_MODELS_DIR) / 'crypto_prediction_model.joblib'
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    # Initialize and run trading system
    trading_system = MultiAssetTradingSystem(
        symbols=args.symbols,
        model_path=model_path,
        exchange_id=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret,
        update_interval=args.interval,
    )
    
    await trading_system.run()


if __name__ == "__main__":
    asyncio.run(main())