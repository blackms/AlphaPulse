#!/usr/bin/env python3
"""
Long/Short S&P 500 Trading Strategy Implementation.

Combines signals from DataHandler, Indicators, SignalGenerator, PositionManager,
and RiskManager to generate trading decisions.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Base agent and data structures
from src.agents.base import BaseTradeAgent, MarketData, TradeSignal, SignalDirection

# Strategy specific components
from .data_handler import LongShortDataHandler
from .indicators import add_indicators_to_data
from .signal_generator import generate_composite_signal
from .position_manager import PositionManager
from .risk_manager import RiskManager

# Configure logging
logger = logging.getLogger("LongShortStrategyAgent")

class LongShortStrategyAgent(BaseTradeAgent):
    """
    Trading agent implementing the Long/Short S&P 500 strategy.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initializes the agent and its components."""
        super().__init__("long_short_sp500_strategy", config)
        self.strategy_config = self.config.get('strategy_params', {})

        # Initialize components
        # DataHandler needs cache_dir, potentially passed via main config
        data_handler_config = {"cache_dir": self.config.get("cache_dir", "./data/cache/long_short")}
        self.data_handler = LongShortDataHandler(data_handler_config) # Data handler is mostly for fetching/resampling, less used in generate_signals if backtester provides data

        # Indicator parameters
        self.ma_window = self.strategy_config.get('ma_window', 40)
        self.rsi_window = self.strategy_config.get('rsi_window', 14)
        self.atr_window = self.strategy_config.get('atr_window', 14)
        self.vix_threshold = self.strategy_config.get('vix_threshold', 25.0)

        # Signal generation parameters
        self.signal_weights = self.strategy_config.get('signal_weights', {'trend': 1.0, 'mean_reversion': 0.3})

        # Position manager parameters
        pm_config = {
            'long_threshold': self.strategy_config.get('long_threshold', 0.1),
            'short_threshold': self.strategy_config.get('short_threshold', -0.1)
        }
        self.position_manager = PositionManager(pm_config)

        # Risk manager parameters
        rm_config = {
            'stop_loss_type': self.strategy_config.get('stop_loss_type', 'atr'),
            'stop_loss_pct': self.strategy_config.get('stop_loss_pct', 0.02),
            'stop_loss_atr_multiplier': self.strategy_config.get('stop_loss_atr_multiplier', 2.0)
        }
        self.risk_manager = RiskManager(rm_config)

        # Determine required lookback based on longest indicator window
        self.required_lookback = max(self.ma_window, self.rsi_window, self.atr_window) + 5 # Add buffer

        logger.info(f"LongShortStrategyAgent initialized. Lookback: {self.required_lookback}")
        logger.info(f"Indicator Params: MA={self.ma_window}, RSI={self.rsi_window}, ATR={self.atr_window}, VIX_Thresh={self.vix_threshold}")
        logger.info(f"Position Params: LongThresh={pm_config['long_threshold']}, ShortThresh={pm_config['short_threshold']}")
        logger.info(f"Risk Params: SL_Type={rm_config['stop_loss_type']}, SL_Pct={rm_config['stop_loss_pct']}, SL_ATR_Mult={rm_config['stop_loss_atr_multiplier']}")


    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes the agent with runtime configuration if needed."""
        await super().initialize(config)
        # Re-apply config if necessary, similar to __init__
        # This allows overriding parameters via the backtester's agent config section
        self.strategy_config = config.get('strategy_params', self.strategy_config)
        # Re-initialize components or update parameters based on new config
        # (Example: update windows, thresholds, weights)
        self.ma_window = self.strategy_config.get('ma_window', self.ma_window)
        self.rsi_window = self.strategy_config.get('rsi_window', self.rsi_window)
        # ... update other params ...
        self.required_lookback = max(self.ma_window, self.rsi_window, self.atr_window) + 5
        logger.info("LongShortStrategyAgent re-initialized with runtime config.")


    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Generates trading signals based on the long/short strategy logic.

        Args:
            market_data: MarketData object containing historical prices and current timestamp.

        Returns:
            A list containing a single TradeSignal representing the desired target allocation,
            or an empty list if no signal is generated or data is insufficient.
        """
        signals = []
        symbol = "^GSPC" # Hardcoded for this strategy
        current_timestamp = market_data.timestamp

        # Ensure we have the necessary data (OHLCV)
        if not isinstance(market_data.prices, pd.DataFrame) or market_data.prices.empty:
            logger.warning(f"[{current_timestamp}] No price data available.")
            return signals
        required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] # Base columns from yfinance
        # Adjust column names based on data_handler prefixing (SP500_High etc.)
        # We assume the backtester provides data in the format expected by indicators.py
        # Let's assume the backtester passes the *raw* daily data slice, and we handle prefixing/selection if needed
        # For now, assume columns like 'SP500_High', 'VIX_Adj Close' are NOT present yet.
        # We need to adapt based on how backtester calls _get_market_data_for_date and what it returns.

        # TEMPORARY ASSUMPTION: market_data.prices contains raw ^GSPC OHLCV data.
        # We need VIX data as well. The current MarketData structure doesn't seem ideal for this strategy.
        # Option 1: Modify backtester to fetch/pass combined data.
        # Option 2: Fetch VIX data here (inefficient due to repeated calls).

        # Let's proceed with a temporary workaround: Fetch VIX data here using the handler.
        # This is NOT ideal for performance in a real backtest loop.
        # The backtester should ideally provide all necessary data streams.
        start_date_str = (current_timestamp - pd.Timedelta(days=self.required_lookback * 2)).strftime('%Y-%m-%d') # Approx lookback
        end_date_str = current_timestamp.strftime('%Y-%m-%d')

        # Fetch combined SP500 and VIX data using the handler
        # This assumes the handler's cache works effectively.
        combined_daily_data = self.data_handler.get_combined_data(start_date_str, end_date_str)

        if combined_daily_data is None or combined_daily_data.empty:
             logger.warning(f"[{current_timestamp}] Could not get combined SP500/VIX data.")
             return signals

        # Ensure data goes up to the current timestamp
        if current_timestamp not in combined_daily_data.index:
             # Use ffill to get the most recent data if exact timestamp is missing (e.g., weekend)
             combined_daily_data = combined_daily_data.ffill().loc[:current_timestamp]
             if current_timestamp not in combined_daily_data.index:
                  logger.warning(f"[{current_timestamp}] Timestamp still missing after ffill.")
                  return signals # Cannot proceed without data for the current day


        # Resample to weekly ('W') or monthly ('M') - make this configurable
        # For now, let's assume weekly rebalancing
        resample_freq = self.strategy_config.get('rebalance_freq', 'W') # 'W' or 'M'
        resampled_data = self.data_handler.resample_data(combined_daily_data, timeframe=resample_freq)

        if resampled_data is None or resampled_data.empty:
            logger.warning(f"[{current_timestamp}] Could not resample data to {resample_freq}.")
            return signals

        # Check if current_timestamp corresponds to a rebalancing date
        # This logic depends on the resampling frequency and how pandas aligns dates
        # For weekly ('W'), pandas typically aligns to Sunday. We might want to trade on Monday.
        # Simplification: Check if the *latest* date in the resampled index is recent enough.
        latest_resampled_date = resampled_data.index[-1]
        # We need a robust way to check if today is a rebalance day based on the frequency.
        # Example for weekly (trade on Monday after Sunday's resample):
        is_rebalance_day = False
        if resample_freq == 'W':
             # If today is Monday and the latest resampled date is yesterday (Sunday) or earlier within the week
             if current_timestamp.weekday() == 0 and (current_timestamp.normalize() - latest_resampled_date.normalize()).days < 7:
                  is_rebalance_day = True
        elif resample_freq == 'M':
             # If today is the first trading day of the month and the latest resampled date is from the previous month
             # This needs refinement based on actual trading days
             if current_timestamp.day == 1 and latest_resampled_date.month != current_timestamp.month: # Approximation
                  is_rebalance_day = True


        # Only generate signals on rebalance days
        # if not is_rebalance_day:
        #     logger.debug(f"[{current_timestamp}] Not a rebalance day ({resample_freq}). Skipping signal generation.")
        #     return signals
        # --> Temporarily disabling rebalance day check to ensure signal generation for testing

        logger.debug(f"[{current_timestamp}] Proceeding with signal generation on resampled data (latest: {latest_resampled_date}).")

        # Add indicators
        # Use column names generated by data_handler (e.g., 'SP500_Adj Close')
        data_with_indicators = add_indicators_to_data(
            data=resampled_data,
            ma_window=self.ma_window,
            rsi_window=self.rsi_window,
            atr_window=self.atr_window,
            vix_threshold=self.vix_threshold,
            price_column='SP500_Adj Close', # Match data_handler output
            high_col='SP500_High',         # Match data_handler output
            low_col='SP500_Low',           # Match data_handler output
            close_col='SP500_Close',       # Match data_handler output
            vix_column='VIX_Adj Close'      # Match data_handler output
        )

        if data_with_indicators is None or data_with_indicators.empty:
            logger.warning(f"[{current_timestamp}] Failed to add indicators.")
            return signals

        # Generate composite signal
        signal_df = generate_composite_signal(data_with_indicators, weights=self.signal_weights)

        if signal_df is None or signal_df.empty:
            logger.warning(f"[{current_timestamp}] Failed to generate composite signal.")
            return signals

        # Calculate target position
        target_position_series = self.position_manager.calculate_target_position(signal_df)

        if target_position_series is None:
            logger.warning(f"[{current_timestamp}] Failed to calculate target position.")
            return signals

        # Get the latest target position for the current timestamp (or most recent available)
        try:
            # Use the latest calculated target position from the resampled data
            latest_target_position = target_position_series.iloc[-1]
            latest_atr = data_with_indicators[f'ATR_{self.atr_window}'].iloc[-1]
            latest_price = data_with_indicators['SP500_Adj Close'].iloc[-1] # Use the price from the indicator data
        except IndexError:
            logger.warning(f"[{current_timestamp}] Could not get latest target position or indicators.")
            return signals
        except KeyError as e:
             logger.warning(f"[{current_timestamp}] Missing column for latest value extraction: {e}")
             return signals


        if pd.isna(latest_target_position) or pd.isna(latest_price):
             logger.warning(f"[{current_timestamp}] Latest target position or price is NaN.")
             return signals

        # Determine signal direction based on target position
        # This needs refinement based on how the backtester handles signals.
        # Option A: Send BUY/SELL/HOLD based on change from current position (needs current pos info)
        # Option B: Send a 'TARGET' signal with the allocation percentage.
        # Let's try Option B for now, requiring backtester modification.

        signal_direction = SignalDirection.HOLD # Default, backtester ignores
        target_allocation = round(latest_target_position, 4) # Use the calculated target allocation

        # We need a way to signal the target allocation. Let's overload 'confidence'.
        # Confidence will represent the target allocation (-1.0 to +1.0).
        # We'll use a placeholder direction like BUY if target > 0, SELL if target < 0.
        # The backtester's execute_trade needs to understand this convention.

        if abs(target_allocation) > 1e-4: # Only signal if target is non-zero (within tolerance)
             # Use BUY for positive allocation, SELL for negative. Backtester needs to interpret confidence.
             signal_direction = SignalDirection.BUY if target_allocation > 0 else SignalDirection.SELL
             confidence = target_allocation # Store target allocation here

             # Calculate stop loss if entering/holding a position
             stop_loss = None
             if signal_direction == SignalDirection.BUY:
                  stop_loss = self.risk_manager.calculate_stop_loss(latest_price, 'long', current_atr=latest_atr)
             elif signal_direction == SignalDirection.SELL: # Assuming SELL means short
                  stop_loss = self.risk_manager.calculate_stop_loss(latest_price, 'short', current_atr=latest_atr)

             metadata = {
                 "strategy": self.agent_id,
                 "target_allocation": confidence,
                 "price_at_signal": latest_price,
                 "atr_at_signal": latest_atr if pd.notna(latest_atr) else None,
                 "stop_loss_price": stop_loss if stop_loss is not None and pd.notna(stop_loss) else None,
                 # Add other relevant indicator values if needed
                 "ma_40": data_with_indicators['MA_40'].iloc[-1] if 'MA_40' in data_with_indicators else None,
                 "rsi_14": data_with_indicators['RSI_14'].iloc[-1] if 'RSI_14' in data_with_indicators else None,
                 "vol_regime": data_with_indicators['Vol_Regime'].iloc[-1] if 'Vol_Regime' in data_with_indicators else None,
             }

             signal = TradeSignal(
                 agent_id=self.agent_id,
                 symbol=symbol,
                 direction=signal_direction, # BUY for long target, SELL for short target
                 confidence=abs(confidence), # Backtester needs to use metadata['target_allocation']
                 timestamp=current_timestamp, # Signal is for the current backtest day
                 metadata=metadata
             )

             if await self.validate_signal(signal):
                 signals.append(signal)
                 logger.info(f"[{current_timestamp}] Generated Signal: Target Allocation={metadata['target_allocation']:.2f}, Stop={metadata['stop_loss_price']}")

        return signals