#!/usr/bin/env python3
"""
Orchestrator for the Long/Short S&P 500 Strategy.

Combines data loading, indicator calculation, signal generation,
position sizing, and risk management.
"""

import logging
import pandas as pd
import numpy as np # Add numpy import
from typing import Optional, Dict, Tuple, List
 
 # Import refactored components from the same directory
from .indicators import add_indicators_to_data
from .signal_generator import generate_composite_signal
from .position_manager import PositionManager
from .risk_manager import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LongShortOrchestrator")

class LongShortOrchestrator:
    """
    Orchestrates the steps involved in generating trading signals and targets
    for the Long/Short strategy.
    """

    def __init__(self, config: Dict):
        """
        Initializes the orchestrator with strategy configuration.

        Args:
            config: A dictionary containing parameters for indicators,
                    signal generation, position management, and risk management.
                    Expected keys: 'indicators', 'signal_weights', 'position_manager', 'risk_manager'.
        """
        self.config = config
        self.indicator_params = config.get('indicators', {})
        self.signal_weights = config.get('signal_weights', None) # Use defaults in generator if None
        self.position_manager = PositionManager(config.get('position_manager', {}))
        self.risk_manager = RiskManager(config.get('risk_manager', {}))
        logger.info("LongShortOrchestrator initialized.")

    def _prepare_input_data(
        self,
        data: Dict[str, pd.DataFrame],
        primary_symbol: str = '^GSPC',
        vix_symbol: str = '^VIX'
    ) -> Optional[pd.DataFrame]:
        """
        Prepares the combined DataFrame needed for indicators.
        Selects relevant columns and aligns data.

        Args:
            data: Dictionary of DataFrames loaded by data_loader (symbol -> OHLCV DF).
            primary_symbol: The main symbol for price/trend indicators (e.g., SP500).
            vix_symbol: The symbol for the VIX index.

        Returns:
            A combined DataFrame with prefixed columns (e.g., SP500_Close, VIX_Close),
            or None if essential data is missing.
        """
        if primary_symbol not in data or data[primary_symbol].empty:
            logger.error(f"Primary symbol '{primary_symbol}' data not found or empty.")
            return None
        if vix_symbol not in data or data[vix_symbol].empty:
            logger.error(f"VIX symbol '{vix_symbol}' data not found or empty.")
            # Allow proceeding without VIX if strategy doesn't strictly require it?
            # For now, assume VIX is required for volatility regime.
            return None

        primary_df = data[primary_symbol][['Open', 'High', 'Low', 'Close', 'Volume']].add_prefix('SP500_')
        vix_df = data[vix_symbol][['Close']].add_prefix('VIX_') # Only need Close for VIX regime

        # Combine using an outer join to keep all dates, then forward-fill VIX
        combined_df = pd.merge(primary_df, vix_df, left_index=True, right_index=True, how='outer')
        combined_df['VIX_Close'] = combined_df['VIX_Close'].ffill()

        # Drop rows where primary data is missing
        combined_df.dropna(subset=['SP500_Close'], inplace=True)

        if combined_df.empty:
            logger.error("Combined DataFrame is empty after merging and cleaning.")
            return None

        logger.debug(f"Prepared combined data shape: {combined_df.shape}")
        return combined_df


    def calculate_signals_and_targets(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """
        Runs the full pipeline: prepare data -> indicators -> signals -> target position -> stop loss.

        Args:
            data: Dictionary of DataFrames loaded by data_loader (symbol -> OHLCV DF).

        Returns:
            A tuple containing:
            1. DataFrame with indicators added.
            2. Series with target allocation (-1.0 to 1.0).
            3. Series with calculated stop-loss prices (or NaN).
            Returns None if any critical step fails.
        """
        logger.info("Starting signal and target calculation pipeline...")

        # 1. Prepare Data
        # TODO: Make symbols configurable
        primary_symbol = '^GSPC'
        vix_symbol = '^VIX'
        combined_data = self._prepare_input_data(data, primary_symbol, vix_symbol)
        if combined_data is None:
            return None

        # 2. Calculate Indicators
        # Pass specific column names based on the _prepare_input_data prefixes
        data_with_indicators = add_indicators_to_data(
            data=combined_data,
            ma_window=self.indicator_params.get('ma_window', 40),
            rsi_window=self.indicator_params.get('rsi_window', 14),
            atr_window=self.indicator_params.get('atr_window', 14),
            vix_threshold=self.indicator_params.get('vix_threshold', 25.0),
            price_column='SP500_Close', # Use the prepared column name
            high_col='SP500_High',
            low_col='SP500_Low',
            close_col='SP500_Close',
            vix_column='VIX_Close'      # Use the prepared column name
        )
        if data_with_indicators is None:
            logger.error("Failed to add indicators.")
            return None
        logger.debug(f"Data with indicators shape: {data_with_indicators.shape}")

        # 3. Generate Composite Signal
        # Pass correct indicator column names
        signal_df = generate_composite_signal(
            data=data_with_indicators,
            weights=self.signal_weights # Pass configured weights
            # generate_composite_signal uses hardcoded indicator names for now, ensure they match add_indicators_to_data
        )
        if signal_df is None:
            logger.error("Failed to generate composite signal.")
            return None
        logger.debug(f"Signal DataFrame shape: {signal_df.shape}")

        # Merge signals back for position/risk calculation
        data_with_signals = pd.merge(data_with_indicators, signal_df, left_index=True, right_index=True, how='left')

        # 4. Calculate Target Position
        target_position = self.position_manager.calculate_target_position(
            signal_data=data_with_signals,
            signal_column='Composite_Signal'
        )
        if target_position is None:
            logger.error("Failed to calculate target position.")
            return None
        logger.debug(f"Target Position Series shape: {target_position.shape}")

        # 5. Calculate Stop Loss (conditionally based on target position)
        stop_loss = pd.Series(np.nan, index=data_with_signals.index)
        atr_col = f"ATR_{self.indicator_params.get('atr_window', 14)}"
        if self.risk_manager.stop_loss_type == 'atr' and atr_col not in data_with_signals.columns:
            logger.error(f"ATR column '{atr_col}' needed for stop loss but not found.")
            # Proceed without stop loss? Or fail? For now, proceed.
        else:
            for i in range(len(data_with_signals)):
                current_target = target_position.iloc[i]
                prev_target = target_position.iloc[i-1] if i > 0 else 0.0

                # Calculate SL only on days where a new position might be entered
                # (target goes from zero/opposite to non-zero, or flips sign)
                if (prev_target == 0.0 and current_target != 0.0) or \
                   (np.sign(prev_target) != np.sign(current_target) and current_target != 0.0):

                    entry_price = data_with_signals['SP500_Close'].iloc[i] # Assume entry at close
                    direction = 'long' if current_target > 0 else 'short'
                    current_atr = data_with_signals[atr_col].iloc[i] if self.risk_manager.stop_loss_type == 'atr' else None

                    sl_price = self.risk_manager.calculate_stop_loss(
                        entry_price=entry_price,
                        direction=direction,
                        current_atr=current_atr
                    )
                    if sl_price is not None:
                        stop_loss.iloc[i] = sl_price
                        logger.debug(f"Index {i} ({stop_loss.index[i].date()}): Target={current_target:.2f}, Prev={prev_target:.2f}. Calculated SL={sl_price:.2f}")

        logger.info("Signal and target calculation pipeline finished.")
        return data_with_signals, target_position, stop_loss

# Example usage (if needed for testing, requires data loader)
# async def run_orchestrator_example():
#     # Load data using data_loader
#     # ... data = await load_ohlcv_data(...) ...
#     # Define config
#     # ... config = {...} ...
#     # orchestrator = LongShortOrchestrator(config)
#     # results = orchestrator.calculate_signals_and_targets(data)
#     # if results:
#     #     df, targets, stops = results
#     #     print(df.tail())
#     #     print(targets.tail())
#     #     print(stops.tail())