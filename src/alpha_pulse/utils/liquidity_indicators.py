"""
Liquidity indicator calculations and utilities.

Provides functions for calculating various liquidity indicators including
microstructure measures, order flow metrics, and liquidity scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from scipy import stats
from numba import jit

logger = logging.getLogger(__name__)


class LiquidityIndicators:
    """Calculate various liquidity indicators from market data."""
    
    @staticmethod
    def calculate_bid_ask_spread(
        bid_prices: pd.Series,
        ask_prices: pd.Series,
        mid_prices: Optional[pd.Series] = None
    ) -> Dict[str, pd.Series]:
        """Calculate various bid-ask spread measures."""
        if mid_prices is None:
            mid_prices = (bid_prices + ask_prices) / 2
        
        # Quoted spread (absolute)
        quoted_spread = ask_prices - bid_prices
        
        # Quoted spread (relative, in bps)
        quoted_spread_bps = (quoted_spread / mid_prices) * 10000
        
        # Log quoted spread
        log_quoted_spread = np.log(ask_prices) - np.log(bid_prices)
        
        # Proportional quoted spread
        prop_quoted_spread = quoted_spread / ((bid_prices + ask_prices) / 2)
        
        return {
            'quoted_spread': quoted_spread,
            'quoted_spread_bps': quoted_spread_bps,
            'log_quoted_spread': log_quoted_spread,
            'proportional_spread': prop_quoted_spread
        }
    
    @staticmethod
    def calculate_effective_spread(
        trade_prices: pd.Series,
        trade_sides: pd.Series,  # 1 for buy, -1 for sell
        mid_prices: pd.Series
    ) -> pd.Series:
        """Calculate effective spread from trades."""
        # Effective spread = 2 * |trade_price - mid_price| * trade_direction
        effective_spread = 2 * trade_sides * (trade_prices - mid_prices)
        effective_spread_bps = (effective_spread / mid_prices) * 10000
        
        return effective_spread_bps.abs()
    
    @staticmethod
    def calculate_realized_spread(
        trade_prices: pd.Series,
        trade_sides: pd.Series,
        future_mid_prices: pd.Series,  # Mid prices some time after trade
        mid_prices: pd.Series
    ) -> pd.Series:
        """Calculate realized spread (temporary component)."""
        # Realized spread = 2 * trade_direction * (trade_price - future_mid_price)
        realized_spread = 2 * trade_sides * (trade_prices - future_mid_prices)
        realized_spread_bps = (realized_spread / mid_prices) * 10000
        
        return realized_spread_bps
    
    @staticmethod
    def calculate_price_impact(
        trade_prices: pd.Series,
        trade_sides: pd.Series,
        mid_prices_before: pd.Series,
        mid_prices_after: pd.Series
    ) -> pd.Series:
        """Calculate price impact of trades."""
        # Price impact = trade_direction * (mid_after - mid_before)
        price_impact = trade_sides * (mid_prices_after - mid_prices_before)
        price_impact_bps = (price_impact / mid_prices_before) * 10000
        
        return price_impact_bps
    
    @staticmethod
    @jit(nopython=True)
    def calculate_roll_measure(price_changes: np.ndarray) -> float:
        """Calculate Roll's implied spread measure."""
        # Roll measure = 2 * sqrt(-cov(price_change_t, price_change_t-1))
        if len(price_changes) < 2:
            return 0.0
        
        # Calculate autocovariance
        cov = np.cov(price_changes[1:], price_changes[:-1])[0, 1]
        
        if cov < 0:
            return 2 * np.sqrt(-cov)
        else:
            return 0.0
    
    @staticmethod
    def calculate_corwin_schultz_spread(
        high_prices: pd.Series,
        low_prices: pd.Series,
        period: int = 2
    ) -> pd.Series:
        """Calculate Corwin-Schultz bid-ask spread estimator."""
        # Log high-low ratios
        log_hl = np.log(high_prices / low_prices)
        
        # Period sum of squared log ratios
        sum_sq = log_hl.rolling(period).apply(lambda x: np.sum(x**2))
        
        # Square of period sum of log ratios
        sq_sum = log_hl.rolling(period).sum() ** 2
        
        # Beta calculation
        beta = sum_sq
        gamma = sq_sum
        
        # Alpha calculation
        k = 2 * (np.sqrt(period) - np.sqrt(period - 1))
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / ((3 - 2 * np.sqrt(2)) * np.sqrt(period)) - \
                np.sqrt(gamma / (period * (3 - 2 * np.sqrt(2))))
        
        # Spread calculation
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        
        # Convert to basis points
        spread_bps = spread * 10000
        
        return spread_bps.fillna(0).clip(lower=0)
    
    @staticmethod
    def calculate_amihud_illiquidity(
        returns: pd.Series,
        volumes: pd.Series,
        prices: pd.Series
    ) -> pd.Series:
        """Calculate Amihud illiquidity ratio."""
        # Amihud = |return| / (volume * price)
        # Measures price impact per unit of trading volume
        
        dollar_volumes = volumes * prices
        
        # Avoid division by zero
        dollar_volumes = dollar_volumes.replace(0, np.nan)
        
        amihud = np.abs(returns) / dollar_volumes
        
        # Scale for interpretability (multiply by 10^6)
        amihud_scaled = amihud * 1e6
        
        return amihud_scaled
    
    @staticmethod
    def calculate_kyle_lambda(
        price_changes: pd.Series,
        signed_volumes: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Estimate Kyle's lambda (price impact coefficient)."""
        # Kyle's lambda from regression: price_change = lambda * signed_volume
        
        def estimate_lambda(data):
            if len(data) < 10:
                return np.nan
            
            price_ch = data['price_change'].values
            signed_vol = data['signed_volume'].values
            
            # Remove outliers
            mask = (np.abs(stats.zscore(price_ch)) < 3) & \
                   (np.abs(stats.zscore(signed_vol)) < 3)
            
            if mask.sum() < 10:
                return np.nan
            
            try:
                slope, _, _, _, _ = stats.linregress(
                    signed_vol[mask], price_ch[mask]
                )
                return slope * 1e6  # Scale for interpretability
            except:
                return np.nan
        
        # Create DataFrame for rolling calculation
        df = pd.DataFrame({
            'price_change': price_changes,
            'signed_volume': signed_volumes
        })
        
        kyle_lambda = df.rolling(window=window).apply(
            lambda x: estimate_lambda(x), raw=False
        )['price_change']
        
        return kyle_lambda
    
    @staticmethod
    def calculate_volume_synchronized_probability_of_informed_trading(
        buy_volumes: pd.Series,
        sell_volumes: pd.Series,
        total_volumes: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Calculate VPIN (Volume-Synchronized Probability of Informed Trading)."""
        # VPIN approximates the probability of informed trading
        
        # Calculate order imbalance
        order_imbalance = np.abs(buy_volumes - sell_volumes)
        
        # VPIN = order_imbalance / total_volume
        vpin = order_imbalance.rolling(window=window).sum() / \
               total_volumes.rolling(window=window).sum()
        
        return vpin.fillna(0).clip(0, 1)
    
    @staticmethod
    def calculate_order_flow_imbalance(
        bid_sizes: pd.Series,
        ask_sizes: pd.Series,
        normalize: bool = True
    ) -> pd.Series:
        """Calculate order flow imbalance."""
        # OFI = (bid_size - ask_size) / (bid_size + ask_size)
        
        total_size = bid_sizes + ask_sizes
        
        if normalize:
            # Normalized OFI between -1 and 1
            ofi = (bid_sizes - ask_sizes) / total_size.replace(0, np.nan)
        else:
            # Raw OFI
            ofi = bid_sizes - ask_sizes
        
        return ofi.fillna(0)
    
    @staticmethod
    def calculate_liquidity_ratio(
        volumes: pd.Series,
        price_ranges: pd.Series,  # High - Low
        window: int = 20
    ) -> pd.Series:
        """Calculate Hui-Heubel liquidity ratio."""
        # LR = (max_price - min_price) / min_price / (volume / shares_outstanding)
        # Simplified version using daily range and volume
        
        # Calculate returns from price ranges
        returns = price_ranges / price_ranges.shift(1)
        
        # Volume ratio (current vs average)
        avg_volume = volumes.rolling(window=window).mean()
        volume_ratio = volumes / avg_volume
        
        # Liquidity ratio
        liquidity_ratio = returns / volume_ratio
        
        return liquidity_ratio.fillna(0)
    
    @staticmethod
    def calculate_market_depth_imbalance(
        bid_depths: List[List[Tuple[float, float]]],  # List of (price, size) pairs
        ask_depths: List[List[Tuple[float, float]]],
        max_levels: int = 5
    ) -> pd.Series:
        """Calculate market depth imbalance at multiple levels."""
        imbalances = []
        
        for bid_book, ask_book in zip(bid_depths, ask_depths):
            # Sum sizes at each level
            bid_depth = sum(size for _, size in bid_book[:max_levels])
            ask_depth = sum(size for _, size in ask_book[:max_levels])
            
            # Calculate imbalance
            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                imbalance = (bid_depth - ask_depth) / total_depth
            else:
                imbalance = 0
            
            imbalances.append(imbalance)
        
        return pd.Series(imbalances)
    
    @staticmethod
    def calculate_quote_slope(
        bid_prices: List[List[float]],
        bid_sizes: List[List[float]],
        ask_prices: List[List[float]],
        ask_sizes: List[List[float]],
        max_levels: int = 5
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate bid and ask quote slopes (price elasticity)."""
        bid_slopes = []
        ask_slopes = []
        
        for bp, bs, ap, as_ in zip(bid_prices, bid_sizes, ask_prices, ask_sizes):
            # Calculate bid slope
            if len(bp) >= 2 and len(bs) >= 2:
                bid_x = np.array(bs[:max_levels])
                bid_y = np.array(bp[:max_levels])
                if len(bid_x) > 1:
                    bid_slope, _ = np.polyfit(bid_x, bid_y, 1)
                else:
                    bid_slope = 0
            else:
                bid_slope = 0
            
            # Calculate ask slope
            if len(ap) >= 2 and len(as_) >= 2:
                ask_x = np.array(as_[:max_levels])
                ask_y = np.array(ap[:max_levels])
                if len(ask_x) > 1:
                    ask_slope, _ = np.polyfit(ask_x, ask_y, 1)
                else:
                    ask_slope = 0
            else:
                ask_slope = 0
            
            bid_slopes.append(abs(bid_slope))
            ask_slopes.append(abs(ask_slope))
        
        return pd.Series(bid_slopes), pd.Series(ask_slopes)
    
    @staticmethod
    def calculate_resilience_measure(
        mid_prices: pd.Series,
        trade_times: pd.Series,
        trade_impacts: pd.Series,
        reversion_window: int = 10  # minutes
    ) -> float:
        """Calculate market resilience (speed of price recovery)."""
        # Measures how quickly prices revert after trades
        
        resilience_scores = []
        
        for i, (trade_time, impact) in enumerate(zip(trade_times, trade_impacts)):
            if abs(impact) < 1:  # Skip small impacts
                continue
            
            # Find prices in reversion window
            future_mask = (mid_prices.index > trade_time) & \
                         (mid_prices.index <= trade_time + timedelta(minutes=reversion_window))
            
            if future_mask.sum() == 0:
                continue
            
            future_prices = mid_prices[future_mask]
            initial_price = mid_prices[mid_prices.index <= trade_time].iloc[-1]
            
            # Calculate reversion
            price_changes = (future_prices - initial_price) / initial_price * 10000
            
            # Measure how much of the impact was reversed
            if impact > 0:
                reversion = (impact - price_changes.iloc[-1]) / impact
            else:
                reversion = (price_changes.iloc[-1] - impact) / abs(impact)
            
            resilience_scores.append(max(0, min(1, reversion)))
        
        return np.mean(resilience_scores) if resilience_scores else 0.5
    
    @staticmethod
    def calculate_turnover_ratio(
        volumes: pd.Series,
        shares_outstanding: Union[float, pd.Series],
        window: int = 20
    ) -> pd.Series:
        """Calculate turnover ratio."""
        # Turnover = Volume / Shares Outstanding
        
        if isinstance(shares_outstanding, (int, float)):
            turnover = volumes / shares_outstanding
        else:
            turnover = volumes / shares_outstanding
        
        # Calculate rolling average
        avg_turnover = turnover.rolling(window=window).mean()
        
        return avg_turnover
    
    @staticmethod
    def calculate_liquidity_score(
        spread_bps: pd.Series,
        volume: pd.Series,
        depth_imbalance: pd.Series,
        amihud_ratio: pd.Series,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """Calculate composite liquidity score."""
        if weights is None:
            weights = {
                'spread': 0.3,
                'volume': 0.3,
                'depth': 0.2,
                'impact': 0.2
            }
        
        # Normalize components (0-100 scale)
        # Lower spread is better
        spread_score = 100 * (1 - spread_bps.clip(0, 100) / 100)
        
        # Higher volume is better (log scale)
        volume_score = 100 * (np.log(volume + 1) / np.log(volume.quantile(0.99) + 1))
        
        # Balanced depth is better
        depth_score = 100 * (1 - np.abs(depth_imbalance))
        
        # Lower impact is better
        amihud_score = 100 * (1 - amihud_ratio.clip(0, 1) / 1)
        
        # Weighted average
        liquidity_score = (
            weights['spread'] * spread_score +
            weights['volume'] * volume_score +
            weights['depth'] * depth_score +
            weights['impact'] * amihud_score
        )
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    def detect_liquidity_shocks(
        liquidity_scores: pd.Series,
        threshold_std: float = 2.0,
        min_duration: int = 5
    ) -> List[Tuple[datetime, datetime, float]]:
        """Detect periods of liquidity shocks."""
        # Calculate rolling statistics
        rolling_mean = liquidity_scores.rolling(window=60).mean()
        rolling_std = liquidity_scores.rolling(window=60).std()
        
        # Identify shocks (significant drops)
        z_scores = (liquidity_scores - rolling_mean) / rolling_std
        shock_mask = z_scores < -threshold_std
        
        # Group consecutive shocks
        shocks = []
        in_shock = False
        shock_start = None
        
        for idx, is_shock in shock_mask.items():
            if is_shock and not in_shock:
                # Shock starts
                in_shock = True
                shock_start = idx
            elif not is_shock and in_shock:
                # Shock ends
                in_shock = False
                duration = (idx - shock_start).total_seconds() / 60
                
                if duration >= min_duration:
                    severity = -z_scores[shock_start:idx].min()
                    shocks.append((shock_start, idx, severity))
        
        return shocks
    
    @staticmethod
    def calculate_execution_shortfall_cost(
        arrival_price: float,
        execution_prices: pd.Series,
        execution_sizes: pd.Series,
        side: str  # 'buy' or 'sell'
    ) -> Dict[str, float]:
        """Calculate execution shortfall and its components."""
        # Calculate weighted average execution price
        total_size = execution_sizes.sum()
        vwap = (execution_prices * execution_sizes).sum() / total_size
        
        # Implementation shortfall
        if side == 'buy':
            shortfall_pct = (vwap - arrival_price) / arrival_price
        else:
            shortfall_pct = (arrival_price - vwap) / arrival_price
        
        shortfall_bps = shortfall_pct * 10000
        
        # Timing component (simplified)
        # Compare to TWAP
        twap = execution_prices.mean()
        timing_cost = (vwap - twap) / arrival_price * 10000
        
        # Market impact (price trend during execution)
        price_trend = (execution_prices.iloc[-1] - execution_prices.iloc[0]) / \
                     execution_prices.iloc[0]
        
        if side == 'buy':
            market_impact = price_trend * 10000
        else:
            market_impact = -price_trend * 10000
        
        return {
            'total_shortfall_bps': shortfall_bps,
            'timing_cost_bps': timing_cost,
            'market_impact_bps': market_impact,
            'vwap': vwap,
            'twap': twap
        }