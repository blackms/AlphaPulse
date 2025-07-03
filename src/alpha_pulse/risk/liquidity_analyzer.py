"""
Liquidity risk analyzer for comprehensive liquidity measurement.

Implements traditional and advanced liquidity metrics including spreads,
depth analysis, and market impact measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
from dataclasses import dataclass

from alpha_pulse.models.liquidity_metrics import (
    LiquidityMetrics, MarketDepth, SpreadMeasurement,
    SpreadType, LiquidityTier, VolumeProfile,
    IntradayLiquidity, LiquidityEvent
)

logger = logging.getLogger(__name__)


class LiquidityAnalyzer:
    """Analyzes market liquidity using multiple metrics and methods."""
    
    def __init__(
        self,
        lookback_days: int = 30,
        intraday_buckets: int = 24,
        depth_levels: int = 10,
        outlier_threshold: float = 3.0
    ):
        """Initialize liquidity analyzer."""
        self.lookback_days = lookback_days
        self.intraday_buckets = intraday_buckets
        self.depth_levels = depth_levels
        self.outlier_threshold = outlier_threshold
        
        # Cache for calculations
        self._cache = {}
        
    def calculate_liquidity_metrics(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        order_book_data: Optional[Dict[str, Any]] = None
    ) -> LiquidityMetrics:
        """Calculate comprehensive liquidity metrics for a symbol."""
        logger.info(f"Calculating liquidity metrics for {symbol}")
        
        metrics = LiquidityMetrics(
            symbol=symbol,
            timestamp=datetime.now()
        )
        
        # Calculate spread metrics
        self._calculate_spread_metrics(metrics, market_data, order_book_data)
        
        # Calculate volume metrics
        self._calculate_volume_metrics(metrics, market_data)
        
        # Calculate depth metrics
        if order_book_data:
            self._calculate_depth_metrics(metrics, order_book_data)
        
        # Calculate impact metrics
        self._calculate_impact_metrics(metrics, market_data)
        
        # Calculate composite score
        metrics.calculate_liquidity_score()
        
        # Calculate intraday patterns
        self._calculate_intraday_patterns(metrics, market_data)
        
        return metrics
    
    def _calculate_spread_metrics(
        self,
        metrics: LiquidityMetrics,
        market_data: pd.DataFrame,
        order_book_data: Optional[Dict[str, Any]] = None
    ):
        """Calculate various spread measures."""
        if 'bid' in market_data.columns and 'ask' in market_data.columns:
            # Quoted spread
            quoted_spreads = (market_data['ask'] - market_data['bid']) / (
                (market_data['ask'] + market_data['bid']) / 2
            ) * 10000  # Convert to basis points
            
            metrics.quoted_spread = quoted_spreads.mean()
            
            # Effective spread (if trade data available)
            if 'trade_price' in market_data.columns and 'mid_price' in market_data.columns:
                effective_spreads = 2 * abs(
                    market_data['trade_price'] - market_data['mid_price']
                ) / market_data['mid_price'] * 10000
                
                metrics.effective_spread = effective_spreads.mean()
        
        # Roll's spread estimator
        if 'close' in market_data.columns:
            metrics.roll_spread = self._calculate_roll_spread(market_data['close'])
        
        # Corwin-Schultz spread estimator
        if 'high' in market_data.columns and 'low' in market_data.columns:
            metrics.roll_spread = self._calculate_corwin_schultz_spread(
                market_data['high'], market_data['low']
            )
    
    def _calculate_roll_spread(self, prices: pd.Series) -> float:
        """Calculate Roll's implied spread from price changes."""
        returns = prices.pct_change().dropna()
        
        # Calculate first-order autocovariance
        if len(returns) > 1:
            autocov = returns.cov(returns.shift(1))
            
            # Roll's model: spread = 2 * sqrt(-cov) if cov < 0
            if autocov < 0:
                spread = 2 * np.sqrt(-autocov) * 10000  # Convert to bps
                return spread
        
        return 0.0
    
    def _calculate_corwin_schultz_spread(
        self,
        highs: pd.Series,
        lows: pd.Series
    ) -> float:
        """Calculate Corwin-Schultz high-low spread estimator."""
        # Log price ratios
        log_hl = np.log(highs / lows)
        
        # Two-day log price ratios
        log_hl_2day = log_hl.rolling(2).sum()
        
        # Beta calculation
        beta = log_hl_2day.mean()
        
        # Gamma calculation
        gamma = log_hl.pow(2).mean()
        
        # Alpha calculation
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        
        # Spread calculation
        if alpha > 0:
            spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)) * 10000
            return spread
        
        return 0.0
    
    def _calculate_volume_metrics(
        self,
        metrics: LiquidityMetrics,
        market_data: pd.DataFrame
    ):
        """Calculate volume-based liquidity metrics."""
        if 'volume' not in market_data.columns:
            return
        
        volumes = market_data['volume']
        
        # Basic volume metrics
        metrics.daily_volume = volumes.iloc[-1] if len(volumes) > 0 else 0
        metrics.average_daily_volume = volumes.mean()
        metrics.volume_volatility = volumes.std() / volumes.mean() if volumes.mean() > 0 else 0
        
        # Turnover ratio (if shares outstanding available)
        if 'shares_outstanding' in market_data.columns:
            shares = market_data['shares_outstanding'].iloc[-1]
            if shares > 0:
                metrics.turnover_ratio = metrics.average_daily_volume / shares
    
    def _calculate_depth_metrics(
        self,
        metrics: LiquidityMetrics,
        order_book_data: Dict[str, Any]
    ):
        """Calculate market depth metrics."""
        if 'bids' not in order_book_data or 'asks' not in order_book_data:
            return
        
        bids = order_book_data['bids']  # List of (price, size)
        asks = order_book_data['asks']  # List of (price, size)
        
        # Create MarketDepth object
        depth = MarketDepth(
            timestamp=datetime.now(),
            symbol=metrics.symbol,
            bid_levels=bids[:self.depth_levels],
            ask_levels=asks[:self.depth_levels],
            total_bid_depth=sum(size for _, size in bids[:self.depth_levels]),
            total_ask_depth=sum(size for _, size in asks[:self.depth_levels])
        )
        
        metrics.market_depth = depth
        metrics.depth_imbalance = depth.get_imbalance()
    
    def _calculate_impact_metrics(
        self,
        metrics: LiquidityMetrics,
        market_data: pd.DataFrame
    ):
        """Calculate price impact metrics."""
        if 'close' not in market_data.columns or 'volume' not in market_data.columns:
            return
        
        prices = market_data['close']
        volumes = market_data['volume']
        returns = prices.pct_change().dropna()
        
        # Amihud illiquidity ratio
        if len(returns) > 0 and len(volumes) > 0:
            # Daily price impact per dollar volume
            daily_illiquidity = abs(returns) / (volumes[1:] * prices[:-1])
            daily_illiquidity = daily_illiquidity.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(daily_illiquidity) > 0:
                metrics.amihud_illiquidity = daily_illiquidity.mean() * 1e6  # Scale for readability
        
        # Kyle's lambda (simplified estimation)
        if len(returns) > 10 and len(volumes) > 10:
            # Regress |returns| on signed volume
            signed_volume = volumes[1:] * np.sign(returns)
            
            # Remove outliers
            mask = (abs(returns) < returns.std() * self.outlier_threshold) & \
                   (abs(signed_volume) < signed_volume.std() * self.outlier_threshold)
            
            if mask.sum() > 10:
                try:
                    slope, _, _, _, _ = stats.linregress(
                        signed_volume[mask],
                        abs(returns)[mask]
                    )
                    metrics.kyle_lambda = abs(slope) * 1e6  # Scale for readability
                except:
                    pass
        
        # Hasbrouck's lambda (simplified)
        # This would require more sophisticated microstructure modeling
        # For now, use Kyle's lambda as proxy
        if metrics.kyle_lambda:
            metrics.hasbrouck_lambda = metrics.kyle_lambda * 0.8  # Adjustment factor
    
    def _calculate_intraday_patterns(
        self,
        metrics: LiquidityMetrics,
        market_data: pd.DataFrame
    ):
        """Calculate intraday liquidity patterns."""
        if 'timestamp' not in market_data.columns:
            return
        
        # Convert index to datetime if needed
        if not isinstance(market_data.index, pd.DatetimeIndex):
            if 'timestamp' in market_data.columns:
                market_data.index = pd.to_datetime(market_data['timestamp'])
            else:
                return
        
        # Calculate hourly patterns
        hourly_spread_pattern = {}
        hourly_volume_pattern = {}
        
        for hour in range(24):
            hour_data = market_data[market_data.index.hour == hour]
            
            if len(hour_data) > 0:
                # Spread pattern
                if 'bid' in hour_data.columns and 'ask' in hour_data.columns:
                    hourly_spreads = (hour_data['ask'] - hour_data['bid']) / (
                        (hour_data['ask'] + hour_data['bid']) / 2
                    ) * 10000
                    hourly_spread_pattern[hour] = hourly_spreads.mean()
                
                # Volume pattern
                if 'volume' in hour_data.columns:
                    hourly_volume_pattern[hour] = hour_data['volume'].mean()
        
        metrics.intraday_spread_pattern = hourly_spread_pattern
        metrics.intraday_volume_pattern = hourly_volume_pattern
    
    def analyze_intraday_liquidity(
        self,
        symbol: str,
        intraday_data: pd.DataFrame
    ) -> IntradayLiquidity:
        """Analyze intraday liquidity patterns."""
        logger.info(f"Analyzing intraday liquidity for {symbol}")
        
        # Ensure datetime index
        if not isinstance(intraday_data.index, pd.DatetimeIndex):
            if 'timestamp' in intraday_data.columns:
                intraday_data.index = pd.to_datetime(intraday_data['timestamp'])
        
        # Calculate hourly metrics
        hourly_spreads = {}
        hourly_volumes = {}
        hourly_depths = {}
        
        for hour in range(24):
            hour_data = intraday_data[intraday_data.index.hour == hour]
            
            if len(hour_data) > 0:
                # Spreads
                if 'bid' in hour_data.columns and 'ask' in hour_data.columns:
                    spread = (hour_data['ask'] - hour_data['bid']) / (
                        (hour_data['ask'] + hour_data['bid']) / 2
                    ) * 10000
                    hourly_spreads[hour] = spread.mean()
                
                # Volumes
                if 'volume' in hour_data.columns:
                    hourly_volumes[hour] = hour_data['volume'].sum()
                
                # Depths (simplified)
                if 'bid_size' in hour_data.columns and 'ask_size' in hour_data.columns:
                    hourly_depths[hour] = (
                        hour_data['bid_size'].mean(),
                        hour_data['ask_size'].mean()
                    )
        
        # Detect liquidity events
        liquidity_gaps = self._detect_liquidity_gaps(intraday_data)
        spread_spikes = self._detect_spread_spikes(intraday_data)
        volume_surges = self._detect_volume_surges(intraday_data)
        
        # Calculate statistics
        all_spreads = list(hourly_spreads.values())
        avg_spread = np.mean(all_spreads) if all_spreads else 0
        spread_volatility = np.std(all_spreads) if len(all_spreads) > 1 else 0
        
        # Volume concentration (Herfindahl index)
        total_volume = sum(hourly_volumes.values())
        if total_volume > 0:
            volume_shares = [v / total_volume for v in hourly_volumes.values()]
            volume_concentration = sum(s**2 for s in volume_shares)
        else:
            volume_concentration = 0
        
        # Composite liquidity factor
        liquidity_factor = self._calculate_liquidity_factor(
            avg_spread, spread_volatility, volume_concentration
        )
        
        return IntradayLiquidity(
            symbol=symbol,
            date=intraday_data.index[0].date() if len(intraday_data) > 0 else datetime.now().date(),
            hourly_spreads=hourly_spreads,
            hourly_volumes=hourly_volumes,
            hourly_depths=hourly_depths,
            liquidity_gaps=liquidity_gaps,
            spread_spikes=spread_spikes,
            volume_surges=volume_surges,
            avg_spread=avg_spread,
            spread_volatility=spread_volatility,
            volume_concentration=volume_concentration,
            liquidity_factor=liquidity_factor
        )
    
    def _detect_liquidity_gaps(
        self,
        data: pd.DataFrame,
        min_gap_minutes: int = 5
    ) -> List[Tuple[datetime, datetime]]:
        """Detect periods of low/no liquidity."""
        gaps = []
        
        if 'volume' not in data.columns:
            return gaps
        
        # Find periods with no volume
        no_volume = data['volume'] == 0
        
        # Group consecutive periods
        gap_groups = (no_volume != no_volume.shift()).cumsum()
        
        for group_id in gap_groups[no_volume].unique():
            gap_data = data[gap_groups == group_id]
            if len(gap_data) >= min_gap_minutes:
                gaps.append((gap_data.index[0], gap_data.index[-1]))
        
        return gaps
    
    def _detect_spread_spikes(
        self,
        data: pd.DataFrame,
        spike_threshold: float = 3.0
    ) -> List[Tuple[datetime, float]]:
        """Detect abnormal spread widening events."""
        spikes = []
        
        if 'bid' not in data.columns or 'ask' not in data.columns:
            return spikes
        
        spreads = (data['ask'] - data['bid']) / ((data['ask'] + data['bid']) / 2) * 10000
        
        # Calculate rolling statistics
        rolling_mean = spreads.rolling(window=60, min_periods=30).mean()
        rolling_std = spreads.rolling(window=60, min_periods=30).std()
        
        # Detect spikes
        z_scores = (spreads - rolling_mean) / rolling_std
        spike_mask = z_scores > spike_threshold
        
        for idx in data.index[spike_mask]:
            if not pd.isna(spreads.loc[idx]):
                spikes.append((idx, float(spreads.loc[idx])))
        
        return spikes
    
    def _detect_volume_surges(
        self,
        data: pd.DataFrame,
        surge_threshold: float = 3.0
    ) -> List[Tuple[datetime, float]]:
        """Detect abnormal volume surges."""
        surges = []
        
        if 'volume' not in data.columns:
            return surges
        
        volumes = data['volume']
        
        # Calculate rolling statistics
        rolling_mean = volumes.rolling(window=60, min_periods=30).mean()
        rolling_std = volumes.rolling(window=60, min_periods=30).std()
        
        # Detect surges
        z_scores = (volumes - rolling_mean) / rolling_std
        surge_mask = z_scores > surge_threshold
        
        for idx in data.index[surge_mask]:
            if not pd.isna(volumes.loc[idx]):
                surges.append((idx, float(volumes.loc[idx])))
        
        return surges
    
    def _calculate_liquidity_factor(
        self,
        avg_spread: float,
        spread_volatility: float,
        volume_concentration: float
    ) -> float:
        """Calculate composite liquidity factor (0-1)."""
        # Normalize components
        spread_score = max(0, 1 - avg_spread / 100)  # Lower spread is better
        volatility_score = max(0, 1 - spread_volatility / 50)  # Lower volatility is better
        concentration_score = max(0, 1 - volume_concentration)  # Lower concentration is better
        
        # Weighted average
        weights = [0.5, 0.3, 0.2]
        scores = [spread_score, volatility_score, concentration_score]
        
        liquidity_factor = sum(w * s for w, s in zip(weights, scores))
        
        return min(1.0, max(0.0, liquidity_factor))
    
    def calculate_volume_profile(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        lookback_days: Optional[int] = None
    ) -> VolumeProfile:
        """Calculate historical volume profile."""
        logger.info(f"Calculating volume profile for {symbol}")
        
        if lookback_days is None:
            lookback_days = self.lookback_days
        
        # Filter recent data
        end_date = historical_data.index[-1]
        start_date = end_date - timedelta(days=lookback_days)
        recent_data = historical_data[historical_data.index >= start_date]
        
        # Calculate hourly volumes
        hourly_volumes = {}
        for hour in range(24):
            hour_data = recent_data[recent_data.index.hour == hour]
            if 'volume' in hour_data.columns:
                hourly_volumes[hour] = hour_data['volume'].mean()
        
        total_volume = sum(hourly_volumes.values())
        avg_volume = recent_data['volume'].mean() if 'volume' in recent_data.columns else 0
        
        # Create normalized volume curve
        if total_volume > 0:
            volume_curve = np.array([
                hourly_volumes.get(h, 0) / total_volume for h in range(24)
            ])
        else:
            volume_curve = np.zeros(24)
        
        return VolumeProfile(
            symbol=symbol,
            date=end_date.date() if hasattr(end_date, 'date') else end_date,
            hourly_volumes=hourly_volumes,
            total_volume=total_volume,
            average_volume=avg_volume,
            volume_curve=volume_curve
        )
    
    def detect_liquidity_events(
        self,
        symbol: str,
        current_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        event_threshold: float = 0.2  # 20% degradation
    ) -> List[LiquidityEvent]:
        """Detect significant liquidity events."""
        logger.info(f"Detecting liquidity events for {symbol}")
        
        events = []
        
        # Calculate current and historical metrics
        current_metrics = self.calculate_liquidity_metrics(symbol, current_data)
        historical_metrics = self.calculate_liquidity_metrics(symbol, historical_data)
        
        # Check for significant degradation
        if current_metrics.liquidity_score and historical_metrics.liquidity_score:
            score_degradation = (historical_metrics.liquidity_score - 
                               current_metrics.liquidity_score) / historical_metrics.liquidity_score
            
            if score_degradation > event_threshold:
                # Determine event type and severity
                event_type = self._classify_liquidity_event(
                    current_metrics, historical_metrics
                )
                severity = self._classify_event_severity(score_degradation)
                
                event = LiquidityEvent(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    event_type=event_type,
                    severity=severity,
                    metrics_before=historical_metrics,
                    metrics_after=current_metrics,
                    duration_seconds=None,  # Would need tick data
                    impact_estimate=current_metrics.amihud_illiquidity
                )
                
                events.append(event)
        
        return events
    
    def _classify_liquidity_event(
        self,
        current: LiquidityMetrics,
        historical: LiquidityMetrics
    ) -> str:
        """Classify type of liquidity event."""
        # Check spread widening
        if current.quoted_spread and historical.quoted_spread:
            if current.quoted_spread > historical.quoted_spread * 1.5:
                return "spread_widening"
        
        # Check volume drop
        if current.daily_volume and historical.average_daily_volume:
            if current.daily_volume < historical.average_daily_volume * 0.5:
                return "volume_drop"
        
        # Check depth depletion
        if current.depth_imbalance and abs(current.depth_imbalance) > 0.5:
            return "depth_depletion"
        
        return "general_degradation"
    
    def _classify_event_severity(self, degradation: float) -> str:
        """Classify severity of liquidity event."""
        if degradation > 0.5:
            return "critical"
        elif degradation > 0.3:
            return "high"
        elif degradation > 0.2:
            return "medium"
        else:
            return "low"
    
    def calculate_liquidity_adjusted_var(
        self,
        portfolio_value: float,
        position_sizes: Dict[str, float],
        liquidity_metrics: Dict[str, LiquidityMetrics],
        confidence_level: float = 0.95,
        liquidation_horizon: int = 10  # days
    ) -> float:
        """Calculate liquidity-adjusted Value at Risk."""
        logger.info("Calculating liquidity-adjusted VaR")
        
        total_lvar = 0.0
        
        for symbol, position_size in position_sizes.items():
            if symbol not in liquidity_metrics:
                continue
            
            metrics = liquidity_metrics[symbol]
            
            # Base VaR calculation (simplified)
            position_value = position_size * portfolio_value
            volatility = 0.02  # 2% daily volatility assumption
            z_score = stats.norm.ppf(confidence_level)
            base_var = position_value * volatility * z_score * np.sqrt(liquidation_horizon)
            
            # Liquidity adjustment
            if metrics.amihud_illiquidity:
                # Estimate liquidation cost
                liquidation_cost = position_value * metrics.amihud_illiquidity * 0.01
                
                # Add spread cost
                if metrics.quoted_spread:
                    spread_cost = position_value * metrics.quoted_spread / 10000
                    liquidation_cost += spread_cost
                
                # Adjust VaR
                lvar = base_var + liquidation_cost
            else:
                lvar = base_var * 1.2  # Default penalty for unknown liquidity
            
            total_lvar += lvar
        
        return total_lvar