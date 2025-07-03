"""
Regime indicator calculations for market regime detection.

Provides calculations for volatility, momentum, liquidity, and sentiment indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
from ta import momentum, volatility, trend

from alpha_pulse.models.market_regime import RegimeIndicator, RegimeIndicatorType

logger = logging.getLogger(__name__)


class RegimeIndicatorCalculator:
    """Calculates various regime indicators from market data."""
    
    def __init__(self):
        """Initialize indicator calculator."""
        self.cache = {}
        
    def calculate_volatility_indicators(
        self,
        market_data: pd.DataFrame
    ) -> Dict[str, RegimeIndicator]:
        """Calculate volatility-based regime indicators."""
        indicators = {}
        
        # VIX level (if available, otherwise use realized vol)
        if 'VIX' in market_data.columns:
            vix_current = market_data['VIX'].iloc[-1]
            vix_ma = market_data['VIX'].rolling(20).mean().iloc[-1]
            
            indicators['vix_level'] = RegimeIndicator(
                name='vix_level',
                indicator_type=RegimeIndicatorType.VOLATILITY,
                value=vix_current,
                normalized_value=self._normalize_vix(vix_current),
                signal=self._vix_to_signal(vix_current),
                weight=1.5  # Higher weight for VIX
            )
            
            # VIX term structure
            if 'VIX9D' in market_data.columns and 'VIX' in market_data.columns:
                term_structure = market_data['VIX9D'].iloc[-1] / market_data['VIX'].iloc[-1] - 1
                indicators['vix_term_structure'] = RegimeIndicator(
                    name='vix_term_structure',
                    indicator_type=RegimeIndicatorType.VOLATILITY,
                    value=term_structure,
                    normalized_value=(term_structure + 0.2) / 0.4,  # Normalize around typical range
                    signal=-term_structure * 5,  # Inverted relationship
                    weight=1.0
                )
        
        # Realized volatility
        returns = market_data.pct_change().dropna()
        if len(returns) > 0:
            # 20-day realized volatility
            realized_vol = returns.iloc[-20:].std() * np.sqrt(252)
            
            indicators['realized_vol'] = RegimeIndicator(
                name='realized_vol',
                indicator_type=RegimeIndicatorType.VOLATILITY,
                value=realized_vol,
                normalized_value=min(realized_vol / 0.40, 1.0),  # Cap at 40% vol
                signal=-realized_vol * 2.5,  # Higher vol = bearish
                weight=1.2
            )
            
            # GARCH volatility estimate (simplified)
            garch_vol = self._estimate_garch_volatility(returns)
            indicators['garch_vol'] = RegimeIndicator(
                name='garch_vol',
                indicator_type=RegimeIndicatorType.VOLATILITY,
                value=garch_vol,
                normalized_value=min(garch_vol / 0.40, 1.0),
                signal=-garch_vol * 2.5,
                weight=0.8
            )
            
            # Volatility of volatility
            vol_series = returns.rolling(20).std() * np.sqrt(252)
            vol_of_vol = vol_series.iloc[-60:].std()
            
            indicators['vol_of_vol'] = RegimeIndicator(
                name='vol_of_vol',
                indicator_type=RegimeIndicatorType.VOLATILITY,
                value=vol_of_vol,
                normalized_value=min(vol_of_vol / 0.10, 1.0),
                signal=-vol_of_vol * 10,  # High vol of vol = instability
                weight=0.6
            )
        
        return indicators
    
    def calculate_momentum_indicators(
        self,
        market_data: pd.DataFrame
    ) -> Dict[str, RegimeIndicator]:
        """Calculate momentum-based regime indicators."""
        indicators = {}
        
        # Use SPY or first column as market proxy
        market_col = 'SPY' if 'SPY' in market_data.columns else market_data.columns[0]
        prices = market_data[market_col]
        
        # Price momentum at different horizons
        for period, name in [(20, '1m'), (60, '3m'), (120, '6m')]:
            if len(prices) > period:
                momentum_value = (prices.iloc[-1] / prices.iloc[-period] - 1)
                
                indicators[f'momentum_{name}'] = RegimeIndicator(
                    name=f'momentum_{name}',
                    indicator_type=RegimeIndicatorType.MOMENTUM,
                    value=momentum_value,
                    normalized_value=(momentum_value + 0.20) / 0.40,  # Normalize to 0-1
                    signal=self._momentum_to_signal(momentum_value),
                    weight=1.0 if name == '3m' else 0.8  # 3-month momentum most important
                )
        
        # Trend strength (ADX-like)
        if len(prices) > 20:
            trend_strength = self._calculate_trend_strength(prices)
            indicators['trend_strength'] = RegimeIndicator(
                name='trend_strength',
                indicator_type=RegimeIndicatorType.MOMENTUM,
                value=trend_strength,
                normalized_value=trend_strength / 100,
                signal=(trend_strength - 25) / 25,  # >25 = trending
                weight=0.7
            )
        
        # Moving average positioning
        if len(prices) > 200:
            ma_50 = prices.rolling(50).mean().iloc[-1]
            ma_200 = prices.rolling(200).mean().iloc[-1]
            
            # Price relative to MAs
            price_to_ma50 = (prices.iloc[-1] / ma_50 - 1)
            price_to_ma200 = (prices.iloc[-1] / ma_200 - 1)
            
            indicators['ma_positioning'] = RegimeIndicator(
                name='ma_positioning',
                indicator_type=RegimeIndicatorType.MOMENTUM,
                value=(price_to_ma50 + price_to_ma200) / 2,
                normalized_value=(price_to_ma50 + price_to_ma200 + 0.20) / 0.40,
                signal=(price_to_ma50 + price_to_ma200) * 2.5,
                weight=0.6
            )
        
        # Momentum factor performance
        if 'MOM' in market_data.columns:
            mom_perf = (market_data['MOM'].iloc[-1] / market_data['MOM'].iloc[-20] - 1)
            indicators['momentum_factor'] = RegimeIndicator(
                name='momentum_factor',
                indicator_type=RegimeIndicatorType.MOMENTUM,
                value=mom_perf,
                normalized_value=(mom_perf + 0.05) / 0.10,
                signal=mom_perf * 20,
                weight=0.5
            )
        
        return indicators
    
    def calculate_liquidity_indicators(
        self,
        market_data: pd.DataFrame
    ) -> Dict[str, RegimeIndicator]:
        """Calculate liquidity-based regime indicators."""
        indicators = {}
        
        # Bid-ask spread (if available)
        if 'bid_ask_spread' in market_data.columns:
            spread = market_data['bid_ask_spread'].iloc[-1]
            spread_ma = market_data['bid_ask_spread'].rolling(20).mean().iloc[-1]
            spread_ratio = spread / spread_ma if spread_ma > 0 else 1.0
            
            indicators['bid_ask_spread'] = RegimeIndicator(
                name='bid_ask_spread',
                indicator_type=RegimeIndicatorType.LIQUIDITY,
                value=spread,
                normalized_value=min(spread / 0.002, 1.0),  # 20bps normalized
                signal=-(spread_ratio - 1) * 5,  # Higher spread = bearish
                weight=0.8
            )
        
        # Volume analysis
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            volume_ma = volume.rolling(20).mean()
            volume_ratio = (volume.iloc[-1] / volume_ma.iloc[-1]) if volume_ma.iloc[-1] > 0 else 1.0
            
            indicators['volume_ratio'] = RegimeIndicator(
                name='volume_ratio',
                indicator_type=RegimeIndicatorType.LIQUIDITY,
                value=volume_ratio,
                normalized_value=min(volume_ratio / 2.0, 1.0),
                signal=(volume_ratio - 1) * 2,  # High volume can be bullish or bearish
                weight=0.6
            )
            
            # Volume trend
            volume_trend = (volume_ma.iloc[-1] / volume_ma.iloc[-20] - 1) if len(volume_ma) > 20 else 0
            indicators['volume_trend'] = RegimeIndicator(
                name='volume_trend',
                indicator_type=RegimeIndicatorType.LIQUIDITY,
                value=volume_trend,
                normalized_value=(volume_trend + 0.50) / 1.0,
                signal=volume_trend * 3,
                weight=0.5
            )
        
        # Market depth proxy (using high-low range)
        if 'high' in market_data.columns and 'low' in market_data.columns:
            hl_range = (market_data['high'] - market_data['low']) / market_data['close']
            avg_range = hl_range.rolling(20).mean().iloc[-1]
            
            indicators['price_range'] = RegimeIndicator(
                name='price_range',
                indicator_type=RegimeIndicatorType.LIQUIDITY,
                value=avg_range,
                normalized_value=min(avg_range / 0.02, 1.0),  # 2% daily range
                signal=-avg_range * 50,  # High range = low liquidity
                weight=0.7
            )
        
        # Composite liquidity score
        liquidity_signals = [ind.signal for ind in indicators.values()]
        if liquidity_signals:
            composite_liquidity = np.mean(liquidity_signals)
            indicators['liquidity_composite'] = RegimeIndicator(
                name='liquidity_composite',
                indicator_type=RegimeIndicatorType.LIQUIDITY,
                value=composite_liquidity,
                normalized_value=(composite_liquidity + 1) / 2,
                signal=composite_liquidity,
                weight=1.0
            )
        
        return indicators
    
    def calculate_sentiment_indicators(
        self,
        market_data: pd.DataFrame,
        additional_data: Optional[Dict[str, float]] = None
    ) -> Dict[str, RegimeIndicator]:
        """Calculate sentiment-based regime indicators."""
        indicators = {}
        
        # Put/Call ratio
        if additional_data and 'put_call_ratio' in additional_data:
            pc_ratio = additional_data['put_call_ratio']
            indicators['put_call_ratio'] = RegimeIndicator(
                name='put_call_ratio',
                indicator_type=RegimeIndicatorType.SENTIMENT,
                value=pc_ratio,
                normalized_value=min(pc_ratio / 1.5, 1.0),
                signal=-(pc_ratio - 1.0) * 3,  # High P/C = bearish sentiment
                weight=0.8
            )
        
        # VIX premium (if spot and futures available)
        if 'VIX' in market_data.columns and 'VIX_FUT' in market_data.columns:
            vix_premium = market_data['VIX_FUT'].iloc[-1] / market_data['VIX'].iloc[-1] - 1
            indicators['vix_premium'] = RegimeIndicator(
                name='vix_premium',
                indicator_type=RegimeIndicatorType.SENTIMENT,
                value=vix_premium,
                normalized_value=(vix_premium + 0.10) / 0.20,
                signal=-vix_premium * 10,  # Contango = complacency
                weight=0.7
            )
        
        # News sentiment (if available)
        if additional_data and 'news_sentiment' in additional_data:
            news_sent = additional_data['news_sentiment']
            indicators['news_sentiment'] = RegimeIndicator(
                name='news_sentiment',
                indicator_type=RegimeIndicatorType.SENTIMENT,
                value=news_sent,
                normalized_value=(news_sent + 1) / 2,
                signal=news_sent,
                weight=0.6
            )
        
        # Social sentiment (if available)
        if additional_data and 'social_sentiment' in additional_data:
            social_sent = additional_data['social_sentiment']
            indicators['social_sentiment'] = RegimeIndicator(
                name='social_sentiment',
                indicator_type=RegimeIndicatorType.SENTIMENT,
                value=social_sent,
                normalized_value=(social_sent + 1) / 2,
                signal=social_sent * 0.8,  # Less weight than news
                weight=0.5
            )
        
        # Fear & Greed proxy (combination of indicators)
        fear_greed = self._calculate_fear_greed_index(market_data, indicators)
        indicators['fear_greed'] = RegimeIndicator(
            name='fear_greed',
            indicator_type=RegimeIndicatorType.SENTIMENT,
            value=fear_greed,
            normalized_value=fear_greed / 100,
            signal=(fear_greed - 50) / 25,
            weight=0.9
        )
        
        # Composite sentiment
        sentiment_signals = [ind.signal * ind.weight for ind in indicators.values()]
        weights = [ind.weight for ind in indicators.values()]
        
        if sentiment_signals and sum(weights) > 0:
            composite_sentiment = sum(sentiment_signals) / sum(weights)
            indicators['sentiment_composite'] = RegimeIndicator(
                name='sentiment_composite',
                indicator_type=RegimeIndicatorType.SENTIMENT,
                value=composite_sentiment,
                normalized_value=(composite_sentiment + 1) / 2,
                signal=composite_sentiment,
                weight=1.0
            )
        
        return indicators
    
    def calculate_technical_indicators(
        self,
        market_data: pd.DataFrame
    ) -> Dict[str, RegimeIndicator]:
        """Calculate technical analysis indicators."""
        indicators = {}
        
        # Use market proxy
        market_col = 'SPY' if 'SPY' in market_data.columns else market_data.columns[0]
        prices = market_data[market_col]
        
        if len(prices) < 20:
            return indicators
        
        # RSI
        rsi = momentum.RSIIndicator(prices, window=14).rsi().iloc[-1]
        indicators['rsi'] = RegimeIndicator(
            name='rsi',
            indicator_type=RegimeIndicatorType.TECHNICAL,
            value=rsi,
            normalized_value=rsi / 100,
            signal=(rsi - 50) / 25,
            weight=0.6
        )
        
        # Bollinger Bands position
        bb = volatility.BollingerBands(prices, window=20, window_dev=2)
        bb_position = (prices.iloc[-1] - bb.bollinger_lband().iloc[-1]) / \
                     (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])
        
        indicators['bb_position'] = RegimeIndicator(
            name='bb_position',
            indicator_type=RegimeIndicatorType.TECHNICAL,
            value=bb_position,
            normalized_value=bb_position,
            signal=(bb_position - 0.5) * 2,
            weight=0.5
        )
        
        # MACD
        macd = trend.MACD(prices)
        macd_signal = macd.macd_signal().iloc[-1]
        macd_diff = macd.macd_diff().iloc[-1]
        
        if not np.isnan(macd_diff):
            indicators['macd_signal'] = RegimeIndicator(
                name='macd_signal',
                indicator_type=RegimeIndicatorType.TECHNICAL,
                value=macd_diff,
                normalized_value=np.tanh(macd_diff * 100),  # Normalize using tanh
                signal=np.tanh(macd_diff * 100),
                weight=0.7
            )
        
        # Market breadth (if available)
        if 'advance_decline' in market_data.columns:
            ad_line = market_data['advance_decline'].iloc[-1]
            ad_ma = market_data['advance_decline'].rolling(20).mean().iloc[-1]
            
            indicators['market_breadth'] = RegimeIndicator(
                name='market_breadth',
                indicator_type=RegimeIndicatorType.TECHNICAL,
                value=ad_line,
                normalized_value=(ad_line + 100) / 200,  # Assume -100 to +100 range
                signal=(ad_line - ad_ma) / 50,
                weight=0.8
            )
        
        return indicators
    
    def _normalize_vix(self, vix: float) -> float:
        """Normalize VIX to 0-1 scale."""
        # VIX typically ranges from 10-80
        return np.clip((vix - 10) / 70, 0, 1)
    
    def _vix_to_signal(self, vix: float) -> float:
        """Convert VIX level to regime signal."""
        # VIX < 20: Bullish (+1 to 0)
        # VIX 20-30: Neutral (0 to -0.5)
        # VIX > 30: Bearish (-0.5 to -1)
        
        if vix < 20:
            return 1 - (vix / 20)
        elif vix < 30:
            return -(vix - 20) / 20
        else:
            return -0.5 - min((vix - 30) / 40, 0.5)
    
    def _momentum_to_signal(self, momentum: float) -> float:
        """Convert momentum to regime signal."""
        # Use tanh to bound signal between -1 and 1
        return np.tanh(momentum * 5)
    
    def _estimate_garch_volatility(self, returns: pd.Series) -> float:
        """Simple GARCH(1,1) volatility estimate."""
        # Simplified EWMA as proxy for GARCH
        ewm_var = returns.ewm(span=20, adjust=False).var()
        return np.sqrt(ewm_var.iloc[-1] * 252) if not ewm_var.empty else 0.15
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength (simplified ADX)."""
        if len(prices) < 20:
            return 0.0
        
        # Use price momentum dispersion as proxy
        returns = prices.pct_change().dropna()
        
        # Positive and negative returns
        pos_returns = returns[returns > 0].sum()
        neg_returns = abs(returns[returns < 0].sum())
        
        # Directional movement
        if pos_returns + neg_returns > 0:
            directional_index = abs(pos_returns - neg_returns) / (pos_returns + neg_returns)
            return directional_index * 100
        
        return 0.0
    
    def _calculate_fear_greed_index(
        self,
        market_data: pd.DataFrame,
        current_indicators: Dict[str, RegimeIndicator]
    ) -> float:
        """Calculate Fear & Greed index (0-100 scale)."""
        components = []
        
        # 1. Market momentum (S&P 500 vs 125-day MA)
        if 'momentum_3m' in current_indicators:
            mom_score = (current_indicators['momentum_3m'].signal + 1) * 50
            components.append(mom_score)
        
        # 2. Stock price strength (new highs vs new lows)
        # Simplified: use recent performance
        market_col = 'SPY' if 'SPY' in market_data.columns else market_data.columns[0]
        if len(market_data) > 252:
            yearly_high = market_data[market_col].iloc[-252:].max()
            current = market_data[market_col].iloc[-1]
            strength_score = (current / yearly_high) * 100
            components.append(strength_score)
        
        # 3. Stock price breadth (advancing vs declining)
        # Use volatility as proxy
        if 'realized_vol' in current_indicators:
            vol_score = 100 - (current_indicators['realized_vol'].normalized_value * 100)
            components.append(vol_score)
        
        # 4. Put/Call ratio
        if 'put_call_ratio' in current_indicators:
            pc_score = (1 - current_indicators['put_call_ratio'].normalized_value) * 100
            components.append(pc_score)
        
        # 5. Market volatility (VIX)
        if 'vix_level' in current_indicators:
            vix_score = 100 - (current_indicators['vix_level'].normalized_value * 100)
            components.append(vix_score)
        
        # 6. Safe haven demand (bonds vs stocks)
        # Simplified: use sentiment
        if 'sentiment_composite' in current_indicators:
            sent_score = (current_indicators['sentiment_composite'].signal + 1) * 50
            components.append(sent_score)
        
        # Average all components
        if components:
            return np.mean(components)
        else:
            return 50.0  # Neutral if no data
    
    def _get_indicator_weights(self) -> Dict[RegimeIndicatorType, float]:
        """Get default indicator type weights."""
        return {
            RegimeIndicatorType.VOLATILITY: 0.30,
            RegimeIndicatorType.MOMENTUM: 0.25,
            RegimeIndicatorType.LIQUIDITY: 0.20,
            RegimeIndicatorType.SENTIMENT: 0.15,
            RegimeIndicatorType.TECHNICAL: 0.10
        }