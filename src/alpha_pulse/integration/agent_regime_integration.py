"""Examples of integrating regime detection into trading agents."""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from ..agents.base_agent import BaseAgent
from ..models.market_regime import MarketRegime
from ..models.signals import Signal, SignalType
from ..integration.regime_integration import RegimeAwareAgent
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class RegimeAwareTechnicalAgent(RegimeAwareAgent):
    """Technical agent that adapts strategies based on market regime."""
    
    def __init__(self, config: Dict, regime_hub=None):
        """Initialize regime-aware technical agent."""
        super().__init__(regime_hub)
        self.config = config
        self.crisis_capable = False  # Can trade in crisis
        
    async def analyze(self, market_data: pd.DataFrame) -> List[Signal]:
        """Analyze market with regime awareness."""
        if not self.should_trade_in_regime():
            logger.info(f"Technical agent skipping - inappropriate regime")
            return []
        
        # Get regime-specific parameters
        regime_params = self.regime_hub.get_regime_params() if self.regime_hub else {}
        
        # Select strategy based on regime
        strategy = regime_params.get('strategy_preference', 'balanced')
        
        if strategy == 'trend_following':
            return await self._trend_following_analysis(market_data, regime_params)
        elif strategy == 'mean_reversion':
            return await self._mean_reversion_analysis(market_data, regime_params)
        elif strategy == 'range_trading':
            return await self._range_trading_analysis(market_data, regime_params)
        elif strategy == 'defensive':
            return await self._defensive_analysis(market_data, regime_params)
        elif strategy == 'momentum':
            return await self._momentum_analysis(market_data, regime_params)
        else:
            return await self._balanced_analysis(market_data, regime_params)
    
    async def _trend_following_analysis(
        self, 
        market_data: pd.DataFrame, 
        regime_params: Dict
    ) -> List[Signal]:
        """Trend following strategy for bull markets."""
        signals = []
        
        # Calculate trend indicators with regime adjustments
        ma_short = market_data['close'].rolling(20).mean()
        ma_long = market_data['close'].rolling(50).mean()
        
        # More aggressive in bull market
        position_size = self.get_regime_adjusted_param('position_size', 1.0)
        
        # Generate signals
        if ma_short.iloc[-1] > ma_long.iloc[-1]:
            signal = Signal(
                symbol=market_data['symbol'].iloc[0],
                signal_type=SignalType.BUY,
                strength=0.8 * self._regime_confidence,
                source="regime_aware_technical",
                metadata={
                    'regime': self._current_regime.value,
                    'regime_confidence': self._regime_confidence,
                    'strategy': 'trend_following',
                    'position_size_multiplier': position_size
                }
            )
            signals.append(signal)
        
        return signals
    
    async def _mean_reversion_analysis(
        self,
        market_data: pd.DataFrame,
        regime_params: Dict
    ) -> List[Signal]:
        """Mean reversion strategy for bear/sideways markets."""
        signals = []
        
        # Calculate mean reversion indicators
        bb_period = 20
        bb_std = 2.0 * regime_params.get('volatility_threshold', 0.2)
        
        sma = market_data['close'].rolling(bb_period).mean()
        std = market_data['close'].rolling(bb_period).std()
        upper_band = sma + (bb_std * std)
        lower_band = sma - (bb_std * std)
        
        current_price = market_data['close'].iloc[-1]
        
        # More conservative in bear market
        position_size = self.get_regime_adjusted_param('position_size', 0.7)
        
        # Generate signals
        if current_price < lower_band.iloc[-1]:
            signal = Signal(
                symbol=market_data['symbol'].iloc[0],
                signal_type=SignalType.BUY,
                strength=0.7 * self._regime_confidence,
                source="regime_aware_technical",
                metadata={
                    'regime': self._current_regime.value,
                    'regime_confidence': self._regime_confidence,
                    'strategy': 'mean_reversion',
                    'position_size_multiplier': position_size
                }
            )
            signals.append(signal)
        elif current_price > upper_band.iloc[-1]:
            signal = Signal(
                symbol=market_data['symbol'].iloc[0],
                signal_type=SignalType.SELL,
                strength=0.7 * self._regime_confidence,
                source="regime_aware_technical",
                metadata={
                    'regime': self._current_regime.value,
                    'regime_confidence': self._regime_confidence,
                    'strategy': 'mean_reversion',
                    'position_size_multiplier': position_size
                }
            )
            signals.append(signal)
        
        return signals
    
    async def _defensive_analysis(
        self,
        market_data: pd.DataFrame,
        regime_params: Dict
    ) -> List[Signal]:
        """Defensive strategy for crisis regime."""
        signals = []
        
        # In crisis, focus on capital preservation
        # Only trade strongest signals with tight stops
        
        rsi = self._calculate_rsi(market_data['close'])
        
        # Very conservative position sizing
        position_size = self.get_regime_adjusted_param('position_size', 0.3)
        
        # Only trade extreme oversold conditions
        if rsi.iloc[-1] < 20:
            signal = Signal(
                symbol=market_data['symbol'].iloc[0],
                signal_type=SignalType.BUY,
                strength=0.5 * self._regime_confidence,
                source="regime_aware_technical",
                metadata={
                    'regime': self._current_regime.value,
                    'regime_confidence': self._regime_confidence,
                    'strategy': 'defensive',
                    'position_size_multiplier': position_size,
                    'stop_loss_tight': True,
                    'take_profit_quick': True
                }
            )
            signals.append(signal)
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class RegimeAwareFundamentalAgent(RegimeAwareAgent):
    """Fundamental agent that adjusts analysis based on market regime."""
    
    async def analyze(self, market_data: Dict, fundamental_data: Dict) -> List[Signal]:
        """Analyze fundamentals with regime context."""
        if not self.should_trade_in_regime(min_confidence=0.7):
            return []
        
        signals = []
        regime_params = self.regime_hub.get_regime_params() if self.regime_hub else {}
        
        # Adjust valuation thresholds based on regime
        if self._current_regime == MarketRegime.BULL:
            # More lenient valuations in bull market
            pe_threshold = 25
            pb_threshold = 3.5
        elif self._current_regime == MarketRegime.BEAR:
            # Stricter valuations in bear market
            pe_threshold = 15
            pb_threshold = 1.5
        elif self._current_regime == MarketRegime.CRISIS:
            # Very strict valuations in crisis
            pe_threshold = 10
            pb_threshold = 1.0
        else:
            # Normal valuations
            pe_threshold = 20
            pb_threshold = 2.5
        
        # Analyze with regime-adjusted thresholds
        if fundamental_data.get('pe_ratio', float('inf')) < pe_threshold:
            if fundamental_data.get('pb_ratio', float('inf')) < pb_threshold:
                signal = Signal(
                    symbol=market_data['symbol'],
                    signal_type=SignalType.BUY,
                    strength=0.7 * self._regime_confidence,
                    source="regime_aware_fundamental",
                    metadata={
                        'regime': self._current_regime.value,
                        'regime_confidence': self._regime_confidence,
                        'pe_ratio': fundamental_data.get('pe_ratio'),
                        'pb_ratio': fundamental_data.get('pb_ratio'),
                        'adjusted_thresholds': True
                    }
                )
                signals.append(signal)
        
        return signals


class RegimeAwareSentimentAgent(RegimeAwareAgent):
    """Sentiment agent that weighs sentiment differently based on regime."""
    
    async def analyze(self, market_data: Dict, sentiment_data: Dict) -> List[Signal]:
        """Analyze sentiment with regime context."""
        if not self.should_trade_in_regime():
            return []
        
        signals = []
        
        # Sentiment matters more in certain regimes
        sentiment_weight = {
            MarketRegime.BULL: 0.7,      # Sentiment less important in strong trends
            MarketRegime.BEAR: 1.2,      # Sentiment more important in fear
            MarketRegime.SIDEWAYS: 1.0,   # Normal weight
            MarketRegime.CRISIS: 1.5,     # Sentiment crucial in crisis
            MarketRegime.RECOVERY: 1.3    # Sentiment important for confirmation
        }.get(self._current_regime, 1.0)
        
        sentiment_score = sentiment_data.get('composite_score', 0)
        adjusted_score = sentiment_score * sentiment_weight
        
        # Generate signals based on adjusted sentiment
        if adjusted_score > 0.7:
            signal_strength = min(adjusted_score, 1.0) * self._regime_confidence
            signal = Signal(
                symbol=market_data['symbol'],
                signal_type=SignalType.BUY,
                strength=signal_strength,
                source="regime_aware_sentiment",
                metadata={
                    'regime': self._current_regime.value,
                    'regime_confidence': self._regime_confidence,
                    'sentiment_weight': sentiment_weight,
                    'raw_sentiment': sentiment_score,
                    'adjusted_sentiment': adjusted_score
                }
            )
            signals.append(signal)
        elif adjusted_score < -0.7:
            signal_strength = min(abs(adjusted_score), 1.0) * self._regime_confidence
            signal = Signal(
                symbol=market_data['symbol'],
                signal_type=SignalType.SELL,
                strength=signal_strength,
                source="regime_aware_sentiment",
                metadata={
                    'regime': self._current_regime.value,
                    'regime_confidence': self._regime_confidence,
                    'sentiment_weight': sentiment_weight,
                    'raw_sentiment': sentiment_score,
                    'adjusted_sentiment': adjusted_score
                }
            )
            signals.append(signal)
        
        return signals


class RegimeAwareValueAgent(RegimeAwareAgent):
    """Value agent that adjusts time horizons based on regime."""
    
    async def analyze(self, market_data: Dict, value_metrics: Dict) -> List[Signal]:
        """Analyze value with regime-adjusted time horizons."""
        if not self.should_trade_in_regime(min_confidence=0.65):
            return []
        
        signals = []
        regime_params = self.regime_hub.get_regime_params() if self.regime_hub else {}
        
        # Adjust holding period based on regime
        holding_period = regime_params.get('holding_period', 'medium')
        
        # Time horizon affects which metrics we prioritize
        if holding_period == 'very_short':
            # Crisis mode - focus on immediate liquidity and safety
            if value_metrics.get('cash_ratio', 0) > 2.0:
                signal = self._create_value_signal(
                    market_data['symbol'],
                    SignalType.BUY,
                    0.5,
                    'high_liquidity_crisis'
                )
                signals.append(signal)
                
        elif holding_period == 'short':
            # Focus on near-term catalysts
            if value_metrics.get('earnings_momentum', 0) > 0.2:
                signal = self._create_value_signal(
                    market_data['symbol'],
                    SignalType.BUY,
                    0.7,
                    'short_term_momentum'
                )
                signals.append(signal)
                
        elif holding_period == 'medium':
            # Traditional value metrics
            if value_metrics.get('intrinsic_value_discount', 0) > 0.3:
                signal = self._create_value_signal(
                    market_data['symbol'],
                    SignalType.BUY,
                    0.8,
                    'medium_term_value'
                )
                signals.append(signal)
                
        else:  # long
            # Deep value opportunities
            if value_metrics.get('deep_value_score', 0) > 0.8:
                signal = self._create_value_signal(
                    market_data['symbol'],
                    SignalType.BUY,
                    0.9,
                    'long_term_value'
                )
                signals.append(signal)
        
        return signals
    
    def _create_value_signal(
        self, 
        symbol: str, 
        signal_type: SignalType,
        base_strength: float,
        strategy: str
    ) -> Signal:
        """Create value signal with regime metadata."""
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=base_strength * self._regime_confidence,
            source="regime_aware_value",
            metadata={
                'regime': self._current_regime.value,
                'regime_confidence': self._regime_confidence,
                'value_strategy': strategy,
                'holding_period': self.regime_hub.get_regime_params().get('holding_period')
            }
        )