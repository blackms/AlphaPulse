"""Regime detection integration module for connecting HMM to trading flow."""

from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio

from ..models.market_regime import MarketRegime
from ..services.regime_detection_service import RegimeDetectionService
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class RegimeIntegrationHub:
    """Central hub for distributing regime information throughout the system."""
    
    def __init__(self, regime_service: RegimeDetectionService):
        """Initialize the regime integration hub."""
        self.regime_service = regime_service
        self._current_regime: Optional[MarketRegime] = None
        self._regime_confidence: float = 0.0
        self._last_update: Optional[datetime] = None
        self._subscribers: List[Any] = []
        
    async def initialize(self):
        """Initialize the hub and start regime monitoring."""
        logger.info("Initializing regime integration hub")
        
        # Register for regime change callbacks
        self.regime_service.add_callback(self._on_regime_change)
        
        # Get initial regime
        await self.update_regime()
        
    async def update_regime(self):
        """Update current regime information."""
        try:
            result = await self.regime_service.detect_regime(datetime.utcnow())
            self._current_regime = result['regime']
            self._regime_confidence = result['confidence']
            self._last_update = datetime.utcnow()
            
            logger.info(
                f"Regime updated: {self._current_regime} "
                f"(confidence: {self._regime_confidence:.2%})"
            )
            
            # Notify subscribers
            await self._notify_subscribers()
            
        except Exception as e:
            logger.error(f"Failed to update regime: {e}")
    
    async def _on_regime_change(self, old_regime: MarketRegime, new_regime: MarketRegime):
        """Handle regime change events."""
        logger.warning(
            f"REGIME CHANGE DETECTED: {old_regime} → {new_regime}"
        )
        
        self._current_regime = new_regime
        await self._notify_subscribers()
    
    async def _notify_subscribers(self):
        """Notify all subscribers of regime update."""
        for subscriber in self._subscribers:
            try:
                if hasattr(subscriber, 'on_regime_update'):
                    await subscriber.on_regime_update(
                        self._current_regime,
                        self._regime_confidence
                    )
            except Exception as e:
                logger.error(f"Failed to notify subscriber {subscriber}: {e}")
    
    def subscribe(self, subscriber: Any):
        """Subscribe to regime updates."""
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)
            logger.info(f"Added regime subscriber: {type(subscriber).__name__}")
    
    def get_current_regime(self) -> Optional[MarketRegime]:
        """Get current market regime."""
        return self._current_regime
    
    def get_regime_confidence(self) -> float:
        """Get confidence in current regime detection."""
        return self._regime_confidence
    
    def get_regime_params(self) -> Dict[str, Any]:
        """Get regime-specific parameters for trading."""
        if not self._current_regime:
            return self._get_default_params()
        
        regime_params = {
            MarketRegime.BULL: {
                'leverage_multiplier': 1.5,
                'risk_tolerance': 'high',
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.5,
                'strategy_preference': 'trend_following',
                'holding_period': 'medium',
                'volatility_threshold': 0.2
            },
            MarketRegime.BEAR: {
                'leverage_multiplier': 0.5,
                'risk_tolerance': 'low',
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,
                'strategy_preference': 'mean_reversion',
                'holding_period': 'short',
                'volatility_threshold': 0.3
            },
            MarketRegime.SIDEWAYS: {
                'leverage_multiplier': 1.0,
                'risk_tolerance': 'medium',
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'strategy_preference': 'range_trading',
                'holding_period': 'short',
                'volatility_threshold': 0.15
            },
            MarketRegime.CRISIS: {
                'leverage_multiplier': 0.2,
                'risk_tolerance': 'very_low',
                'position_size_multiplier': 0.3,
                'stop_loss_multiplier': 0.5,
                'strategy_preference': 'defensive',
                'holding_period': 'very_short',
                'volatility_threshold': 0.5
            },
            MarketRegime.RECOVERY: {
                'leverage_multiplier': 1.2,
                'risk_tolerance': 'medium_high',
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 1.2,
                'strategy_preference': 'momentum',
                'holding_period': 'medium',
                'volatility_threshold': 0.25
            }
        }
        
        return regime_params.get(self._current_regime, self._get_default_params())
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters when regime is unknown."""
        return {
            'leverage_multiplier': 0.8,
            'risk_tolerance': 'low',
            'position_size_multiplier': 0.8,
            'stop_loss_multiplier': 0.9,
            'strategy_preference': 'balanced',
            'holding_period': 'short',
            'volatility_threshold': 0.2
        }


class RegimeAwareComponent:
    """Base class for components that need regime information."""
    
    def __init__(self, regime_hub: Optional[RegimeIntegrationHub] = None):
        """Initialize regime-aware component."""
        self.regime_hub = regime_hub
        self._current_regime: Optional[MarketRegime] = None
        self._regime_confidence: float = 0.0
        
        # Subscribe to regime updates
        if regime_hub:
            regime_hub.subscribe(self)
    
    async def on_regime_update(self, regime: MarketRegime, confidence: float):
        """Handle regime update notification."""
        self._current_regime = regime
        self._regime_confidence = confidence
        
        # Override in subclasses for specific behavior
        await self._handle_regime_change(regime, confidence)
    
    async def _handle_regime_change(self, regime: MarketRegime, confidence: float):
        """Handle regime change - override in subclasses."""
        pass
    
    def get_regime_adjusted_param(self, param_name: str, base_value: float) -> float:
        """Get parameter adjusted for current regime."""
        if not self.regime_hub:
            return base_value
        
        params = self.regime_hub.get_regime_params()
        multiplier = params.get(f"{param_name}_multiplier", 1.0)
        
        return base_value * multiplier
    
    def should_trade_in_regime(self, min_confidence: float = 0.6) -> bool:
        """Check if we should trade in current regime."""
        if not self._current_regime:
            return False
        
        # Don't trade if confidence is too low
        if self._regime_confidence < min_confidence:
            return False
        
        # Don't trade in crisis unless specifically designed for it
        if self._current_regime == MarketRegime.CRISIS:
            return hasattr(self, 'crisis_capable') and self.crisis_capable
        
        return True


class RegimeAwareAgent(RegimeAwareComponent):
    """Base class for regime-aware trading agents."""
    
    async def analyze_with_regime(self, market_data: Any) -> Any:
        """Analyze market data with regime context."""
        if not self.should_trade_in_regime():
            logger.info(
                f"Skipping analysis - regime: {self._current_regime}, "
                f"confidence: {self._regime_confidence:.2%}"
            )
            return None
        
        # Get regime-specific parameters
        params = self.regime_hub.get_regime_params() if self.regime_hub else {}
        
        # Perform analysis with regime context
        return await self._analyze_with_params(market_data, params)
    
    async def _analyze_with_params(self, market_data: Any, regime_params: Dict) -> Any:
        """Perform analysis with regime parameters - override in subclasses."""
        raise NotImplementedError


class RegimeAwareRiskManager(RegimeAwareComponent):
    """Risk manager that adjusts based on market regime."""
    
    def calculate_position_size_with_regime(self, base_size: float) -> float:
        """Calculate position size adjusted for regime."""
        return self.get_regime_adjusted_param('position_size', base_size)
    
    def get_leverage_limit(self) -> float:
        """Get leverage limit for current regime."""
        if not self.regime_hub:
            return 1.0
        
        params = self.regime_hub.get_regime_params()
        return params.get('leverage_multiplier', 1.0)
    
    def get_stop_loss_multiplier(self) -> float:
        """Get stop loss adjustment for current regime."""
        if not self.regime_hub:
            return 1.0
        
        params = self.regime_hub.get_regime_params()
        return params.get('stop_loss_multiplier', 1.0)


class RegimeAwarePortfolioOptimizer(RegimeAwareComponent):
    """Portfolio optimizer that adjusts based on market regime."""
    
    def get_regime_weights(self) -> Dict[str, float]:
        """Get asset class weights for current regime."""
        if not self._current_regime:
            return self._get_default_weights()
        
        regime_weights = {
            MarketRegime.BULL: {
                'stocks': 0.7,
                'bonds': 0.2,
                'commodities': 0.05,
                'cash': 0.05
            },
            MarketRegime.BEAR: {
                'stocks': 0.3,
                'bonds': 0.4,
                'commodities': 0.1,
                'cash': 0.2
            },
            MarketRegime.SIDEWAYS: {
                'stocks': 0.5,
                'bonds': 0.3,
                'commodities': 0.1,
                'cash': 0.1
            },
            MarketRegime.CRISIS: {
                'stocks': 0.1,
                'bonds': 0.3,
                'commodities': 0.2,
                'cash': 0.4
            },
            MarketRegime.RECOVERY: {
                'stocks': 0.6,
                'bonds': 0.25,
                'commodities': 0.1,
                'cash': 0.05
            }
        }
        
        return regime_weights.get(self._current_regime, self._get_default_weights())
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default weights when regime is unknown."""
        return {
            'stocks': 0.4,
            'bonds': 0.4,
            'commodities': 0.1,
            'cash': 0.1
        }
    
    async def _handle_regime_change(self, regime: MarketRegime, confidence: float):
        """Handle regime change by triggering rebalancing."""
        logger.info(
            f"Portfolio optimizer detected regime change to {regime}. "
            f"Triggering rebalancing..."
        )
        # Trigger portfolio rebalancing logic here