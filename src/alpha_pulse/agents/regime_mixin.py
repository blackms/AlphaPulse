"""
Regime Detection Mixin for Trading Agents.

Provides centralized market regime detection capabilities to trading agents.
"""
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
from loguru import logger

from alpha_pulse.services.regime_detection_service import RegimeDetectionService
from alpha_pulse.ml.regime.regime_classifier import RegimeInfo, RegimeType


class RegimeAwareMixin(ABC):
    """
    Mixin that provides regime detection capabilities to trading agents.
    
    This mixin allows agents to access the centralized regime detection service
    and adjust their strategies based on the current market regime.
    """
    
    def __init__(self, *args, regime_service: Optional[RegimeDetectionService] = None, **kwargs):
        """Initialize the regime-aware mixin."""
        super().__init__(*args, **kwargs)
        self.regime_service = regime_service
        self._last_regime_info: Optional[RegimeInfo] = None
        self._regime_cache_time: Optional[datetime] = None
        self._regime_cache_duration = 300  # 5 minutes cache
        
        # Regime-based strategy adjustments
        self.regime_strategy_adjustments = {
            RegimeType.BULL: {
                "signal_multiplier": 1.2,
                "confidence_boost": 0.1,
                "risk_tolerance": "high"
            },
            RegimeType.BEAR: {
                "signal_multiplier": 0.8,
                "confidence_boost": -0.1,
                "risk_tolerance": "low"
            },
            RegimeType.VOLATILE: {
                "signal_multiplier": 0.6,
                "confidence_boost": -0.2,
                "risk_tolerance": "very_low"
            },
            RegimeType.RANGING: {
                "signal_multiplier": 0.9,
                "confidence_boost": 0.0,
                "risk_tolerance": "moderate"
            }
        }
    
    async def get_current_regime(self) -> Optional[RegimeInfo]:
        """
        Get current market regime with caching.
        
        Returns:
            Current regime information or None if unavailable
        """
        try:
            # Check cache first
            if (self._last_regime_info and self._regime_cache_time and 
                (datetime.now() - self._regime_cache_time).total_seconds() < self._regime_cache_duration):
                return self._last_regime_info
            
            # Get from service if available
            if self.regime_service:
                regime_info = await self.regime_service.get_current_regime()
                if regime_info:
                    self._last_regime_info = regime_info
                    self._regime_cache_time = datetime.now()
                    return regime_info
            
            # Fallback to local detection if service unavailable
            logger.debug(f"Regime service unavailable for {self.agent_id}, using fallback")
            return await self._fallback_regime_detection()
            
        except Exception as e:
            logger.warning(f"Error getting regime for {self.agent_id}: {e}")
            return await self._fallback_regime_detection()
    
    @abstractmethod
    async def _fallback_regime_detection(self) -> Optional[RegimeInfo]:
        """
        Fallback regime detection when service is unavailable.
        
        Each agent should implement its own fallback regime detection.
        """
        pass
    
    def adjust_signal_for_regime(self, signal_strength: float, confidence: float, regime_info: Optional[RegimeInfo]) -> Tuple[float, float]:
        """
        Adjust signal strength and confidence based on current market regime.
        
        Args:
            signal_strength: Original signal strength (-1 to 1)
            confidence: Original confidence (0 to 1)
            regime_info: Current regime information
            
        Returns:
            Tuple of (adjusted_signal_strength, adjusted_confidence)
        """
        if not regime_info:
            return signal_strength, confidence
        
        adjustments = self.regime_strategy_adjustments.get(regime_info.regime_type, {})
        
        # Adjust signal strength
        multiplier = adjustments.get("signal_multiplier", 1.0)
        adjusted_signal = signal_strength * multiplier
        
        # Adjust confidence
        confidence_boost = adjustments.get("confidence_boost", 0.0)
        adjusted_confidence = max(0.0, min(1.0, confidence + confidence_boost))
        
        # Apply regime confidence factor
        regime_confidence_factor = regime_info.confidence
        adjusted_confidence *= regime_confidence_factor
        
        return adjusted_signal, adjusted_confidence
    
    def get_regime_strategy_context(self, regime_info: Optional[RegimeInfo]) -> Dict[str, Any]:
        """
        Get strategy context based on current regime.
        
        Args:
            regime_info: Current regime information
            
        Returns:
            Dictionary with regime-based strategy context
        """
        if not regime_info:
            return {
                "regime_type": "unknown",
                "risk_tolerance": "moderate",
                "strategy_mode": "neutral"
            }
        
        adjustments = self.regime_strategy_adjustments.get(regime_info.regime_type, {})
        
        return {
            "regime_type": regime_info.regime_type.value,
            "regime_confidence": regime_info.confidence,
            "risk_tolerance": adjustments.get("risk_tolerance", "moderate"),
            "strategy_mode": self._get_strategy_mode(regime_info.regime_type),
            "expected_duration": getattr(regime_info, 'expected_remaining_duration', None),
            "transition_probability": getattr(regime_info, 'transition_probability', None)
        }
    
    def _get_strategy_mode(self, regime_type: RegimeType) -> str:
        """Get strategy mode based on regime type."""
        strategy_modes = {
            RegimeType.BULL: "trend_following",
            RegimeType.BEAR: "defensive",
            RegimeType.VOLATILE: "mean_reversion",
            RegimeType.RANGING: "range_trading"
        }
        return strategy_modes.get(regime_type, "neutral")
    
    async def log_regime_based_decision(self, symbol: str, original_signal: float, adjusted_signal: float, regime_info: Optional[RegimeInfo]):
        """
        Log regime-based signal adjustments for analysis.
        
        Args:
            symbol: Trading symbol
            original_signal: Original signal strength
            adjusted_signal: Regime-adjusted signal strength
            regime_info: Current regime information
        """
        if regime_info and abs(original_signal - adjusted_signal) > 0.01:
            logger.info(
                f"Regime adjustment for {self.agent_id} on {symbol}: "
                f"signal {original_signal:.3f} → {adjusted_signal:.3f} "
                f"in {regime_info.regime_type.value} regime "
                f"(confidence: {regime_info.confidence:.2%})"
            ) 