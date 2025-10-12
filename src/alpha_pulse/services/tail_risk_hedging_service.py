"""
Tail Risk Hedging Service for AlphaPulse.

This service coordinates tail risk analysis and hedging recommendations
for the portfolio, integrating with the hedge manager and portfolio optimizer.
"""
import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
from enum import Enum

from alpha_pulse.portfolio.data_models import PortfolioData
from alpha_pulse.services.exceptions import ServiceConfigurationError

try:  # pragma: no cover - optional dependency
    from alpha_pulse.monitoring.base import AlertManager, AlertLevel
except ImportError:  # pragma: no cover
    AlertManager = Any  # type: ignore

    class AlertLevel(Enum):
        WARNING = "warning"


@dataclass
class TailRiskAnalysis:
    """Results of tail risk analysis."""
    tail_risk_score: float
    confidence: float
    risk_factors: Dict[str, float]
    recommended_hedges: List[Dict[str, Any]]
    timestamp: datetime


class TailRiskHedgingService:
    """Service for monitoring and managing tail risk hedging."""
    
    def __init__(
        self,
        hedge_manager: Any,
        alert_manager: AlertManager,
        config: Dict[str, Any],
        portfolio_provider: Optional[Callable[[], Awaitable[Optional[PortfolioData]]]] = None,
    ):
        """
        Initialize tail risk hedging service.
        
        Args:
            hedge_manager: Hedge manager instance
            alert_manager: Alert manager for notifications
            config: Service configuration
        """
        self.hedge_manager = hedge_manager
        self.alert_manager = alert_manager
        self.config = config
        self.portfolio_provider = portfolio_provider
        
        # Configuration
        self.enabled = config.get('enabled', True)
        self.tail_risk_threshold = config.get('threshold', 0.05)  # 5% default
        self.check_interval = config.get('check_interval_minutes', 60)
        self.max_hedge_cost = config.get('max_hedge_cost', 0.02)  # 2% of portfolio
        
        # State
        self.monitoring_task = None
        self.last_analysis: Optional[TailRiskAnalysis] = None
        self.active_hedges: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized TailRiskHedgingService with threshold: {self.tail_risk_threshold}")
    
    async def start(self):
        """Start the tail risk monitoring service."""
        if not self.enabled:
            logger.info("Tail risk hedging service is disabled")
            return
            
        logger.info("Starting tail risk hedging service")
        self.monitoring_task = asyncio.create_task(self._monitor_tail_risk())
    
    async def stop(self):
        """Stop the tail risk monitoring service."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Tail risk hedging service stopped")
    
    async def _monitor_tail_risk(self):
        """Continuously monitor portfolio tail risk."""
        while True:
            try:
                # Wait for next check interval
                await asyncio.sleep(self.check_interval * 60)
                
                # Analyze current tail risk
                analysis = await self.analyze_portfolio_tail_risk()
                
                if analysis and analysis.tail_risk_score > self.tail_risk_threshold:
                    # Send alert for elevated tail risk
                    await self.alert_manager.send_alert(
                        alert_id=f"tail_risk_{datetime.now().timestamp()}",
                        title="Elevated Tail Risk Detected",
                        message=f"Tail risk score: {analysis.tail_risk_score:.2%} (threshold: {self.tail_risk_threshold:.2%})",
                        level=AlertLevel.WARNING,
                        context={
                            "tail_risk_score": analysis.tail_risk_score,
                            "risk_factors": analysis.risk_factors,
                            "recommended_hedges": len(analysis.recommended_hedges)
                        }
                    )
                    
            except Exception as e:
                logger.error(f"Error in tail risk monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def analyze_portfolio_tail_risk(
        self,
        portfolio_data: Optional[PortfolioData] = None
    ) -> Optional[TailRiskAnalysis]:
        """
        Analyze portfolio tail risk.
        
        Args:
            portfolio_data: Current portfolio data (will fetch if not provided)
            
        Returns:
            Tail risk analysis results
        """
        try:
            if not portfolio_data:
                portfolio_data = await self._resolve_portfolio_data()
                if not portfolio_data:
                    raise ServiceConfigurationError(
                        "TailRiskHedgingService requires portfolio data but none was provided."
                    )
            
            # Prepare portfolio for hedge analysis
            portfolio_dict = {
                'positions': [
                    {
                        'symbol': getattr(pos, 'symbol', None) or getattr(pos, 'asset_id', ''),
                        'quantity': float(getattr(pos, 'quantity', 0)),
                        'current_price': float(getattr(pos, 'current_price', 0) or 0),
                        'value': float(getattr(pos, 'quantity', 0)) * float(getattr(pos, 'current_price', 0) or 0)
                    }
                    for pos in portfolio_data.positions
                ],
                'total_value': float(portfolio_data.total_value),
                'cash': float(portfolio_data.cash_balance)
            }
            
            # Analyze tail risk using hedge manager
            risk_assessment = self.hedge_manager.risk_analyzer.analyze_tail_risk(
                portfolio_dict['positions']
            )
            
            # Get hedge recommendations if risk is elevated
            recommended_hedges = []
            if risk_assessment['tail_risk_score'] > self.tail_risk_threshold:
                hedge_recommendations = self.hedge_manager.recommend_hedges(
                    portfolio_dict,
                    risk_tolerance='moderate',
                    max_cost=self.max_hedge_cost * portfolio_dict['total_value']
                )
                recommended_hedges = hedge_recommendations.get('hedges', [])
            
            # Create analysis result
            analysis = TailRiskAnalysis(
                tail_risk_score=risk_assessment['tail_risk_score'],
                confidence=risk_assessment.get('confidence', 0.8),
                risk_factors=risk_assessment.get('risk_factors', {}),
                recommended_hedges=recommended_hedges,
                timestamp=datetime.now()
            )
            
            self.last_analysis = analysis
            return analysis
            
        except ServiceConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Error analyzing tail risk: {e}")
            return None
    
    async def get_hedge_recommendations(
        self,
        portfolio_data: PortfolioData,
        risk_tolerance: str = 'moderate'
    ) -> Dict[str, Any]:
        """
        Get hedge recommendations for the portfolio.
        
        Args:
            portfolio_data: Current portfolio data
            risk_tolerance: Risk tolerance level
            
        Returns:
            Hedge recommendations
        """
        try:
            # Convert portfolio data for hedge manager
            portfolio_dict = {
                'positions': [
                    {
                        'symbol': getattr(pos, 'symbol', None) or getattr(pos, 'asset_id', ''),
                        'quantity': float(getattr(pos, 'quantity', 0)),
                        'current_price': float(getattr(pos, 'current_price', 0) or 0),
                        'value': float(getattr(pos, 'quantity', 0)) * float(getattr(pos, 'current_price', 0) or 0)
                    }
                    for pos in portfolio_data.positions
                ],
                'total_value': float(portfolio_data.total_value),
                'cash': float(portfolio_data.cash_balance)
            }
            
            # Get recommendations from hedge manager
            recommendations = self.hedge_manager.recommend_hedges(
                portfolio_dict,
                risk_tolerance=risk_tolerance,
                max_cost=self.max_hedge_cost * portfolio_dict['total_value']
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting hedge recommendations: {e}")
            return {'hedges': [], 'error': str(e)}
    
    async def execute_hedge(self, hedge_trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a hedge trade.
        
        Args:
            hedge_trade: Hedge trade details
            
        Returns:
            Execution result
        """
        try:
            # In production, this would interface with the execution system
            result = {
                'hedge_id': f"hedge_{datetime.now().timestamp()}",
                'trade': hedge_trade,
                'status': 'simulated',
                'timestamp': datetime.now().isoformat()
            }
            
            # Track active hedge
            self.active_hedges.append(result)
            
            logger.info(f"Executed hedge trade: {hedge_trade}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing hedge: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_active_hedges(self) -> List[Dict[str, Any]]:
        """Get list of active hedges."""
        return self.active_hedges
    
    def get_last_analysis(self) -> Optional[TailRiskAnalysis]:
        """Get the last tail risk analysis."""
        return self.last_analysis

    async def _resolve_portfolio_data(self) -> Optional[PortfolioData]:
        """Retrieve portfolio data from configured provider."""
        if not self.portfolio_provider:
            logger.error("Portfolio provider not configured for TailRiskHedgingService")
            return None
        try:
            return await self.portfolio_provider()
        except Exception as exc:
            logger.error(f"Failed to retrieve portfolio snapshot: {exc}")
            return None
    
    async def calculate_hedge_effectiveness(self) -> Dict[str, float]:
        """
        Calculate effectiveness of active hedges.
        
        Returns:
            Hedge effectiveness metrics
        """
        if not self.active_hedges:
            return {'total_hedges': 0, 'average_effectiveness': 0.0}
        
        # In production, calculate actual P&L of hedges vs portfolio losses
        # This is a placeholder implementation
        effectiveness_scores = []
        for hedge in self.active_hedges:
            # Simulate effectiveness score
            effectiveness_scores.append(0.75)  # 75% effective placeholder
        
        return {
            'total_hedges': len(self.active_hedges),
            'average_effectiveness': sum(effectiveness_scores) / len(effectiveness_scores),
            'total_hedge_value': sum(h.get('trade', {}).get('value', 0) for h in self.active_hedges)
        }
