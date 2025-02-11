"""
Bill Ackman-inspired activist investing agent implementation.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .interfaces import (
    BaseTradeAgent,
    MarketData,
    TradeSignal,
    SignalDirection,
    AgentMetrics
)


class ActivistAgent(BaseTradeAgent):
    """
    Implements activist investing strategies inspired by Bill Ackman's approach.
    Focuses on identifying companies with potential for structural improvements
    and corporate actions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize activist agent."""
        super().__init__("activist_agent", config)
        self.min_market_cap = self.config.get("min_market_cap", 1e9)  # $1B minimum
        self.max_market_cap = self.config.get("max_market_cap", 50e9)  # $50B maximum
        self.min_ownership_target = self.config.get("min_ownership_target", 0.05)  # 5%
        self.holding_period = self.config.get("holding_period", 360)  # Days
        self.value_metrics = {
            'pe_ratio': self.config.get("max_pe_ratio", 20),
            'pb_ratio': self.config.get("max_pb_ratio", 2),
            'debt_to_equity': self.config.get("max_debt_to_equity", 1.5),
            'fcf_yield': self.config.get("min_fcf_yield", 0.05)
        }
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize agent with configuration."""
        await super().initialize(config)
        # Additional initialization specific to activist strategy
        self.target_sectors = config.get("target_sectors", [
            "Technology", "Consumer", "Industrial", "Healthcare"
        ])
        self.corporate_action_weights = config.get("corporate_action_weights", {
            "spinoff": 0.3,
            "restructuring": 0.25,
            "management_change": 0.2,
            "strategic_alternatives": 0.15,
            "capital_return": 0.1
        })
        
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Generate trading signals based on activist investing criteria.
        
        Args:
            market_data: Market data including fundamentals and corporate actions
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not market_data.fundamentals:
            return signals
            
        for symbol, fundamentals in market_data.fundamentals.items():
            # Skip if market cap is outside our range
            market_cap = fundamentals.get("market_cap", 0)
            if not (self.min_market_cap <= market_cap <= self.max_market_cap):
                continue
                
            # Calculate activist opportunity score
            opportunity_score = await self._calculate_opportunity_score(fundamentals)
            
            if opportunity_score >= 0.7:  # High conviction threshold
                # Generate buy signal
                signals.append(TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    confidence=opportunity_score,
                    timestamp=datetime.now(),
                    target_price=await self._calculate_target_price(fundamentals),
                    stop_loss=await self._calculate_stop_loss(market_data, symbol),
                    metadata={
                        "strategy": "activist",
                        "holding_period": self.holding_period,
                        "opportunity_metrics": await self._get_opportunity_metrics(fundamentals),
                        "catalyst_probability": await self._estimate_catalyst_probability(fundamentals)
                    }
                ))
            elif opportunity_score <= 0.3:  # Exit threshold
                # Generate sell signal for existing positions
                signals.append(TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    confidence=1 - opportunity_score,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "activist",
                        "exit_reason": "deteriorating_thesis"
                    }
                ))
                
        return signals
        
    async def _calculate_opportunity_score(self, fundamentals: Dict) -> float:
        """Calculate overall activist opportunity score."""
        scores = []
        
        # Valuation score
        valuation_score = await self._calculate_valuation_score(fundamentals)
        scores.append(valuation_score * 0.3)  # 30% weight
        
        # Corporate governance score
        governance_score = await self._calculate_governance_score(fundamentals)
        scores.append(governance_score * 0.25)  # 25% weight
        
        # Operational improvement potential
        operational_score = await self._calculate_operational_score(fundamentals)
        scores.append(operational_score * 0.25)  # 25% weight
        
        # Catalyst probability
        catalyst_score = await self._estimate_catalyst_probability(fundamentals)
        scores.append(catalyst_score * 0.2)  # 20% weight
        
        return sum(scores)
        
    async def _calculate_valuation_score(self, fundamentals: Dict) -> float:
        """Calculate valuation attractiveness score."""
        scores = []
        
        # P/E ratio score
        pe_ratio = fundamentals.get("pe_ratio", float('inf'))
        if pe_ratio <= self.value_metrics["pe_ratio"]:
            scores.append(1 - (pe_ratio / self.value_metrics["pe_ratio"]))
            
        # P/B ratio score
        pb_ratio = fundamentals.get("pb_ratio", float('inf'))
        if pb_ratio <= self.value_metrics["pb_ratio"]:
            scores.append(1 - (pb_ratio / self.value_metrics["pb_ratio"]))
            
        # FCF yield score
        fcf_yield = fundamentals.get("fcf_yield", 0)
        if fcf_yield >= self.value_metrics["fcf_yield"]:
            scores.append(min(fcf_yield / self.value_metrics["fcf_yield"], 2))
            
        return np.mean(scores) if scores else 0
        
    async def _calculate_governance_score(self, fundamentals: Dict) -> float:
        """Calculate corporate governance score."""
        governance_metrics = fundamentals.get("governance", {})
        
        factors = {
            "board_independence": 0.3,
            "insider_ownership": 0.2,
            "voting_rights": 0.2,
            "shareholder_rights": 0.3
        }
        
        score = 0
        for factor, weight in factors.items():
            metric_value = governance_metrics.get(factor, 0)
            score += metric_value * weight
            
        return score
        
    async def _calculate_operational_score(self, fundamentals: Dict) -> float:
        """Calculate operational improvement potential score."""
        peer_metrics = fundamentals.get("peer_comparison", {})
        
        metrics = {
            "operating_margin": {"weight": 0.3, "higher_better": True},
            "asset_turnover": {"weight": 0.2, "higher_better": True},
            "roic": {"weight": 0.3, "higher_better": True},
            "sg&a_ratio": {"weight": 0.2, "higher_better": False}
        }
        
        score = 0
        for metric, config in metrics.items():
            company_value = fundamentals.get(metric, 0)
            peer_value = peer_metrics.get(metric, company_value)
            
            if config["higher_better"]:
                metric_score = 1 - (company_value / peer_value) if peer_value > 0 else 0
            else:
                metric_score = (company_value / peer_value) - 1 if peer_value > 0 else 0
                
            score += max(min(metric_score, 1), 0) * config["weight"]
            
        return score
        
    async def _estimate_catalyst_probability(self, fundamentals: Dict) -> float:
        """Estimate probability of activist catalysts."""
        corporate_actions = fundamentals.get("corporate_actions", {})
        
        probability = 0
        for action, weight in self.corporate_action_weights.items():
            if action in corporate_actions:
                probability += weight * corporate_actions[action]
                
        return min(probability, 1.0)
        
    async def _calculate_target_price(self, fundamentals: Dict) -> float:
        """Calculate target price based on activist thesis."""
        current_price = fundamentals.get("price", 0)
        
        # Sum of parts valuation with activist improvements
        operational_improvement = await self._calculate_operational_score(fundamentals)
        strategic_value = await self._estimate_catalyst_probability(fundamentals)
        
        # Target multiple expansion
        current_pe = fundamentals.get("pe_ratio", 15)
        peer_pe = fundamentals.get("peer_comparison", {}).get("pe_ratio", current_pe)
        multiple_expansion = min(peer_pe / current_pe, 1.5) if current_pe > 0 else 1
        
        # Calculate target price with margin of safety
        target_price = current_price * (1 + operational_improvement) * \
                      (1 + strategic_value) * multiple_expansion
        
        return target_price * 0.8  # 20% margin of safety
        
    async def _calculate_stop_loss(self, market_data: MarketData, symbol: str) -> float:
        """Calculate stop loss level based on volatility and support levels."""
        if symbol not in market_data.prices.columns:
            return 0
            
        prices = market_data.prices[symbol].dropna()
        if len(prices) < 20:
            return 0
            
        current_price = prices.iloc[-1]
        volatility = prices.pct_change().std()
        
        # Dynamic stop loss based on volatility
        stop_loss = current_price * (1 - (volatility * 2))
        
        # Find recent support levels
        support_level = prices.rolling(20).min().iloc[-1]
        
        # Use the higher of volatility-based stop and support level
        return max(stop_loss, support_level)
        
    async def _get_opportunity_metrics(self, fundamentals: Dict) -> Dict:
        """Get detailed opportunity metrics for signal metadata."""
        return {
            "valuation_score": await self._calculate_valuation_score(fundamentals),
            "governance_score": await self._calculate_governance_score(fundamentals),
            "operational_score": await self._calculate_operational_score(fundamentals),
            "catalyst_probability": await self._estimate_catalyst_probability(fundamentals),
            "target_price": await self._calculate_target_price(fundamentals)
        }