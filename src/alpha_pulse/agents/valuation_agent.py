"""
Valuation analysis agent implementation.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats

from .interfaces import (
    BaseTradeAgent,
    MarketData,
    TradeSignal,
    SignalDirection,
    AgentMetrics
)


class ValuationAgent(BaseTradeAgent):
    """
    Implements valuation analysis strategies focusing on company valuations,
    price ratios, and fair value estimates.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize valuation analysis agent."""
        super().__init__("valuation_agent", config)
        self.valuation_methods = {
            'dcf': self.config.get("dcf_weight", 0.3),
            'multiples': self.config.get("multiples_weight", 0.3),
            'asset_based': self.config.get("asset_weight", 0.2),
            'dividend': self.config.get("dividend_weight", 0.2)
        }
        self.ratio_thresholds = {
            'pe_ratio': self.config.get("max_pe", 25),
            'pb_ratio': self.config.get("max_pb", 3),
            'ps_ratio': self.config.get("max_ps", 5),
            'ev_ebitda': self.config.get("max_ev_ebitda", 15),
            'dividend_yield': self.config.get("min_dividend_yield", 0.02)
        }
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize agent with configuration."""
        await super().initialize(config)
        self.discount_rate = config.get("discount_rate", 0.1)
        self.growth_rates = config.get("growth_rates", {
            "high_growth": 0.15,
            "moderate_growth": 0.08,
            "stable_growth": 0.03
        })
        self.valuation_history = {}
        
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Generate trading signals based on valuation analysis.
        
        Args:
            market_data: Market data including financial metrics and valuations
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not market_data.fundamentals:
            return signals
            
        for symbol, fundamentals in market_data.fundamentals.items():
            # Calculate intrinsic value using multiple methods
            dcf_value = await self._calculate_dcf_value(fundamentals)
            multiples_value = await self._calculate_multiples_value(fundamentals)
            asset_value = await self._calculate_asset_value(fundamentals)
            dividend_value = await self._calculate_dividend_value(fundamentals)
            
            # Weighted average of valuation methods
            intrinsic_value = (
                dcf_value * self.valuation_methods['dcf'] +
                multiples_value * self.valuation_methods['multiples'] +
                asset_value * self.valuation_methods['asset_based'] +
                dividend_value * self.valuation_methods['dividend']
            )
            
            # Calculate valuation metrics
            current_price = fundamentals.get("price", 0)
            if current_price > 0:
                valuation_gap = (intrinsic_value - current_price) / current_price
                
                # Store valuation history
                if symbol not in self.valuation_history:
                    self.valuation_history[symbol] = []
                self.valuation_history[symbol].append(valuation_gap)
                
                # Generate signal based on valuation gap
                signal = await self._generate_valuation_signal(
                    symbol,
                    valuation_gap,
                    fundamentals,
                    current_price
                )
                
                if signal:
                    signals.append(signal)
                    
        return signals
        
    async def _calculate_dcf_value(self, fundamentals: Dict) -> float:
        """Calculate discounted cash flow value."""
        try:
            fcf = fundamentals.get("free_cash_flow", 0)
            if fcf <= 0:
                return 0
                
            # Determine growth phase
            revenue_growth = fundamentals.get("revenue_growth", 0)
            if revenue_growth > 0.12:
                growth_rate = self.growth_rates["high_growth"]
                transition_year = 5
            elif revenue_growth > 0.06:
                growth_rate = self.growth_rates["moderate_growth"]
                transition_year = 7
            else:
                growth_rate = self.growth_rates["stable_growth"]
                transition_year = 10
                
            # Project cash flows
            terminal_growth = self.growth_rates["stable_growth"]
            projected_fcf = []
            
            for year in range(1, transition_year + 1):
                if year <= transition_year:
                    growth = growth_rate
                else:
                    growth = terminal_growth
                    
                projected_fcf.append(fcf * (1 + growth) ** year)
                
            # Calculate terminal value
            terminal_value = (projected_fcf[-1] * (1 + terminal_growth)) / \
                           (self.discount_rate - terminal_growth)
                           
            # Discount cash flows
            present_value = 0
            for year, cash_flow in enumerate(projected_fcf, 1):
                present_value += cash_flow / (1 + self.discount_rate) ** year
                
            # Add terminal value
            present_value += terminal_value / (1 + self.discount_rate) ** transition_year
            
            # Adjust for shares outstanding
            shares = fundamentals.get("shares_outstanding", 1)
            return present_value / shares
            
        except (TypeError, ZeroDivisionError):
            return 0
            
    async def _calculate_multiples_value(self, fundamentals: Dict) -> float:
        """Calculate value using comparable company multiples."""
        try:
            values = []
            
            # P/E based valuation
            eps = fundamentals.get("eps", 0)
            industry_pe = fundamentals.get("industry_pe", self.ratio_thresholds["pe_ratio"])
            if eps > 0:
                values.append(eps * industry_pe)
                
            # P/B based valuation
            bvps = fundamentals.get("book_value_per_share", 0)
            industry_pb = fundamentals.get("industry_pb", self.ratio_thresholds["pb_ratio"])
            if bvps > 0:
                values.append(bvps * industry_pb)
                
            # EV/EBITDA based valuation
            ebitda_per_share = fundamentals.get("ebitda_per_share", 0)
            industry_ev_ebitda = fundamentals.get(
                "industry_ev_ebitda",
                self.ratio_thresholds["ev_ebitda"]
            )
            if ebitda_per_share > 0:
                values.append(ebitda_per_share * industry_ev_ebitda)
                
            return np.median(values) if values else 0
            
        except (TypeError, ZeroDivisionError):
            return 0
            
    async def _calculate_asset_value(self, fundamentals: Dict) -> float:
        """Calculate asset-based value."""
        try:
            # Net asset value
            total_assets = fundamentals.get("total_assets", 0)
            total_liabilities = fundamentals.get("total_liabilities", 0)
            shares = fundamentals.get("shares_outstanding", 1)
            
            if shares > 0:
                nav = (total_assets - total_liabilities) / shares
                
                # Adjust for intangibles
                intangibles = fundamentals.get("intangible_assets", 0) / shares
                tangible_nav = nav - intangibles
                
                # Add growth premium
                growth_rate = fundamentals.get("revenue_growth", 0)
                growth_premium = max(0, growth_rate * nav)
                
                return tangible_nav + growth_premium
                
            return 0
            
        except (TypeError, ZeroDivisionError):
            return 0
            
    async def _calculate_dividend_value(self, fundamentals: Dict) -> float:
        """Calculate dividend-based value."""
        try:
            dividend = fundamentals.get("dividend_per_share", 0)
            if dividend <= 0:
                return 0
                
            # Gordon Growth Model
            growth_rate = min(
                fundamentals.get("dividend_growth", 0),
                self.growth_rates["stable_growth"]
            )
            
            if self.discount_rate > growth_rate:
                return dividend * (1 + growth_rate) / (self.discount_rate - growth_rate)
                
            return 0
            
        except (TypeError, ZeroDivisionError):
            return 0
            
    async def _generate_valuation_signal(
        self,
        symbol: str,
        valuation_gap: float,
        fundamentals: Dict,
        current_price: float
    ) -> Optional[TradeSignal]:
        """Generate trading signal based on valuation analysis."""
        # Minimum valuation gap threshold
        if abs(valuation_gap) < 0.15:  # 15% minimum gap
            return None
            
        # Calculate confidence based on valuation consistency
        valuation_history = self.valuation_history.get(symbol, [])
        if len(valuation_history) > 1:
            confidence = 1 - min(1, np.std(valuation_history))
        else:
            confidence = 0.5
            
        # Adjust confidence based on quality metrics
        quality_score = await self._calculate_quality_score(fundamentals)
        confidence *= quality_score
        
        # Generate signal if confidence is sufficient
        if confidence >= 0.6:
            direction = SignalDirection.BUY if valuation_gap > 0 else SignalDirection.SELL
            
            # Calculate target price and stop loss
            if direction == SignalDirection.BUY:
                target_price = current_price * (1 + abs(valuation_gap))
                stop_loss = current_price * 0.85  # 15% stop loss
            else:
                target_price = current_price * (1 - abs(valuation_gap))
                stop_loss = current_price * 1.15  # 15% stop loss
                
            return TradeSignal(
                agent_id=self.agent_id,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                timestamp=datetime.now(),
                target_price=target_price,
                stop_loss=stop_loss,
                metadata={
                    "strategy": "valuation",
                    "valuation_gap": valuation_gap,
                    "quality_score": quality_score,
                    "valuation_metrics": await self._get_valuation_metrics(fundamentals),
                    "historical_gaps": valuation_history[-5:] if valuation_history else []
                }
            )
            
        return None
        
    async def _calculate_quality_score(self, fundamentals: Dict) -> float:
        """Calculate company quality score for confidence adjustment."""
        scores = []
        
        # Profitability
        roe = fundamentals.get("roe", 0)
        if roe > 0:
            scores.append(min(roe / 0.15, 1))  # Cap at 15% ROE
            
        # Financial health
        current_ratio = fundamentals.get("current_ratio", 0)
        if current_ratio > 0:
            scores.append(min(current_ratio / 2, 1))  # Cap at 2x
            
        # Growth stability
        growth_rates = [
            fundamentals.get("revenue_growth", 0),
            fundamentals.get("earnings_growth", 0),
            fundamentals.get("fcf_growth", 0)
        ]
        growth_stability = 1 - np.std(growth_rates) if growth_rates else 0
        scores.append(growth_stability)
        
        return np.mean(scores) if scores else 0.5
        
    async def _get_valuation_metrics(self, fundamentals: Dict) -> Dict:
        """Get detailed valuation metrics for signal metadata."""
        return {
            "pe_ratio": fundamentals.get("pe_ratio", 0),
            "pb_ratio": fundamentals.get("pb_ratio", 0),
            "ps_ratio": fundamentals.get("ps_ratio", 0),
            "ev_ebitda": fundamentals.get("ev_ebitda", 0),
            "dividend_yield": fundamentals.get("dividend_yield", 0),
            "fcf_yield": fundamentals.get("fcf_yield", 0),
            "growth_rate": fundamentals.get("revenue_growth", 0),
            "profit_margin": fundamentals.get("net_margin", 0)
        }