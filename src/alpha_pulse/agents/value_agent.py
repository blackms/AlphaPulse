"""
Warren Buffett-inspired value investing agent implementation.
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
from alpha_pulse.decorators.audit_decorators import audit_agent_signal


class ValueAgent(BaseTradeAgent):
    """
    Implements value investing strategies inspired by Warren Buffett's approach.
    Focuses on identifying high-quality businesses with strong fundamentals,
    competitive advantages, and attractive valuations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize value investing agent."""
        super().__init__("value_agent", config)
        self.quality_metrics = {
            'min_roe': self.config.get("min_roe", 0.15),  # 15% minimum ROE
            'min_roic': self.config.get("min_roic", 0.12),  # 12% minimum ROIC
            'max_debt_to_equity': self.config.get("max_debt_to_equity", 0.5),
            'min_interest_coverage': self.config.get("min_interest_coverage", 5),
            'min_operating_margin': self.config.get("min_operating_margin", 0.15)
        }
        self.valuation_metrics = {
            'max_pe_ratio': self.config.get("max_pe_ratio", 15),
            'max_pb_ratio': self.config.get("max_pb_ratio", 3),
            'min_dividend_yield': self.config.get("min_dividend_yield", 0.02),
            'min_fcf_yield': self.config.get("min_fcf_yield", 0.05)
        }
        self.holding_period = self.config.get("holding_period", 720)  # Days
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize agent with configuration."""
        await super().initialize(config)
        # Additional initialization for value strategy
        self.preferred_sectors = config.get("preferred_sectors", [
            "Consumer Staples", "Financials", "Healthcare", "Industrials"
        ])
        self.moat_indicators = config.get("moat_indicators", {
            "brand_value": 0.2,
            "market_share": 0.2,
            "switching_costs": 0.2,
            "network_effects": 0.2,
            "patents": 0.2
        })
        
    @audit_agent_signal(agent_type='value', include_market_data=True)
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Generate trading signals based on value investing criteria.
        
        Args:
            market_data: Market data including fundamentals and financial metrics
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not market_data.fundamentals:
            return signals
            
        for symbol, fundamentals in market_data.fundamentals.items():
            # Calculate business quality score
            quality_score = await self._calculate_quality_score(fundamentals)
            
            # Calculate valuation attractiveness
            valuation_score = await self._calculate_valuation_score(fundamentals)
            
            # Calculate moat strength
            moat_score = await self._calculate_moat_score(fundamentals)
            
            # Combined investment score
            investment_score = (quality_score * 0.4 + 
                              valuation_score * 0.3 + 
                              moat_score * 0.3)
            
            if investment_score >= 0.8:  # High conviction threshold
                # Generate buy signal
                signals.append(TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    confidence=investment_score,
                    timestamp=datetime.now(),
                    target_price=await self._calculate_intrinsic_value(fundamentals),
                    stop_loss=await self._calculate_margin_of_safety(fundamentals),
                    metadata={
                        "strategy": "value_investing",
                        "holding_period": self.holding_period,
                        "quality_metrics": await self._get_quality_metrics(fundamentals),
                        "moat_analysis": await self._get_moat_analysis(fundamentals)
                    }
                ))
            elif investment_score <= 0.4:  # Exit threshold
                # Generate sell signal
                signals.append(TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    confidence=1 - investment_score,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "value_investing",
                        "exit_reason": "deteriorating_fundamentals"
                    }
                ))
                
        return signals
        
    async def _calculate_quality_score(self, fundamentals: Dict) -> float:
        """Calculate business quality score."""
        scores = []
        
        # Return on Equity
        roe = fundamentals.get("roe", 0)
        if roe >= self.quality_metrics["min_roe"]:
            scores.append(min(roe / self.quality_metrics["min_roe"], 2))
            
        # Return on Invested Capital
        roic = fundamentals.get("roic", 0)
        if roic >= self.quality_metrics["min_roic"]:
            scores.append(min(roic / self.quality_metrics["min_roic"], 2))
            
        # Debt to Equity
        debt_to_equity = fundamentals.get("debt_to_equity", float('inf'))
        if debt_to_equity <= self.quality_metrics["max_debt_to_equity"]:
            scores.append(1 - (debt_to_equity / self.quality_metrics["max_debt_to_equity"]))
            
        # Interest Coverage
        interest_coverage = fundamentals.get("interest_coverage", 0)
        if interest_coverage >= self.quality_metrics["min_interest_coverage"]:
            scores.append(min(interest_coverage / self.quality_metrics["min_interest_coverage"], 2))
            
        # Operating Margin
        operating_margin = fundamentals.get("operating_margin", 0)
        if operating_margin >= self.quality_metrics["min_operating_margin"]:
            scores.append(min(operating_margin / self.quality_metrics["min_operating_margin"], 2))
            
        return np.mean(scores) if scores else 0
        
    async def _calculate_valuation_score(self, fundamentals: Dict) -> float:
        """Calculate valuation attractiveness score."""
        scores = []
        
        # P/E ratio
        pe_ratio = fundamentals.get("pe_ratio", float('inf'))
        if pe_ratio <= self.valuation_metrics["max_pe_ratio"]:
            scores.append(1 - (pe_ratio / self.valuation_metrics["max_pe_ratio"]))
            
        # P/B ratio
        pb_ratio = fundamentals.get("pb_ratio", float('inf'))
        if pb_ratio <= self.valuation_metrics["max_pb_ratio"]:
            scores.append(1 - (pb_ratio / self.valuation_metrics["max_pb_ratio"]))
            
        # Dividend Yield
        dividend_yield = fundamentals.get("dividend_yield", 0)
        if dividend_yield >= self.valuation_metrics["min_dividend_yield"]:
            scores.append(min(dividend_yield / self.valuation_metrics["min_dividend_yield"], 2))
            
        # Free Cash Flow Yield
        fcf_yield = fundamentals.get("fcf_yield", 0)
        if fcf_yield >= self.valuation_metrics["min_fcf_yield"]:
            scores.append(min(fcf_yield / self.valuation_metrics["min_fcf_yield"], 2))
            
        return np.mean(scores) if scores else 0
        
    async def _calculate_moat_score(self, fundamentals: Dict) -> float:
        """Calculate economic moat strength score."""
        moat_metrics = fundamentals.get("moat_metrics", {})
        
        score = 0
        for indicator, weight in self.moat_indicators.items():
            metric_value = moat_metrics.get(indicator, 0)
            score += metric_value * weight
            
        return score
        
    async def _calculate_intrinsic_value(self, fundamentals: Dict) -> float:
        """Calculate intrinsic value using discounted cash flow."""
        try:
            # Get required metrics
            fcf = fundamentals.get("free_cash_flow", 0)
            growth_rate = fundamentals.get("growth_rate", 0.05)
            discount_rate = self.config.get("discount_rate", 0.1)
            terminal_growth = self.config.get("terminal_growth", 0.02)
            
            # Project cash flows for 10 years
            projected_fcf = []
            for year in range(1, 11):
                projected_fcf.append(fcf * (1 + growth_rate) ** year)
                
            # Calculate terminal value
            terminal_value = (projected_fcf[-1] * (1 + terminal_growth)) / \
                           (discount_rate - terminal_growth)
                           
            # Discount all cash flows
            present_value = 0
            for year, cash_flow in enumerate(projected_fcf, 1):
                present_value += cash_flow / (1 + discount_rate) ** year
                
            # Add discounted terminal value
            present_value += terminal_value / (1 + discount_rate) ** 10
            
            # Add net cash and divide by shares outstanding
            shares_outstanding = fundamentals.get("shares_outstanding", 1)
            net_cash = fundamentals.get("net_cash", 0)
            
            intrinsic_value = (present_value + net_cash) / shares_outstanding
            current_price = fundamentals.get("price", intrinsic_value)
            
            return max(intrinsic_value, current_price * 1.2)  # At least 20% upside
            
        except (TypeError, ZeroDivisionError):
            return 0
            
    async def _calculate_margin_of_safety(self, fundamentals: Dict) -> float:
        """Calculate stop loss with margin of safety."""
        try:
            book_value = fundamentals.get("book_value_per_share", 0)
            tangible_book = fundamentals.get("tangible_book_per_share", book_value)
            current_price = fundamentals.get("price", 0)
            
            # Use the higher of tangible book value or 30% below current price
            return max(tangible_book, current_price * 0.7)
            
        except (TypeError, ZeroDivisionError):
            return 0
            
    async def _get_quality_metrics(self, fundamentals: Dict) -> Dict:
        """Get detailed quality metrics for signal metadata."""
        return {
            "roe": fundamentals.get("roe", 0),
            "roic": fundamentals.get("roic", 0),
            "operating_margin": fundamentals.get("operating_margin", 0),
            "debt_to_equity": fundamentals.get("debt_to_equity", 0),
            "interest_coverage": fundamentals.get("interest_coverage", 0)
        }
        
    async def _get_moat_analysis(self, fundamentals: Dict) -> Dict:
        """Get detailed moat analysis for signal metadata."""
        moat_metrics = fundamentals.get("moat_metrics", {})
        return {
            indicator: moat_metrics.get(indicator, 0)
            for indicator in self.moat_indicators.keys()
        }