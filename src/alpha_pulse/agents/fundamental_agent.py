"""
Fundamentals-focused trading agent implementation.
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
from alpha_pulse.decorators.audit_decorators import audit_agent_signal


class FundamentalAgent(BaseTradeAgent):
    """
    Implements fundamental analysis strategies focusing on financial statements,
    economic indicators, and quantitative metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize fundamental analysis agent."""
        super().__init__("fundamental_agent", config)
        self.financial_metrics = {
            'revenue_growth': self.config.get("min_revenue_growth", 0.1),
            'gross_margin': self.config.get("min_gross_margin", 0.3),
            'ebitda_margin': self.config.get("min_ebitda_margin", 0.15),
            'net_margin': self.config.get("min_net_margin", 0.1),
            'current_ratio': self.config.get("min_current_ratio", 1.5),
            'quick_ratio': self.config.get("min_quick_ratio", 1.0),
            'asset_turnover': self.config.get("min_asset_turnover", 0.5)
        }
        self.macro_weights = {
            'gdp_growth': 0.2,
            'inflation': 0.15,
            'interest_rates': 0.15,
            'sector_performance': 0.25,
            'market_sentiment': 0.25
        }
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize agent with configuration."""
        await super().initialize(config)
        # Additional initialization for fundamental analysis
        self.analysis_timeframes = config.get("analysis_timeframes", {
            "short_term": 90,    # 3 months
            "medium_term": 180,  # 6 months
            "long_term": 360     # 1 year
        })
        self.sector_correlations = {}
        self.historical_zscore_threshold = config.get("zscore_threshold", 2.0)
        
    @audit_agent_signal(agent_type='fundamental', include_market_data=True)
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Generate trading signals based on fundamental analysis.
        
        Args:
            market_data: Market data including financial statements and economic indicators
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not market_data.fundamentals:
            return signals
            
        # Update sector correlations
        await self._update_sector_correlations(market_data)
        
        for symbol, fundamentals in market_data.fundamentals.items():
            # Calculate fundamental scores
            financial_score = await self._analyze_financials(fundamentals)
            macro_score = await self._analyze_macro_environment(market_data)
            sector_score = await self._analyze_sector_dynamics(symbol, market_data)
            
            # Calculate combined score with time series analysis
            historical_score = await self._analyze_historical_trends(symbol, market_data)
            
            # Weighted combination of scores
            total_score = (
                financial_score * 0.4 +
                macro_score * 0.2 +
                sector_score * 0.2 +
                historical_score * 0.2
            )
            
            # Generate signals based on score
            if total_score >= 0.7:  # Strong buy signal
                signals.append(TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    confidence=total_score,
                    timestamp=datetime.now(),
                    target_price=await self._calculate_price_target(fundamentals),
                    stop_loss=await self._calculate_risk_level(symbol, market_data),
                    metadata={
                        "strategy": "fundamental",
                        "financial_metrics": await self._get_financial_metrics(fundamentals),
                        "macro_indicators": await self._get_macro_indicators(market_data),
                        "sector_analysis": await self._get_sector_analysis(symbol, market_data)
                    }
                ))
            elif total_score <= 0.3:  # Strong sell signal
                signals.append(TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    confidence=1 - total_score,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "fundamental",
                        "exit_reason": "deteriorating_fundamentals"
                    }
                ))
                
        return signals
        
    async def _analyze_financials(self, fundamentals: Dict) -> float:
        """Analyze company financial metrics."""
        scores = []
        
        # Revenue growth analysis
        revenue_growth = fundamentals.get("revenue_growth", 0)
        if revenue_growth >= self.financial_metrics["revenue_growth"]:
            scores.append(min(revenue_growth / self.financial_metrics["revenue_growth"], 2))
            
        # Margin analysis
        for margin_type in ["gross_margin", "ebitda_margin", "net_margin"]:
            margin = fundamentals.get(margin_type, 0)
            if margin >= self.financial_metrics[margin_type]:
                scores.append(min(margin / self.financial_metrics[margin_type], 2))
                
        # Liquidity analysis
        for ratio_type in ["current_ratio", "quick_ratio"]:
            ratio = fundamentals.get(ratio_type, 0)
            if ratio >= self.financial_metrics[ratio_type]:
                scores.append(min(ratio / self.financial_metrics[ratio_type], 2))
                
        # Efficiency analysis
        asset_turnover = fundamentals.get("asset_turnover", 0)
        if asset_turnover >= self.financial_metrics["asset_turnover"]:
            scores.append(min(asset_turnover / self.financial_metrics["asset_turnover"], 2))
            
        return np.mean(scores) if scores else 0
        
    async def _analyze_macro_environment(self, market_data: MarketData) -> float:
        """Analyze macroeconomic indicators."""
        macro_data = market_data.fundamentals.get("macro_indicators", {})
        score = 0
        
        for indicator, weight in self.macro_weights.items():
            if indicator in macro_data:
                indicator_score = await self._score_macro_indicator(
                    indicator,
                    macro_data[indicator]
                )
                score += indicator_score * weight
                
        return score
        
    async def _score_macro_indicator(self, indicator: str, value: float) -> float:
        """Score individual macro indicators."""
        if indicator == "gdp_growth":
            return min(max(value / 0.03, 0), 1)  # Normalize around 3% growth
        elif indicator == "inflation":
            return 1 - min(max(value / 0.03, 0), 1)  # Lower is better, target 2-3%
        elif indicator == "interest_rates":
            return 1 - min(max(value / 0.05, 0), 1)  # Lower is better
        elif indicator == "sector_performance":
            return min(max(value / 0.1, 0), 1)  # Normalize around 10% return
        elif indicator == "market_sentiment":
            return min(max((value + 1) / 2, 0), 1)  # Convert [-1, 1] to [0, 1]
        return 0
        
    async def _analyze_sector_dynamics(self, symbol: str, market_data: MarketData) -> float:
        """Analyze sector dynamics and relative performance."""
        sector_data = market_data.fundamentals.get("sector_data", {})
        symbol_sector = sector_data.get(symbol, {}).get("sector")
        
        if not symbol_sector:
            return 0
            
        scores = []
        
        # Sector growth rate
        sector_growth = sector_data.get("growth_rate", 0)
        scores.append(min(max(sector_growth / 0.1, 0), 1))  # Normalize around 10% growth
        
        # Sector profitability
        sector_margin = sector_data.get("profit_margin", 0)
        scores.append(min(max(sector_margin / 0.15, 0), 1))  # Normalize around 15% margin
        
        # Sector momentum
        sector_momentum = sector_data.get("momentum", 0)
        scores.append(min(max((sector_momentum + 1) / 2, 0), 1))  # Convert [-1, 1] to [0, 1]
        
        # Relative strength
        if symbol in self.sector_correlations:
            relative_strength = self.sector_correlations[symbol]
            scores.append(min(max((relative_strength + 1) / 2, 0), 1))
            
        return np.mean(scores) if scores else 0
        
    async def _analyze_historical_trends(self, symbol: str, market_data: MarketData) -> float:
        """Analyze historical trends using time series analysis."""
        if symbol not in market_data.prices.columns:
            return 0
            
        prices = market_data.prices[symbol].dropna()
        if len(prices) < 60:  # Minimum 60 days of data
            return 0
            
        scores = []
        
        # Calculate z-score of recent performance
        returns = prices.pct_change()
        recent_zscore = stats.zscore(returns)[-1]
        scores.append(1 / (1 + np.exp(-recent_zscore)))  # Sigmoid transformation
        
        # Trend analysis
        for timeframe in self.analysis_timeframes.values():
            if len(prices) >= timeframe:
                trend = self._calculate_trend_strength(prices[-timeframe:])
                scores.append(trend)
                
        return np.mean(scores) if scores else 0
        
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using linear regression."""
        try:
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            trend_strength = r_value ** 2 * np.sign(slope)
            return (trend_strength + 1) / 2  # Convert [-1, 1] to [0, 1]
        except:
            return 0
            
    async def _update_sector_correlations(self, market_data: MarketData) -> None:
        """Update sector correlation metrics."""
        if not market_data.prices.empty:
            returns = market_data.prices.pct_change()
            sector_returns = returns.mean(axis=1)  # Use mean as sector proxy
            
            for symbol in returns.columns:
                if len(returns[symbol].dropna()) > 0:
                    correlation = returns[symbol].corr(sector_returns)
                    self.sector_correlations[symbol] = correlation
                    
    async def _calculate_price_target(self, fundamentals: Dict) -> float:
        """Calculate price target using multiple valuation methods."""
        try:
            # Earnings-based valuation
            eps = fundamentals.get("eps", 0)
            industry_pe = fundamentals.get("industry_pe", 15)
            earnings_target = eps * industry_pe
            
            # Book value-based valuation
            book_value = fundamentals.get("book_value_per_share", 0)
            industry_pb = fundamentals.get("industry_pb", 2)
            book_target = book_value * industry_pb
            
            # Cash flow-based valuation
            fcf_per_share = fundamentals.get("fcf_per_share", 0)
            industry_pfcf = fundamentals.get("industry_pfcf", 12)
            fcf_target = fcf_per_share * industry_pfcf
            
            # Weight the different valuation methods
            price_target = (
                earnings_target * 0.4 +
                book_target * 0.3 +
                fcf_target * 0.3
            )
            
            current_price = fundamentals.get("price", price_target)
            return max(price_target, current_price * 1.15)  # At least 15% upside
            
        except (TypeError, ZeroDivisionError):
            return 0
            
    async def _calculate_risk_level(self, symbol: str, market_data: MarketData) -> float:
        """Calculate risk-based stop loss level."""
        try:
            prices = market_data.prices[symbol].dropna()
            if len(prices) < 20:
                return 0
                
            current_price = prices.iloc[-1]
            volatility = prices.pct_change().std()
            
            # Dynamic stop loss based on volatility and trend
            trend_strength = self._calculate_trend_strength(prices[-60:])  # 60-day trend
            volatility_adjustment = 2 + (1 - trend_strength)  # More buffer in weak trends
            
            stop_loss = current_price * (1 - (volatility * volatility_adjustment))
            
            # Find support levels
            support_level = prices.rolling(20).min().iloc[-1]
            
            return max(stop_loss, support_level)
            
        except (KeyError, IndexError):
            return 0
            
    async def _get_financial_metrics(self, fundamentals: Dict) -> Dict:
        """Get detailed financial metrics for signal metadata."""
        return {
            metric: fundamentals.get(metric, 0)
            for metric in self.financial_metrics.keys()
        }
        
    async def _get_macro_indicators(self, market_data: MarketData) -> Dict:
        """Get macro indicators for signal metadata."""
        macro_data = market_data.fundamentals.get("macro_indicators", {})
        return {
            indicator: macro_data.get(indicator, 0)
            for indicator in self.macro_weights.keys()
        }
        
    async def _get_sector_analysis(self, symbol: str, market_data: MarketData) -> Dict:
        """Get sector analysis for signal metadata."""
        sector_data = market_data.fundamentals.get("sector_data", {})
        return {
            "sector": sector_data.get(symbol, {}).get("sector", "unknown"),
            "sector_growth": sector_data.get("growth_rate", 0),
            "sector_margin": sector_data.get("profit_margin", 0),
            "sector_momentum": sector_data.get("momentum", 0),
            "relative_strength": self.sector_correlations.get(symbol, 0)
        }