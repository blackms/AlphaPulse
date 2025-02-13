"""
Self-supervised fundamental analysis agent implementation.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

from ..interfaces import MarketData, TradeSignal, SignalDirection
from .base import BaseSelfSupervisedAgent


class SelfSupervisedFundamentalAgent(BaseSelfSupervisedAgent):
    """
    Self-supervised fundamental analysis agent that can optimize its own parameters
    and adapt to changing market conditions.
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize self-supervised fundamental agent."""
        super().__init__(agent_id, config)
        
        # Fundamental analysis parameters
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
        
        # Self-supervision parameters
        self._metric_performance = {
            metric: [] for metric in self.financial_metrics.keys()
        }
        self._macro_performance = {
            indicator: [] for indicator in self.macro_weights.keys()
        }
        self._prediction_history = []
        self._market_cycle = "expansion"  # expansion, peak, contraction, trough
        
    async def optimize(self) -> None:
        """
        Optimize fundamental analysis parameters based on performance metrics.
        This includes:
        1. Adjusting financial metric thresholds based on predictive power
        2. Optimizing macro weights based on market cycle
        3. Adapting valuation methods based on sector performance
        """
        await super().optimize()
        
        try:
            # Optimize financial metric thresholds
            for metric, history in self._metric_performance.items():
                if len(history) >= 50:  # Need sufficient history
                    recent_history = history[-50:]
                    
                    # Calculate predictive accuracy
                    true_positives = sum(1 for h in recent_history if h['predicted'] and h['actual'])
                    false_positives = sum(1 for h in recent_history if h['predicted'] and not h['actual'])
                    
                    if true_positives + false_positives > 0:
                        precision = true_positives / (true_positives + false_positives)
                        
                        # Adjust threshold based on precision
                        if precision < 0.6:  # Below target precision
                            self.financial_metrics[metric] *= 1.1  # Increase threshold
                        elif precision > 0.8:  # Above target precision
                            self.financial_metrics[metric] *= 0.9  # Decrease threshold
                            
            logger.info(f"Updated financial metric thresholds: {self.financial_metrics}")
            
            # Optimize macro weights based on market cycle
            if self._market_cycle == "expansion":
                # Focus more on growth metrics
                self.macro_weights['gdp_growth'] = 0.25
                self.macro_weights['sector_performance'] = 0.3
                self.macro_weights['market_sentiment'] = 0.2
                self.macro_weights['inflation'] = 0.15
                self.macro_weights['interest_rates'] = 0.1
            elif self._market_cycle == "contraction":
                # Focus more on defensive metrics
                self.macro_weights['interest_rates'] = 0.25
                self.macro_weights['inflation'] = 0.25
                self.macro_weights['market_sentiment'] = 0.2
                self.macro_weights['gdp_growth'] = 0.15
                self.macro_weights['sector_performance'] = 0.15
                
            logger.info(f"Updated macro weights for {self._market_cycle} cycle: {self.macro_weights}")
            
            # Store optimization result
            self._optimization_history.append({
                'timestamp': datetime.now(),
                'market_cycle': self._market_cycle,
                'financial_metrics': self.financial_metrics.copy(),
                'macro_weights': self.macro_weights.copy()
            })
            
        except Exception as e:
            logger.error(f"Error in fundamental agent optimization: {str(e)}")
            raise
            
    async def self_evaluate(self) -> Dict[str, float]:
        """
        Evaluate agent's performance and market adaptation.
        Returns performance metrics specific to fundamental analysis.
        """
        metrics = await super().self_evaluate()
        
        try:
            # Calculate metric-specific performance
            for metric, history in self._metric_performance.items():
                if history:
                    recent_history = history[-50:]  # Look at last 50 predictions
                    metrics[f"{metric}_accuracy"] = np.mean([
                        1 if h['predicted'] == h['actual'] else 0
                        for h in recent_history
                    ])
                    
            # Calculate macro indicator performance
            for indicator, history in self._macro_performance.items():
                if history:
                    recent_history = history[-50:]
                    metrics[f"{indicator}_accuracy"] = np.mean([
                        1 if h['predicted'] == h['actual'] else 0
                        for h in recent_history
                    ])
                    
            # Calculate adaptation metrics
            if len(self._optimization_history) >= 2:
                last_opt = self._optimization_history[-1]
                prev_opt = self._optimization_history[-2]
                
                # Calculate threshold change magnitude
                threshold_changes = [
                    abs(last_opt['financial_metrics'][k] - prev_opt['financial_metrics'][k])
                    for k in last_opt['financial_metrics']
                ]
                metrics['threshold_adaptation'] = np.mean(threshold_changes)
                
                # Calculate weight change magnitude
                weight_changes = [
                    abs(last_opt['macro_weights'][k] - prev_opt['macro_weights'][k])
                    for k in last_opt['macro_weights']
                ]
                metrics['weight_adaptation'] = np.mean(weight_changes)
                
            logger.debug(f"Fundamental agent self-evaluation metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error in fundamental agent self-evaluation: {str(e)}")
            raise
            
        return metrics
        
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """Generate trading signals with self-supervision capabilities."""
        try:
            # Update market cycle
            self._market_cycle = await self._detect_market_cycle(market_data)
            logger.info(f"Current market cycle: {self._market_cycle}")
            
            # Generate signals using parent implementation
            signals = await super().generate_signals(market_data)
            
            # Track metric performance
            if market_data.fundamentals:
                for symbol, fundamentals in market_data.fundamentals.items():
                    for metric, threshold in self.financial_metrics.items():
                        value = fundamentals.get(metric, 0)
                        prediction = value >= threshold
                        
                        self._metric_performance[metric].append({
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'predicted': prediction,
                            'actual': None,  # Will be updated in next iteration
                            'value': value
                        })
                        
            # Update previous predictions with actual outcomes
            current_prices = {
                symbol: prices.iloc[-1]
                for symbol, prices in market_data.prices.items()
            }
            
            for metric, history in self._metric_performance.items():
                for record in history:
                    if record['actual'] is None and record['symbol'] in current_prices:
                        symbol = record['symbol']
                        if len(market_data.prices[symbol]) > 1:
                            price_change = (
                                current_prices[symbol] -
                                market_data.prices[symbol].iloc[-2]
                            )
                            record['actual'] = price_change > 0
                            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals in fundamental agent: {str(e)}")
            raise
            
    async def _detect_market_cycle(self, market_data: MarketData) -> str:
        """
        Detect the current market cycle using fundamental indicators.
        Returns: "expansion", "peak", "contraction", or "trough"
        """
        try:
            if not market_data.fundamentals:
                return self._market_cycle
                
            macro = market_data.fundamentals.get("macro_indicators", {})
            
            # Economic indicators
            gdp_growth = macro.get("gdp_growth", 0)
            inflation = macro.get("inflation", 0)
            interest_rates = macro.get("interest_rates", 0)
            unemployment = macro.get("unemployment", 0)
            
            # Determine cycle phase
            if gdp_growth > 0.02 and inflation < 0.03:
                return "expansion"
            elif gdp_growth > 0.02 and inflation > 0.03:
                return "peak"
            elif gdp_growth < 0.01 and unemployment > 0.05:
                return "contraction"
            elif gdp_growth < 0 and interest_rates < 0.02:
                return "trough"
                
            return self._market_cycle
            
        except Exception as e:
            logger.error(f"Error detecting market cycle: {str(e)}")
            return self._market_cycle