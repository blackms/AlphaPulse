"""Performance metrics utilities for ensemble evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class PerformanceMetrics:
    """Calculate various performance metrics for trading signals."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, 
                               target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < target_return]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
            
        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
        
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """Calculate maximum drawdown and duration."""
        if len(equity_curve) == 0:
            return 0.0, 0, 0
            
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Find start of drawdown
        start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                start_idx = i
                break
                
        # Find recovery point
        recovery_idx = len(drawdown) - 1
        for i in range(max_dd_idx, len(drawdown)):
            if equity_curve[i] >= running_max[max_dd_idx]:
                recovery_idx = i
                break
                
        duration = recovery_idx - start_idx
        
        return float(max_dd), start_idx, duration
        
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) == 0 or len(equity_curve) == 0:
            return 0.0
            
        annual_return = np.mean(returns) * 252
        max_dd, _, _ = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
            
        return annual_return / abs(max_dd)
        
    @staticmethod
    def calculate_win_rate(returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns)."""
        if len(returns) == 0:
            return 0.0
            
        return np.sum(returns > 0) / len(returns)
        
    @staticmethod
    def calculate_profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(returns) == 0:
            return 1.0
            
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 1.0
            
        return gross_profit / gross_loss
        
    @staticmethod
    def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
            
        active_returns = returns - benchmark_returns
        
        if np.std(active_returns) == 0:
            return 0.0
            
        return np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
        
    @staticmethod
    def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        if len(returns) == 0:
            return 1.0
            
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if np.sum(losses) == 0:
            return float('inf') if np.sum(gains) > 0 else 1.0
            
        return np.sum(gains) / np.sum(losses)
        
    @staticmethod
    def calculate_tail_ratio(returns: np.ndarray, percentile: float = 5.0) -> float:
        """Calculate tail ratio (ratio of percentile gains to losses)."""
        if len(returns) == 0:
            return 1.0
            
        right_tail = np.percentile(returns, 100 - percentile)
        left_tail = np.percentile(returns, percentile)
        
        if abs(left_tail) < 1e-8:
            return float('inf') if right_tail > 0 else 1.0
            
        return abs(right_tail / left_tail)
        
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
            
        return np.percentile(returns, (1 - confidence) * 100)
        
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
            
        var = PerformanceMetrics.calculate_var(returns, confidence)
        return np.mean(returns[returns <= var])
        
    @staticmethod
    def calculate_recovery_factor(returns: np.ndarray, equity_curve: np.ndarray) -> float:
        """Calculate recovery factor (net profit / max drawdown)."""
        if len(returns) == 0 or len(equity_curve) == 0:
            return 0.0
            
        net_profit = equity_curve[-1] - equity_curve[0] if len(equity_curve) > 0 else 0
        max_dd, _, _ = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return float('inf') if net_profit > 0 else 0.0
            
        return net_profit / abs(max_dd)
        
    @staticmethod
    def calculate_ulcer_index(equity_curve: np.ndarray, lookback: int = 14) -> float:
        """Calculate Ulcer Index (measures downside volatility)."""
        if len(equity_curve) < lookback:
            return 0.0
            
        # Calculate percentage drawdown from rolling maximum
        drawdowns = []
        
        for i in range(lookback, len(equity_curve)):
            window = equity_curve[i-lookback:i+1]
            max_value = np.max(window)
            current_dd = (window[-1] - max_value) / max_value * 100
            drawdowns.append(current_dd)
            
        if not drawdowns:
            return 0.0
            
        # Root mean square of drawdowns
        return np.sqrt(np.mean(np.square(drawdowns)))
        
    @staticmethod
    def calculate_comprehensive_metrics(returns: np.ndarray, 
                                      equity_curve: Optional[np.ndarray] = None,
                                      benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive set of performance metrics."""
        if equity_curve is None:
            equity_curve = np.cumprod(1 + returns)
            
        metrics = {
            'total_return': float(np.prod(1 + returns) - 1) if len(returns) > 0 else 0.0,
            'annual_return': float(np.mean(returns) * 252) if len(returns) > 0 else 0.0,
            'volatility': float(np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0.0,
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(returns),
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(returns, equity_curve),
            'win_rate': PerformanceMetrics.calculate_win_rate(returns),
            'profit_factor': PerformanceMetrics.calculate_profit_factor(returns),
            'omega_ratio': PerformanceMetrics.calculate_omega_ratio(returns),
            'tail_ratio': PerformanceMetrics.calculate_tail_ratio(returns),
            'var_95': PerformanceMetrics.calculate_var(returns, 0.95),
            'cvar_95': PerformanceMetrics.calculate_cvar(returns, 0.95),
            'recovery_factor': PerformanceMetrics.calculate_recovery_factor(returns, equity_curve),
            'ulcer_index': PerformanceMetrics.calculate_ulcer_index(equity_curve)
        }
        
        # Add drawdown metrics
        max_dd, dd_start, dd_duration = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        metrics['max_drawdown'] = float(max_dd)
        metrics['max_drawdown_duration'] = int(dd_duration)
        
        # Add benchmark comparison if available
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            metrics['information_ratio'] = PerformanceMetrics.calculate_information_ratio(
                returns, benchmark_returns
            )
            metrics['tracking_error'] = float(np.std(returns - benchmark_returns) * np.sqrt(252))
            metrics['alpha'] = float(np.mean(returns - benchmark_returns) * 252)
            
        return metrics


class SignalMetrics:
    """Metrics specific to trading signals."""
    
    @staticmethod
    def calculate_signal_accuracy(signals: np.ndarray, outcomes: np.ndarray) -> float:
        """Calculate directional accuracy of signals."""
        if len(signals) == 0 or len(outcomes) == 0:
            return 0.0
            
        correct = np.sum(np.sign(signals) == np.sign(outcomes))
        return correct / len(signals)
        
    @staticmethod
    def calculate_signal_correlation(signals: np.ndarray, outcomes: np.ndarray) -> float:
        """Calculate correlation between signals and outcomes."""
        if len(signals) < 2 or len(outcomes) < 2:
            return 0.0
            
        return float(np.corrcoef(signals, outcomes)[0, 1])
        
    @staticmethod
    def calculate_signal_stability(signals: np.ndarray, window: int = 20) -> float:
        """Calculate signal stability (inverse of flip rate)."""
        if len(signals) < 2:
            return 1.0
            
        # Count sign changes
        sign_changes = np.sum(np.diff(np.sign(signals)) != 0)
        
        # Normalize by number of possible changes
        max_changes = len(signals) - 1
        flip_rate = sign_changes / max_changes if max_changes > 0 else 0
        
        return 1.0 - flip_rate
        
    @staticmethod
    def calculate_signal_conviction(signals: np.ndarray) -> float:
        """Calculate average signal conviction (magnitude)."""
        if len(signals) == 0:
            return 0.0
            
        return float(np.mean(np.abs(signals)))
        
    @staticmethod
    def calculate_signal_consistency(signals: np.ndarray, confidences: np.ndarray) -> float:
        """Calculate consistency between signal strength and confidence."""
        if len(signals) == 0 or len(confidences) == 0:
            return 0.0
            
        # Correlation between absolute signal and confidence
        return float(np.corrcoef(np.abs(signals), confidences)[0, 1])
        
    @staticmethod
    def calculate_signal_metrics(signals: np.ndarray, outcomes: np.ndarray,
                               confidences: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive signal metrics."""
        metrics = {
            'accuracy': SignalMetrics.calculate_signal_accuracy(signals, outcomes),
            'correlation': SignalMetrics.calculate_signal_correlation(signals, outcomes),
            'stability': SignalMetrics.calculate_signal_stability(signals),
            'conviction': SignalMetrics.calculate_signal_conviction(signals)
        }
        
        if confidences is not None:
            metrics['consistency'] = SignalMetrics.calculate_signal_consistency(signals, confidences)
            metrics['avg_confidence'] = float(np.mean(confidences))
            
        return metrics