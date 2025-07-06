"""
GPU-accelerated backtesting engine for AlphaPulse.

This module extends the base backtester with GPU acceleration for
computationally intensive operations like calculating returns,
drawdowns, and running Monte Carlo simulations.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
import asyncio

from .backtester import Backtester, BacktestResult
from ..ml.gpu.gpu_service import GPUService
from ..ml.gpu.gpu_config import get_inference_config


class GPUBacktester(Backtester):
    """GPU-accelerated backtesting engine."""
    
    def __init__(
        self,
        commission: float = 0.002,
        initial_capital: float = 100000.0,
        slippage: float = 0.001,
        use_fixed_position_sizing: bool = True,
        stop_loss_slippage: float = 0.002,
        market_impact_factor: float = 0.0001,
        benchmark_symbol: str = "^GSPC",
        use_gpu: bool = True,
        gpu_batch_size: int = 10000
    ):
        """
        Initialize GPU-accelerated backtester.
        
        Args:
            commission: Trading commission as a fraction
            initial_capital: Starting capital for the backtest
            slippage: Slippage as a fraction
            use_fixed_position_sizing: Whether to use initial capital for position sizing
            stop_loss_slippage: Additional slippage for stop loss orders
            market_impact_factor: Market impact as a fraction of position size
            benchmark_symbol: Symbol for benchmark comparison
            use_gpu: Whether to use GPU acceleration
            gpu_batch_size: Batch size for GPU operations
        """
        super().__init__(
            commission=commission,
            initial_capital=initial_capital,
            slippage=slippage,
            use_fixed_position_sizing=use_fixed_position_sizing,
            stop_loss_slippage=stop_loss_slippage,
            market_impact_factor=market_impact_factor,
            benchmark_symbol=benchmark_symbol
        )
        
        self.use_gpu = use_gpu
        self.gpu_batch_size = gpu_batch_size
        self.gpu_service = None
        
        if self.use_gpu:
            try:
                # Initialize GPU service with inference configuration
                config = get_inference_config()
                self.gpu_service = GPUService(config)
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.gpu_service.start())
                logger.info("GPU acceleration enabled for backtesting")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}. Falling back to CPU.")
                self.use_gpu = False
                self.gpu_service = None

    def _calculate_returns_gpu(self, prices: np.ndarray) -> np.ndarray:
        """Calculate returns using GPU acceleration."""
        if self.gpu_service:
            try:
                returns = self.gpu_service.cuda_ops.calculate_returns(prices)
                return returns
            except Exception as e:
                logger.warning(f"GPU returns calculation failed: {e}. Using CPU.")
        
        # Fallback to CPU
        return np.diff(prices) / prices[:-1]

    def _calculate_drawdown_gpu(self, equity_curve: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculate drawdown series and maximum drawdown using GPU."""
        if self.gpu_service:
            try:
                # Calculate running maximum
                running_max = self.gpu_service.cuda_ops.cumulative_max(equity_curve)
                
                # Calculate drawdown
                drawdown = (equity_curve - running_max) / running_max
                max_drawdown = float(np.min(drawdown))
                
                return drawdown, abs(max_drawdown)
            except Exception as e:
                logger.warning(f"GPU drawdown calculation failed: {e}. Using CPU.")
        
        # Fallback to CPU
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return drawdown, abs(np.min(drawdown))

    def _calculate_technical_indicators_batch(
        self,
        prices: pd.Series,
        indicators: List[str]
    ) -> Dict[str, pd.Series]:
        """Calculate multiple technical indicators in batch using GPU."""
        if not self.gpu_service:
            return {}
        
        try:
            # Convert to numpy array
            price_array = prices.values
            
            # Calculate indicators on GPU
            gpu_results = self.gpu_service.calculate_technical_indicators(
                price_array, indicators
            )
            
            # Convert back to pandas Series
            results = {}
            for indicator, values in gpu_results.items():
                # Handle different output lengths
                if len(values) == len(prices):
                    results[indicator] = pd.Series(values, index=prices.index)
                elif len(values) == len(prices) - 1:
                    # For returns and similar indicators
                    results[indicator] = pd.Series(values, index=prices.index[1:])
                else:
                    # For indicators with different lengths, align to end
                    idx_start = len(prices) - len(values)
                    results[indicator] = pd.Series(
                        values, index=prices.index[idx_start:]
                    )
            
            return results
        except Exception as e:
            logger.warning(f"GPU indicator calculation failed: {e}")
            return {}

    def _run_monte_carlo_paths(
        self,
        initial_price: float,
        returns: np.ndarray,
        n_simulations: int = 1000,
        time_horizon: Optional[int] = None
    ) -> np.ndarray:
        """Run Monte Carlo simulations for price paths using GPU."""
        if not self.gpu_service:
            # CPU fallback
            return self._run_monte_carlo_cpu(
                initial_price, returns, n_simulations, time_horizon
            )
        
        try:
            # Calculate drift and volatility from historical returns
            drift = np.mean(returns)
            volatility = np.std(returns)
            
            if time_horizon is None:
                time_horizon = len(returns)
            
            # Run GPU Monte Carlo
            paths = self.gpu_service.run_monte_carlo(
                initial_price=initial_price,
                drift=drift,
                volatility=volatility,
                time_horizon=time_horizon,
                n_simulations=n_simulations
            )
            
            return paths
        except Exception as e:
            logger.warning(f"GPU Monte Carlo failed: {e}. Using CPU.")
            return self._run_monte_carlo_cpu(
                initial_price, returns, n_simulations, time_horizon
            )

    def _run_monte_carlo_cpu(
        self,
        initial_price: float,
        returns: np.ndarray,
        n_simulations: int,
        time_horizon: int
    ) -> np.ndarray:
        """CPU fallback for Monte Carlo simulations."""
        drift = np.mean(returns)
        volatility = np.std(returns)
        
        dt = 1  # Daily time step
        paths = np.zeros((n_simulations, time_horizon))
        paths[:, 0] = initial_price
        
        for t in range(1, time_horizon):
            Z = np.random.randn(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z
            )
        
        return paths

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio with GPU acceleration if available."""
        if self.gpu_service and len(returns) > self.gpu_batch_size:
            try:
                # Use GPU for large datasets
                mean_return = float(self.gpu_service.cuda_ops.mean(returns))
                std_return = float(self.gpu_service.cuda_ops.std(returns))
                
                if std_return > 0:
                    sharpe = mean_return / std_return * np.sqrt(252)
                else:
                    sharpe = 0.0
                
                return sharpe
            except Exception as e:
                logger.warning(f"GPU Sharpe calculation failed: {e}. Using CPU.")
        
        # CPU calculation (original method)
        return super()._calculate_sharpe_ratio(returns)

    def backtest_with_indicators(
        self,
        prices: pd.Series,
        signals: pd.Series,
        stop_losses: Optional[pd.Series] = None,
        benchmark_prices: Optional[pd.Series] = None,
        calculate_indicators: Optional[List[str]] = None
    ) -> Tuple[BacktestResult, Dict[str, pd.Series]]:
        """
        Run backtest with optional technical indicator calculation.
        
        Args:
            prices: Time series of asset prices
            signals: Time series of target allocation signals
            stop_losses: Optional time series of stop-loss prices
            benchmark_prices: Optional benchmark prices
            calculate_indicators: List of indicators to calculate
            
        Returns:
            Tuple of (BacktestResult, indicator_dict)
        """
        # Calculate indicators if requested
        indicators = {}
        if calculate_indicators and self.gpu_service:
            indicators = self._calculate_technical_indicators_batch(
                prices, calculate_indicators
            )
            logger.info(f"Calculated {len(indicators)} indicators using GPU")
        
        # Run standard backtest
        result = self.backtest(prices, signals, stop_losses, benchmark_prices)
        
        # Add GPU metrics to result if available
        if self.gpu_service:
            gpu_metrics = self.gpu_service.get_metrics()
            if gpu_metrics and 'devices' in gpu_metrics:
                for device_id, device_info in gpu_metrics['devices'].items():
                    logger.info(
                        f"GPU {device_id} utilization during backtest: "
                        f"{device_info['utilization']:.1%}"
                    )
        
        return result, indicators

    def run_monte_carlo_backtest(
        self,
        prices: pd.Series,
        signals: pd.Series,
        n_simulations: int = 1000,
        confidence_levels: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95]
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation of backtest results.
        
        Args:
            prices: Historical price series
            signals: Trading signals
            n_simulations: Number of Monte Carlo paths
            confidence_levels: Percentiles to calculate
            
        Returns:
            Dictionary with Monte Carlo results
        """
        # Calculate historical returns
        returns = prices.pct_change().dropna().values
        
        # Run Monte Carlo simulations
        initial_price = float(prices.iloc[0])
        mc_paths = self._run_monte_carlo_paths(
            initial_price, returns, n_simulations, len(prices)
        )
        
        # Run backtest on each path
        mc_results = []
        
        for i in range(min(n_simulations, 100)):  # Limit to 100 for performance
            # Create price series from Monte Carlo path
            mc_prices = pd.Series(mc_paths[i], index=prices.index)
            
            # Run backtest
            try:
                result = self.backtest(mc_prices, signals)
                mc_results.append({
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate
                })
            except Exception as e:
                logger.warning(f"MC simulation {i} failed: {e}")
        
        # Calculate statistics
        if mc_results:
            df_results = pd.DataFrame(mc_results)
            
            stats = {
                'n_simulations': len(mc_results),
                'metrics': {}
            }
            
            for metric in df_results.columns:
                stats['metrics'][metric] = {
                    'mean': float(df_results[metric].mean()),
                    'std': float(df_results[metric].std()),
                    'percentiles': {}
                }
                
                for level in confidence_levels:
                    stats['metrics'][metric]['percentiles'][f"p{int(level*100)}"] = \
                        float(df_results[metric].quantile(level))
            
            return stats
        else:
            return {'error': 'No successful Monte Carlo simulations'}

    def optimize_parameters_gpu(
        self,
        prices: pd.Series,
        parameter_grid: Dict[str, List[Any]],
        objective: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using GPU acceleration.
        
        Args:
            prices: Historical prices
            parameter_grid: Dictionary of parameters to optimize
            objective: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Best parameters and results
        """
        # This is a placeholder for GPU-accelerated parameter optimization
        # In practice, this would use the distributed backtester
        logger.info("GPU parameter optimization not yet implemented")
        return {}

    def __del__(self):
        """Cleanup GPU resources."""
        if self.gpu_service:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.gpu_service.stop())
            else:
                try:
                    loop.run_until_complete(self.gpu_service.stop())
                except Exception:
                    pass  # Ignore errors during cleanup