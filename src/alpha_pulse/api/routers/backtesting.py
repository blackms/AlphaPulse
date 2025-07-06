"""
Backtesting API endpoints with data lake integration.

Provides REST API access to the enhanced backtesting system,
including data source comparison and historical analysis capabilities.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, validator
from loguru import logger

from alpha_pulse.backtesting.enhanced_backtester import (
    EnhancedBacktester, 
    DataSource, 
    create_enhanced_backtester
)
from alpha_pulse.backtesting.strategy import BaseStrategy


router = APIRouter(prefix="/backtesting", tags=["backtesting"])


class BacktestRequest(BaseModel):
    """Request model for backtesting."""
    symbols: List[str]
    timeframe: str = "1d"
    start_date: datetime
    end_date: datetime
    strategy_type: str = "simple_ma"
    strategy_params: Dict[str, Any] = {}
    initial_capital: float = 100000.0
    commission: float = 0.002
    slippage: float = 0.001
    data_source: str = "auto"
    benchmark_symbol: str = "SPY"
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol is required")
        if len(v) > 10:
            raise ValueError("Maximum 10 symbols allowed")
        return v
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        if v not in valid_timeframes:
            raise ValueError(f"Timeframe must be one of: {valid_timeframes}")
        return v
    
    @validator('data_source')
    def validate_data_source(cls, v):
        valid_sources = ['auto', 'database', 'data_lake']
        if v not in valid_sources:
            raise ValueError(f"Data source must be one of: {valid_sources}")
        return v


class BacktestComparisonRequest(BaseModel):
    """Request model for data source comparison."""
    symbols: List[str]
    timeframe: str = "1d"
    start_date: datetime
    end_date: datetime
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol is required")
        return v


class FeatureBacktestRequest(BaseModel):
    """Request model for feature-enhanced backtesting."""
    symbols: List[str]
    timeframe: str = "1d"
    start_date: datetime
    end_date: datetime
    strategy_type: str = "technical_features"
    strategy_params: Dict[str, Any] = {}
    feature_config: Dict[str, List[str]]
    feature_dataset: str = "technical_features"
    initial_capital: float = 100000.0
    commission: float = 0.002


class BacktestResult(BaseModel):
    """Response model for backtest results."""
    symbol: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    execution_time_ms: float = 0.0


class BacktestResponse(BaseModel):
    """Response model for backtesting API."""
    request_id: str
    status: str
    results: Dict[str, BacktestResult]
    data_source_used: str
    execution_time_ms: float
    metadata: Dict[str, Any] = {}


class DataSourceComparison(BaseModel):
    """Response model for data source comparison."""
    comparison_results: Dict[str, Dict[str, Any]]
    recommendation: str
    performance_summary: Dict[str, Any]


# Simple strategy implementations for API use
class SimpleMovingAverageStrategy(BaseStrategy):
    """Simple moving average strategy for API."""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window
        self.position_open = False
    
    def should_enter(self, signal) -> bool:
        if len(signal) < self.long_window:
            return False
        
        short_ma = signal.rolling(window=self.short_window).mean()
        long_ma = signal.rolling(window=self.long_window).mean()
        
        if len(short_ma) >= 2 and len(long_ma) >= 2:
            crossover = (short_ma.iloc[-2] <= long_ma.iloc[-2]) and (short_ma.iloc[-1] > long_ma.iloc[-1])
            if crossover and not self.position_open:
                self.position_open = True
                return True
        return False
    
    def should_exit(self, signal, position) -> bool:
        if len(signal) < self.long_window:
            return False
        
        short_ma = signal.rolling(window=self.short_window).mean()
        long_ma = signal.rolling(window=self.long_window).mean()
        
        if len(short_ma) >= 2 and len(long_ma) >= 2:
            crossover = (short_ma.iloc[-2] >= long_ma.iloc[-2]) and (short_ma.iloc[-1] < long_ma.iloc[-1])
            if crossover and self.position_open:
                self.position_open = False
                return True
        return False


class RSIStrategy(BaseStrategy):
    """RSI-based strategy for API."""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_open = False
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def should_enter(self, signal) -> bool:
        if len(signal) < self.rsi_period:
            return False
        
        rsi = self._calculate_rsi(signal, self.rsi_period)
        
        if not rsi.empty and rsi.iloc[-1] < self.oversold and not self.position_open:
            self.position_open = True
            return True
        return False
    
    def should_exit(self, signal, position) -> bool:
        if len(signal) < self.rsi_period:
            return False
        
        rsi = self._calculate_rsi(signal, self.rsi_period)
        
        if not rsi.empty and rsi.iloc[-1] > self.overbought and self.position_open:
            self.position_open = False
            return True
        return False


def _create_strategy(strategy_type: str, params: Dict[str, Any]) -> BaseStrategy:
    """Create strategy instance from type and parameters."""
    if strategy_type == "simple_ma":
        return SimpleMovingAverageStrategy(
            short_window=params.get('short_window', 20),
            long_window=params.get('long_window', 50)
        )
    elif strategy_type == "rsi":
        return RSIStrategy(
            rsi_period=params.get('rsi_period', 14),
            oversold=params.get('oversold', 30),
            overbought=params.get('overbought', 70)
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Run a backtest with the specified parameters.
    
    Supports multiple data sources (database, data lake, auto) and various strategies.
    """
    import time
    import uuid
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Running backtest {request_id} for symbols: {request.symbols}")
    
    try:
        # Map data source string to enum
        data_source_map = {
            'auto': DataSource.AUTO,
            'database': DataSource.DATABASE,
            'data_lake': DataSource.DATA_LAKE
        }
        data_source = data_source_map[request.data_source]
        
        # Create enhanced backtester
        backtester = create_enhanced_backtester(
            commission=request.commission,
            initial_capital=request.initial_capital,
            data_source=data_source
        )
        
        # Create strategy
        strategy = _create_strategy(request.strategy_type, request.strategy_params)
        
        # Run backtest
        results = await backtester.run_backtest(
            strategy=strategy,
            symbols=request.symbols,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Convert results to response format
        result_dict = {}
        for symbol, result in results.items():
            result_dict[symbol] = BacktestResult(
                symbol=symbol,
                total_return=result.total_return,
                sharpe_ratio=result.sharpe_ratio,
                max_drawdown=result.max_drawdown,
                total_trades=result.total_trades,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                benchmark_return=result.benchmark_return,
                alpha=result.alpha,
                beta=result.beta
            )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Get data source stats
        stats = backtester.get_data_source_stats()
        
        response = BacktestResponse(
            request_id=request_id,
            status="completed",
            results=result_dict,
            data_source_used=request.data_source,
            execution_time_ms=execution_time,
            metadata={
                "strategy_type": request.strategy_type,
                "strategy_params": request.strategy_params,
                "data_source_stats": stats
            }
        )
        
        logger.info(f"Backtest {request_id} completed in {execution_time:.0f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Backtest {request_id} failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {str(e)}"
        )


@router.post("/compare-sources", response_model=DataSourceComparison)
async def compare_data_sources(request: BacktestComparisonRequest) -> DataSourceComparison:
    """
    Compare performance and data availability across different data sources.
    
    Helps users understand which data source is optimal for their use case.
    """
    logger.info(f"Comparing data sources for symbols: {request.symbols}")
    
    try:
        # Create backtester for comparison
        backtester = create_enhanced_backtester(data_source=DataSource.AUTO)
        
        # Run comparison
        comparison = await backtester.compare_data_sources(
            symbols=request.symbols,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Generate recommendation
        recommendation = "Use database as fallback"
        
        if 'speedup' in comparison:
            faster_source = comparison['speedup']['faster_source']
            speedup = comparison['speedup']['data_lake_vs_database']
            
            if faster_source == 'data_lake' and speedup > 1.5:
                recommendation = f"Use data lake (recommended) - {speedup:.1f}x faster"
            elif faster_source == 'database':
                recommendation = "Use database - more reliable for this query"
            else:
                recommendation = "Both sources perform similarly - use auto selection"
        
        # Create performance summary
        performance_summary = {}
        
        if 'data_lake' in comparison and comparison['data_lake']['success']:
            performance_summary['data_lake'] = {
                'available': True,
                'load_time': comparison['data_lake']['load_time'],
                'record_count': comparison['data_lake']['record_count']
            }
        
        if 'database' in comparison and comparison['database']['success']:
            performance_summary['database'] = {
                'available': True,
                'load_time': comparison['database']['load_time'],
                'record_count': comparison['database']['record_count']
            }
        
        return DataSourceComparison(
            comparison_results=comparison,
            recommendation=recommendation,
            performance_summary=performance_summary
        )
        
    except Exception as e:
        logger.error(f"Data source comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )


@router.post("/feature-enhanced", response_model=BacktestResponse)
async def run_feature_enhanced_backtest(request: FeatureBacktestRequest) -> BacktestResponse:
    """
    Run backtest with pre-computed technical features from data lake.
    
    Requires data lake with technical features dataset.
    """
    import time
    import uuid
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Running feature-enhanced backtest {request_id}")
    
    try:
        # Create backtester with data lake
        backtester = create_enhanced_backtester(data_source=DataSource.DATA_LAKE)
        
        # Create strategy
        strategy = _create_strategy(request.strategy_type, request.strategy_params)
        
        # Run feature-enhanced backtest
        results = await backtester.run_feature_enhanced_backtest(
            strategy=strategy,
            symbols=request.symbols,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            feature_config=request.feature_config,
            feature_dataset=request.feature_dataset
        )
        
        # Convert results
        result_dict = {}
        for symbol, result in results.items():
            result_dict[symbol] = BacktestResult(
                symbol=symbol,
                total_return=result.total_return,
                sharpe_ratio=result.sharpe_ratio,
                max_drawdown=result.max_drawdown,
                total_trades=result.total_trades,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                benchmark_return=result.benchmark_return,
                alpha=result.alpha,
                beta=result.beta
            )
        
        execution_time = (time.time() - start_time) * 1000
        
        response = BacktestResponse(
            request_id=request_id,
            status="completed",
            results=result_dict,
            data_source_used="data_lake",
            execution_time_ms=execution_time,
            metadata={
                "strategy_type": request.strategy_type,
                "feature_config": request.feature_config,
                "feature_dataset": request.feature_dataset
            }
        )
        
        logger.info(f"Feature-enhanced backtest {request_id} completed")
        return response
        
    except Exception as e:
        logger.error(f"Feature-enhanced backtest {request_id} failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature-enhanced backtest failed: {str(e)}"
        )


@router.get("/strategies")
async def get_available_strategies() -> Dict[str, Any]:
    """Get list of available backtesting strategies and their parameters."""
    strategies = {
        "simple_ma": {
            "name": "Simple Moving Average Crossover",
            "description": "Enters long when short MA crosses above long MA",
            "parameters": {
                "short_window": {
                    "type": "integer",
                    "default": 20,
                    "min": 5,
                    "max": 50,
                    "description": "Short moving average window"
                },
                "long_window": {
                    "type": "integer",
                    "default": 50,
                    "min": 20,
                    "max": 200,
                    "description": "Long moving average window"
                }
            }
        },
        "rsi": {
            "name": "RSI Strategy",
            "description": "Buys oversold and sells overbought based on RSI",
            "parameters": {
                "rsi_period": {
                    "type": "integer",
                    "default": 14,
                    "min": 7,
                    "max": 30,
                    "description": "RSI calculation period"
                },
                "oversold": {
                    "type": "float",
                    "default": 30.0,
                    "min": 10.0,
                    "max": 40.0,
                    "description": "Oversold threshold for buying"
                },
                "overbought": {
                    "type": "float",
                    "default": 70.0,
                    "min": 60.0,
                    "max": 90.0,
                    "description": "Overbought threshold for selling"
                }
            }
        }
    }
    
    return {
        "strategies": strategies,
        "data_sources": ["auto", "database", "data_lake"],
        "timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    }


@router.get("/data-lake/status")
async def get_data_lake_status() -> Dict[str, Any]:
    """Get data lake availability and status information."""
    try:
        backtester = create_enhanced_backtester(data_source=DataSource.AUTO)
        stats = backtester.get_data_source_stats()
        
        return {
            "data_lake_available": stats.get('data_lake_available', False),
            "available_datasets": stats.get('available_datasets', []),
            "cache_stats": stats.get('data_lake_cache', {}),
            "recommended_source": "data_lake" if stats.get('data_lake_available') else "database"
        }
        
    except Exception as e:
        logger.error(f"Failed to get data lake status: {e}")
        return {
            "data_lake_available": False,
            "error": str(e),
            "recommended_source": "database"
        }


@router.delete("/cache")
async def clear_cache() -> Dict[str, str]:
    """Clear all backtesting caches."""
    try:
        backtester = create_enhanced_backtester(data_source=DataSource.AUTO)
        backtester.clear_cache()
        
        return {"status": "success", "message": "All caches cleared"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return {"status": "error", "message": str(e)}