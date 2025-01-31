"""
Prometheus metrics collection for AlphaPulse trading system.
"""
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from functools import wraps
import time

# Trading Metrics
TRADE_COUNTER = Counter('alphapulse_trades_total', 'Total number of trades executed',
                       ['symbol', 'direction'])
PNL_GAUGE = Gauge('alphapulse_pnl_current', 'Current PnL in base currency',
                  ['symbol'])
POSITION_SIZE_GAUGE = Gauge('alphapulse_position_size', 'Current position size',
                           ['symbol'])
WIN_RATE_GAUGE = Gauge('alphapulse_win_rate', 'Rolling win rate percentage',
                      ['symbol', 'timeframe'])

# Model Metrics
PREDICTION_HISTOGRAM = Histogram('alphapulse_model_predictions', 
                               'Distribution of model predictions',
                               ['model_name'])
MODEL_LATENCY = Histogram('alphapulse_model_latency_seconds',
                         'Model inference latency in seconds',
                         ['model_name'])

# System Metrics
ERROR_COUNTER = Counter('alphapulse_errors_total', 'Total number of errors',
                       ['type'])
API_LATENCY = Histogram('alphapulse_api_latency_seconds',
                       'API request latency in seconds',
                       ['endpoint'])

def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics server"""
    start_http_server(port)

def track_latency(metric: Histogram):
    """Decorator to track function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            metric.observe(duration)
            return result
        return wrapper
    return decorator

class MetricsTracker:
    """Helper class to track trading metrics"""
    
    @staticmethod
    def record_trade(symbol: str, direction: str):
        """Record a new trade"""
        TRADE_COUNTER.labels(symbol=symbol, direction=direction).inc()
    
    @staticmethod
    def update_pnl(symbol: str, pnl: float):
        """Update current PnL"""
        PNL_GAUGE.labels(symbol=symbol).set(pnl)
    
    @staticmethod
    def update_position(symbol: str, size: float):
        """Update position size"""
        POSITION_SIZE_GAUGE.labels(symbol=symbol).set(size)
    
    @staticmethod
    def update_win_rate(symbol: str, timeframe: str, rate: float):
        """Update win rate"""
        WIN_RATE_GAUGE.labels(symbol=symbol, timeframe=timeframe).set(rate)
    
    @staticmethod
    def record_prediction(model_name: str, prediction: float):
        """Record model prediction"""
        PREDICTION_HISTOGRAM.labels(model_name=model_name).observe(prediction)
    
    @staticmethod
    def record_error(error_type: str):
        """Record an error"""
        ERROR_COUNTER.labels(type=error_type).inc()

# Example usage:
# @track_latency(MODEL_LATENCY.labels(model_name='crypto_prediction'))
# def predict(features):
#     return model.predict(features)