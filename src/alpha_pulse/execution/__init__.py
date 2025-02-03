"""
Trading execution module.
"""
from .broker_interface import (
    BrokerInterface,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position
)
from .paper_broker import PaperBroker
from .broker_factory import create_broker, TradingMode


__all__ = [
    # Interfaces
    'BrokerInterface',
    'Order',
    'OrderResult',
    'OrderSide',
    'OrderStatus',
    'OrderType',
    'Position',
    
    # Implementations
    'PaperBroker',
    
    # Factory
    'create_broker',
    'TradingMode'
]