"""API routers for AlphaPulse."""
from fastapi import APIRouter

# Import routers
from .metrics import router as metrics_router
from .portfolio import router as portfolio_router
from .alerts import router as alerts_router
from .trades import router as trades_router
from .system import router as system_router
from .hedging import router as hedging_router
from .risk import router as risk_router
from .positions import router as positions_router

# Export routers
metrics = metrics_router
portfolio = portfolio_router
alerts = alerts_router
trades = trades_router
system = system_router
hedging = hedging_router
risk = risk_router
positions = positions_router

__all__ = [
    "metrics",
    "portfolio",
    "alerts",
    "trades",
    "system",
    "hedging",
    "risk",
    "positions"
]