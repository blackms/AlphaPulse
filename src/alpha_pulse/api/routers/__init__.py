"""
Router package for AlphaPulse API.
"""
from .positions import router as positions_router
from .portfolio import router as portfolio_router
from .hedging import router as hedging_router
from .risk import router as risk_router

# Export the routers for cleaner imports
positions = positions_router
portfolio = portfolio_router
hedging = hedging_router
risk = risk_router