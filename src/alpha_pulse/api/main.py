"""
Main FastAPI application module for AlphaPulse.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from alpha_pulse.api.middleware import LoggingMiddleware, RateLimitMiddleware
from alpha_pulse.api.dependencies import cleanup_exchange
from .routers import positions

# Create FastAPI application
app = FastAPI(
    title="AlphaPulse API",
    description="REST API for AlphaPulse Trading System",
    version="1.0.0",
)

# Configure middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Starting AlphaPulse API")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down AlphaPulse API")
    await cleanup_exchange()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AlphaPulse API",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AlphaPulse API",
        "version": "1.0.0",
        "description": "Trading system API providing access to positions, portfolio analysis, and trading operations."
    }

# Include routers
app.include_router(positions, prefix="/api/v1/positions", tags=["positions"])

# Additional routers will be added as they are implemented:
# - Portfolio analysis
# - Hedging operations
# - Risk management
# - Trading execution