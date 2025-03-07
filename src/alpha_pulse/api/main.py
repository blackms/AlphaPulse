"""
Main API application.

This module defines the FastAPI application and routes.
"""
import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm

# Import routers
from .routers import metrics, alerts, portfolio, system, trades
from .websockets import endpoints as ws_endpoints
from .websockets.subscription import subscription_manager

# Import dependencies
from .dependencies import get_current_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AlphaPulse API",
    description="API for the AlphaPulse AI Hedge Fund",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])
app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])
app.include_router(trades.router, prefix="/api/v1", tags=["trades"])
app.include_router(system.router, prefix="/api/v1", tags=["system"])

# Include WebSocket routers
app.include_router(ws_endpoints.router, tags=["websockets"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to AlphaPulse API"}


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Get an access token.
    
    In a real implementation, this would validate the username and password
    and return a JWT token.
    """
    # For testing purposes, accept any username/password
    return {
        "access_token": "mock_token",
        "token_type": "bearer"
    }


@app.on_event("startup")
async def startup_event():
    """Run when the application starts."""
    logger.info("Starting AlphaPulse API")
    
    # Start the subscription manager
    await subscription_manager.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Run when the application shuts down."""
    logger.info("Shutting down AlphaPulse API")
    
    # Stop the subscription manager
    await subscription_manager.stop()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )