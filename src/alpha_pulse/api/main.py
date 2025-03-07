"""
Main FastAPI application module for AlphaPulse.
"""
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .config import config
from .middleware import setup_middleware
from .dependencies import cleanup_exchange
from .routers import metrics, portfolio, alerts, trades, system, hedging, risk, positions
from .websockets.manager import ConnectionManager
from .websockets.auth import WebSocketAuthenticator
from .websockets.subscription import SubscriptionManager

# Configure logging
logging_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

logging.basicConfig(
    level=logging_levels.get(config.logging.level.lower(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" if config.logging.format == "text" else None
)

# Create FastAPI application
app = FastAPI(
    title="AlphaPulse API",
    description="REST API for AlphaPulse Trading System",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.allowed_origins,
    allow_credentials=config.cors.allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up middleware
setup_middleware(app)

# Initialize websocket components
connection_manager = ConnectionManager.get_instance()
websocket_auth = WebSocketAuthenticator()
subscription_manager = SubscriptionManager.get_instance()

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Starting AlphaPulse API")
    await subscription_manager.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down AlphaPulse API")
    await subscription_manager.stop()
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
app.include_router(metrics, prefix="/api/v1/metrics", tags=["metrics"])
app.include_router(portfolio, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(alerts, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(trades, prefix="/api/v1/trades", tags=["trades"])
app.include_router(system, prefix="/api/v1/system", tags=["system"])
app.include_router(hedging, prefix="/api/v1/hedging", tags=["hedging"])
app.include_router(risk, prefix="/api/v1/risk", tags=["risk"])
app.include_router(positions, prefix="/api/v1/positions", tags=["positions"])

# WebSocket endpoints
@app.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    """WebSocket for real-time metrics updates."""
    # Accept connection
    await websocket.accept()
    client_id = f"client-{id(websocket)}"
    
    try:
        # Authenticate user
        user = await websocket_auth.authenticate(websocket)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        # Register connection
        await connection_manager.connect(websocket, client_id)
        
        # Subscribe to metrics channel
        await connection_manager.subscribe(websocket, "metrics")
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Ping-pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Handle disconnect
        logger.info(f"Client {client_id} disconnected")
    finally:
        await connection_manager.disconnect(websocket, client_id)

@app.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """WebSocket for real-time alerts updates."""
    # Accept connection
    await websocket.accept()
    client_id = f"client-{id(websocket)}"
    
    try:
        # Authenticate user
        user = await websocket_auth.authenticate(websocket)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        # Register connection
        await connection_manager.connect(websocket, client_id)
        
        # Subscribe to alerts channel
        await connection_manager.subscribe(websocket, "alerts")
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Ping-pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Handle disconnect
        logger.info(f"Client {client_id} disconnected")
    finally:
        await connection_manager.disconnect(websocket, client_id)

@app.websocket("/ws/portfolio")
async def portfolio_websocket(websocket: WebSocket):
    """WebSocket for real-time portfolio updates."""
    # Accept connection
    await websocket.accept()
    client_id = f"client-{id(websocket)}"
    
    try:
        # Authenticate user
        user = await websocket_auth.authenticate(websocket)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        # Register connection
        await connection_manager.connect(websocket, client_id)
        
        # Subscribe to portfolio channel
        await connection_manager.subscribe(websocket, "portfolio")
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Ping-pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Handle disconnect
        logger.info(f"Client {client_id} disconnected")
    finally:
        await connection_manager.disconnect(websocket, client_id)

@app.websocket("/ws/trades")
async def trades_websocket(websocket: WebSocket):
    """WebSocket for real-time trade updates."""
    # Accept connection
    await websocket.accept()
    client_id = f"client-{id(websocket)}"
    
    try:
        # Authenticate user
        user = await websocket_auth.authenticate(websocket)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        # Register connection
        await connection_manager.connect(websocket, client_id)
        
        # Subscribe to trades channel
        await connection_manager.subscribe(websocket, "trades")
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Ping-pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Handle disconnect
        logger.info(f"Client {client_id} disconnected")
    finally:
        await connection_manager.disconnect(websocket, client_id)