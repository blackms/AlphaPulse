"""
WebSocket endpoints.

This module defines the WebSocket endpoints for the API.
"""
import logging
import json
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from .manager import connection_manager
from .auth import websocket_auth
from .subscription import subscription_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for metrics.
    
    This endpoint provides real-time metrics updates.
    """
    await connection_manager.connect(websocket)
    
    try:
        # Wait for authentication message
        auth_message = await websocket.receive_text()
        user = await websocket_auth.authenticate(websocket, auth_message)
        
        if not user:
            # Authentication failed
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        # Check permission
        if "view_metrics" not in user.get("permissions", []):
            await websocket.close(code=1008, reason="Permission denied")
            return
        
        # Subscribe to metrics channel
        await connection_manager.subscribe(websocket, "metrics")
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "message": "Connected to metrics channel",
            "user": user["username"]
        }))
        
        # Handle messages
        while True:
            message = await websocket.receive_text()
            # Process messages if needed
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from metrics channel")
    except Exception as e:
        logger.error(f"Error in metrics WebSocket: {e}")
    finally:
        await connection_manager.disconnect(websocket)


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for alerts.
    
    This endpoint provides real-time alerts updates.
    """
    await connection_manager.connect(websocket)
    
    try:
        # Wait for authentication message
        auth_message = await websocket.receive_text()
        user = await websocket_auth.authenticate(websocket, auth_message)
        
        if not user:
            # Authentication failed
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        # Check permission
        if "view_alerts" not in user.get("permissions", []):
            await websocket.close(code=1008, reason="Permission denied")
            return
        
        # Subscribe to alerts channel
        await connection_manager.subscribe(websocket, "alerts")
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "message": "Connected to alerts channel",
            "user": user["username"]
        }))
        
        # Handle messages
        while True:
            message = await websocket.receive_text()
            # Process messages if needed
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from alerts channel")
    except Exception as e:
        logger.error(f"Error in alerts WebSocket: {e}")
    finally:
        await connection_manager.disconnect(websocket)


@router.websocket("/ws/portfolio")
async def websocket_portfolio(websocket: WebSocket):
    """
    WebSocket endpoint for portfolio.
    
    This endpoint provides real-time portfolio updates.
    """
    await connection_manager.connect(websocket)
    
    try:
        # Wait for authentication message
        auth_message = await websocket.receive_text()
        user = await websocket_auth.authenticate(websocket, auth_message)
        
        if not user:
            # Authentication failed
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        # Check permission
        if "view_portfolio" not in user.get("permissions", []):
            await websocket.close(code=1008, reason="Permission denied")
            return
        
        # Subscribe to portfolio channel
        await connection_manager.subscribe(websocket, "portfolio")
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "message": "Connected to portfolio channel",
            "user": user["username"]
        }))
        
        # Handle messages
        while True:
            message = await websocket.receive_text()
            # Process messages if needed
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from portfolio channel")
    except Exception as e:
        logger.error(f"Error in portfolio WebSocket: {e}")
    finally:
        await connection_manager.disconnect(websocket)


@router.websocket("/ws/trades")
async def websocket_trades(websocket: WebSocket):
    """
    WebSocket endpoint for trades.
    
    This endpoint provides real-time trade updates.
    """
    await connection_manager.connect(websocket)
    
    try:
        # Wait for authentication message
        auth_message = await websocket.receive_text()
        user = await websocket_auth.authenticate(websocket, auth_message)
        
        if not user:
            # Authentication failed
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        # Check permission
        if "view_trades" not in user.get("permissions", []):
            await websocket.close(code=1008, reason="Permission denied")
            return
        
        # Subscribe to trades channel
        await connection_manager.subscribe(websocket, "trades")
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "message": "Connected to trades channel",
            "user": user["username"]
        }))
        
        # Handle messages
        while True:
            message = await websocket.receive_text()
            # Process messages if needed
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from trades channel")
    except Exception as e:
        logger.error(f"Error in trades WebSocket: {e}")
    finally:
        await connection_manager.disconnect(websocket)