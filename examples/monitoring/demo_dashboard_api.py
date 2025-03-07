"""
Demo script for the Dashboard API.

This script demonstrates how to use the Dashboard API to access metrics, alerts, and portfolio data.
"""
import asyncio
import json
import os
import sys
import logging
from datetime import datetime, timedelta
import requests
import websockets
import jwt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("demo_dashboard_api")

# API configuration
API_BASE_URL = "http://localhost:8080/api/v1"
WS_BASE_URL = "ws://localhost:8080/ws"
JWT_SECRET = "dev-secret-key"  # For demo only, use environment variable in production


def create_token(username: str, role: str = "admin") -> str:
    """Create a JWT token for authentication."""
    payload = {
        "sub": username,
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


async def fetch_metrics():
    """Fetch metrics from the API."""
    token = create_token("demo_user")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Fetch latest performance metrics
    response = requests.get(
        f"{API_BASE_URL}/metrics/performance/latest",
        headers=headers
    )
    
    if response.status_code == 200:
        logger.info("Latest performance metrics:")
        logger.info(json.dumps(response.json(), indent=2))
    else:
        logger.error(f"Failed to fetch metrics: {response.status_code} - {response.text}")
    
    # Fetch historical metrics
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    response = requests.get(
        f"{API_BASE_URL}/metrics/portfolio_value",
        headers=headers,
        params={
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "aggregation": "mean"
        }
    )
    
    if response.status_code == 200:
        logger.info("Historical portfolio value metrics:")
        logger.info(json.dumps(response.json(), indent=2))
    else:
        logger.error(f"Failed to fetch historical metrics: {response.status_code} - {response.text}")


async def fetch_portfolio():
    """Fetch portfolio data from the API."""
    token = create_token("demo_user")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Fetch portfolio with history
    response = requests.get(
        f"{API_BASE_URL}/portfolio",
        headers=headers,
        params={"include_history": True}
    )
    
    if response.status_code == 200:
        logger.info("Portfolio data:")
        logger.info(json.dumps(response.json(), indent=2))
    else:
        logger.error(f"Failed to fetch portfolio: {response.status_code} - {response.text}")


async def fetch_alerts():
    """Fetch alerts from the API."""
    token = create_token("demo_user")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Fetch recent alerts
    response = requests.get(
        f"{API_BASE_URL}/alerts",
        headers=headers,
        params={"acknowledged": False}
    )
    
    if response.status_code == 200:
        alerts = response.json()
        logger.info(f"Found {len(alerts)} unacknowledged alerts")
        
        # Acknowledge first alert if any
        if alerts:
            alert_id = alerts[0]["alert_id"]
            ack_response = requests.post(
                f"{API_BASE_URL}/alerts/{alert_id}/acknowledge",
                headers=headers
            )
            
            if ack_response.status_code == 200:
                logger.info(f"Successfully acknowledged alert {alert_id}")
            else:
                logger.error(f"Failed to acknowledge alert: {ack_response.status_code} - {ack_response.text}")
    else:
        logger.error(f"Failed to fetch alerts: {response.status_code} - {response.text}")


async def listen_websocket(channel: str):
    """Listen to WebSocket updates."""
    token = create_token("demo_user")
    
    try:
        # Connect to WebSocket
        async with websockets.connect(f"{WS_BASE_URL}/{channel}") as websocket:
            # Authenticate
            await websocket.send(json.dumps({"token": token}))
            auth_response = await websocket.recv()
            logger.info(f"Authentication response: {auth_response}")
            
            # Listen for updates
            logger.info(f"Listening for {channel} updates...")
            for _ in range(5):  # Listen for 5 updates
                update = await websocket.recv()
                logger.info(f"Received {channel} update: {update}")
                
                # Send ping to keep connection alive
                await websocket.send("ping")
                pong = await websocket.recv()
                assert pong == "pong"
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")


async def main():
    """Run the demo."""
    logger.info("Starting Dashboard API demo")
    
    # Fetch data from REST API
    await fetch_metrics()
    await fetch_portfolio()
    await fetch_alerts()
    
    # Listen to WebSocket updates
    await listen_websocket("metrics")
    
    logger.info("Demo completed")


if __name__ == "__main__":
    asyncio.run(main())