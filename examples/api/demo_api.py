#!/usr/bin/env python
"""
Demo script for AlphaPulse API.

This script demonstrates how to use the AlphaPulse API.
"""
import os
import sys
import asyncio
import logging
import json
import websockets
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demo_api')

# API base URL
API_BASE_URL = "http://localhost:8000/api/v1"
WS_BASE_URL = "ws://localhost:8000/ws"


async def demo_rest_api():
    """Demonstrate REST API endpoints."""
    logger.info("Demonstrating REST API endpoints...")
    
    # Get token
    token_response = requests.post(
        "http://localhost:8000/token",
        data={"username": "admin", "password": "admin"}
    )
    token_data = token_response.json()
    token = token_data["access_token"]
    
    # Set headers with token
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Get metrics
    logger.info("Getting metrics...")
    metrics_response = requests.get(
        f"{API_BASE_URL}/metrics/portfolio_value",
        headers=headers
    )
    logger.info(f"Metrics response: {metrics_response.status_code}")
    if metrics_response.status_code == 200:
        logger.info(f"Metrics data: {metrics_response.json()}")
    
    # Get latest metric
    logger.info("Getting latest metric...")
    latest_metric_response = requests.get(
        f"{API_BASE_URL}/metrics/portfolio_value/latest",
        headers=headers
    )
    logger.info(f"Latest metric response: {latest_metric_response.status_code}")
    if latest_metric_response.status_code == 200:
        logger.info(f"Latest metric data: {latest_metric_response.json()}")
    
    # Get alerts
    logger.info("Getting alerts...")
    alerts_response = requests.get(
        f"{API_BASE_URL}/alerts",
        headers=headers
    )
    logger.info(f"Alerts response: {alerts_response.status_code}")
    if alerts_response.status_code == 200:
        logger.info(f"Alerts data: {alerts_response.json()}")
    
    # Get portfolio
    logger.info("Getting portfolio...")
    portfolio_response = requests.get(
        f"{API_BASE_URL}/portfolio",
        headers=headers
    )
    logger.info(f"Portfolio response: {portfolio_response.status_code}")
    if portfolio_response.status_code == 200:
        logger.info(f"Portfolio data: {portfolio_response.json()}")
    
    # Get trades
    logger.info("Getting trades...")
    trades_response = requests.get(
        f"{API_BASE_URL}/trades",
        headers=headers
    )
    logger.info(f"Trades response: {trades_response.status_code}")
    if trades_response.status_code == 200:
        logger.info(f"Trades data: {trades_response.json()}")
    
    # Get system metrics
    logger.info("Getting system metrics...")
    system_response = requests.get(
        f"{API_BASE_URL}/system",
        headers=headers
    )
    logger.info(f"System response: {system_response.status_code}")
    if system_response.status_code == 200:
        logger.info(f"System data: {system_response.json()}")


async def demo_websocket_api():
    """Demonstrate WebSocket API endpoints."""
    logger.info("Demonstrating WebSocket API endpoints...")
    
    # Connect to metrics WebSocket
    logger.info("Connecting to metrics WebSocket...")
    async with websockets.connect(f"{WS_BASE_URL}/metrics") as websocket:
        # Send authentication message
        auth_message = json.dumps({
            "token": "mock_token"
        })
        await websocket.send(auth_message)
        
        # Receive welcome message
        welcome = await websocket.recv()
        logger.info(f"Received welcome message: {welcome}")
        
        # Receive updates for a while
        for _ in range(3):
            message = await websocket.recv()
            logger.info(f"Received metrics update: {message}")
    
    # Connect to alerts WebSocket
    logger.info("Connecting to alerts WebSocket...")
    async with websockets.connect(f"{WS_BASE_URL}/alerts") as websocket:
        # Send authentication message
        auth_message = json.dumps({
            "token": "mock_token"
        })
        await websocket.send(auth_message)
        
        # Receive welcome message
        welcome = await websocket.recv()
        logger.info(f"Received welcome message: {welcome}")
        
        # Receive updates for a while
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            logger.info(f"Received alerts update: {message}")
        except asyncio.TimeoutError:
            logger.info("No alerts received within timeout")
    
    # Connect to portfolio WebSocket
    logger.info("Connecting to portfolio WebSocket...")
    async with websockets.connect(f"{WS_BASE_URL}/portfolio") as websocket:
        # Send authentication message
        auth_message = json.dumps({
            "token": "mock_token"
        })
        await websocket.send(auth_message)
        
        # Receive welcome message
        welcome = await websocket.recv()
        logger.info(f"Received welcome message: {welcome}")
        
        # Receive updates for a while
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            logger.info(f"Received portfolio update: {message}")
        except asyncio.TimeoutError:
            logger.info("No portfolio updates received within timeout")
    
    # Connect to trades WebSocket
    logger.info("Connecting to trades WebSocket...")
    async with websockets.connect(f"{WS_BASE_URL}/trades") as websocket:
        # Send authentication message
        auth_message = json.dumps({
            "token": "mock_token"
        })
        await websocket.send(auth_message)
        
        # Receive welcome message
        welcome = await websocket.recv()
        logger.info(f"Received welcome message: {welcome}")
        
        # Receive updates for a while
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            logger.info(f"Received trades update: {message}")
        except asyncio.TimeoutError:
            logger.info("No trades received within timeout")


async def main():
    """Main function to run the demo."""
    logger.info("Starting API demo...")
    
    try:
        # Demonstrate REST API
        await demo_rest_api()
        
        # Demonstrate WebSocket API
        await demo_websocket_api()
        
        logger.info("API demo completed successfully!")
    except Exception as e:
        logger.error(f"Error in API demo: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)