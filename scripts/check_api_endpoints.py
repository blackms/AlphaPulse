#!/usr/bin/env python3
"""
Script to check if all required API endpoints are available.
This helps verify that the API is properly configured for the demo.
"""
import asyncio
import aiohttp
import sys
from datetime import datetime

API_BASE_URL = "http://localhost:8000"
API_USERNAME = "admin"
API_PASSWORD = "password"

REQUIRED_ENDPOINTS = [
    # Authentication
    {"method": "POST", "url": "/token", "auth": False, "name": "Authentication Token"},
    
    # API Endpoints
    {"method": "GET", "url": "/api/v1/metrics/portfolio", "auth": True, "name": "Portfolio Metrics"},
    {"method": "GET", "url": "/api/v1/alerts", "auth": True, "name": "Alerts List"},
    {"method": "GET", "url": "/api/v1/portfolio", "auth": True, "name": "Portfolio Data"},
    {"method": "GET", "url": "/api/v1/trades", "auth": True, "name": "Trades List"},
    {"method": "GET", "url": "/api/v1/system", "auth": True, "name": "System Status"},
    
    # WebSocket endpoints
    {"method": "GET", "url": "/ws", "auth": False, "name": "WebSocket Connection", "websocket": True},
]

async def get_auth_token():
    """Get authentication token from API."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_BASE_URL}/token",
            data={"username": API_USERNAME, "password": API_PASSWORD}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data["access_token"]
            else:
                print(f"❌ Failed to get auth token: {response.status}")
                return None

async def check_endpoint(session, endpoint, token=None):
    """Check if an endpoint is available."""
    headers = {}
    if endpoint.get("auth") and token:
        headers["Authorization"] = f"Bearer {token}"
    
    if endpoint.get("websocket"):
        try:
            ws_url = f"ws://localhost:8000{endpoint['url']}"
            async with session.ws_connect(ws_url, headers=headers) as ws:
                await ws.close()
            return True
        except Exception as e:
            print(f"❌ WebSocket endpoint {endpoint['url']} error: {str(e)}")
            return False
    else:
        try:
            method = endpoint["method"].lower()
            request_func = getattr(session, method)
            async with request_func(f"{API_BASE_URL}{endpoint['url']}", headers=headers) as response:
                return response.status < 400
        except Exception as e:
            print(f"❌ Endpoint {endpoint['url']} error: {str(e)}")
            return False

async def main():
    """Main function to check all endpoints."""
    print(f"Checking API endpoints at {API_BASE_URL} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Get auth token
    token = await get_auth_token()
    if not token and any(endpoint.get("auth") for endpoint in REQUIRED_ENDPOINTS):
        print("❌ Authentication failed. Cannot check authenticated endpoints.")
        sys.exit(1)
    
    # Check all endpoints
    results = []
    async with aiohttp.ClientSession() as session:
        for endpoint in REQUIRED_ENDPOINTS:
            print(f"Checking {endpoint['name']} ({endpoint['method']} {endpoint['url']})...", end="", flush=True)
            success = await check_endpoint(session, endpoint, token)
            results.append((endpoint, success))
            print(" ✅ OK" if success else " ❌ Failed")
    
    # Print summary
    print("\nSummary:")
    print("-" * 80)
    success_count = sum(1 for _, success in results if success)
    print(f"✅ {success_count}/{len(results)} endpoints available")
    
    if success_count < len(results):
        print("\nFailed endpoints:")
        for endpoint, success in results:
            if not success:
                print(f"❌ {endpoint['name']} ({endpoint['method']} {endpoint['url']})")
        sys.exit(1)
    else:
        print("\n✅ All required API endpoints are available!")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())