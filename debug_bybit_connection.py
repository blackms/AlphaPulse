#!/usr/bin/env python3
"""
Debug script for Bybit API connection issues.

This script helps diagnose connection issues with the Bybit API by:
1. Testing network connectivity to Bybit API endpoints
2. Validating API credentials
3. Testing basic API operations
4. Providing detailed error information and troubleshooting steps

Usage:
    python debug_bybit_connection.py

Environment Variables:
    BYBIT_API_KEY: Your Bybit API key
    BYBIT_API_SECRET: Your Bybit API secret
"""
import os
import sys
import time
import json
import logging
import asyncio
import argparse
import traceback
from typing import Dict, Any, Optional, List, Tuple
import socket
import ssl
import urllib.request
import urllib.error
import requests
from requests.exceptions import RequestException
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Bybit API endpoints
BYBIT_API_ENDPOINTS = {
    "mainnet": {
        "base": "https://api.bybit.com",
        "v5": "https://api.bybit.com/v5",
        "asset": "https://api.bybit.com/v5/asset/coin/query-info",
        "market": "https://api.bybit.com/v5/market/tickers",
        "account": "https://api.bybit.com/v5/account/wallet-balance",
    },
    "testnet": {
        "base": "https://api-testnet.bybit.com",
        "v5": "https://api-testnet.bybit.com/v5",
        "asset": "https://api-testnet.bybit.com/v5/asset/coin/query-info",
        "market": "https://api-testnet.bybit.com/v5/market/tickers",
        "account": "https://api-testnet.bybit.com/v5/account/wallet-balance",
    }
}

def get_credentials() -> Tuple[str, str, bool]:
    """Get Bybit API credentials from environment variables."""
    api_key = os.environ.get("BYBIT_API_KEY", "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")
    testnet = os.environ.get("BYBIT_TESTNET", "false").lower() == "true"
    
    # Check if credentials are set
    if not api_key or not api_secret:
        logger.warning("API credentials not found in environment variables")
        logger.info("Please set BYBIT_API_KEY and BYBIT_API_SECRET environment variables")
        
        # Try to load from credentials file
        home_dir = os.path.expanduser("~")
        creds_file = os.path.join(home_dir, ".alpha_pulse", "credentials.json")
        if os.path.exists(creds_file):
            try:
                with open(creds_file, "r") as f:
                    creds_data = json.load(f)
                    if "bybit" in creds_data:
                        bybit_creds = creds_data["bybit"]
                        api_key = bybit_creds.get("api_key", "")
                        api_secret = bybit_creds.get("api_secret", "")
                        testnet = bybit_creds.get("testnet", False)
                        logger.info(f"Loaded credentials from {creds_file}")
                        logger.info(f"Using testnet: {testnet}")
            except Exception as e:
                logger.error(f"Error loading credentials from file: {e}")
    
    return api_key, api_secret, testnet

def check_network_connectivity(endpoints: Dict[str, str]) -> Dict[str, bool]:
    """Check network connectivity to Bybit API endpoints."""
    logger.info("Checking network connectivity to Bybit API endpoints...")
    results = {}
    
    for name, url in endpoints.items():
        try:
            logger.info(f"Testing connection to {name} endpoint: {url}")
            response = requests.get(url, timeout=5)
            if response.status_code < 500:  # Accept any non-server error
                results[name] = True
                logger.info(f"✅ Successfully connected to {name} endpoint")
            else:
                results[name] = False
                logger.error(f"❌ Failed to connect to {name} endpoint: HTTP {response.status_code}")
        except RequestException as e:
            results[name] = False
            logger.error(f"❌ Failed to connect to {name} endpoint: {str(e)}")
    
    return results

def check_dns_resolution(host: str) -> bool:
    """Check DNS resolution for a host."""
    logger.info(f"Checking DNS resolution for {host}...")
    try:
        ip_address = socket.gethostbyname(host)
        logger.info(f"✅ Successfully resolved {host} to {ip_address}")
        return True
    except socket.gaierror as e:
        logger.error(f"❌ Failed to resolve {host}: {str(e)}")
        return False

def check_ssl_certificate(host: str, port: int = 443) -> bool:
    """Check SSL certificate for a host."""
    logger.info(f"Checking SSL certificate for {host}...")
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port)) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                subject = dict(x[0] for x in cert['subject'])
                issued_to = subject.get('commonName', 'Unknown')
                issuer = dict(x[0] for x in cert['issuer'])
                issued_by = issuer.get('commonName', 'Unknown')
                expires = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                
                logger.info(f"✅ SSL certificate for {host}:")
                logger.info(f"   Issued to: {issued_to}")
                logger.info(f"   Issued by: {issued_by}")
                logger.info(f"   Expires: {expires}")
                
                return True
    except Exception as e:
        logger.error(f"❌ Failed to check SSL certificate for {host}: {str(e)}")
        return False

def generate_signature(api_key: str, api_secret: str, timestamp: int) -> str:
    """Generate signature for Bybit API authentication."""
    import hmac
    import hashlib
    
    param_str = f"api_key={api_key}&timestamp={timestamp}"
    signature = hmac.new(
        bytes(api_secret, "utf-8"),
        bytes(param_str, "utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    return signature

def test_authentication(api_key: str, api_secret: str, testnet: bool) -> bool:
    """Test authentication with Bybit API."""
    logger.info("Testing authentication with Bybit API...")
    
    if not api_key or not api_secret:
        logger.error("❌ API key or secret is missing")
        return False
    
    # Determine which endpoint set to use
    endpoints = BYBIT_API_ENDPOINTS["testnet" if testnet else "mainnet"]
    
    # Generate signature
    timestamp = int(time.time() * 1000)
    signature = generate_signature(api_key, api_secret, timestamp)
    
    # Set up headers
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-SIGN": signature,
        "X-BAPI-TIMESTAMP": str(timestamp),
        "X-BAPI-RECV-WINDOW": "5000"
    }
    
    # Test authentication with account endpoint
    try:
        logger.info(f"Testing authentication with account endpoint: {endpoints['account']}")
        response = requests.get(endpoints['account'], headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                logger.info("✅ Authentication successful")
                return True
            else:
                logger.error(f"❌ Authentication failed: {data.get('retMsg', 'Unknown error')}")
                logger.error(f"Response: {json.dumps(data, indent=2)}")
                return False
        else:
            logger.error(f"❌ Authentication failed: HTTP {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Authentication test failed: {str(e)}")
        return False

def test_market_data(testnet: bool) -> bool:
    """Test fetching market data from Bybit API."""
    logger.info("Testing market data from Bybit API...")
    
    # Determine which endpoint set to use
    endpoints = BYBIT_API_ENDPOINTS["testnet" if testnet else "mainnet"]
    
    # Test market data endpoint
    try:
        logger.info(f"Testing market data endpoint: {endpoints['market']}?category=spot")
        response = requests.get(f"{endpoints['market']}?category=spot", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                result = data.get("result", {})
                tickers = result.get("list", [])
                if tickers:
                    logger.info(f"✅ Successfully fetched {len(tickers)} tickers")
                    logger.info(f"First ticker: {json.dumps(tickers[0], indent=2)}")
                    return True
                else:
                    logger.warning("⚠️ No tickers found in response")
                    return True  # Still consider this a success as the API responded correctly
            else:
                logger.error(f"❌ Market data fetch failed: {data.get('retMsg', 'Unknown error')}")
                logger.error(f"Response: {json.dumps(data, indent=2)}")
                return False
        else:
            logger.error(f"❌ Market data fetch failed: HTTP {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Market data test failed: {str(e)}")
        return False

def test_asset_info(testnet: bool) -> bool:
    """Test fetching asset info from Bybit API."""
    logger.info("Testing asset info from Bybit API...")
    
    # Determine which endpoint set to use
    endpoints = BYBIT_API_ENDPOINTS["testnet" if testnet else "mainnet"]
    
    # Test asset info endpoint
    try:
        logger.info(f"Testing asset info endpoint: {endpoints['asset']}")
        response = requests.get(endpoints['asset'], timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                result = data.get("result", {})
                rows = result.get("rows", [])
                if rows:
                    logger.info(f"✅ Successfully fetched info for {len(rows)} assets")
                    logger.info(f"First asset: {json.dumps(rows[0], indent=2)}")
                    return True
                else:
                    logger.warning("⚠️ No assets found in response")
                    return True  # Still consider this a success as the API responded correctly
            else:
                logger.error(f"❌ Asset info fetch failed: {data.get('retMsg', 'Unknown error')}")
                logger.error(f"Response: {json.dumps(data, indent=2)}")
                return False
        else:
            logger.error(f"❌ Asset info fetch failed: HTTP {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Asset info test failed: {str(e)}")
        return False

def print_troubleshooting_steps(network_ok: bool, dns_ok: bool, ssl_ok: bool, auth_ok: bool, market_ok: bool, asset_ok: bool) -> None:
    """Print troubleshooting steps based on test results."""
    logger.info("\n=== Troubleshooting Steps ===")
    
    if not network_ok:
        logger.info("Network Connectivity Issues:")
        logger.info("1. Check your internet connection")
        logger.info("2. Check if your firewall is blocking outbound connections to api.bybit.com")
        logger.info("3. Try using a different network (e.g., mobile hotspot)")
        logger.info("4. Check if Bybit API is down: https://status.bybit.com/")
    
    if not dns_ok:
        logger.info("DNS Resolution Issues:")
        logger.info("1. Check your DNS settings")
        logger.info("2. Try using a different DNS server (e.g., 8.8.8.8 or 1.1.1.1)")
        logger.info("3. Flush your DNS cache")
    
    if not ssl_ok:
        logger.info("SSL Certificate Issues:")
        logger.info("1. Check your system date and time (incorrect time can cause SSL validation failures)")
        logger.info("2. Update your SSL certificates")
        logger.info("3. Check if your antivirus or security software is intercepting SSL connections")
    
    if not auth_ok:
        logger.info("Authentication Issues:")
        logger.info("1. Verify your API key and secret are correct")
        logger.info("2. Check if your API key has the necessary permissions")
        logger.info("3. Ensure you're using the correct network (mainnet vs testnet)")
        logger.info("4. Check if your API key has IP restrictions")
        logger.info("5. Generate a new API key if necessary")
    
    if not market_ok or not asset_ok:
        logger.info("API Endpoint Issues:")
        logger.info("1. Check if the Bybit API endpoints have changed")
        logger.info("2. Verify you're using the correct API version")
        logger.info("3. Check if the specific endpoint you're trying to access is available")
    
    if network_ok and dns_ok and ssl_ok and auth_ok and market_ok and asset_ok:
        logger.info("All tests passed! If you're still experiencing issues:")
        logger.info("1. Check your application code for logical errors")
        logger.info("2. Verify you're handling API responses correctly")
        logger.info("3. Check for rate limiting issues")
        logger.info("4. Ensure you're using the correct parameters for your API calls")

def main():
    """Main function to run the debug script."""
    parser = argparse.ArgumentParser(description="Debug Bybit API connection issues")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=== Bybit API Connection Diagnostics ===")
    
    # Get credentials
    api_key, api_secret, testnet = get_credentials()
    logger.info(f"Using testnet: {testnet}")
    
    # Determine which endpoint set to use
    endpoints = BYBIT_API_ENDPOINTS["testnet" if testnet else "mainnet"]
    
    # Extract hostname from base URL
    base_url = endpoints["base"]
    hostname = base_url.replace("https://", "").replace("http://", "").split("/")[0]
    
    # Check DNS resolution
    dns_ok = check_dns_resolution(hostname)
    
    # Check SSL certificate
    ssl_ok = check_ssl_certificate(hostname)
    
    # Check network connectivity
    network_results = check_network_connectivity(endpoints)
    network_ok = all(network_results.values())
    
    # Test authentication if network connectivity is OK
    auth_ok = False
    if network_ok:
        auth_ok = test_authentication(api_key, api_secret, testnet)
    
    # Test market data if network connectivity is OK
    market_ok = False
    if network_ok:
        market_ok = test_market_data(testnet)
    
    # Test asset info if network connectivity is OK
    asset_ok = False
    if network_ok:
        asset_ok = test_asset_info(testnet)
    
    # Print summary
    logger.info("\n=== Diagnostic Summary ===")
    logger.info(f"DNS Resolution: {'✅ OK' if dns_ok else '❌ Failed'}")
    logger.info(f"SSL Certificate: {'✅ OK' if ssl_ok else '❌ Failed'}")
    logger.info(f"Network Connectivity: {'✅ OK' if network_ok else '❌ Failed'}")
    logger.info(f"Authentication: {'✅ OK' if auth_ok else '❌ Failed'}")
    logger.info(f"Market Data: {'✅ OK' if market_ok else '❌ Failed'}")
    logger.info(f"Asset Info: {'✅ OK' if asset_ok else '❌ Failed'}")
    
    # Print troubleshooting steps
    print_troubleshooting_steps(network_ok, dns_ok, ssl_ok, auth_ok, market_ok, asset_ok)
    
    # Return exit code
    if network_ok and dns_ok and ssl_ok and auth_ok and market_ok and asset_ok:
        logger.info("\n✅ All tests passed! Your Bybit API connection is working correctly.")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Please check the troubleshooting steps above.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nDiagnostic interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)