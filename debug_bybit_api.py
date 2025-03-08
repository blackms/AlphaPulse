#!/usr/bin/env python3
"""
Debug script for Bybit API connection issues.

This script performs a series of tests to diagnose common issues with Bybit API connections:
1. Checks API credentials from environment variables
2. Tests network connectivity to Bybit API endpoints
3. Verifies API permissions
4. Tests basic API operations
5. Provides detailed error information and troubleshooting steps
"""
import asyncio
import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

from loguru import logger
import ccxt.async_support as ccxt
import aiohttp

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Bybit API endpoints for connectivity testing
BYBIT_API_ENDPOINTS = {
    "mainnet": [
        "https://api.bybit.com/v5/market/tickers?category=spot",
        "https://api.bybit.com/v5/asset/coin/query-info",
        "https://api.bybit.com/v5/market/time"
    ],
    "testnet": [
        "https://api-testnet.bybit.com/v5/market/tickers?category=spot",
        "https://api-testnet.bybit.com/v5/asset/coin/query-info",
        "https://api-testnet.bybit.com/v5/market/time"
    ]
}

class BybitAPIDebugger:
    """Diagnoses and troubleshoots Bybit API connection issues."""
    
    def __init__(self):
        self.results = {
            "credentials_check": None,
            "network_check": None,
            "api_operations": None,
            "issues_found": [],
            "recommendations": []
        }
        self.api_key = None
        self.api_secret = None
        self.testnet = None
        self.exchange = None
    
    async def run_diagnostics(self):
        """Run all diagnostic tests."""
        logger.info("Starting Bybit API diagnostics")
        
        # Step 1: Check credentials
        await self.check_credentials()
        
        # Step 2: Test network connectivity
        await self.test_network_connectivity()
        
        # Step 3: Test API operations (if credentials are available)
        if self.api_key and self.api_secret:
            await self.test_api_operations()
        
        # Generate recommendations based on test results
        self.generate_recommendations()
        
        # Print summary
        self.print_summary()
    
    async def check_credentials(self):
        """Check if API credentials are available and valid."""
        logger.info("Checking API credentials")
        
        # Check environment variables
        self.api_key = os.environ.get('BYBIT_API_KEY', os.environ.get('EXCHANGE_API_KEY', ''))
        self.api_secret = os.environ.get('BYBIT_API_SECRET', os.environ.get('EXCHANGE_API_SECRET', ''))
        
        # Check testnet setting
        if 'BYBIT_TESTNET' in os.environ:
            self.testnet = os.environ.get('BYBIT_TESTNET', '').lower() == 'true'
        else:
            self.testnet = os.environ.get('EXCHANGE_TESTNET', 'false').lower() == 'true'
        
        # Log results (with masked credentials)
        if self.api_key:
            masked_key = self.api_key[:4] + '...' + self.api_key[-4:] if len(self.api_key) > 8 else '***'
            logger.info(f"Found API key: {masked_key}")
        else:
            logger.warning("No API key found in environment variables")
            self.results["issues_found"].append("Missing API key")
        
        if self.api_secret:
            masked_secret = self.api_secret[:4] + '...' + self.api_secret[-4:] if len(self.api_secret) > 8 else '***'
            logger.info(f"Found API secret: {masked_secret}")
        else:
            logger.warning("No API secret found in environment variables")
            self.results["issues_found"].append("Missing API secret")
        
        logger.info(f"Using testnet: {self.testnet}")
        
        # Store results
        self.results["credentials_check"] = {
            "api_key_found": bool(self.api_key),
            "api_secret_found": bool(self.api_secret),
            "testnet": self.testnet
        }
    
    async def test_network_connectivity(self):
        """Test network connectivity to Bybit API endpoints."""
        logger.info("Testing network connectivity to Bybit API endpoints")
        
        # Determine which endpoints to test based on testnet setting
        endpoint_type = "testnet" if self.testnet else "mainnet"
        endpoints = BYBIT_API_ENDPOINTS[endpoint_type]
        
        results = []
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    start_time = time.time()
                    async with session.get(endpoint, timeout=10) as response:
                        elapsed = time.time() - start_time
                        status = response.status
                        if status == 200:
                            data = await response.json()
                            logger.info(f"✅ Successfully connected to {endpoint} (status: {status}, time: {elapsed:.2f}s)")
                            results.append({
                                "endpoint": endpoint,
                                "status": status,
                                "time": elapsed,
                                "success": True
                            })
                        else:
                            text = await response.text()
                            logger.error(f"❌ Failed to connect to {endpoint} (status: {status}, time: {elapsed:.2f}s)")
                            logger.error(f"Response: {text}")
                            results.append({
                                "endpoint": endpoint,
                                "status": status,
                                "time": elapsed,
                                "success": False,
                                "error": text
                            })
                            self.results["issues_found"].append(f"Failed to connect to {endpoint} (status: {status})")
                except aiohttp.ClientError as e:
                    logger.error(f"❌ Connection error for {endpoint}: {str(e)}")
                    results.append({
                        "endpoint": endpoint,
                        "success": False,
                        "error": str(e)
                    })
                    self.results["issues_found"].append(f"Connection error for {endpoint}: {str(e)}")
                except Exception as e:
                    logger.error(f"❌ Unexpected error for {endpoint}: {str(e)}")
                    results.append({
                        "endpoint": endpoint,
                        "success": False,
                        "error": str(e)
                    })
                    self.results["issues_found"].append(f"Unexpected error for {endpoint}: {str(e)}")
        
        # Store results
        self.results["network_check"] = {
            "endpoints_tested": len(endpoints),
            "successful_connections": sum(1 for r in results if r.get("success", False)),
            "details": results
        }
    
    async def test_api_operations(self):
        """Test basic API operations using the provided credentials."""
        logger.info("Testing API operations")
        
        if not self.api_key or not self.api_secret:
            logger.warning("Skipping API operations test due to missing credentials")
            return
        
        operations_results = []
        
        try:
            # Initialize exchange
            logger.info(f"Initializing Bybit exchange (testnet={self.testnet})")
            self.exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                }
            })
            
            # Set testnet mode if needed
            if self.testnet:
                self.exchange.set_sandbox_mode(True)
                logger.info("Using testnet mode")
            
            # Test 1: Load markets
            try:
                logger.info("Testing: Load markets")
                start_time = time.time()
                markets = await self.exchange.load_markets()
                elapsed = time.time() - start_time
                logger.info(f"✅ Successfully loaded {len(markets)} markets (time: {elapsed:.2f}s)")
                operations_results.append({
                    "operation": "load_markets",
                    "success": True,
                    "time": elapsed,
                    "details": f"Loaded {len(markets)} markets"
                })
            except Exception as e:
                logger.error(f"❌ Failed to load markets: {str(e)}")
                operations_results.append({
                    "operation": "load_markets",
                    "success": False,
                    "error": str(e)
                })
                self.results["issues_found"].append(f"Failed to load markets: {str(e)}")
            
            # Test 2: Fetch balance (requires authentication)
            try:
                logger.info("Testing: Fetch balance (requires authentication)")
                start_time = time.time()
                balance = await self.exchange.fetch_balance()
                elapsed = time.time() - start_time
                
                # Count non-zero balances
                non_zero = sum(1 for asset, total in balance['total'].items() if total > 0)
                
                logger.info(f"✅ Successfully fetched balance with {non_zero} non-zero assets (time: {elapsed:.2f}s)")
                operations_results.append({
                    "operation": "fetch_balance",
                    "success": True,
                    "time": elapsed,
                    "details": f"Found {non_zero} non-zero assets"
                })
            except Exception as e:
                logger.error(f"❌ Failed to fetch balance: {str(e)}")
                operations_results.append({
                    "operation": "fetch_balance",
                    "success": False,
                    "error": str(e)
                })
                self.results["issues_found"].append(f"Failed to fetch balance: {str(e)}")
            
            # Test 3: Fetch ticker for BTC/USDT
            try:
                logger.info("Testing: Fetch ticker for BTC/USDT")
                start_time = time.time()
                ticker = await self.exchange.fetch_ticker('BTC/USDT')
                elapsed = time.time() - start_time
                logger.info(f"✅ Successfully fetched BTC/USDT ticker: {ticker['last']} (time: {elapsed:.2f}s)")
                operations_results.append({
                    "operation": "fetch_ticker",
                    "success": True,
                    "time": elapsed,
                    "details": f"BTC/USDT price: {ticker['last']}"
                })
            except Exception as e:
                logger.error(f"❌ Failed to fetch ticker: {str(e)}")
                operations_results.append({
                    "operation": "fetch_ticker",
                    "success": False,
                    "error": str(e)
                })
                self.results["issues_found"].append(f"Failed to fetch ticker: {str(e)}")
            
            # Test 4: Fetch order history (if available)
            try:
                logger.info("Testing: Fetch order history")
                start_time = time.time()
                
                # For Bybit, we need to use specific parameters
                params = {
                    'category': 'spot',
                    'limit': 10
                }
                
                # Try to fetch closed orders
                orders = await self.exchange.fetch_closed_orders('BTC/USDT', None, 10, params)
                elapsed = time.time() - start_time
                
                logger.info(f"✅ Successfully fetched {len(orders)} orders (time: {elapsed:.2f}s)")
                operations_results.append({
                    "operation": "fetch_closed_orders",
                    "success": True,
                    "time": elapsed,
                    "details": f"Found {len(orders)} orders"
                })
            except Exception as e:
                logger.error(f"❌ Failed to fetch order history: {str(e)}")
                operations_results.append({
                    "operation": "fetch_closed_orders",
                    "success": False,
                    "error": str(e)
                })
                # This is not critical, so we don't add it to issues_found
            
        except Exception as e:
            logger.error(f"❌ Error during API operations test: {str(e)}")
            logger.error(traceback.format_exc())
            self.results["issues_found"].append(f"Error during API operations test: {str(e)}")
        finally:
            # Close exchange connection
            if self.exchange:
                try:
                    await self.exchange.close()
                except Exception as e:
                    logger.error(f"Error closing exchange connection: {str(e)}")
        
        # Store results
        self.results["api_operations"] = {
            "operations_tested": len(operations_results),
            "successful_operations": sum(1 for r in operations_results if r.get("success", False)),
            "details": operations_results
        }
    
    def generate_recommendations(self):
        """Generate recommendations based on test results."""
        issues = self.results["issues_found"]
        recommendations = []
        
        # Check for credential issues
        if not self.api_key or not self.api_secret:
            recommendations.append(
                "Set API credentials in environment variables:\n"
                "  export BYBIT_API_KEY=your_api_key\n"
                "  export BYBIT_API_SECRET=your_api_secret"
            )
        
        # Check for network connectivity issues
        network_check = self.results.get("network_check", {})
        if network_check and network_check.get("successful_connections", 0) < network_check.get("endpoints_tested", 0):
            recommendations.append(
                "Network connectivity issues detected:\n"
                "  - Check your internet connection\n"
                "  - Verify that your firewall allows connections to Bybit API endpoints\n"
                "  - Try using a VPN if your location might be restricted"
            )
        
        # Check for API operation issues
        api_ops = self.results.get("api_operations", {})
        if api_ops and api_ops.get("successful_operations", 0) < api_ops.get("operations_tested", 0):
            # Check for authentication errors
            auth_errors = [
                r for r in api_ops.get("details", []) 
                if not r.get("success", False) and "auth" in r.get("error", "").lower()
            ]
            
            if auth_errors:
                recommendations.append(
                    "Authentication errors detected:\n"
                    "  - Verify that your API key and secret are correct\n"
                    "  - Check if your API key has the necessary permissions\n"
                    "  - Ensure you're using the correct testnet setting"
                )
            
            # Check for rate limit errors
            rate_limit_errors = [
                r for r in api_ops.get("details", []) 
                if not r.get("success", False) and "rate" in r.get("error", "").lower()
            ]
            
            if rate_limit_errors:
                recommendations.append(
                    "Rate limit errors detected:\n"
                    "  - Reduce the frequency of your API requests\n"
                    "  - Implement rate limiting in your code\n"
                    "  - Consider using a different API key"
                )
        
        # Add general recommendations if there are any issues
        if issues:
            recommendations.append(
                "General troubleshooting steps:\n"
                "  - Run debug_bybit_auth.py to test authentication specifically\n"
                "  - Check if Bybit is experiencing any API issues\n"
                "  - Verify that your system time is synchronized (important for API signatures)\n"
                "  - Try creating a new API key with appropriate permissions"
            )
        
        # If no issues found, add a success message
        if not issues and not recommendations:
            recommendations.append(
                "No issues detected! Your Bybit API connection is working correctly."
            )
        
        self.results["recommendations"] = recommendations
    
    def print_summary(self):
        """Print a summary of the diagnostic results."""
        print("\n" + "="*80)
        print(" BYBIT API DIAGNOSTICS SUMMARY ".center(80, "="))
        print("="*80 + "\n")
        
        # Print credentials check
        creds = self.results["credentials_check"]
        print("API CREDENTIALS:")
        print(f"  API Key:    {'✅ Found' if creds['api_key_found'] else '❌ Missing'}")
        print(f"  API Secret: {'✅ Found' if creds['api_secret_found'] else '❌ Missing'}")
        print(f"  Testnet:    {'Yes' if creds['testnet'] else 'No'}")
        print()
        
        # Print network check
        net = self.results["network_check"]
        print("NETWORK CONNECTIVITY:")
        print(f"  Endpoints Tested: {net['endpoints_tested']}")
        print(f"  Successful:       {net['successful_connections']}/{net['endpoints_tested']}")
        if net['successful_connections'] < net['endpoints_tested']:
            print("  Failed endpoints:")
            for detail in net['details']:
                if not detail.get('success', False):
                    print(f"    - {detail['endpoint']}: {detail.get('error', 'Unknown error')}")
        print()
        
        # Print API operations
        if self.results["api_operations"]:
            api_ops = self.results["api_operations"]
            print("API OPERATIONS:")
            print(f"  Operations Tested: {api_ops['operations_tested']}")
            print(f"  Successful:        {api_ops['successful_operations']}/{api_ops['operations_tested']}")
            if api_ops['successful_operations'] < api_ops['operations_tested']:
                print("  Failed operations:")
                for detail in api_ops['details']:
                    if not detail.get('success', False):
                        print(f"    - {detail['operation']}: {detail.get('error', 'Unknown error')}")
            print()
        
        # Print issues found
        if self.results["issues_found"]:
            print("ISSUES FOUND:")
            for i, issue in enumerate(self.results["issues_found"], 1):
                print(f"  {i}. {issue}")
            print()
        
        # Print recommendations
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(self.results["recommendations"], 1):
            print(f"  {i}. {rec}")
            print()
        
        print("="*80)
        print(" END OF DIAGNOSTICS ".center(80, "="))
        print("="*80 + "\n")

async def main():
    """Main entry point."""
    logger.info("Starting Bybit API diagnostics")
    
    debugger = BybitAPIDebugger()
    await debugger.run_diagnostics()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bybit_api_diagnostics_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(debugger.results, f, indent=2)
    
    logger.info(f"Diagnostic results saved to {filename}")

if __name__ == "__main__":
    asyncio.run(main())