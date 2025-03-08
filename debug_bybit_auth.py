#!/usr/bin/env python3
"""
Script to debug Bybit authentication by testing the API directly.
This script attempts to authenticate with Bybit using the credentials
from different sources to identify which ones work.
"""
import asyncio
import os
import json
import sys
from pathlib import Path
from loguru import logger
import ccxt.async_support as ccxt

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

class CredentialTester:
    """Tests Bybit credentials from different sources."""
    
    def __init__(self):
        self.results = {}
    
    async def test_credentials(self, source_name, api_key, api_secret, testnet=False):
        """
        Test a set of Bybit credentials.
        
        Args:
            source_name: Name of the credential source
            api_key: API key to test
            api_secret: API secret to test
            testnet: Whether to use testnet
        """
        logger.info(f"Testing credentials from {source_name}")
        logger.debug(f"API Key: {api_key}")
        logger.debug(f"API Secret: {api_secret}")
        logger.debug(f"Testnet: {testnet}")
        
        if not api_key or not api_secret:
            logger.warning(f"Missing credentials from {source_name}, skipping test")
            self.results[source_name] = {"success": False, "error": "Missing credentials"}
            return
        
        try:
            # Create Bybit exchange instance with these credentials
            exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                }
            })
            
            # Configure testnet if enabled
            if testnet:
                logger.debug(f"Setting sandbox mode to True")
                exchange.set_sandbox_mode(True)
            
            # Try to fetch balance which requires authentication
            logger.info(f"Attempting to fetch balance with {source_name} credentials")
            balance = await exchange.fetch_balance()
            
            # If we get here, authentication succeeded
            logger.info(f"Authentication successful with {source_name} credentials!")
            total_balance = sum(float(bal) for bal in balance['total'].values())
            self.results[source_name] = {
                "success": True, 
                "total_balance": total_balance,
                "currencies": len(balance['total']),
                "testnet": testnet
            }
            
        except Exception as e:
            # Authentication failed
            logger.error(f"Authentication failed with {source_name} credentials: {str(e)}")
            self.results[source_name] = {"success": False, "error": str(e)}
        finally:
            # Close exchange connection
            if 'exchange' in locals():
                await exchange.close()
    
    async def test_env_credentials(self):
        """Test credentials from environment variables."""
        # Check BYBIT specific variables first
        api_key = os.environ.get('BYBIT_API_KEY', '')
        api_secret = os.environ.get('BYBIT_API_SECRET', '')
        
        if api_key and api_secret:
            testnet = os.environ.get('BYBIT_TESTNET', 'true').lower() == 'true'
            await self.test_credentials("BYBIT_ENV_VARS", api_key, api_secret, testnet)
        
        # Check generic exchange variables
        api_key = os.environ.get('EXCHANGE_API_KEY', '')
        api_secret = os.environ.get('EXCHANGE_API_SECRET', '')
        
        if api_key and api_secret:
            testnet = os.environ.get('EXCHANGE_TESTNET', 'true').lower() == 'true'
            await self.test_credentials("GENERIC_ENV_VARS", api_key, api_secret, testnet)
    
    async def test_credentials_file(self):
        """Test credentials from ~/.alpha_pulse/credentials.json."""
        config_path = os.path.expanduser("~/.alpha_pulse/credentials.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                
                if "bybit" in data:
                    creds = data["bybit"]
                    api_key = creds.get("api_key", "")
                    api_secret = creds.get("api_secret", "")
                    testnet = creds.get("testnet", False)
                    
                    await self.test_credentials("HOME_CREDENTIALS_FILE", api_key, api_secret, testnet)
            except Exception as e:
                logger.error(f"Error loading credentials from {config_path}: {str(e)}")
        else:
            logger.warning(f"No credentials file found at {config_path}")
    
    async def test_local_credentials_file(self):
        """Test credentials from src/alpha_pulse/exchanges/credentials/bybit_credentials.json."""
        config_path = "src/alpha_pulse/exchanges/credentials/bybit_credentials.json"
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                
                api_key = data.get("api_key", "")
                api_secret = data.get("api_secret", "")
                testnet = data.get("testnet", False)
                
                await self.test_credentials("LOCAL_CREDENTIALS_FILE", api_key, api_secret, testnet)
                # Also test with testnet=True
                await self.test_credentials("LOCAL_CREDENTIALS_FILE_TESTNET", api_key, api_secret, True)
            except Exception as e:
                logger.error(f"Error loading credentials from {config_path}: {str(e)}")
        else:
            logger.warning(f"No credentials file found at {config_path}")
    
    async def test_custom_credentials(self):
        """Test with custom input credentials."""
        print("\nWould you like to test with custom credentials? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            print("\nEnter API key: ", end="")
            api_key = input().strip()
            print("Enter API secret: ", end="")
            api_secret = input().strip()
            print("Use testnet? (y/n): ", end="")
            testnet = input().strip().lower() == 'y'
            
            await self.test_credentials("CUSTOM_INPUT", api_key, api_secret, testnet)
    
    def print_results(self):
        """Print test results."""
        print("\n=== BYBIT AUTHENTICATION TEST RESULTS ===\n")
        
        for source, result in self.results.items():
            if result["success"]:
                print(f"✅ {source}: SUCCESS")
                print(f"   Total Balance: {result.get('total_balance', 'N/A')}")
                print(f"   Currencies: {result.get('currencies', 'N/A')}")
                print(f"   Testnet: {result.get('testnet', 'N/A')}")
            else:
                print(f"❌ {source}: FAILED")
                print(f"   Error: {result.get('error', 'Unknown error')}")
            print()
        
        # Print summary
        successful = sum(1 for result in self.results.values() if result["success"])
        total = len(self.results)
        
        print(f"Summary: {successful}/{total} credential sources authenticated successfully")
        
        if successful == 0:
            print("\n❌ ALL AUTHENTICATION ATTEMPTS FAILED")
            print("Possible reasons:")
            print("1. Invalid API key/secret")
            print("2. API key has been revoked or expired")
            print("3. Incorrect testnet/mainnet configuration")
            print("4. IP restrictions on the API key")
            print("5. Network connectivity issues")
            print("\nRecommendation: Generate a new API key on Bybit and update your credentials")
        else:
            print("\n✅ FOUND WORKING CREDENTIALS")
            print("Use the successful credential source in your application")

async def main():
    """Main entry point."""
    logger.info("Starting Bybit authentication test")
    
    tester = CredentialTester()
    
    # Test all credential sources
    await tester.test_env_credentials()
    await tester.test_credentials_file()
    await tester.test_local_credentials_file()
    await tester.test_custom_credentials()
    
    # Print results
    tester.print_results()

if __name__ == "__main__":
    asyncio.run(main())