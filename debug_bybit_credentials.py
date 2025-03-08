#!/usr/bin/env python3
"""
Debug script to safely print Bybit API credentials being used.
"""
import asyncio
import os
import json
from pathlib import Path
from loguru import logger

def mask_key(key, visible_chars=4):
    """Show only a few characters of the key for debugging."""
    if not key:
        return "(empty)"
    if len(key) <= visible_chars * 2:
        return "***" 
    return f"{key[:visible_chars]}...{key[-visible_chars:]}"

async def debug_credentials():
    """Print information about Bybit credentials."""
    logger.info("=== BYBIT CREDENTIALS DEBUG ===")
    
    # 1. Check credentials file in home directory
    logger.info("Checking home directory credentials file...")
    home_creds_path = os.path.expanduser("~/.alpha_pulse/credentials.json")
    logger.info(f"Home credentials path: {home_creds_path}")
    home_creds_exist = os.path.exists(home_creds_path)
    logger.info(f"Home credentials file exists: {home_creds_exist}")
    
    if home_creds_exist:
        try:
            with open(home_creds_path, "r") as f:
                home_creds = json.load(f)
            if "bybit" in home_creds:
                bybit_creds = home_creds["bybit"]
                logger.info(f"Found Bybit credentials in ~/.alpha_pulse/credentials.json")
                logger.info(f"API Key (masked): {mask_key(bybit_creds.get('api_key', ''))}")
                logger.info(f"API Secret (masked): {mask_key(bybit_creds.get('api_secret', ''))}")
                logger.info(f"Testnet: {bybit_creds.get('testnet', False)}")
            else:
                logger.warning("No Bybit credentials found in ~/.alpha_pulse/credentials.json")
        except Exception as e:
            logger.error(f"Error reading home credentials: {str(e)}")
    
    # 2. Check local credentials file
    logger.info("\nChecking local credentials file...")
    local_creds_path = "src/alpha_pulse/exchanges/credentials/bybit_credentials.json"
    logger.info(f"Local credentials path: {local_creds_path}")
    local_creds_exist = os.path.exists(local_creds_path)
    logger.info(f"Local credentials file exists: {local_creds_exist}")
    
    if local_creds_exist:
        try:
            with open(local_creds_path, "r") as f:
                local_creds = json.load(f)
            logger.info(f"API Key (masked): {mask_key(local_creds.get('api_key', ''))}")
            logger.info(f"API Secret (masked): {mask_key(local_creds.get('api_secret', ''))}")
        except Exception as e:
            logger.error(f"Error reading local credentials: {str(e)}")
    
    # 3. Check environment variables
    logger.info("\nChecking environment variables...")
    env_api_key = os.environ.get('BYBIT_API_KEY', os.environ.get('EXCHANGE_API_KEY', ''))
    env_secret = os.environ.get('BYBIT_API_SECRET', os.environ.get('EXCHANGE_API_SECRET', ''))
    env_testnet = os.environ.get('EXCHANGE_TESTNET', 'true').lower() == 'true'
    
    if env_api_key:
        logger.info(f"API Key from env (masked): {mask_key(env_api_key)}")
        logger.info(f"API Secret from env (masked): {mask_key(env_secret)}")
        logger.info(f"Testnet from env: {env_testnet}")
    else:
        logger.warning("No API credentials found in environment variables")
    
    # 4. Summary of what will be used
    logger.info("\n=== SUMMARY ===")
    if home_creds_exist and "bybit" in home_creds:
        logger.info("System will prioritize credentials from ~/.alpha_pulse/credentials.json")
    elif env_api_key:
        logger.info("System will use credentials from environment variables")
    elif local_creds_exist:
        logger.info("NOTE: Local credentials file exists but is NOT used by the application")
        logger.info("The system does not check src/alpha_pulse/exchanges/credentials/bybit_credentials.json")
    else:
        logger.error("NO VALID CREDENTIALS FOUND IN ANY LOCATION")

if __name__ == "__main__":
    asyncio.run(debug_credentials())
