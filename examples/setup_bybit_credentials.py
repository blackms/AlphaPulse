#!/usr/bin/env python3
"""
Script to set up Bybit credentials in the AlphaPulse credentials manager.
"""
import os
import json
from pathlib import Path

# Define the credentials file path
CREDENTIALS_FILE = os.path.expanduser("~/.alpha_pulse/credentials.json")

def main():
    """Set up Bybit credentials."""
    print("AlphaPulse Bybit Credentials Setup")
    print("==================================")
    
    # Load existing credentials if available
    credentials = {}
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, "r") as f:
                credentials = json.load(f)
            print(f"Loaded existing credentials from {CREDENTIALS_FILE}")
        except Exception as e:
            print(f"Error loading existing credentials: {str(e)}")
    
    # Ask for Bybit API credentials
    print("\nPlease enter your Bybit API credentials:")
    api_key = input("API Key: ")
    api_secret = input("API Secret: ")
    use_testnet = input("Use Testnet (y/n): ").lower() == "y"
    
    # Add Bybit credentials
    credentials["bybit"] = {
        "api_key": api_key,
        "api_secret": api_secret,
        "testnet": use_testnet
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
    
    # Save credentials
    try:
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump(credentials, f, indent=4)
        print(f"\nCredentials saved to {CREDENTIALS_FILE}")
        print("Bybit credentials have been added to the AlphaPulse credentials manager.")
    except Exception as e:
        print(f"Error saving credentials: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    main()