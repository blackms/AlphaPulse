"""
Script to set up exchange credentials.
"""
from alpha_pulse.exchanges.credentials.manager import credentials_manager

def setup_credentials():
    """Set up Binance credentials."""
    # Set Binance credentials
    credentials_manager.set_credentials(
        exchange="binance",
        api_key="Jq1z3ZfGDavVnD2WsyaJnXNE381yvSkazV8iQjKZm7PG8epsLHcNBg1DToEdisrP",
        api_secret="eAIVE9Sj4Tgg2T7SftGgvX1MVAzNUik4vzsuMib0JUYDYSo07k9cqiEVuD0DUsvi",
        testnet=False
    )
    
    print("Credentials set up successfully")
    print("Available exchanges:", credentials_manager.list_exchanges())

if __name__ == "__main__":
    setup_credentials()