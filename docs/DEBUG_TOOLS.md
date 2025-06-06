# Debugging Tools for Exchange Connectivity

This directory contains several debugging tools to help diagnose and fix issues with exchange connectivity, particularly for Bybit integration.

## Available Debug Tools

### 1. `debug_bybit_api.py`

A comprehensive diagnostic tool for Bybit API connectivity issues. This script:

- Checks API credentials from environment variables
- Tests network connectivity to Bybit API endpoints
- Verifies API permissions
- Tests basic API operations
- Provides detailed error information and troubleshooting steps

**Usage:**
```bash
python debug_bybit_api.py
```

### 2. `debug_bybit_auth.py`

Specifically tests authentication with Bybit using credentials from different sources:
- Environment variables
- Credentials file
- Custom input

**Usage:**
```bash
python debug_bybit_auth.py
```

### 3. `debug_bybit_credentials.py`

Prints information about Bybit credentials from different sources to help identify which credentials are being used.

**Usage:**
```bash
python debug_bybit_credentials.py
```

### 4. `debug_exchange_connection.py`

Tests the exchange connection using the current environment variables, attempting to initialize the exchange and fetch basic data.

**Usage:**
```bash
python debug_exchange_connection.py
```

### 5. `debug_bybit_connection.py`

A more advanced diagnostic tool that performs comprehensive connectivity tests:
- DNS resolution checks
- SSL certificate validation
- Network connectivity tests to multiple endpoints
- Authentication verification
- Market data retrieval tests
- Asset information retrieval tests
- Detailed troubleshooting steps based on test results

**Usage:**
```bash
python debug_bybit_connection.py
# For more verbose output
python debug_bybit_connection.py --verbose
```

## Common Issues and Solutions

### 1. Missing or Invalid API Credentials

**Symptoms:**
- "Authentication error" messages
- "No API credentials available" warnings
- Empty balances returned

**Solutions:**
- Set the correct environment variables:
  ```bash
  export BYBIT_API_KEY=your_api_key
  export BYBIT_API_SECRET=your_api_secret
  ```
- Verify that your API key has the necessary permissions (read access at minimum)
- Create a new API key if necessary

### 2. Network Connectivity Issues

**Symptoms:**
- Timeout errors
- Connection refused errors
- DNS resolution failures

**Solutions:**
- Check your internet connection
- Verify that your firewall allows connections to Bybit API endpoints
- Try using a VPN if your location might be restricted

### 3. Testnet vs. Mainnet Configuration

**Symptoms:**
- "Invalid API key" errors despite having correct credentials
- Empty data returned from API calls

**Solutions:**
- Ensure your testnet setting matches your API key:
  ```bash
  # For mainnet
  export BYBIT_TESTNET=false
  
  # For testnet
  export BYBIT_TESTNET=true
  ```
- Verify that you're using the correct API key for the selected network
- Note: The system now always uses mainnet mode (testnet=false) for simplicity

### 4. Rate Limiting Issues

**Symptoms:**
- "Too many requests" errors
- Intermittent failures

**Solutions:**
- Reduce the frequency of your API requests
- Implement exponential backoff for retries
- Consider using a different API key

### 5. Event Loop Issues

**Symptoms:**
- "Task got Future attached to a different loop" errors
- Hanging API requests
- Timeouts during initialization
- Database connection errors

**Solutions:**
- Ensure you're using the singleton instance of ExchangeDataSynchronizer
- Avoid creating multiple instances of components that use event loops
- Use proper error handling for asyncio operations
- Implement timeouts for all API calls
- Use thread-specific event loops when working with multiple threads

## Troubleshooting Process

1. Run `debug_bybit_credentials.py` to check which credentials are being used
2. Run `debug_bybit_connection.py` for a comprehensive network and API diagnosis
3. Run `debug_bybit_api.py` for more detailed API functionality testing
4. Fix any issues identified by the diagnostic tools
5. Run `debug_exchange_connection.py` to verify that the exchange connection works with the application code
6. If issues persist, check the Bybit API documentation and status page

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BYBIT_API_KEY` | Bybit API key | - |
| `BYBIT_API_SECRET` | Bybit API secret | - |
| `BYBIT_TESTNET` | Whether to use testnet (true/false) | false |
| `EXCHANGE_API_KEY` | Generic exchange API key (fallback) | - |
| `EXCHANGE_API_SECRET` | Generic exchange API secret (fallback) | - |
| `EXCHANGE_TESTNET` | Generic testnet setting (fallback) | true |
| `EXCHANGE_TYPE` | Type of exchange to use | bybit |