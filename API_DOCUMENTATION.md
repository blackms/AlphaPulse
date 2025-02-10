# AlphaPulse API Documentation

## Overview
The AlphaPulse API provides RESTful endpoints to interact with the trading system. The API is built using FastAPI and provides secure access to trading data and operations.

## Authentication
All API endpoints (except health check) require authentication using an API key. Include the API key in the request header:
```
X-API-Key: your_api_key
```

## Base URL
```
http://localhost:18001
```

## Endpoints

### Health Check
```
GET /health
```
Returns the API service status. No authentication required.

### Positions

#### Get Position Metrics
```
GET /api/v1/positions/metrics
```
Returns detailed metrics for all positions including:
- Spot positions
- Futures positions
- Net exposure
- Hedge ratios

#### Get Spot Positions
```
GET /api/v1/positions/spot
```
Returns current spot positions.

#### Get Futures Positions
```
GET /api/v1/positions/futures
```
Returns current futures positions.

## Configuration
The API uses environment variables for configuration:
```env
# API Configuration
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
ALPHA_PULSE_BYBIT_TESTNET=true/false

# Logging
LOG_LEVEL=INFO
```

## Next Steps
Future endpoints to be implemented:
1. Portfolio Analysis
   - GET /api/v1/portfolio/analysis
   - GET /api/v1/portfolio/metrics
   - GET /api/v1/portfolio/performance

2. Hedging Operations
   - POST /api/v1/hedging/analyze
   - POST /api/v1/hedging/execute
   - POST /api/v1/hedging/close

3. Risk Management
   - GET /api/v1/risk/exposure
   - GET /api/v1/risk/metrics
   - GET /api/v1/risk/limits

4. Trading Operations
   - POST /api/v1/trading/orders
   - DELETE /api/v1/trading/orders/{id}
   - GET /api/v1/trading/orders/status/{id}

## Development
To run the API server in development mode:
```bash
python src/scripts/run_api.py
```

The server will start on port 18001 with auto-reload enabled for development.