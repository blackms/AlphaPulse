# AlphaPulse API Documentation

## Overview
The AlphaPulse API provides RESTful endpoints to interact with the trading system. The API is built using FastAPI and provides secure access to trading data, portfolio management, risk analysis, and hedging operations.

## Authentication
The API supports two authentication methods:

### API Key Authentication
Include the API key in the request header:
```
X-API-Key: your_api_key
```

### OAuth2 Authentication
1. Obtain a token by sending a POST request to `/token` with username and password:
```
POST /token
Content-Type: application/x-www-form-urlencoded

username=your_username&password=your_password
```

2. Include the token in the Authorization header:
```
Authorization: Bearer your_access_token
```

## Base URL
```
http://localhost:18001
```

## REST Endpoints

### Health Check
```
GET /health
```
Returns the API service status. No authentication required.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### Positions

#### Get Spot Positions
```
GET /api/v1/positions/spot
```
Returns current spot positions.

**Response:**
```json
[
  {
    "symbol": "BTC",
    "quantity": 0.5,
    "value_usd": 25000.0,
    "avg_entry_price": 50000.0
  }
]
```

#### Get Futures Positions
```
GET /api/v1/positions/futures
```
Returns current futures positions.

**Response:**
```json
[
  {
    "symbol": "BTC-PERP",
    "quantity": -0.5,
    "value_usd": -25000.0,
    "avg_entry_price": 50000.0,
    "leverage": 3.0
  }
]
```

#### Get Position Metrics
```
GET /api/v1/positions/metrics
```
Returns detailed metrics for all positions including spot positions, futures positions, net exposure, and hedge ratios.

**Response:**
```json
[
  {
    "symbol": "BTC",
    "spot_value": 25000.0,
    "spot_qty": 0.5,
    "futures_value": -25000.0,
    "futures_qty": -0.5,
    "net_exposure": 0.0,
    "hedge_ratio": 1.0
  }
]
```

### Risk Management

#### Get Risk Exposure
```
GET /api/v1/risk/exposure
```
Returns current risk exposure metrics.

**Response:**
```json
{
  "BTC": 0.25,
  "ETH": 0.15,
  "total": 0.4
}
```

#### Get Risk Metrics
```
GET /api/v1/risk/metrics
```
Returns detailed risk metrics including volatility, VaR, CVaR, and ratios.

**Response:**
```json
{
  "volatility": 0.25,
  "var_95": 0.15,
  "cvar_95": 0.18,
  "max_drawdown": 0.3,
  "sharpe_ratio": 1.2,
  "sortino_ratio": 1.5,
  "calmar_ratio": 0.8
}
```

#### Get Risk Limits
```
GET /api/v1/risk/limits
```
Returns current risk limits.

**Response:**
```json
{
  "position_limits": {
    "default": 20000.0
  },
  "margin_limits": {
    "total": 150000.0
  },
  "exposure_limits": {
    "total": 100000.0
  },
  "drawdown_limits": {
    "max": 25000.0
  }
}
```

#### Get Position Size Recommendation
```
GET /api/v1/risk/position-size/{asset}
```
Returns position size recommendation for a specific asset.

**Parameters:**
- `asset`: Asset symbol (e.g., BTC, ETH)

**Response:**
```json
{
  "asset": "BTC",
  "recommended_size": 0.1,
  "max_size": 0.5,
  "risk_per_trade": 1000.0,
  "stop_loss": 45000.0,
  "take_profit": 55000.0
}
```

### Hedging

#### Analyze Hedge Positions
```
GET /api/v1/hedging/analysis
```
Analyzes current hedge positions and provides recommendations.

**Response:**
```json
{
  "commentary": "Current portfolio is 80% hedged. Consider increasing hedge ratio for BTC.",
  "adjustments": [
    {
      "asset": "BTC",
      "action": "Increase short position by 0.1 BTC",
      "priority": "high"
    }
  ],
  "current_net_exposure": 0.2,
  "target_net_exposure": 0.0,
  "risk_metrics": {
    "hedge_ratio": 0.8,
    "correlation": -0.95
  }
}
```

#### Execute Hedge Adjustments
```
POST /api/v1/hedging/execute
```
Executes recommended hedge adjustments.

**Response:**
```json
{
  "status": "completed",
  "executed_trades": [
    {
      "asset": "BTC-PERP",
      "side": "sell",
      "amount": "0.1",
      "price": "50000.0"
    }
  ],
  "message": "Successfully executed hedge adjustments"
}
```

#### Close All Hedges
```
POST /api/v1/hedging/close
```
Closes all hedge positions.

**Response:**
```json
{
  "status": "completed",
  "executed_trades": [
    {
      "asset": "BTC-PERP",
      "side": "buy",
      "amount": "0.5",
      "price": "50000.0"
    }
  ],
  "message": "Successfully closed all hedge positions"
}
```

### Portfolio

#### Get Portfolio
```
GET /api/v1/portfolio
```
Returns current portfolio data.

**Parameters:**
- `include_history` (optional): Include portfolio history (default: false)
- `refresh` (optional): Force refresh from exchange (default: false)

**Response:**
```json
{
  "total_value": 100000.0,
  "cash": 50000.0,
  "positions": [
    {
      "symbol": "BTC",
      "quantity": 0.5,
      "value": 25000.0,
      "allocation": 0.25
    },
    {
      "symbol": "ETH",
      "quantity": 10.0,
      "value": 25000.0,
      "allocation": 0.25
    }
  ],
  "history": [
    {
      "timestamp": "2025-04-21T00:00:00Z",
      "total_value": 98000.0
    }
  ]
}
```

#### Reload Exchange Data
```
POST /api/v1/portfolio/reload
```
Forces a reload of exchange data.

**Response:**
```json
{
  "status": "success",
  "message": "Exchange data reloaded",
  "timestamp": "2025-04-22T10:30:00Z"
}
```

### Metrics

#### Get Metrics
```
GET /api/v1/metrics/{metric_type}
```
Returns metrics data of a specific type.

**Parameters:**
- `metric_type`: Type of metric to retrieve (e.g., portfolio_value, performance, risk)
- `start_time` (optional): Start time for the query (default: 24 hours ago)
- `end_time` (optional): End time for the query (default: now)
- `aggregation` (optional): Aggregation function (avg, min, max, sum, count) (default: avg)

**Response:**
```json
[
  {
    "timestamp": "2025-04-22T09:00:00Z",
    "value": 100000.0
  },
  {
    "timestamp": "2025-04-22T10:00:00Z",
    "value": 101000.0
  }
]
```

#### Get Latest Metrics
```
GET /api/v1/metrics/{metric_type}/latest
```
Returns the latest metrics of a specific type.

**Parameters:**
- `metric_type`: Type of metric to retrieve (e.g., portfolio_value, performance, risk)

**Response:**
```json
{
  "timestamp": "2025-04-22T10:30:00Z",
  "value": 101000.0,
  "change_24h": 0.01
}
```

### Alerts
The API provides endpoints for managing alerts. Detailed documentation to be added.

### System
The API provides endpoints for system information. Detailed documentation to be added.

### Trades
The API provides endpoints for trade data. Detailed documentation to be added.

## WebSocket Endpoints

The API provides real-time updates via WebSocket connections. All WebSocket connections require authentication.

### Authentication
Send an authentication message immediately after connecting:
```json
{
  "type": "auth",
  "token": "your_access_token"
}
```

### Available Channels

#### Metrics Channel
```
WebSocket: /ws/metrics
```
Provides real-time metrics updates.

**Example message:**
```json
{
  "type": "metrics_update",
  "data": {
    "portfolio_value": 101000.0,
    "timestamp": "2025-04-22T10:30:00Z"
  }
}
```

#### Alerts Channel
```
WebSocket: /ws/alerts
```
Provides real-time alerts updates.

**Example message:**
```json
{
  "type": "new_alert",
  "data": {
    "id": "alert-001",
    "severity": "critical",
    "message": "Portfolio drawdown exceeded threshold",
    "timestamp": "2025-04-22T10:30:00Z"
  }
}
```

#### Portfolio Channel
```
WebSocket: /ws/portfolio
```
Provides real-time portfolio updates.

**Example message:**
```json
{
  "type": "portfolio_update",
  "data": {
    "total_value": 101000.0,
    "cash": 50000.0,
    "positions": [
      {
        "symbol": "BTC",
        "quantity": 0.5,
        "value": 25500.0
      }
    ],
    "timestamp": "2025-04-22T10:30:00Z"
  }
}
```

#### Trades Channel
```
WebSocket: /ws/trades
```
Provides real-time trade updates.

**Example message:**
```json
{
  "type": "new_trade",
  "data": {
    "id": "trade-001",
    "symbol": "BTC",
    "side": "buy",
    "quantity": 0.1,
    "price": 51000.0,
    "timestamp": "2025-04-22T10:30:00Z"
  }
}
```

## Configuration
The API uses environment variables for configuration:
```env
# API Configuration
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
ALPHA_PULSE_BYBIT_TESTNET=true/false

# OpenAI API Key (for LLM-based hedging analysis)
OPENAI_API_KEY=your_openai_api_key

# Authentication
JWT_SECRET=your_jwt_secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
```

## Development
To run the API server in development mode:
```bash
python src/scripts/run_api.py
```

The server will start on port 18001 with auto-reload enabled for development.

## Error Handling
The API returns standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

Error responses include a detail message:
```json
{
  "detail": "Error message"
}