# Dashboard Integration Testing with Real Data

This document provides instructions for running the complete AI Hedge Fund system with the dashboard frontend connected to the actual backend API, using real data instead of mocks.

## Prerequisites

1. Ensure you have all dependencies installed:
   - Backend: Python 3.9+, FastAPI, etc. (see `requirements.txt`)
   - Frontend: Node.js 16+, npm/yarn (see `dashboard/package.json`)

2. Database setup:
   - Ensure the database is initialized with schema
   - Sample data should be loaded for testing (if needed)

## Step 1: Start the Backend API Server

First, we need to start the backend API server that will provide real data to our frontend:

```bash
# From the project root directory
cd src
python -m alpha_pulse.api.main
```

The API server should start and display a message indicating it's running, typically at http://localhost:8000.

Verify the API is working by accessing the OpenAPI docs in your browser:
```
http://localhost:8000/docs
```

## Step 2: Configure the Frontend

Update the frontend configuration to connect to the real backend instead of using mocked data:

1. Create or edit the `.env` file in the `dashboard` directory:

```bash
cd dashboard
cp .env.example .env
```

2. Edit the `.env` file to point to your local backend:

```
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_WS_URL=ws://localhost:8000/ws
REACT_APP_USE_MOCK_DATA=false
```

## Step 3: Start the Frontend Development Server

Start the React development server:

```bash
cd dashboard
npm install  # Only needed the first time
npm start
```

The dashboard should open automatically in your browser at http://localhost:3000.

## Step 4: Login to the Dashboard

Use the following test credentials to log in:

- Username: `admin`
- Password: `password`

## Step 5: Generate Real-time Data

To see the dashboard working with real-time data, you can run the data generation script:

```bash
# From the project root directory
cd examples
python trading/demo_ai_hedge_fund.py --live-data
```

This script will:
1. Generate simulated market data
2. Process it through the AI agents
3. Generate trading signals
4. Update the portfolio
5. Send alerts and notifications

All of these changes should reflect in real-time on the dashboard via WebSocket connections.

## Step 6: Testing Different Features

### 1. Portfolio Management

- Navigate to the Portfolio section
- Check asset allocation
- View position details
- Test rebalancing functionality

### 2. Trading Activity

- View recent trades
- Filter by asset or date range
- Examine trade details and performance

### 3. Alerts Monitoring

- Check alert notifications
- Test alert acknowledgment
- Verify severity levels are displayed correctly

### 4. System Status

- Monitor system health metrics
- Check component status
- Verify performance indicators

## Troubleshooting

### API Connection Issues

If the dashboard cannot connect to the backend:

1. Verify the API server is running
2. Check that the URLs in the `.env` file are correct
3. Look for CORS-related errors in the browser console
4. Ensure the API and WebSocket ports are not blocked by firewall

### Authentication Issues

If you cannot log in:

1. Verify the authentication service is running
2. Check credentials are correct
3. Look for token-related errors in console

### Data Issues

If no data appears:

1. Run the data generation script mentioned in Step 5
2. Check database connectivity
3. Verify the data pipeline is processing information
4. Look for data-related errors in the API logs

## Demo Script

For convenience, you can use the following shell script to start both the backend and frontend:

```bash
#!/bin/bash
# File: run_demo.sh

# Start backend API server
echo "Starting API server..."
cd src
python -m alpha_pulse.api.main &
API_PID=$!
cd ..

# Wait for API to start
echo "Waiting for API to initialize..."
sleep 5

# Start frontend
echo "Starting dashboard frontend..."
cd dashboard
npm start &
FRONTEND_PID=$!
cd ..

# Start data generation
echo "Starting data generation..."
cd examples
python trading/demo_ai_hedge_fund.py --live-data &
DATA_PID=$!
cd ..

echo "Demo is running!"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all components"

# Wait for user to press Ctrl+C
trap "kill $API_PID $FRONTEND_PID $DATA_PID; exit" INT
wait
```

Make this script executable and run it:

```bash
chmod +x run_demo.sh
./run_demo.sh
```

## Next Steps

After successfully testing the integrated system, consider:

1. Deploying to a test environment
2. Load testing with higher volumes of data
3. Adding additional features from the Phase 3 implementation plan
4. Gathering user feedback on the interface and functionality