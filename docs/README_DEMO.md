# AI Hedge Fund Demo

This README provides instructions for running the AI Hedge Fund demo with real data visualization through the dashboard.

## Prerequisites

- Python 3.9+ with required packages installed
- Node.js 16+ for the frontend
- All dependencies installed (see below)

## Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install frontend dependencies**:
   ```bash
   cd dashboard
   npm install
   cd ..
   ```

## Running the Demo

The easiest way to run the complete demo is to use the provided script:

```bash
./run_demo.sh
```

This script will:
1. Start the backend API server
2. Configure and start the frontend dashboard
3. Start the data generation script to provide real-time data
4. Set up proper cleanup when you exit with Ctrl+C

## Accessing the Dashboard

Once the demo is running, you can access the dashboard at:
```
http://localhost:3000
```

Use the following credentials to log in:
- Username: `admin`
- Password: `password`

## Components

The demo consists of several components working together:

1. **Backend API** (http://localhost:8000):
   - Provides data endpoints for the dashboard
   - Handles authentication
   - Manages WebSocket connections for real-time updates

2. **Frontend Dashboard** (http://localhost:3000):
   - Visualizes portfolio performance
   - Displays trading activity
   - Shows system status and alerts
   - Updates in real-time via WebSockets

3. **Data Generation**:
   - Simulates market data
   - Generates trading signals through AI agents
   - Executes portfolio decisions
   - Sends data to the API for dashboard display

## Manual Component Startup

If you prefer to start components individually:

1. **Start the API server**:
   ```bash
   cd src
   python -m alpha_pulse.api.main
   ```

2. **Start the frontend**:
   ```bash
   cd dashboard
   npm start
   ```

3. **Start data generation**:
   ```bash
   cd examples
   python trading/demo_ai_hedge_fund_live.py --live-data
   ```

## Troubleshooting

If you encounter any issues:

1. **API server won't start**:
   - Check if port 8000 is already in use
   - Verify Python dependencies are installed
   - Check logs for specific errors

2. **Frontend won't start**:
   - Check if port 3000 is already in use
   - Verify Node.js dependencies are installed
   - Check for JavaScript errors in the console

3. **No data appears in dashboard**:
   - Verify the API server is running
   - Check that the data generation script is running
   - Look for connection errors in the browser console

4. **Authentication issues**:
   - Ensure you're using the correct credentials
   - Check for token-related errors in the console

## Stopping the Demo

To stop all components, press `Ctrl+C` in the terminal where you started the demo script.