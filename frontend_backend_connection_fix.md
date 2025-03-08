# Frontend to Backend Connection Configuration

## Current Setup
The frontend running on http://localhost:3000 connects to the backend API running on https://platform.aigenconsult.com.

### Configuration
- API requests from the frontend are proxied to the backend to avoid CORS issues
- The proxy configuration handles routing requests to the correct backend endpoint
- All API endpoints use relative URLs in the frontend code

### Root Cause of Previous Issues
Browser security policies block requests between different origins (including different ports on localhost). This is known as the "Same-Origin Policy" restriction.

## Solution Implemented

### 1. Proxy Configuration
Updated the proxy configuration in the frontend's `package.json`:
```json
{
  "proxy": "https://platform.aigenconsult.com"
}
```

This tells the React development server to proxy any API requests to the specified backend server, avoiding CORS issues entirely.

### 2. Updated API Client
Modified the API client to use relative URLs instead of absolute URLs:

- Changed the baseURL from `http://localhost:8000` to an empty string `''`
- Updated token fetch URL to use a relative path `/token` 
- Updated all API requests to use relative paths

### 3. Created Debugging Tools
- `debug_connection.sh`: Tests connectivity between frontend and backend
- `debug_api.js`: Adds detailed request/response logging to the frontend
- `restart_frontend.sh`: Restarts the frontend with the new configuration

## How It Works
With the proxy configuration:

1. Frontend sends request to `/api/v1/system` (relative URL)
2. React dev server receives the request and forwards it to `http://localhost:8000/api/v1/system`
3. The backend processes the request and sends the response back
4. React dev server forwards the response to the frontend

This bypasses CORS completely because, from the browser's perspective, the request is made to the same origin (localhost:3000).

## Testing the Fix
To verify the fix:
1. Ensure the backend is running: `./run_api_sqlite.sh`
2. Restart the frontend with new configuration: `./restart_frontend.sh`
3. Open browser to http://localhost:3000
4. Check browser console for successful API responses

## Common Issues
- If you change the proxy setting, you need to restart the frontend development server
- Environment variables override proxy settings, so ensure REACT_APP_API_URL is not set
- The proxy only works in development mode, for production builds you'll need to configure CORS properly