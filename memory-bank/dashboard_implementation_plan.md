# Dashboard Frontend Implementation Plan

This document outlines the specific implementation steps for completing the AI Hedge Fund Dashboard Frontend (Task 1.5 from our implementation plan).

## Current Status

The dashboard backend API has been implemented with endpoints for:
- Authentication and user management
- Portfolio data and metrics
- Trading history and execution
- Alerts and notification management
- System metrics and health

The dashboard frontend design has been documented but implementation has not begun.

## Implementation Tasks

### 1. Project Setup and Configuration (1 day)

- [ ] Initialize React project with Create React App and TypeScript
```bash
npx create-react-app dashboard --template typescript
```

- [ ] Add core dependencies
```bash
npm install react-router-dom axios recharts @reduxjs/toolkit socket.io-client react-query tailwindcss
```

- [ ] Configure Tailwind CSS
- [ ] Set up project structure following the design document
- [ ] Create environment configuration (dev/prod)

### 2. Authentication Implementation (1 day)

- [ ] Create login page component
- [ ] Implement token-based authentication
- [ ] Add persistent login state with localStorage
- [ ] Implement protected routes
- [ ] Add user context/provider
- [ ] Create auth utility functions

### 3. Core Layout and Navigation (1 day)

- [ ] Create main layout component
- [ ] Implement responsive sidebar
- [ ] Create navigation menu with active state
- [ ] Add header with user dropdown
- [ ] Implement dark/light mode toggle
- [ ] Create loading and error states

### 4. Dashboard Home Page (1 day)

- [ ] Create dashboard grid layout
- [ ] Implement key metrics cards
- [ ] Add portfolio value chart
- [ ] Create asset allocation chart
- [ ] Add recent alerts panel
- [ ] Create recent trades panel

### 5. Portfolio View (2 days)

- [ ] Create portfolio summary section
- [ ] Implement portfolio composition chart
- [ ] Build asset table with details
- [ ] Add portfolio performance chart
- [ ] Create portfolio metrics section
- [ ] Implement filtering and timeframe selection

### 6. Trades View (1 day)

- [ ] Create trade history table
- [ ] Implement sorting and filtering
- [ ] Add trade details modal/panel
- [ ] Create trade visualization chart
- [ ] Implement trade statistics

### 7. Alerts Management (1 day)

- [ ] Create alerts list with severity indicators
- [ ] Add filtering and sorting options
- [ ] Implement alert acknowledgment
- [ ] Create alert details view
- [ ] Add alert history section

### 8. Real-time Updates (1 day)

- [ ] Set up WebSocket connection management
- [ ] Implement connection status indicator
- [ ] Create subscription management
- [ ] Add real-time updates for metrics
- [ ] Implement notification system for alerts
- [ ] Add reconnection logic

### 9. Charts and Visualizations (2 days)

- [ ] Create reusable chart components
- [ ] Implement time series chart with zoom
- [ ] Add candlestick charts for price data
- [ ] Create bar/column charts for comparisons
- [ ] Implement pie/donut charts for allocations
- [ ] Add tooltips and interactive elements

### 10. Responsive Design and Styling (1 day)

- [ ] Refine responsive layouts for all viewports
- [ ] Implement consistent spacing and typography
- [ ] Add animations and transitions
- [ ] Create loading skeletons
- [ ] Polish UI details and accessibility

### 11. Testing and Documentation (1 day)

- [ ] Write component tests
- [ ] Add integration tests
- [ ] Create storybook documentation
- [ ] Write API integration documentation
- [ ] Create user guide

## Component Architecture

```
dashboard/
├── src/
│   ├── components/
│   │   ├── common/
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── Table.tsx
│   │   │   ├── Alert.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Modal.tsx
│   │   │   └── ...
│   │   ├── layout/
│   │   │   ├── MainLayout.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   └── Footer.tsx
│   │   ├── charts/
│   │   │   ├── TimeSeriesChart.tsx
│   │   │   ├── PieChart.tsx
│   │   │   ├── CandlestickChart.tsx
│   │   │   └── ...
│   │   ├── dashboard/
│   │   │   ├── MetricCard.tsx
│   │   │   ├── PortfolioSummary.tsx
│   │   │   └── ...
│   │   ├── portfolio/
│   │   │   ├── AssetTable.tsx
│   │   │   ├── AllocationChart.tsx
│   │   │   └── ...
│   │   ├── trades/
│   │   │   ├── TradeTable.tsx
│   │   │   ├── TradeDetails.tsx
│   │   │   └── ...
│   │   └── alerts/
│   │       ├── AlertList.tsx
│   │       ├── AlertBadge.tsx
│   │       └── ...
│   ├── pages/
│   │   ├── Login.tsx
│   │   ├── Dashboard.tsx
│   │   ├── Portfolio.tsx
│   │   ├── Trades.tsx
│   │   ├── Alerts.tsx
│   │   └── Settings.tsx
│   ├── hooks/
│   │   ├── useAuth.ts
│   │   ├── useAPI.ts
│   │   ├── useWebSocket.ts
│   │   └── ...
│   ├── store/
│   │   ├── index.ts
│   │   ├── authSlice.ts
│   │   ├── metricsSlice.ts
│   │   ├── portfolioSlice.ts
│   │   ├── tradesSlice.ts
│   │   └── alertsSlice.ts
│   ├── api/
│   │   ├── client.ts
│   │   ├── auth.ts
│   │   ├── metrics.ts
│   │   ├── portfolio.ts
│   │   ├── trades.ts
│   │   └── alerts.ts
│   ├── utils/
│   │   ├── formatting.ts
│   │   ├── calculations.ts
│   │   ├── dates.ts
│   │   └── ...
│   ├── App.tsx
│   └── index.tsx
```

## API Integration

### Authentication
```typescript
// src/api/auth.ts
import client from './client';

export const login = async (username: string, password: string) => {
  const response = await client.post('/token', { username, password });
  return response.data;
};

export const getUser = async () => {
  const response = await client.get('/api/v1/user');
  return response.data;
};
```

### Metrics
```typescript
// src/api/metrics.ts
import client from './client';

export const getMetricHistory = async (metricName: string, timeRange: string) => {
  const response = await client.get(`/api/v1/metrics/${metricName}`, {
    params: { timeRange }
  });
  return response.data;
};

export const getLatestMetric = async (metricName: string) => {
  const response = await client.get(`/api/v1/metrics/${metricName}/latest`);
  return response.data;
};
```

### WebSocket Integration
```typescript
// src/hooks/useWebSocket.ts
import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { useSelector } from 'react-redux';
import { RootState } from '../store';

export const useWebSocket = (channels: string[]) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const { token } = useSelector((state: RootState) => state.auth);
  
  useEffect(() => {
    if (!token) return;
    
    const newSocket = io(process.env.REACT_APP_WS_URL || 'ws://localhost:8000', {
      auth: { token },
      transports: ['websocket']
    });
    
    newSocket.on('connect', () => {
      setConnected(true);
      newSocket.emit('subscribe', { channels });
    });
    
    newSocket.on('disconnect', () => {
      setConnected(false);
    });
    
    setSocket(newSocket);
    
    return () => {
      newSocket.disconnect();
    };
  }, [token, channels]);
  
  return { socket, connected };
};
```

## Redux Store Setup

```typescript
// src/store/index.ts
import { configureStore } from '@reduxjs/toolkit';
import authReducer from './authSlice';
import metricsReducer from './metricsSlice';
import portfolioReducer from './portfolioSlice';
import tradesReducer from './tradesSlice';
import alertsReducer from './alertsSlice';
import uiReducer from './uiSlice';

export const store = configureStore({
  reducer: {
    auth: authReducer,
    metrics: metricsReducer,
    portfolio: portfolioReducer,
    trades: tradesReducer,
    alerts: alertsReducer,
    ui: uiReducer
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false
    })
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

## Responsive Design Considerations

1. **Mobile-First Approach**
   - Design layouts starting with mobile viewport
   - Use fluid grids and flexible components
   - Implement responsive typography

2. **Breakpoints**
   - Small: < 640px (mobile)
   - Medium: 640px - 1024px (tablet)
   - Large: 1024px - 1440px (desktop)
   - XL: > 1440px (large desktop)

3. **Layout Adaptations**
   - Stack cards vertically on mobile
   - Collapse sidebar to menu icon on small screens
   - Adjust table displays for smaller screens
   - Use appropriate touch targets for mobile

## Performance Optimizations

1. **Code Splitting**
   - Split bundles by route
   - Lazy load components
   ```typescript
   const Portfolio = React.lazy(() => import('./pages/Portfolio'));
   ```

2. **Rendering Optimization**
   - Memoize expensive components
   - Use virtualization for long lists
   - Optimize re-renders

3. **Data Caching**
   - Implement React Query for API caching
   - Use appropriate stale times
   - Prefetch critical data

## Development Approach

1. **Sprint 1: Foundation** (3 days)
   - Project setup and core utilities
   - Authentication implementation
   - Main layout and navigation

2. **Sprint 2: Core Views** (3 days)
   - Dashboard home page
   - Portfolio view basics
   - Initial charts

3. **Sprint 3: Advanced Features** (3 days)
   - Trades and alerts views
   - WebSocket integration
   - Real-time updates

4. **Sprint 4: Polish** (3 days)
   - Responsive design refinement
   - Additional charts
   - Testing and documentation

## Next Steps

1. Set up the project repository
2. Create initial project structure
3. Implement authentication
4. Build the core layout
5. Schedule a review meeting after the foundation is complete