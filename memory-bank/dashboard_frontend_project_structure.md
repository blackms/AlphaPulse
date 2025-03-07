# Dashboard Frontend Initial Project Structure

This document outlines the initial file and directory structure for the Dashboard Frontend implementation.

## Project Setup Instructions

### 1. Create React Application with TypeScript

```bash
npx create-react-app alpha-pulse-dashboard --template typescript
cd alpha-pulse-dashboard
```

### 2. Install Core Dependencies

```bash
npm install @reduxjs/toolkit react-redux react-router-dom axios socket.io-client
npm install @mui/material @mui/icons-material @emotion/react @emotion/styled
npm install chart.js react-chartjs-2 d3
npm install jwt-decode date-fns lodash
```

### 3. Install Development Dependencies

```bash
npm install --save-dev @testing-library/react @testing-library/jest-dom @testing-library/user-event
npm install --save-dev eslint prettier eslint-config-prettier eslint-plugin-react-hooks
npm install --save-dev msw cypress
```

## Project Directory Structure

```
alpha-pulse-dashboard/
├── public/
│   ├── favicon.ico
│   ├── index.html
│   ├── logo192.png
│   ├── logo512.png
│   ├── manifest.json
│   └── robots.txt
├── src/
│   ├── assets/
│   │   ├── images/
│   │   │   ├── logo.svg
│   │   │   └── dashboard-bg.jpg
│   │   ├── styles/
│   │   │   ├── global.css
│   │   │   └── variables.css
│   │   └── fonts/
│   │       └── (custom fonts)
│   ├── components/
│   │   ├── alerts/
│   │   │   ├── AlertBadge.tsx
│   │   │   ├── AlertItem.tsx
│   │   │   ├── AlertList.tsx
│   │   │   ├── AlertsOverview.tsx
│   │   │   └── index.ts
│   │   ├── charts/
│   │   │   ├── AreaChart.tsx
│   │   │   ├── BarChart.tsx
│   │   │   ├── LineChart.tsx
│   │   │   ├── PieChart.tsx
│   │   │   ├── SparkLine.tsx
│   │   │   └── index.ts
│   │   ├── common/
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── Dropdown.tsx
│   │   │   ├── Loading.tsx
│   │   │   ├── Modal.tsx
│   │   │   ├── Pagination.tsx
│   │   │   ├── Tooltip.tsx
│   │   │   └── index.ts
│   │   ├── layout/
│   │   │   ├── AppBar.tsx
│   │   │   ├── Footer.tsx
│   │   │   ├── MainLayout.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── index.ts
│   │   ├── portfolio/
│   │   │   ├── AssetAllocation.tsx
│   │   │   ├── PerformanceChart.tsx
│   │   │   ├── PortfolioOverview.tsx
│   │   │   ├── PositionTable.tsx
│   │   │   └── index.ts
│   │   └── trading/
│   │       ├── TradeBadge.tsx
│   │       ├── TradeHistory.tsx
│   │       ├── TradeItem.tsx
│   │       ├── TradingOverview.tsx
│   │       └── index.ts
│   ├── hooks/
│   │   ├── useAlerts.ts
│   │   ├── useAuth.ts
│   │   ├── useMetrics.ts
│   │   ├── usePortfolio.ts
│   │   ├── useSocket.ts
│   │   ├── useTrading.ts
│   │   └── index.ts
│   ├── pages/
│   │   ├── alerts/
│   │   │   ├── AlertsDetail.tsx
│   │   │   ├── AlertsPage.tsx
│   │   │   └── index.ts
│   │   ├── auth/
│   │   │   ├── LoginPage.tsx
│   │   │   ├── ProfilePage.tsx
│   │   │   └── index.ts
│   │   ├── dashboard/
│   │   │   ├── DashboardPage.tsx
│   │   │   └── index.ts
│   │   ├── portfolio/
│   │   │   ├── PortfolioDetailPage.tsx
│   │   │   ├── PortfolioPage.tsx
│   │   │   └── index.ts
│   │   ├── settings/
│   │   │   ├── ApiKeysPage.tsx
│   │   │   ├── PreferencesPage.tsx
│   │   │   ├── SettingsPage.tsx
│   │   │   └── index.ts
│   │   ├── system/
│   │   │   ├── SystemMetricsPage.tsx
│   │   │   ├── SystemStatusPage.tsx
│   │   │   └── index.ts
│   │   └── trading/
│   │       ├── TradeDetailPage.tsx
│   │       ├── TradingPage.tsx
│   │       └── index.ts
│   ├── services/
│   │   ├── api/
│   │   │   ├── alertsApi.ts
│   │   │   ├── apiClient.ts
│   │   │   ├── metricsApi.ts
│   │   │   ├── portfolioApi.ts
│   │   │   ├── systemApi.ts
│   │   │   ├── tradingApi.ts
│   │   │   └── index.ts
│   │   ├── auth/
│   │   │   ├── authService.ts
│   │   │   ├── tokenService.ts
│   │   │   └── index.ts
│   │   └── socket/
│   │       ├── socketClient.ts
│   │       ├── socketEvents.ts
│   │       └── index.ts
│   ├── store/
│   │   ├── slices/
│   │   │   ├── alertsSlice.ts
│   │   │   ├── authSlice.ts
│   │   │   ├── metricsSlice.ts
│   │   │   ├── portfolioSlice.ts
│   │   │   ├── systemSlice.ts
│   │   │   ├── tradingSlice.ts
│   │   │   ├── uiSlice.ts
│   │   │   └── index.ts
│   │   ├── selectors/
│   │   │   ├── alertsSelectors.ts
│   │   │   ├── portfolioSelectors.ts
│   │   │   ├── tradingSelectors.ts
│   │   │   └── index.ts
│   │   ├── middleware/
│   │   │   ├── apiMiddleware.ts
│   │   │   ├── socketMiddleware.ts
│   │   │   └── index.ts
│   │   ├── rootReducer.ts
│   │   └── store.ts
│   ├── types/
│   │   ├── alerts.ts
│   │   ├── api.ts
│   │   ├── auth.ts
│   │   ├── metrics.ts
│   │   ├── portfolio.ts
│   │   ├── system.ts
│   │   ├── trading.ts
│   │   └── index.ts
│   ├── utils/
│   │   ├── constants.ts
│   │   ├── dates.ts
│   │   ├── formatting.ts
│   │   ├── localStorage.ts
│   │   ├── numbers.ts
│   │   ├── validation.ts
│   │   └── index.ts
│   ├── App.tsx
│   ├── index.tsx
│   ├── routes.tsx
│   ├── setupTests.ts
│   └── react-app-env.d.ts
├── .eslintrc.js
├── .prettierrc
├── .gitignore
├── package.json
├── tsconfig.json
├── README.md
└── cypress/
    ├── fixtures/
    │   └── example.json
    ├── integration/
    │   ├── alerts.spec.ts
    │   ├── auth.spec.ts
    │   ├── dashboard.spec.ts
    │   ├── portfolio.spec.ts
    │   └── trading.spec.ts
    ├── plugins/
    │   └── index.js
    └── support/
        ├── commands.js
        └── index.js
```

## Key Files and Their Purposes

### Core Application Files

1. **src/index.tsx**: Application entry point
2. **src/App.tsx**: Main application component
3. **src/routes.tsx**: Application routing configuration

### State Management

1. **src/store/store.ts**: Redux store configuration
2. **src/store/rootReducer.ts**: Combined reducer for all slices
3. **src/store/slices/**: Feature-specific Redux slices
4. **src/store/selectors/**: Memoized selectors for accessing state

### API Communication

1. **src/services/api/apiClient.ts**: Base API client with axios
2. **src/services/api/*.ts**: Domain-specific API services
3. **src/services/socket/socketClient.ts**: WebSocket client

### Authentication

1. **src/services/auth/authService.ts**: Authentication logic
2. **src/services/auth/tokenService.ts**: JWT token management

### UI Components

1. **src/components/layout/**: Application layout components
2. **src/components/common/**: Reusable UI elements
3. **src/components/charts/**: Data visualization components
4. **src/components/alerts/**, **src/components/portfolio/**, etc.: Domain-specific components

### Pages

1. **src/pages/dashboard/DashboardPage.tsx**: Main dashboard page
2. **src/pages/*/**: Feature-specific pages

## Initial File Templates

### 1. src/index.tsx

```tsx
import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { store } from './store/store';
import App from './App';
import './assets/styles/global.css';

ReactDOM.render(
  <React.StrictMode>
    <Provider store={store}>
      <App />
    </Provider>
  </React.StrictMode>,
  document.getElementById('root')
);
```

### 2. src/App.tsx

```tsx
import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import AppRoutes from './routes';

const theme = createTheme({
  palette: {
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppRoutes />
      </Router>
    </ThemeProvider>
  );
};

export default App;
```

### 3. src/routes.tsx

```tsx
import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './hooks/useAuth';
import MainLayout from './components/layout/MainLayout';
import LoginPage from './pages/auth/LoginPage';
import DashboardPage from './pages/dashboard/DashboardPage';
import AlertsPage from './pages/alerts/AlertsPage';
import PortfolioPage from './pages/portfolio/PortfolioPage';
import TradingPage from './pages/trading/TradingPage';
import SystemStatusPage from './pages/system/SystemStatusPage';
import SettingsPage from './pages/settings/SettingsPage';

// Protected route wrapper
const ProtectedRoute: React.FC<{ element: React.ReactElement }> = ({ element }) => {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? element : <Navigate to="/login" />;
};

const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/" element={<ProtectedRoute element={<MainLayout />} />}>
        <Route index element={<DashboardPage />} />
        <Route path="alerts" element={<AlertsPage />} />
        <Route path="portfolio" element={<PortfolioPage />} />
        <Route path="trading" element={<TradingPage />} />
        <Route path="system" element={<SystemStatusPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
};

export default AppRoutes;
```

### 4. src/store/store.ts

```tsx
import { configureStore } from '@reduxjs/toolkit';
import rootReducer from './rootReducer';
import { apiMiddleware } from './middleware/apiMiddleware';
import { socketMiddleware } from './middleware/socketMiddleware';

export const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(apiMiddleware, socketMiddleware),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

### 5. src/services/api/apiClient.ts

```tsx
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { tokenService } from '../auth';

// Base API configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding token to requests
apiClient.interceptors.request.use(
  (config: AxiosRequestConfig) => {
    const token = tokenService.getToken();
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for handling token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    // If the error is unauthorized and not already retrying
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        // Try to refresh the token
        await tokenService.refreshToken();
        
        // Update authorization header
        const token = tokenService.getToken();
        if (token) {
          originalRequest.headers.Authorization = `Bearer ${token}`;
        }
        
        // Retry the original request
        return apiClient(originalRequest);
      } catch (refreshError) {
        // If refresh fails, redirect to login
        tokenService.clearToken();
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }
    
    return Promise.reject(error);
  }
);

export default apiClient;
```

## Next Steps After Structure Creation

1. Create the basic project structure
2. Implement the authentication system
3. Set up core API services
4. Create the main layout components
5. Implement the dashboard page
6. Add WebSocket integration
7. Build out individual feature pages