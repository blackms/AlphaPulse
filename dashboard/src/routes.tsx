import React from 'react';
import { Navigate, useRoutes } from 'react-router-dom';

// Layouts
import DashboardLayout from './layouts/DashboardLayout';
import AuthLayout from './layouts/AuthLayout';

// Pages
import DashboardPage from './pages/dashboard/DashboardPage';
import PortfolioPage from './pages/portfolio/PortfolioPage';
import TradingPage from './pages/trading/TradingPage';
import AlertsPage from './pages/alerts/AlertsPage';
import SystemStatusPage from './pages/system/SystemStatusPage';
import SettingsPage from './pages/settings/SettingsPage';
import LoginPage from './pages/auth/LoginPage';
import NotFoundPage from './pages/NotFoundPage';

export default function Router() {
  return useRoutes([
    {
      path: '/dashboard',
      element: <DashboardLayout />,
      children: [
        { path: '', element: <DashboardPage /> },
        { path: 'portfolio', element: <PortfolioPage /> },
        { path: 'trading', element: <TradingPage /> },
        { path: 'alerts', element: <AlertsPage /> },
        { path: 'system', element: <SystemStatusPage /> },
        { path: 'settings', element: <SettingsPage /> },
      ],
    },
    {
      path: '/',
      element: <AuthLayout />,
      children: [
        { path: '/', element: <Navigate to="/dashboard" /> },
        { path: 'login', element: <LoginPage /> },
        { path: '404', element: <NotFoundPage /> },
        { path: '*', element: <Navigate to="/404" /> },
      ],
    },
    { path: '*', element: <Navigate to="/404" replace /> },
  ]);
}