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
interface ProtectedRouteProps {
  element: React.ReactElement;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ element }) => {
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