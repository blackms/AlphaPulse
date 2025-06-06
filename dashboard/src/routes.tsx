import React from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import DashboardLayout from './layouts/DashboardLayout';
import AuthLayout from './layouts/AuthLayout';
import LoginPage from './pages/auth/LoginPage';
import DashboardPage from './pages/dashboard/DashboardPage';
import PortfolioPage from './pages/portfolio/PortfolioPage';
import AlertsPage from './pages/alerts/AlertsPage';
import SystemStatusPage from './pages/system/SystemStatusPage';
import DiagnosticPage from './pages/system/DiagnosticPage';
import SettingsPage from './pages/settings/SettingsPage';
import NotFoundPage from './pages/NotFoundPage';

// Check if user is authenticated - always return true for testing
const isAuthenticated = () => {
  // For testing purposes, always return true
  localStorage.setItem('isAuthenticated', 'true');
  return true;
};

// Protected route component - bypassed for testing
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  // Always render children for testing
  return <>{children}</>;
};

const Router: React.FC = () => {
  return (
    <Routes>
      {/* Auth routes */}
      <Route element={<AuthLayout />}>
        <Route path="/login" element={<LoginPage />} />
      </Route>

      {/* Protected dashboard routes */}
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <DashboardLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<DashboardPage />} />
        <Route path="portfolio" element={<PortfolioPage />} />
        <Route path="trading" element={<DashboardPage />} /> {/* No TradingPage yet, using DashboardPage as fallback */}
        <Route path="alerts" element={<AlertsPage />} />
        <Route path="system" element={<SystemStatusPage />} />
        <Route path="system/diagnostics" element={<DiagnosticPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>

      {/* Root redirect */}
      <Route path="/" element={<Navigate to="/dashboard" />} />

      {/* 404 Not Found */}
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
};

export default Router;