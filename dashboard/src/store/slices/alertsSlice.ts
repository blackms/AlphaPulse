import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type AlertSeverity = 'success' | 'info' | 'warning' | 'error';
export type AlertCategory = 'system' | 'trading' | 'portfolio' | 'security' | 'market';

export interface Alert {
  id: string;
  title: string;
  message: string;
  severity: AlertSeverity;
  category: AlertCategory;
  timestamp: number;
  read: boolean;
  actionRequired: boolean;
  actionLink?: string;
  actionText?: string;
  relatedAsset?: string;
}

interface AlertsState {
  alerts: Alert[];
  unreadCount: number;
  loading: boolean;
  lastChecked: number | null;
}

const initialState: AlertsState = {
  alerts: [
    {
      id: 'a1',
      title: 'Market Volatility Alert',
      message: 'Unusual market volatility detected in crypto assets. Risk management has automatically adjusted position sizes.',
      severity: 'warning',
      category: 'market',
      timestamp: Date.now() - 45 * 60000, // 45 minutes ago
      read: false,
      actionRequired: false,
    },
    {
      id: 'a2',
      title: 'New Trading Signal',
      message: 'Technical Agent has generated a strong buy signal for ETH based on breakout pattern.',
      severity: 'info',
      category: 'trading',
      timestamp: Date.now() - 120 * 60000, // 2 hours ago
      read: false,
      actionRequired: true,
      actionLink: '/dashboard/trading',
      actionText: 'Review Signal',
      relatedAsset: 'eth',
    },
    {
      id: 'a3',
      title: 'Portfolio Rebalancing Complete',
      message: 'Automated portfolio rebalancing has been completed successfully. Asset weights adjusted to maintain target allocation.',
      severity: 'success',
      category: 'portfolio',
      timestamp: Date.now() - 240 * 60000, // 4 hours ago
      read: true,
      actionRequired: false,
    },
    {
      id: 'a4',
      title: 'System Maintenance',
      message: 'Scheduled system maintenance will occur on Saturday at 02:00 UTC. No trading will be affected.',
      severity: 'info',
      category: 'system',
      timestamp: Date.now() - 360 * 60000, // 6 hours ago
      read: true,
      actionRequired: false,
    },
    {
      id: 'a5',
      title: 'Stop Loss Triggered',
      message: 'Stop loss has been triggered for SOL position. 15% of position has been sold to limit downside risk.',
      severity: 'warning',
      category: 'trading',
      timestamp: Date.now() - 510 * 60000, // 8.5 hours ago
      read: true,
      actionRequired: false,
      relatedAsset: 'sol',
    }
  ],
  unreadCount: 2, // Matches the number of unread alerts in initialState
  loading: false,
  lastChecked: Date.now() - 30 * 60000, // 30 minutes ago
};

const alertsSlice = createSlice({
  name: 'alerts',
  initialState,
  reducers: {
    fetchAlertsStart: (state) => {
      state.loading = true;
    },
    fetchAlertsSuccess: (state, action: PayloadAction<Alert[]>) => {
      state.alerts = action.payload;
      state.unreadCount = action.payload.filter(alert => !alert.read).length;
      state.loading = false;
      state.lastChecked = Date.now();
    },
    fetchAlertsFailure: (state) => {
      state.loading = false;
    },
    markAlertAsRead: (state, action: PayloadAction<string>) => {
      const alert = state.alerts.find(a => a.id === action.payload);
      if (alert && !alert.read) {
        alert.read = true;
        state.unreadCount = Math.max(0, state.unreadCount - 1);
      }
    },
    markAllAlertsAsRead: (state) => {
      state.alerts.forEach(alert => {
        alert.read = true;
      });
      state.unreadCount = 0;
    },
    addAlert: (state, action: PayloadAction<Omit<Alert, 'id' | 'timestamp' | 'read'>>) => {
      const newAlert: Alert = {
        ...action.payload,
        id: `a${Date.now()}`,
        timestamp: Date.now(),
        read: false,
      };
      state.alerts.unshift(newAlert);
      state.unreadCount += 1;
    },
    removeAlert: (state, action: PayloadAction<string>) => {
      const alertIndex = state.alerts.findIndex(a => a.id === action.payload);
      if (alertIndex !== -1) {
        const wasUnread = !state.alerts[alertIndex].read;
        state.alerts.splice(alertIndex, 1);
        if (wasUnread) {
          state.unreadCount = Math.max(0, state.unreadCount - 1);
        }
      }
    },
    clearAlerts: (state) => {
      state.alerts = [];
      state.unreadCount = 0;
    },
  },
});

export const {
  fetchAlertsStart,
  fetchAlertsSuccess,
  fetchAlertsFailure,
  markAlertAsRead,
  markAllAlertsAsRead,
  addAlert,
  removeAlert,
  clearAlerts,
} = alertsSlice.actions;

// Selectors
export const selectAlerts = (state: RootState) => state.alerts.alerts;
export const selectUnreadCount = (state: RootState) => state.alerts.unreadCount;
export const selectIsLoading = (state: RootState) => state.alerts.loading;
export const selectLastChecked = (state: RootState) => state.alerts.lastChecked;
export const selectAlertsByCategory = (category: AlertCategory) => 
  (state: RootState) => state.alerts.alerts.filter(alert => alert.category === category);
export const selectAlertsByAsset = (asset: string) =>
  (state: RootState) => state.alerts.alerts.filter(alert => alert.relatedAsset === asset);

export default alertsSlice.reducer;