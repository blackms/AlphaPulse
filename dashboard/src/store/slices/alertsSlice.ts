import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Alert, AlertsState } from '../../types/alerts';

const initialState: AlertsState = {
  alerts: [],
  loading: false,
  error: null,
  lastUpdated: null,
};

const alertsSlice = createSlice({
  name: 'alerts',
  initialState,
  reducers: {
    fetchAlertsStart(state) {
      state.loading = true;
      state.error = null;
    },
    fetchAlertsSuccess(state, action: PayloadAction<Alert[]>) {
      state.alerts = action.payload;
      state.loading = false;
      state.error = null;
      state.lastUpdated = new Date().toISOString();
    },
    fetchAlertsFailure(state, action: PayloadAction<string>) {
      state.loading = false;
      state.error = action.payload;
    },
    addAlert(state, action: PayloadAction<Alert>) {
      state.alerts = [action.payload, ...state.alerts].slice(0, 100); // Keep last 100 alerts
      state.lastUpdated = new Date().toISOString();
    },
    removeAlert(state, action: PayloadAction<string>) {
      state.alerts = state.alerts.filter(alert => alert.id !== action.payload);
      state.lastUpdated = new Date().toISOString();
    },
    clearAlerts(state) {
      state.alerts = [];
      state.lastUpdated = new Date().toISOString();
    },
    acknowledgeAlert(state, action: PayloadAction<string>) {
      const alertIndex = state.alerts.findIndex(alert => alert.id === action.payload);
      if (alertIndex >= 0) {
        state.alerts[alertIndex].acknowledged = true;
        state.lastUpdated = new Date().toISOString();
      }
    },
  },
});

export const {
  fetchAlertsStart,
  fetchAlertsSuccess,
  fetchAlertsFailure,
  addAlert,
  removeAlert,
  clearAlerts,
  acknowledgeAlert,
} = alertsSlice.actions;

export default alertsSlice.reducer;