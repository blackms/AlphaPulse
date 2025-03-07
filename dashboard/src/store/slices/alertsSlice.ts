import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../rootReducer';
import { Alert, AlertPreferences, AlertRule, AlertSeverity } from '../../types/alerts';

interface AlertsState {
  alerts: Alert[];
  rules: AlertRule[];
  preferences: AlertPreferences;
  loading: boolean;
  error: string | null;
}

const initialState: AlertsState = {
  alerts: [],
  rules: [],
  preferences: {
    emailNotifications: true,
    pushNotifications: true,
    soundEnabled: true,
    alertThreshold: 'medium',
    channels: {
      email: true,
      sms: false,
      push: true,
      slack: false
    },
    preferences: {
      includeCritical: true,
      includeHigh: true,
      includeMedium: true,
      includeLow: false
    }
  },
  loading: false,
  error: null
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
    },
    fetchAlertsFailure(state, action: PayloadAction<string>) {
      state.loading = false;
      state.error = action.payload;
    },
    fetchRulesSuccess(state, action: PayloadAction<AlertRule[]>) {
      state.rules = action.payload;
    },
    fetchPreferencesSuccess(state, action: PayloadAction<AlertPreferences>) {
      state.preferences = action.payload;
    },
    addAlert(state, action: PayloadAction<Alert>) {
      state.alerts.unshift(action.payload);
    },
    updateAlertStatus(state, action: PayloadAction<{ id: string; acknowledged: boolean }>) {
      const { id, acknowledged } = action.payload;
      const alert = state.alerts.find(a => a.id === id);
      if (alert) {
        alert.acknowledged = acknowledged;
      }
    },
    markAllAsAcknowledged(state) {
      state.alerts.forEach(alert => {
        alert.acknowledged = true;
      });
    },
    clearAllAlerts(state) {
      state.alerts = [];
    },
    addRule(state, action: PayloadAction<AlertRule>) {
      state.rules.push(action.payload);
    },
    updateRule(state, action: PayloadAction<AlertRule>) {
      const index = state.rules.findIndex(r => r.id === action.payload.id);
      if (index !== -1) {
        state.rules[index] = action.payload;
      }
    },
    deleteRule(state, action: PayloadAction<string>) {
      state.rules = state.rules.filter(r => r.id !== action.payload);
    },
    toggleRuleEnabled(state, action: PayloadAction<{ id: string; enabled: boolean }>) {
      const { id, enabled } = action.payload;
      const rule = state.rules.find(r => r.id === id);
      if (rule) {
        rule.enabled = enabled;
      }
    },
    updatePreferences(state, action: PayloadAction<Partial<AlertPreferences>>) {
      state.preferences = { ...state.preferences, ...action.payload };
    },
    updateNotificationSettings(state, action: PayloadAction<AlertPreferences>) {
      state.preferences = action.payload;
    }
  }
});

export const {
  fetchAlertsStart,
  fetchAlertsSuccess,
  fetchAlertsFailure,
  fetchRulesSuccess,
  fetchPreferencesSuccess,
  addAlert,
  updateAlertStatus,
  markAllAsAcknowledged,
  clearAllAlerts,
  addRule,
  updateRule,
  deleteRule,
  toggleRuleEnabled,
  updatePreferences,
  updateNotificationSettings
} = alertsSlice.actions;

// Selectors
export const selectAllAlerts = (state: RootState) => state.alerts.alerts;
export const selectAlerts = selectAllAlerts; // Alias for backward compatibility
export const selectUnreadCount = (state: RootState) => 
  state.alerts.alerts.filter(alert => !alert.acknowledged).length;
export const selectUnreadAlertCount = selectUnreadCount; // Alias for backward compatibility
export const selectAlertById = (id: string) => (state: RootState) => 
  state.alerts.alerts.find(alert => alert.id === id);
export const selectAlertsByCategory = (category: string) => (state: RootState) => 
  state.alerts.alerts.filter(alert => alert.category === category);
export const selectAlertsBySeverity = (severity: string) => (state: RootState) => 
  state.alerts.alerts.filter(alert => alert.severity === severity);
export const selectActiveAlerts = (state: RootState) => 
  state.alerts.alerts.filter(alert => !alert.resolved);
export const selectAlertsLoading = (state: RootState) => state.alerts.loading;
export const selectAlertsError = (state: RootState) => state.alerts.error;
export const selectAllRules = (state: RootState) => state.alerts.rules;
export const selectRuleById = (id: string) => (state: RootState) => 
  state.alerts.rules.find(rule => rule.id === id);
export const selectEnabledRules = (state: RootState) => 
  state.alerts.rules.filter(rule => rule.enabled);
export const selectAlertPreferences = (state: RootState) => state.alerts.preferences;

// Re-export types properly
export type { Alert, AlertRule, AlertPreferences };
export { AlertSeverity };

export default alertsSlice.reducer;
