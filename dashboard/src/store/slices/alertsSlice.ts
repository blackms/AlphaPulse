import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

// Define the alert interface
export interface Alert {
  id: string;
  message: string;
  severity: 'info' | 'warning' | 'critical';
  source: string;
  timestamp: string;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
}

// Define the alerts state interface
interface AlertsState {
  alerts: Alert[];
  filteredAlerts: Alert[];
  selectedAlert: Alert | null;
  isLoading: boolean;
  error: string | null;
  filters: {
    severity: string[];
    acknowledged: boolean | null;
    timeRange: {
      start: string | null;
      end: string | null;
    };
    search: string;
  };
}

// Initial state
const initialState: AlertsState = {
  alerts: [],
  filteredAlerts: [],
  selectedAlert: null,
  isLoading: false,
  error: null,
  filters: {
    severity: [],
    acknowledged: null,
    timeRange: {
      start: null,
      end: null,
    },
    search: '',
  },
};

// Helper function to apply filters
const applyFilters = (alerts: Alert[], filters: AlertsState['filters']): Alert[] => {
  return alerts.filter((alert) => {
    // Filter by severity
    if (filters.severity.length > 0 && !filters.severity.includes(alert.severity)) {
      return false;
    }
    
    // Filter by acknowledged status
    if (filters.acknowledged !== null && alert.acknowledged !== filters.acknowledged) {
      return false;
    }
    
    // Filter by time range
    if (filters.timeRange.start && new Date(alert.timestamp) < new Date(filters.timeRange.start)) {
      return false;
    }
    
    if (filters.timeRange.end && new Date(alert.timestamp) > new Date(filters.timeRange.end)) {
      return false;
    }
    
    // Filter by search term
    if (filters.search && !alert.message.toLowerCase().includes(filters.search.toLowerCase()) && 
        !alert.source.toLowerCase().includes(filters.search.toLowerCase())) {
      return false;
    }
    
    return true;
  });
};

// Create the alerts slice
const alertsSlice = createSlice({
  name: 'alerts',
  initialState,
  reducers: {
    fetchAlertsRequest: (state) => {
      state.isLoading = true;
      state.error = null;
    },
    fetchAlertsSuccess: (state, action: PayloadAction<Alert[]>) => {
      state.isLoading = false;
      state.alerts = action.payload;
      state.filteredAlerts = applyFilters(action.payload, state.filters);
      state.error = null;
    },
    fetchAlertsFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;
    },
    selectAlert: (state, action: PayloadAction<string>) => {
      state.selectedAlert = state.alerts.find((alert) => alert.id === action.payload) || null;
    },
    clearSelectedAlert: (state) => {
      state.selectedAlert = null;
    },
    acknowledgeAlertRequest: (state, action: PayloadAction<string>) => {
      state.isLoading = true;
      state.error = null;
    },
    acknowledgeAlertSuccess: (state, action: PayloadAction<Alert>) => {
      state.isLoading = false;
      
      // Update the alert in the alerts array
      const index = state.alerts.findIndex((alert) => alert.id === action.payload.id);
      if (index !== -1) {
        state.alerts[index] = action.payload;
      }
      
      // Update the selected alert if it's the acknowledged alert
      if (state.selectedAlert && state.selectedAlert.id === action.payload.id) {
        state.selectedAlert = action.payload;
      }
      
      // Apply filters to update filteredAlerts
      state.filteredAlerts = applyFilters(state.alerts, state.filters);
      
      state.error = null;
    },
    acknowledgeAlertFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;
    },
    receiveNewAlert: (state, action: PayloadAction<Alert>) => {
      // Add the new alert to the beginning of the array
      state.alerts = [action.payload, ...state.alerts];
      
      // Apply filters to update filteredAlerts
      state.filteredAlerts = applyFilters(state.alerts, state.filters);
    },
    updateAlertFilters: (state, action: PayloadAction<Partial<AlertsState['filters']>>) => {
      // Update filters
      state.filters = {
        ...state.filters,
        ...action.payload,
      };
      
      // Apply filters
      state.filteredAlerts = applyFilters(state.alerts, state.filters);
    },
    clearAlertFilters: (state) => {
      state.filters = initialState.filters;
      state.filteredAlerts = state.alerts;
    },
  },
});

// Export actions
export const {
  fetchAlertsRequest,
  fetchAlertsSuccess,
  fetchAlertsFailure,
  selectAlert,
  clearSelectedAlert,
  acknowledgeAlertRequest,
  acknowledgeAlertSuccess,
  acknowledgeAlertFailure,
  receiveNewAlert,
  updateAlertFilters,
  clearAlertFilters,
} = alertsSlice.actions;

// Export selectors
export const selectAlerts = (state: RootState) => state.alerts.alerts;
export const selectFilteredAlerts = (state: RootState) => state.alerts.filteredAlerts;
export const selectSelectedAlert = (state: RootState) => state.alerts.selectedAlert;
export const selectAlertsLoading = (state: RootState) => state.alerts.isLoading;
export const selectAlertsError = (state: RootState) => state.alerts.error;
export const selectAlertFilters = (state: RootState) => state.alerts.filters;

// Export reducer
export default alertsSlice.reducer;