#!/bin/bash
# Script to fix critical TypeScript issues in the dashboard

echo "==============================================="
echo "  FIXING DASHBOARD TYPESCRIPT ISSUES"
echo "==============================================="
echo ""

# Navigate to dashboard directory
cd dashboard || {
  echo "Error: dashboard directory not found!"
  echo "Make sure you're running this script from the project root."
  exit 1
}

# 1. Fix missing recharts dependency
echo "Installing missing dependencies..."
npm install --save recharts

# 2. Fix alertsSlice.ts
echo "Fixing alertsSlice.ts..."
cat > src/store/slices/alertsSlice.ts << 'EOL'
import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../rootReducer';
import { Alert, AlertPreferences, AlertRule } from '../../types/alerts';

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
    alertThreshold: 'medium'
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
  updatePreferences
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

export default alertsSlice.reducer;
EOL

# 3. Fix uiSlice.ts
echo "Fixing uiSlice.ts..."
cat > src/store/slices/uiSlice.ts << 'EOL'
import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../rootReducer';

interface UiState {
  darkMode: boolean;
  sidebarOpen: boolean;
  sidebarSize: 'normal' | 'compact';
  notifications: {
    show: boolean;
    message: string;
    type: 'success' | 'error' | 'info' | 'warning';
  };
  confirmDialog: {
    open: boolean;
    title: string;
    message: string;
    confirmText: string;
    cancelText: string;
    onConfirm: (() => void) | null;
  };
}

const initialState: UiState = {
  darkMode: false,
  sidebarOpen: true,
  sidebarSize: 'normal',
  notifications: {
    show: false,
    message: '',
    type: 'info'
  },
  confirmDialog: {
    open: false,
    title: '',
    message: '',
    confirmText: 'Confirm',
    cancelText: 'Cancel',
    onConfirm: null
  }
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleDarkMode(state) {
      state.darkMode = !state.darkMode;
    },
    setDarkMode(state, action: PayloadAction<boolean>) {
      state.darkMode = action.payload;
    },
    toggleSidebar(state) {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen(state, action: PayloadAction<boolean>) {
      state.sidebarOpen = action.payload;
    },
    setSidebarSize(state, action: PayloadAction<'normal' | 'compact'>) {
      state.sidebarSize = action.payload;
    },
    showNotification(state, action: PayloadAction<{ message: string; type: 'success' | 'error' | 'info' | 'warning' }>) {
      state.notifications = {
        show: true,
        message: action.payload.message,
        type: action.payload.type
      };
    },
    hideNotification(state) {
      state.notifications.show = false;
    },
    showConfirmDialog(state, action: PayloadAction<{
      title: string;
      message: string;
      confirmText?: string;
      cancelText?: string;
      onConfirm?: () => void;
    }>) {
      state.confirmDialog = {
        open: true,
        title: action.payload.title,
        message: action.payload.message,
        confirmText: action.payload.confirmText || 'Confirm',
        cancelText: action.payload.cancelText || 'Cancel',
        onConfirm: action.payload.onConfirm || null
      };
    },
    hideConfirmDialog(state) {
      state.confirmDialog.open = false;
    }
  }
});

export const {
  toggleDarkMode,
  setDarkMode,
  toggleSidebar,
  setSidebarOpen,
  setSidebarSize,
  showNotification,
  hideNotification,
  showConfirmDialog,
  hideConfirmDialog
} = uiSlice.actions;

// Selectors
export const selectDarkMode = (state: RootState) => state.ui.darkMode;
export const selectSidebarOpen = (state: RootState) => state.ui.sidebarOpen;
export const selectSidebarSize = (state: RootState) => state.ui.sidebarSize;
export const selectNotification = (state: RootState) => state.ui.notifications;
export const selectConfirmDialog = (state: RootState) => state.ui.confirmDialog;

export default uiSlice.reducer;
EOL

# 4. Fix systemSlice.ts
echo "Fixing systemSlice.ts..."
cat > src/store/slices/systemSlice.ts << 'EOL'
import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../rootReducer';
import { SystemComponent, SystemLog, SystemMetric } from '../../types/system';

export enum SystemStatus {
  OPERATIONAL = 'operational',
  DEGRADED = 'degraded',
  MAINTENANCE = 'maintenance',
  OUTAGE = 'outage'
}

export enum ComponentStatus {
  HEALTHY = 'healthy',
  WARNING = 'warning',
  ERROR = 'error',
  OFFLINE = 'offline'
}

interface SystemState {
  status: SystemStatus;
  components: SystemComponent[];
  logs: SystemLog[];
  metrics: SystemMetric[];
  lastUpdated: string | null;
  performance: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
  };
  uptime: number;
  statusLoading: boolean;
  statusError: string | null;
  logsLoading: boolean;
  logsError: string | null;
  metricsLoading: boolean;
  metricsError: string | null;
}

const initialState: SystemState = {
  status: SystemStatus.OPERATIONAL,
  components: [],
  logs: [],
  metrics: [],
  lastUpdated: null,
  performance: {
    cpu: 0,
    memory: 0,
    disk: 0,
    network: 0
  },
  uptime: 0,
  statusLoading: false,
  statusError: null,
  logsLoading: false,
  logsError: null,
  metricsLoading: false,
  metricsError: null
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    fetchSystemStatusStart(state) {
      state.statusLoading = true;
      state.statusError = null;
    },
    fetchSystemStatusSuccess(state, action: PayloadAction<{
      status: SystemStatus;
      components: SystemComponent[];
      lastUpdated: string;
      uptime: number;
    }>) {
      state.status = action.payload.status;
      state.components = action.payload.components;
      state.lastUpdated = action.payload.lastUpdated;
      state.uptime = action.payload.uptime;
      state.statusLoading = false;
    },
    fetchSystemStatusFailure(state, action: PayloadAction<string>) {
      state.statusLoading = false;
      state.statusError = action.payload;
    },
    fetchSystemLogsStart(state) {
      state.logsLoading = true;
      state.logsError = null;
    },
    fetchSystemLogsSuccess(state, action: PayloadAction<SystemLog[]>) {
      state.logs = action.payload;
      state.logsLoading = false;
    },
    fetchSystemLogsFailure(state, action: PayloadAction<string>) {
      state.logsLoading = false;
      state.logsError = action.payload;
    },
    fetchSystemMetricsStart(state) {
      state.metricsLoading = true;
      state.metricsError = null;
    },
    fetchSystemMetricsSuccess(state, action: PayloadAction<SystemMetric[]>) {
      state.metrics = action.payload;
      state.metricsLoading = false;
    },
    fetchSystemMetricsFailure(state, action: PayloadAction<string>) {
      state.metricsLoading = false;
      state.metricsError = action.payload;
    },
    updateComponentStatus(state, action: PayloadAction<{
      id: string;
      status: ComponentStatus;
    }>) {
      const component = state.components.find(c => c.id === action.payload.id);
      if (component) {
        component.status = action.payload.status;
      }
    },
    updateSystemPerformance(state, action: PayloadAction<{
      cpu: number;
      memory: number;
      disk: number;
      network: number;
    }>) {
      state.performance = action.payload;
    },
    addSystemLog(state, action: PayloadAction<SystemLog>) {
      state.logs.unshift(action.payload);
      // Keep only the latest 100 logs
      if (state.logs.length > 100) {
        state.logs = state.logs.slice(0, 100);
      }
    },
    clearSystemLogs(state) {
      state.logs = [];
    }
  }
});

export const {
  fetchSystemStatusStart,
  fetchSystemStatusSuccess,
  fetchSystemStatusFailure,
  fetchSystemLogsStart,
  fetchSystemLogsSuccess,
  fetchSystemLogsFailure,
  fetchSystemMetricsStart,
  fetchSystemMetricsSuccess,
  fetchSystemMetricsFailure,
  updateComponentStatus,
  updateSystemPerformance,
  addSystemLog,
  clearSystemLogs
} = systemSlice.actions;

// Alias for backward compatibility
export const fetchSystemStart = fetchSystemStatusStart;

// Selectors
export const selectSystemStatus = (state: RootState) => state.system.status;
export const selectSystemComponents = (state: RootState) => state.system.components;
export const selectSystemLogs = (state: RootState) => state.system.logs;
export const selectSystemMetrics = (state: RootState) => state.system.metrics;
export const selectSystemLastUpdated = (state: RootState) => state.system.lastUpdated;
export const selectSystemUptime = (state: RootState) => state.system.uptime;
export const selectSystemPerformance = (state: RootState) => state.system.performance;
export const selectStatusLoading = (state: RootState) => state.system.statusLoading;
export const selectStatusError = (state: RootState) => state.system.statusError;
export const selectLogsLoading = (state: RootState) => state.system.logsLoading;
export const selectLogsError = (state: RootState) => state.system.logsError;
export const selectMetricsLoading = (state: RootState) => state.system.metricsLoading;
export const selectMetricsError = (state: RootState) => state.system.metricsError;
export const selectSystemLoading = (state: RootState) => 
  state.system.statusLoading || state.system.logsLoading || state.system.metricsLoading;
export const selectComponentById = (id: string) => (state: RootState) => 
  state.system.components.find(c => c.id === id);
export const selectComponentsByType = (type: string) => (state: RootState) => 
  state.system.components.filter(c => c.type === type);
export const selectAllLogs = (state: RootState) => state.system.logs;
export const selectRecentLogs = (count: number) => (state: RootState) => 
  state.system.logs.slice(0, count);
export const selectSystemOverallStatus = (state: RootState) => {
  const components = state.system.components;
  if (components.some(c => c.status === ComponentStatus.ERROR)) {
    return ComponentStatus.ERROR;
  } else if (components.some(c => c.status === ComponentStatus.WARNING)) {
    return ComponentStatus.WARNING;
  } else if (components.some(c => c.status === ComponentStatus.OFFLINE)) {
    return ComponentStatus.OFFLINE;
  } else {
    return ComponentStatus.HEALTHY;
  }
};

export default systemSlice.reducer;
EOL

# 5. Create system types
echo "Creating system types..."
mkdir -p src/types
cat > src/types/system.ts << 'EOL'
export interface SystemComponent {
  id: string;
  name: string;
  type: string;
  status: string;
  healthScore: number;
  lastUpdated: string;
  description: string;
  version?: string;
  dependencies?: string[];
}

export interface SystemLog {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  source: string;
  message: string;
  details?: string;
}

export interface SystemMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  timestamp: string;
  change?: number;
  changePercent?: number;
  target?: number;
  status?: 'good' | 'warning' | 'error';
}
EOL

echo "Fixes applied successfully!"
echo "Now you can run the dashboard with: ./run_dashboard.sh"