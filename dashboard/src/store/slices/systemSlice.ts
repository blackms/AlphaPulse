import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../rootReducer';
import { SystemComponent, SystemLog, SystemMetric, SystemStatus, ComponentStatus } from '../../types/system';

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

export { SystemStatus, ComponentStatus };

export default systemSlice.reducer;
