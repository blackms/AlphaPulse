import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type SystemStatus = 'operational' | 'degraded' | 'maintenance' | 'outage';
export type ComponentStatus = 'operational' | 'degraded' | 'down' | 'maintenance';

export interface SystemComponent {
  id: string;
  name: string;
  status: ComponentStatus;
  lastUpdated: string;
  message?: string;
  metrics?: Record<string, number>;
}

export interface SystemResources {
  cpu: number;
  memory: number;
  disk: number;
  network: {
    in: number;
    out: number;
  };
}

export interface ApiStatus {
  responseTime: number;
  requests: {
    total: number;
    success: number;
    error: number;
  };
  lastError?: string;
}

interface SystemState {
  status: {
    overall: SystemStatus;
    message?: string;
  };
  components: SystemComponent[];
  resources: SystemResources;
  api: ApiStatus;
  lastErrors: Array<{
    timestamp: string;
    component: string;
    message: string;
    level: 'info' | 'warning' | 'error' | 'critical';
  }>;
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: SystemState = {
  status: {
    overall: 'operational',
  },
  components: [],
  resources: {
    cpu: 0,
    memory: 0,
    disk: 0,
    network: {
      in: 0,
      out: 0,
    },
  },
  api: {
    responseTime: 0,
    requests: {
      total: 0,
      success: 0,
      error: 0,
    },
  },
  lastErrors: [],
  loading: false,
  error: null,
  lastUpdated: null,
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    fetchSystemStatusStart(state) {
      state.loading = true;
      state.error = null;
    },
    fetchSystemStatusSuccess(state, action: PayloadAction<Partial<SystemState>>) {
      return {
        ...state,
        ...action.payload,
        loading: false,
        error: null,
        lastUpdated: new Date().toISOString(),
      };
    },
    fetchSystemStatusFailure(state, action: PayloadAction<string>) {
      state.loading = false;
      state.error = action.payload;
    },
    updateSystemStatus(state, action: PayloadAction<Partial<SystemState>>) {
      return {
        ...state,
        ...action.payload,
        lastUpdated: new Date().toISOString(),
      };
    },
    updateComponent(state, action: PayloadAction<SystemComponent>) {
      const index = state.components.findIndex(c => c.id === action.payload.id);
      if (index >= 0) {
        state.components[index] = action.payload;
      } else {
        state.components.push(action.payload);
      }
      
      // Recalculate overall status based on component statuses
      const statuses = state.components.map(c => c.status);
      if (statuses.includes('down')) {
        state.status.overall = 'outage';
      } else if (statuses.includes('degraded')) {
        state.status.overall = 'degraded';
      } else if (statuses.includes('maintenance')) {
        state.status.overall = 'maintenance';
      } else {
        state.status.overall = 'operational';
      }
      
      state.lastUpdated = new Date().toISOString();
    },
    addSystemError(state, action: PayloadAction<SystemState['lastErrors'][0]>) {
      state.lastErrors = [action.payload, ...state.lastErrors].slice(0, 50); // Keep last 50 errors
      state.lastUpdated = new Date().toISOString();
    },
  },
});

export const {
  fetchSystemStatusStart,
  fetchSystemStatusSuccess,
  fetchSystemStatusFailure,
  updateSystemStatus,
  updateComponent,
  addSystemError,
} = systemSlice.actions;

export default systemSlice.reducer;