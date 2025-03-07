import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type SystemStatus = 'operational' | 'degraded' | 'maintenance' | 'outage';

export interface SystemComponent {
  id: string;
  name: string;
  status: SystemStatus;
  message?: string;
  lastUpdated: number | null;
}

interface SystemState {
  status: SystemStatus;
  message: string | null;
  components: SystemComponent[];
  lastChecked: number | null;
  loading: boolean;
}

const initialState: SystemState = {
  status: 'operational',
  message: null,
  components: [
    {
      id: 'data-pipeline',
      name: 'Data Pipeline',
      status: 'operational',
      lastUpdated: Date.now() - 300000, // 5 minutes ago
    },
    {
      id: 'trading-engine',
      name: 'Trading Engine',
      status: 'operational',
      lastUpdated: Date.now() - 180000, // 3 minutes ago
    },
    {
      id: 'agent-system',
      name: 'AI Agent System',
      status: 'operational',
      lastUpdated: Date.now() - 240000, // 4 minutes ago
    },
    {
      id: 'portfolio-manager',
      name: 'Portfolio Manager',
      status: 'operational',
      lastUpdated: Date.now() - 120000, // 2 minutes ago
    },
    {
      id: 'risk-manager',
      name: 'Risk Management',
      status: 'operational',
      lastUpdated: Date.now() - 210000, // 3.5 minutes ago
    },
    {
      id: 'execution-broker',
      name: 'Execution Broker',
      status: 'operational',
      lastUpdated: Date.now() - 270000, // 4.5 minutes ago
    },
  ],
  lastChecked: Date.now(),
  loading: false,
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    fetchSystemStatusStart: (state) => {
      state.loading = true;
    },
    fetchSystemStatusSuccess: (state, action: PayloadAction<{
      status: SystemStatus;
      message: string | null;
      components: SystemComponent[];
    }>) => {
      state.status = action.payload.status;
      state.message = action.payload.message;
      state.components = action.payload.components;
      state.lastChecked = Date.now();
      state.loading = false;
    },
    fetchSystemStatusFailure: (state) => {
      state.loading = false;
    },
    updateComponentStatus: (state, action: PayloadAction<{
      componentId: string;
      status: SystemStatus;
      message?: string;
    }>) => {
      const component = state.components.find(c => c.id === action.payload.componentId);
      if (component) {
        component.status = action.payload.status;
        component.message = action.payload.message;
        component.lastUpdated = Date.now();
      }
      
      // Recalculate overall system status
      if (state.components.some(c => c.status === 'outage')) {
        state.status = 'outage';
      } else if (state.components.some(c => c.status === 'degraded')) {
        state.status = 'degraded';
      } else if (state.components.some(c => c.status === 'maintenance')) {
        state.status = 'maintenance';
      } else {
        state.status = 'operational';
      }
    },
    setSystemMessage: (state, action: PayloadAction<string | null>) => {
      state.message = action.payload;
    },
  },
});

export const {
  fetchSystemStatusStart,
  fetchSystemStatusSuccess,
  fetchSystemStatusFailure,
  updateComponentStatus,
  setSystemMessage,
} = systemSlice.actions;

// Selectors
export const selectSystemStatus = (state: RootState) => state.system.status;
export const selectSystemMessage = (state: RootState) => state.system.message;
export const selectSystemComponents = (state: RootState) => state.system.components;
export const selectLastChecked = (state: RootState) => state.system.lastChecked;
export const selectIsLoading = (state: RootState) => state.system.loading;

export default systemSlice.reducer;