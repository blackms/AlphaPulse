import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type SystemStatus = 'operational' | 'degraded' | 'down' | 'maintenance';
export type ComponentStatus = 'operational' | 'degraded' | 'down' | 'maintenance' | 'unknown';
export type ComponentType = 'agent' | 'data' | 'portfolio' | 'risk' | 'execution' | 'monitoring';
export type LogLevel = 'info' | 'warning' | 'error' | 'critical' | 'debug';

export interface SystemComponent {
  id: string;
  name: string;
  type: ComponentType;
  status: ComponentStatus;
  lastUpdated: number;
  healthScore: number; // 0-100
  description: string;
  metrics?: {
    [key: string]: number | string;
  };
}

export interface SystemLog {
  id: string;
  timestamp: number;
  level: LogLevel;
  component: string;
  message: string;
  details?: string;
}

export interface SystemMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  timestamp: number;
  change?: number;
  changePercent?: number;
  min?: number;
  max?: number;
  target?: number;
  status?: 'good' | 'warning' | 'critical';
}

interface SystemState {
  status: SystemStatus;
  components: SystemComponent[];
  logs: SystemLog[];
  metrics: SystemMetric[];
  lastUpdated: number | null;
  loading: boolean;
  error: string | null;
}

const initialState: SystemState = {
  status: 'operational',
  components: [
    {
      id: 'technical_agent',
      name: 'Technical Agent',
      type: 'agent',
      status: 'operational',
      lastUpdated: Date.now() - 5 * 60 * 1000,
      healthScore: 95,
      description: 'Analyzes market patterns and momentum',
      metrics: {
        signals: 48,
        accuracy: 76.5,
        latency: 235,
      },
    },
    {
      id: 'fundamental_agent',
      name: 'Fundamental Agent',
      type: 'agent',
      status: 'operational',
      lastUpdated: Date.now() - 10 * 60 * 1000,
      healthScore: 92,
      description: 'Evaluates on-chain metrics and fundamentals',
      metrics: {
        signals: 32,
        accuracy: 81.2,
        latency: 310,
      },
    },
    {
      id: 'sentiment_agent',
      name: 'Sentiment Agent',
      type: 'agent',
      status: 'degraded',
      lastUpdated: Date.now() - 15 * 60 * 1000,
      healthScore: 78,
      description: 'Processes market sentiment data',
      metrics: {
        signals: 22,
        accuracy: 69.8,
        latency: 450,
      },
    },
    {
      id: 'value_agent',
      name: 'Value Agent',
      type: 'agent',
      status: 'operational',
      lastUpdated: Date.now() - 8 * 60 * 1000,
      healthScore: 90,
      description: 'Calculates intrinsic value metrics',
      metrics: {
        signals: 18,
        accuracy: 83.7,
        latency: 275,
      },
    },
    {
      id: 'data_pipeline',
      name: 'Data Pipeline',
      type: 'data',
      status: 'operational',
      lastUpdated: Date.now() - 3 * 60 * 1000,
      healthScore: 98,
      description: 'Manages data ingestion and processing',
      metrics: {
        throughput: 1250,
        errorRate: 0.03,
        latency: 68,
      },
    },
    {
      id: 'portfolio_manager',
      name: 'Portfolio Manager',
      type: 'portfolio',
      status: 'operational',
      lastUpdated: Date.now() - 7 * 60 * 1000,
      healthScore: 94,
      description: 'Handles portfolio allocation and rebalancing',
      metrics: {
        allocations: 6,
        rebalances: 2,
        optimizationRuns: 24,
      },
    },
    {
      id: 'risk_manager',
      name: 'Risk Manager',
      type: 'risk',
      status: 'operational',
      lastUpdated: Date.now() - 6 * 60 * 1000,
      healthScore: 97,
      description: 'Monitors and manages risk exposure',
      metrics: {
        checks: 145,
        blocks: 3,
        adjustments: 8,
      },
    },
    {
      id: 'execution_broker',
      name: 'Execution Broker',
      type: 'execution',
      status: 'operational',
      lastUpdated: Date.now() - 2 * 60 * 1000,
      healthScore: 96,
      description: 'Executes trades and manages orders',
      metrics: {
        orders: 12,
        fills: 9,
        latency: 125,
      },
    },
    {
      id: 'monitoring_system',
      name: 'Monitoring System',
      type: 'monitoring',
      status: 'operational',
      lastUpdated: Date.now() - 1 * 60 * 1000,
      healthScore: 99,
      description: 'Tracks system performance and alerts',
      metrics: {
        metrics: 87,
        alerts: 2,
        checks: 215,
      },
    },
  ],
  logs: [
    {
      id: 'log1',
      timestamp: Date.now() - 5 * 60 * 1000,
      level: 'warning',
      component: 'sentiment_agent',
      message: 'Increased latency in sentiment data processing',
      details: 'Social media API rate limiting affecting data collection speed',
    },
    {
      id: 'log2',
      timestamp: Date.now() - 15 * 60 * 1000,
      level: 'info',
      component: 'portfolio_manager',
      message: 'Portfolio rebalancing completed',
      details: 'Adjusted allocations to match target weights; transaction costs: $12.50',
    },
    {
      id: 'log3',
      timestamp: Date.now() - 30 * 60 * 1000,
      level: 'error',
      component: 'data_pipeline',
      message: 'Temporary data connection failure',
      details: 'Connection to price feed dropped for 12 seconds; automatic recovery successful',
    },
    {
      id: 'log4',
      timestamp: Date.now() - 45 * 60 * 1000,
      level: 'info',
      component: 'risk_manager',
      message: 'Position size adjustment',
      details: 'Reduced BTC position size due to increased market volatility',
    },
    {
      id: 'log5',
      timestamp: Date.now() - 60 * 60 * 1000,
      level: 'info',
      component: 'technical_agent',
      message: 'New trading signal generated',
      details: 'BUY signal for ETH with 82% confidence based on breakout pattern',
    },
  ],
  metrics: [
    {
      id: 'metric1',
      name: 'System Uptime',
      value: 99.98,
      unit: '%',
      timestamp: Date.now(),
      change: 0.01,
      changePercent: 0.01,
      min: 0,
      max: 100,
      target: 99.95,
      status: 'good',
    },
    {
      id: 'metric2',
      name: 'Signal Generation Rate',
      value: 24.5,
      unit: 'signals/hour',
      timestamp: Date.now(),
      change: 2.1,
      changePercent: 9.37,
      status: 'good',
    },
    {
      id: 'metric3',
      name: 'Average Signal Latency',
      value: 310,
      unit: 'ms',
      timestamp: Date.now(),
      change: -15,
      changePercent: -4.62,
      target: 500,
      status: 'good',
    },
    {
      id: 'metric4',
      name: 'Agent Accuracy (7-day)',
      value: 78.3,
      unit: '%',
      timestamp: Date.now(),
      change: 1.2,
      changePercent: 1.56,
      min: 0,
      max: 100,
      target: 75,
      status: 'good',
    },
    {
      id: 'metric5',
      name: 'Data Processing Volume',
      value: 2340,
      unit: 'MB/min',
      timestamp: Date.now(),
      change: 120,
      changePercent: 5.41,
      status: 'good',
    },
    {
      id: 'metric6',
      name: 'API Error Rate',
      value: 0.12,
      unit: '%',
      timestamp: Date.now(),
      change: 0.04,
      changePercent: 50.0,
      min: 0,
      max: 5,
      target: 0.5,
      status: 'good',
    },
    {
      id: 'metric7',
      name: 'Order Execution Time',
      value: 165,
      unit: 'ms',
      timestamp: Date.now(),
      change: 10,
      changePercent: 6.45,
      target: 250,
      status: 'good',
    },
  ],
  lastUpdated: Date.now(),
  loading: false,
  error: null,
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    fetchSystemStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    fetchSystemSuccess: (state, action: PayloadAction<{
      status: SystemStatus;
      components: SystemComponent[];
      logs: SystemLog[];
      metrics: SystemMetric[];
    }>) => {
      state.status = action.payload.status;
      state.components = action.payload.components;
      state.logs = action.payload.logs;
      state.metrics = action.payload.metrics;
      state.lastUpdated = Date.now();
      state.loading = false;
    },
    fetchSystemFailure: (state, action: PayloadAction<string>) => {
      state.loading = false;
      state.error = action.payload;
    },
    updateComponentStatus: (state, action: PayloadAction<{
      componentId: string;
      status: ComponentStatus;
      healthScore?: number;
      metrics?: {
        [key: string]: number | string;
      };
    }>) => {
      const component = state.components.find(c => c.id === action.payload.componentId);
      if (component) {
        component.status = action.payload.status;
        component.lastUpdated = Date.now();
        
        if (action.payload.healthScore !== undefined) {
          component.healthScore = action.payload.healthScore;
        }
        
        if (action.payload.metrics) {
          component.metrics = {
            ...component.metrics,
            ...action.payload.metrics,
          };
        }
        
        // Update overall system status based on components
        if (state.components.some(c => c.status === 'down')) {
          state.status = 'down';
        } else if (state.components.some(c => c.status === 'degraded')) {
          state.status = 'degraded';
        } else if (state.components.some(c => c.status === 'maintenance')) {
          state.status = 'maintenance';
        } else {
          state.status = 'operational';
        }
      }
    },
    addSystemLog: (state, action: PayloadAction<Omit<SystemLog, 'id'>>) => {
      const newLog: SystemLog = {
        id: `log${Date.now()}`,
        ...action.payload,
      };
      
      state.logs.unshift(newLog);
      
      // Keep only the most recent 100 logs
      if (state.logs.length > 100) {
        state.logs = state.logs.slice(0, 100);
      }
    },
    updateMetric: (state, action: PayloadAction<{
      id: string;
      value: number;
      change?: number;
      changePercent?: number;
      status?: 'good' | 'warning' | 'critical';
    }>) => {
      const metric = state.metrics.find(m => m.id === action.payload.id);
      if (metric) {
        const oldValue = metric.value;
        metric.value = action.payload.value;
        metric.timestamp = Date.now();
        
        if (action.payload.change !== undefined) {
          metric.change = action.payload.change;
        } else {
          metric.change = action.payload.value - oldValue;
        }
        
        if (action.payload.changePercent !== undefined) {
          metric.changePercent = action.payload.changePercent;
        } else if (oldValue !== 0) {
          metric.changePercent = ((action.payload.value - oldValue) / oldValue) * 100;
        }
        
        if (action.payload.status !== undefined) {
          metric.status = action.payload.status;
        } else {
          // Auto-determine status if target exists
          if (metric.target !== undefined) {
            if (metric.value < metric.target * 0.8) {
              metric.status = 'critical';
            } else if (metric.value < metric.target * 0.9) {
              metric.status = 'warning';
            } else {
              metric.status = 'good';
            }
          }
        }
      }
    },
  },
});

export const {
  fetchSystemStart,
  fetchSystemSuccess,
  fetchSystemFailure,
  updateComponentStatus,
  addSystemLog,
  updateMetric,
} = systemSlice.actions;

// Selectors
export const selectSystemStatus = (state: RootState) => state.system.status;
export const selectSystemComponents = (state: RootState) => state.system.components;
export const selectSystemLogs = (state: RootState) => state.system.logs;
export const selectSystemMetrics = (state: RootState) => state.system.metrics;
export const selectSystemLastUpdated = (state: RootState) => state.system.lastUpdated;
export const selectSystemLoading = (state: RootState) => state.system.loading;
export const selectSystemError = (state: RootState) => state.system.error;
export const selectComponentById = (id: string) => 
  (state: RootState) => state.system.components.find(c => c.id === id);
export const selectComponentsByType = (type: ComponentType) => 
  (state: RootState) => state.system.components.filter(c => c.type === type);
export const selectMetricById = (id: string) => 
  (state: RootState) => state.system.metrics.find(m => m.id === id);

export default systemSlice.reducer;