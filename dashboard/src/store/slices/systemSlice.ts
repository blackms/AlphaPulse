import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type SystemComponentStatus = 'operational' | 'degraded' | 'outage' | 'maintenance' | 'unknown';
export type SystemComponentType = 'agent' | 'data' | 'risk' | 'portfolio' | 'execution' | 'infrastructure';

export interface SystemComponent {
  id: string;
  name: string;
  type: SystemComponentType;
  status: SystemComponentStatus;
  uptime: number; // In seconds
  lastUpdated: number; // Timestamp
  metrics: {
    cpu: number; // Percentage
    memory: number; // Percentage
    throughput: number; // Operations per second
    latency: number; // In milliseconds
  };
  dependencies: string[]; // IDs of dependent components
  metadata?: Record<string, any>;
}

export interface SystemLog {
  id: string;
  timestamp: number;
  level: 'debug' | 'info' | 'warning' | 'error' | 'critical';
  component: string;
  message: string;
  details?: Record<string, any>;
}

export interface SystemMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: number;
}

export interface SystemStatus {
  overall: SystemComponentStatus;
  components: SystemComponent[];
  recentLogs: SystemLog[];
  performance: {
    cpuUtilization: number;
    memoryUtilization: number;
    diskUtilization: number;
    networkThroughput: number;
    requestsPerSecond: number;
    responseTime: number;
  };
  uptime: number; // In seconds
  lastFullRestart: number; // Timestamp
  lastUpdated: number; // Timestamp
}

interface SystemState {
  status: SystemStatus;
  loadingStatus: boolean;
  statusError: string | null;
  logs: SystemLog[];
  loadingLogs: boolean;
  logsError: string | null;
  metrics: SystemMetric[];
  loadingMetrics: boolean;
  metricsError: string | null;
}

// Initial state with mock data
const initialState: SystemState = {
  status: {
    overall: 'operational',
    components: [
      {
        id: 'technical-agent',
        name: 'Technical Agent',
        type: 'agent',
        status: 'operational',
        uptime: 345600, // 4 days
        lastUpdated: Date.now() - 300000, // 5 minutes ago
        metrics: {
          cpu: 12.5,
          memory: 18.2,
          throughput: 425,
          latency: 45
        },
        dependencies: ['data-pipeline']
      },
      {
        id: 'fundamental-agent',
        name: 'Fundamental Agent',
        type: 'agent',
        status: 'operational',
        uptime: 345600, // 4 days
        lastUpdated: Date.now() - 240000, // 4 minutes ago
        metrics: {
          cpu: 14.2,
          memory: 22.5,
          throughput: 180,
          latency: 120
        },
        dependencies: ['data-pipeline']
      },
      {
        id: 'sentiment-agent',
        name: 'Sentiment Agent',
        type: 'agent',
        status: 'operational',
        uptime: 345600, // 4 days
        lastUpdated: Date.now() - 180000, // 3 minutes ago
        metrics: {
          cpu: 18.7,
          memory: 27.3,
          throughput: 210,
          latency: 85
        },
        dependencies: ['data-pipeline']
      },
      {
        id: 'value-agent',
        name: 'Value Agent',
        type: 'agent',
        status: 'operational',
        uptime: 345600, // 4 days
        lastUpdated: Date.now() - 360000, // 6 minutes ago
        metrics: {
          cpu: 10.8,
          memory: 16.5,
          throughput: 175,
          latency: 65
        },
        dependencies: ['data-pipeline']
      },
      {
        id: 'activist-agent',
        name: 'Activist Agent',
        type: 'agent',
        status: 'operational',
        uptime: 345600, // 4 days
        lastUpdated: Date.now() - 420000, // 7 minutes ago
        metrics: {
          cpu: 8.3,
          memory: 12.7,
          throughput: 85,
          latency: 95
        },
        dependencies: ['data-pipeline']
      },
      {
        id: 'risk-manager',
        name: 'Risk Manager',
        type: 'risk',
        status: 'operational',
        uptime: 432000, // 5 days
        lastUpdated: Date.now() - 180000, // 3 minutes ago
        metrics: {
          cpu: 11.2,
          memory: 16.8,
          throughput: 320,
          latency: 30
        },
        dependencies: ['technical-agent', 'fundamental-agent', 'sentiment-agent', 'value-agent']
      },
      {
        id: 'portfolio-manager',
        name: 'Portfolio Manager',
        type: 'portfolio',
        status: 'operational',
        uptime: 432000, // 5 days
        lastUpdated: Date.now() - 150000, // 2.5 minutes ago
        metrics: {
          cpu: 16.5,
          memory: 24.8,
          throughput: 95,
          latency: 75
        },
        dependencies: ['risk-manager']
      },
      {
        id: 'portfolio-optimizer',
        name: 'Portfolio Optimizer',
        type: 'portfolio',
        status: 'operational',
        uptime: 432000, // 5 days
        lastUpdated: Date.now() - 210000, // 3.5 minutes ago
        metrics: {
          cpu: 25.3,
          memory: 35.7,
          throughput: 15,
          latency: 850
        },
        dependencies: ['portfolio-manager']
      },
      {
        id: 'rebalancer',
        name: 'Rebalancer',
        type: 'portfolio',
        status: 'operational',
        uptime: 432000, // 5 days
        lastUpdated: Date.now() - 390000, // 6.5 minutes ago
        metrics: {
          cpu: 5.2,
          memory: 8.4,
          throughput: 5,
          latency: 230
        },
        dependencies: ['portfolio-optimizer']
      },
      {
        id: 'execution-broker',
        name: 'Execution Broker',
        type: 'execution',
        status: 'operational',
        uptime: 518400, // 6 days
        lastUpdated: Date.now() - 120000, // 2 minutes ago
        metrics: {
          cpu: 8.7,
          memory: 12.6,
          throughput: 25,
          latency: 45
        },
        dependencies: ['portfolio-manager', 'rebalancer']
      },
      {
        id: 'data-pipeline',
        name: 'Data Pipeline',
        type: 'data',
        status: 'operational',
        uptime: 518400, // 6 days
        lastUpdated: Date.now() - 60000, // 1 minute ago
        metrics: {
          cpu: 28.5,
          memory: 42.3,
          throughput: 870,
          latency: 110
        },
        dependencies: []
      },
      {
        id: 'database',
        name: 'Database',
        type: 'infrastructure',
        status: 'operational',
        uptime: 864000, // 10 days
        lastUpdated: Date.now() - 90000, // 1.5 minutes ago
        metrics: {
          cpu: 35.2,
          memory: 48.7,
          throughput: 1250,
          latency: 15
        },
        dependencies: []
      },
      {
        id: 'api-gateway',
        name: 'API Gateway',
        type: 'infrastructure',
        status: 'operational',
        uptime: 691200, // 8 days
        lastUpdated: Date.now() - 75000, // 1.25 minutes ago
        metrics: {
          cpu: 22.8,
          memory: 31.5,
          throughput: 485,
          latency: 25
        },
        dependencies: ['database']
      }
    ],
    recentLogs: [
      {
        id: 'log1',
        timestamp: Date.now() - 60000, // 1 minute ago
        level: 'info',
        component: 'data-pipeline',
        message: 'Market data refreshed successfully',
        details: {
          assets: ['BTC', 'ETH', 'SOL', 'AVAX', 'DOT', 'LINK', 'MATIC'],
          timeframe: '1h'
        }
      },
      {
        id: 'log2',
        timestamp: Date.now() - 180000, // 3 minutes ago
        level: 'info',
        component: 'technical-agent',
        message: 'New technical signal generated',
        details: {
          asset: 'BTC',
          signal: 'buy',
          confidence: 0.82,
          indicators: {
            rsi: 65,
            macd: 'bullish'
          }
        }
      },
      {
        id: 'log3',
        timestamp: Date.now() - 300000, // 5 minutes ago
        level: 'warning',
        component: 'execution-broker',
        message: 'Increased order execution latency detected',
        details: {
          latency: '120ms',
          threshold: '100ms',
          action: 'monitoring'
        }
      },
      {
        id: 'log4',
        timestamp: Date.now() - 600000, // 10 minutes ago
        level: 'error',
        component: 'sentiment-agent',
        message: 'API rate limit reached for sentiment data provider',
        details: {
          provider: 'SocialMetrics',
          retryAfter: '300 seconds',
          fallbackUsed: true
        }
      },
      {
        id: 'log5',
        timestamp: Date.now() - 1800000, // 30 minutes ago
        level: 'info',
        component: 'portfolio-manager',
        message: 'Portfolio rebalancing completed',
        details: {
          changedAssets: ['BTC', 'ETH', 'SOL'],
          duration: '45 seconds',
          status: 'success'
        }
      }
    ],
    performance: {
      cpuUtilization: 32.5,
      memoryUtilization: 48.2,
      diskUtilization: 65.7,
      networkThroughput: 42.8,
      requestsPerSecond: 385,
      responseTime: 85
    },
    uptime: 518400, // 6 days in seconds
    lastFullRestart: Date.now() - 518400000, // 6 days ago
    lastUpdated: Date.now() - 30000 // 30 seconds ago
  },
  loadingStatus: false,
  statusError: null,
  logs: [
    {
      id: 'log1',
      timestamp: Date.now() - 60000, // 1 minute ago
      level: 'info',
      component: 'data-pipeline',
      message: 'Market data refreshed successfully',
      details: {
        assets: ['BTC', 'ETH', 'SOL', 'AVAX', 'DOT', 'LINK', 'MATIC'],
        timeframe: '1h'
      }
    },
    {
      id: 'log2',
      timestamp: Date.now() - 180000, // 3 minutes ago
      level: 'info',
      component: 'technical-agent',
      message: 'New technical signal generated',
      details: {
        asset: 'BTC',
        signal: 'buy',
        confidence: 0.82,
        indicators: {
          rsi: 65,
          macd: 'bullish'
        }
      }
    },
    {
      id: 'log3',
      timestamp: Date.now() - 300000, // 5 minutes ago
      level: 'warning',
      component: 'execution-broker',
      message: 'Increased order execution latency detected',
      details: {
        latency: '120ms',
        threshold: '100ms',
        action: 'monitoring'
      }
    },
    {
      id: 'log4',
      timestamp: Date.now() - 600000, // 10 minutes ago
      level: 'error',
      component: 'sentiment-agent',
      message: 'API rate limit reached for sentiment data provider',
      details: {
        provider: 'SocialMetrics',
        retryAfter: '300 seconds',
        fallbackUsed: true
      }
    },
    {
      id: 'log5',
      timestamp: Date.now() - 1800000, // 30 minutes ago
      level: 'info',
      component: 'portfolio-manager',
      message: 'Portfolio rebalancing completed',
      details: {
        changedAssets: ['BTC', 'ETH', 'SOL'],
        duration: '45 seconds',
        status: 'success'
      }
    },
    {
      id: 'log6',
      timestamp: Date.now() - 3600000, // 1 hour ago
      level: 'info',
      component: 'risk-manager',
      message: 'Daily risk assessment completed',
      details: {
        portfolioRisk: 'medium',
        varDaily: '4.8%',
        maxExposure: '35.2%',
        action: 'maintain'
      }
    },
    {
      id: 'log7',
      timestamp: Date.now() - 7200000, // 2 hours ago
      level: 'info',
      component: 'fundamental-agent',
      message: 'Weekly on-chain metrics updated',
      details: {
        assets: ['BTC', 'ETH'],
        metrics: ['network_activity', 'transaction_volume', 'active_addresses']
      }
    },
    {
      id: 'log8',
      timestamp: Date.now() - 14400000, // 4 hours ago
      level: 'warning',
      component: 'database',
      message: 'Database running near capacity',
      details: {
        utilization: '85%',
        action: 'scale up',
        scheduledAt: 'next maintenance window'
      }
    },
    {
      id: 'log9',
      timestamp: Date.now() - 28800000, // 8 hours ago
      level: 'info',
      component: 'activist-agent',
      message: 'Governance proposal analysis completed',
      details: {
        protocol: 'Uniswap',
        proposal: 'UIP-123',
        assessment: 'positive',
        confidenceScore: 0.76
      }
    },
    {
      id: 'log10',
      timestamp: Date.now() - 86400000, // 24 hours ago
      level: 'info',
      component: 'system',
      message: 'Daily system health check completed',
      details: {
        status: 'healthy',
        checks: 42,
        warnings: 2,
        critical: 0
      }
    }
  ],
  loadingLogs: false,
  logsError: null,
  metrics: [
    {
      name: 'system.cpu.utilization',
      value: 32.5,
      unit: 'percent',
      timestamp: Date.now() - 30000 // 30 seconds ago
    },
    {
      name: 'system.memory.utilization',
      value: 48.2,
      unit: 'percent',
      timestamp: Date.now() - 30000 // 30 seconds ago
    },
    {
      name: 'system.disk.utilization',
      value: 65.7,
      unit: 'percent',
      timestamp: Date.now() - 30000 // 30 seconds ago
    },
    {
      name: 'system.network.throughput',
      value: 42.8,
      unit: 'mbps',
      timestamp: Date.now() - 30000 // 30 seconds ago
    },
    {
      name: 'api.requests_per_second',
      value: 385,
      unit: 'count',
      timestamp: Date.now() - 30000 // 30 seconds ago
    },
    {
      name: 'api.response_time',
      value: 85,
      unit: 'ms',
      timestamp: Date.now() - 30000 // 30 seconds ago
    },
    {
      name: 'database.connections',
      value: 124,
      unit: 'count',
      timestamp: Date.now() - 60000 // 1 minute ago
    },
    {
      name: 'database.query_time',
      value: 12.5,
      unit: 'ms',
      timestamp: Date.now() - 60000 // 1 minute ago
    },
    {
      name: 'agents.signals_per_hour',
      value: 45,
      unit: 'count',
      timestamp: Date.now() - 300000 // 5 minutes ago
    },
    {
      name: 'trading.orders_per_hour',
      value: 8,
      unit: 'count',
      timestamp: Date.now() - 300000 // 5 minutes ago
    }
  ],
  loadingMetrics: false,
  metricsError: null
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    fetchSystemStatusStart: (state) => {
      state.loadingStatus = true;
      state.statusError = null;
    },
    fetchSystemStatusSuccess: (state, action: PayloadAction<SystemStatus>) => {
      state.status = action.payload;
      state.loadingStatus = false;
    },
    fetchSystemStatusFailure: (state, action: PayloadAction<string>) => {
      state.loadingStatus = false;
      state.statusError = action.payload;
    },
    fetchSystemLogsStart: (state) => {
      state.loadingLogs = true;
      state.logsError = null;
    },
    fetchSystemLogsSuccess: (state, action: PayloadAction<SystemLog[]>) => {
      state.logs = action.payload;
      state.loadingLogs = false;
    },
    fetchSystemLogsFailure: (state, action: PayloadAction<string>) => {
      state.loadingLogs = false;
      state.logsError = action.payload;
    },
    fetchSystemMetricsStart: (state) => {
      state.loadingMetrics = true;
      state.metricsError = null;
    },
    fetchSystemMetricsSuccess: (state, action: PayloadAction<SystemMetric[]>) => {
      state.metrics = action.payload;
      state.loadingMetrics = false;
    },
    fetchSystemMetricsFailure: (state, action: PayloadAction<string>) => {
      state.loadingMetrics = false;
      state.metricsError = action.payload;
    },
    updateComponentStatus: (state, action: PayloadAction<{
      componentId: string;
      status: SystemComponentStatus;
    }>) => {
      const component = state.status.components.find(c => c.id === action.payload.componentId);
      if (component) {
        component.status = action.payload.status;
        component.lastUpdated = Date.now();
        
        // Recalculate overall status
        if (state.status.components.some(c => c.status === 'outage')) {
          state.status.overall = 'outage';
        } else if (state.status.components.some(c => c.status === 'degraded')) {
          state.status.overall = 'degraded';
        } else if (state.status.components.some(c => c.status === 'maintenance')) {
          state.status.overall = 'maintenance';
        } else {
          state.status.overall = 'operational';
        }
      }
    },
    addSystemLog: (state, action: PayloadAction<Omit<SystemLog, 'id'>>) => {
      const newLog = {
        id: `log_${Date.now()}`,
        ...action.payload
      };
      
      // Add to both logs array and recent logs
      state.logs.unshift(newLog);
      state.status.recentLogs.unshift(newLog);
      
      // Keep recent logs limited to 5
      if (state.status.recentLogs.length > 5) {
        state.status.recentLogs = state.status.recentLogs.slice(0, 5);
      }
    },
    clearSystemLogs: (state) => {
      state.logs = [];
    },
    updateSystemPerformance: (state, action: PayloadAction<Partial<SystemStatus['performance']>>) => {
      state.status.performance = {
        ...state.status.performance,
        ...action.payload
      };
      state.status.lastUpdated = Date.now();
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
  addSystemLog,
  clearSystemLogs,
  updateSystemPerformance
} = systemSlice.actions;

// Selectors
export const selectSystemStatus = (state: RootState) => state.system.status;
export const selectSystemOverallStatus = (state: RootState) => state.system.status.overall;
export const selectSystemComponents = (state: RootState) => state.system.status.components;
export const selectComponentById = (componentId: string) => 
  (state: RootState) => state.system.status.components.find(c => c.id === componentId);
export const selectComponentsByType = (type: SystemComponentType) => 
  (state: RootState) => state.system.status.components.filter(c => c.type === type);
export const selectRecentLogs = (state: RootState) => state.system.status.recentLogs;
export const selectAllLogs = (state: RootState) => state.system.logs;
export const selectSystemPerformance = (state: RootState) => state.system.status.performance;
export const selectSystemMetrics = (state: RootState) => state.system.metrics;
export const selectSystemUptime = (state: RootState) => state.system.status.uptime;
export const selectSystemLastUpdated = (state: RootState) => state.system.status.lastUpdated;
export const selectStatusLoading = (state: RootState) => state.system.loadingStatus;
export const selectLogsLoading = (state: RootState) => state.system.loadingLogs;
export const selectMetricsLoading = (state: RootState) => state.system.loadingMetrics;
export const selectStatusError = (state: RootState) => state.system.statusError;
export const selectLogsError = (state: RootState) => state.system.logsError;
export const selectMetricsError = (state: RootState) => state.system.metricsError;

export default systemSlice.reducer;