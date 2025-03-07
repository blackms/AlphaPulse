import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export interface PerformanceMetric {
  id: string;
  name: string;
  value: number;
  previousValue: number;
  change: number;
  changePercentage: number;
  timestamp: number;
  period: 'daily' | 'weekly' | 'monthly' | 'yearly';
}

export interface HistoricalDataPoint {
  timestamp: number;
  value: number;
}

export interface MetricHistory {
  id: string;
  data: HistoricalDataPoint[];
}

interface MetricsState {
  metrics: PerformanceMetric[];
  history: Record<string, HistoricalDataPoint[]>;
  loading: boolean;
  lastUpdated: number | null;
}

const initialState: MetricsState = {
  metrics: [
    {
      id: 'portfolio-value',
      name: 'Portfolio Value',
      value: 125000,
      previousValue: 120000,
      change: 5000,
      changePercentage: 4.17,
      timestamp: Date.now(),
      period: 'daily',
    },
    {
      id: 'daily-return',
      name: 'Daily Return',
      value: 1.2,
      previousValue: 0.8,
      change: 0.4,
      changePercentage: 50,
      timestamp: Date.now(),
      period: 'daily',
    },
    {
      id: 'monthly-return',
      name: 'Monthly Return',
      value: 8.4,
      previousValue: 7.2,
      change: 1.2,
      changePercentage: 16.67,
      timestamp: Date.now(),
      period: 'monthly',
    },
    {
      id: 'yearly-return',
      name: 'Yearly Return',
      value: -2.1,
      previousValue: -4.3,
      change: 2.2,
      changePercentage: 51.16,
      timestamp: Date.now(),
      period: 'yearly',
    },
  ],
  history: {},
  loading: false,
  lastUpdated: Date.now(),
};

const metricsSlice = createSlice({
  name: 'metrics',
  initialState,
  reducers: {
    fetchMetricsStart: (state) => {
      state.loading = true;
    },
    fetchMetricsSuccess: (state, action: PayloadAction<PerformanceMetric[]>) => {
      state.metrics = action.payload;
      state.loading = false;
      state.lastUpdated = Date.now();
    },
    fetchMetricsFailure: (state) => {
      state.loading = false;
    },
    fetchHistoricalDataSuccess: (state, action: PayloadAction<{ 
      id: string; 
      data: HistoricalDataPoint[]; 
    }>) => {
      state.history[action.payload.id] = action.payload.data;
    },
    updateMetric: (state, action: PayloadAction<PerformanceMetric>) => {
      const index = state.metrics.findIndex(m => m.id === action.payload.id);
      if (index !== -1) {
        state.metrics[index] = action.payload;
      } else {
        state.metrics.push(action.payload);
      }
      state.lastUpdated = Date.now();
    },
  },
});

export const {
  fetchMetricsStart,
  fetchMetricsSuccess,
  fetchMetricsFailure,
  fetchHistoricalDataSuccess,
  updateMetric,
} = metricsSlice.actions;

// Selectors
export const selectMetrics = (state: RootState) => state.metrics.metrics;
export const selectMetricById = (id: string) => (state: RootState) => 
  state.metrics.metrics.find(m => m.id === id);
export const selectHistoricalData = (id: string) => (state: RootState) => 
  state.metrics.history[id] || [];
export const selectIsLoading = (state: RootState) => state.metrics.loading;
export const selectLastUpdated = (state: RootState) => state.metrics.lastUpdated;

export default metricsSlice.reducer;