import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface MetricsState {
  performance: {
    daily: number;
    weekly: number;
    monthly: number;
    yearly: number;
  };
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: MetricsState = {
  performance: {
    daily: 0,
    weekly: 0,
    monthly: 0,
    yearly: 0,
  },
  volatility: 0,
  sharpeRatio: 0,
  maxDrawdown: 0,
  loading: false,
  error: null,
  lastUpdated: null,
};

const metricsSlice = createSlice({
  name: 'metrics',
  initialState,
  reducers: {
    fetchMetricsStart(state) {
      state.loading = true;
      state.error = null;
    },
    fetchMetricsSuccess(state, action: PayloadAction<Partial<MetricsState>>) {
      return {
        ...state,
        ...action.payload,
        loading: false,
        error: null,
        lastUpdated: new Date().toISOString(),
      };
    },
    fetchMetricsFailure(state, action: PayloadAction<string>) {
      state.loading = false;
      state.error = action.payload;
    },
    updateMetrics(state, action: PayloadAction<Partial<MetricsState>>) {
      return {
        ...state,
        ...action.payload,
        lastUpdated: new Date().toISOString(),
      };
    },
  },
});

export const {
  fetchMetricsStart,
  fetchMetricsSuccess,
  fetchMetricsFailure,
  updateMetrics,
} = metricsSlice.actions;

export default metricsSlice.reducer;