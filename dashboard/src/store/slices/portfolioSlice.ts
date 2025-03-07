import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface PortfolioPosition {
  symbol: string;
  quantity: number;
  averageEntryPrice: number;
  currentPrice: number;
  value: number;
  pnl: number;
  pnlPercentage: number;
  allocation: number;
}

interface PortfolioState {
  totalValue: number;
  cashBalance: number;
  positions: PortfolioPosition[];
  performance: {
    daily: number;
    weekly: number;
    monthly: number;
    yearly: number;
  };
  riskMetrics: {
    volatility: number;
    sharpeRatio: number;
    maxDrawdown: number;
    beta: number;
  };
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: PortfolioState = {
  totalValue: 0,
  cashBalance: 0,
  positions: [],
  performance: {
    daily: 0,
    weekly: 0,
    monthly: 0,
    yearly: 0,
  },
  riskMetrics: {
    volatility: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    beta: 0,
  },
  loading: false,
  error: null,
  lastUpdated: null,
};

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    fetchPortfolioStart(state) {
      state.loading = true;
      state.error = null;
    },
    fetchPortfolioSuccess(state, action: PayloadAction<Partial<PortfolioState>>) {
      return {
        ...state,
        ...action.payload,
        loading: false,
        error: null,
        lastUpdated: new Date().toISOString(),
      };
    },
    fetchPortfolioFailure(state, action: PayloadAction<string>) {
      state.loading = false;
      state.error = action.payload;
    },
    updatePortfolio(state, action: PayloadAction<Partial<PortfolioState>>) {
      return {
        ...state,
        ...action.payload,
        lastUpdated: new Date().toISOString(),
      };
    },
  },
});

export const {
  fetchPortfolioStart,
  fetchPortfolioSuccess,
  fetchPortfolioFailure,
  updatePortfolio,
} = portfolioSlice.actions;

export default portfolioSlice.reducer;