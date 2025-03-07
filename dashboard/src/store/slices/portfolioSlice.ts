import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export interface AssetAllocation {
  assetId: string;
  symbol: string;
  name: string;
  allocation: number;
  value: number;
  quantity: number;
  price: number;
  dayChange: number;
  dayChangePercent: number;
}

export interface PortfolioPerformance {
  period: 'day' | 'week' | 'month' | 'quarter' | 'year' | 'all';
  returnValue: number;
  returnPercent: number;
}

export interface PortfolioHistoryPoint {
  timestamp: number;
  value: number;
}

interface PortfolioState {
  totalValue: number;
  cashBalance: number;
  assets: AssetAllocation[];
  performance: PortfolioPerformance[];
  history: PortfolioHistoryPoint[];
  loading: boolean;
  lastUpdated: number | null;
}

const initialState: PortfolioState = {
  totalValue: 125000,
  cashBalance: 25000,
  assets: [
    {
      assetId: 'btc',
      symbol: 'BTC',
      name: 'Bitcoin',
      allocation: 30,
      value: 37500,
      quantity: 0.75,
      price: 50000,
      dayChange: 1200,
      dayChangePercent: 2.4,
    },
    {
      assetId: 'eth',
      symbol: 'ETH',
      name: 'Ethereum',
      allocation: 25,
      value: 31250,
      quantity: 10,
      price: 3125,
      dayChange: 750,
      dayChangePercent: 2.1,
    },
    {
      assetId: 'sol',
      symbol: 'SOL',
      name: 'Solana',
      allocation: 15,
      value: 18750,
      quantity: 200,
      price: 93.75,
      dayChange: -375,
      dayChangePercent: -1.8,
    },
    {
      assetId: 'ada',
      symbol: 'ADA',
      name: 'Cardano',
      allocation: 10,
      value: 12500,
      quantity: 10000,
      price: 1.25,
      dayChange: 150,
      dayChangePercent: 1.2,
    },
  ],
  performance: [
    { period: 'day', returnValue: 1500, returnPercent: 1.2 },
    { period: 'week', returnValue: 3750, returnPercent: 3.1 },
    { period: 'month', returnValue: 10500, returnPercent: 9.2 },
    { period: 'quarter', returnValue: 15000, returnPercent: 13.6 },
    { period: 'year', returnValue: -2625, returnPercent: -2.1 },
    { period: 'all', returnValue: 25000, returnPercent: 25 },
  ],
  history: [
    // Example historical data points (would normally have many more)
    { timestamp: Date.now() - 86400000 * 30, value: 115000 },
    { timestamp: Date.now() - 86400000 * 25, value: 117500 },
    { timestamp: Date.now() - 86400000 * 20, value: 121000 },
    { timestamp: Date.now() - 86400000 * 15, value: 119500 },
    { timestamp: Date.now() - 86400000 * 10, value: 122000 },
    { timestamp: Date.now() - 86400000 * 5, value: 120000 },
    { timestamp: Date.now(), value: 125000 },
  ],
  loading: false,
  lastUpdated: Date.now(),
};

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    fetchPortfolioStart: (state) => {
      state.loading = true;
    },
    fetchPortfolioSuccess: (state, action: PayloadAction<{
      totalValue: number;
      cashBalance: number;
      assets: AssetAllocation[];
      performance: PortfolioPerformance[];
    }>) => {
      state.totalValue = action.payload.totalValue;
      state.cashBalance = action.payload.cashBalance;
      state.assets = action.payload.assets;
      state.performance = action.payload.performance;
      state.loading = false;
      state.lastUpdated = Date.now();
    },
    fetchPortfolioFailure: (state) => {
      state.loading = false;
    },
    fetchPortfolioHistorySuccess: (state, action: PayloadAction<PortfolioHistoryPoint[]>) => {
      state.history = action.payload;
    },
    updateAsset: (state, action: PayloadAction<AssetAllocation>) => {
      const index = state.assets.findIndex(a => a.assetId === action.payload.assetId);
      if (index !== -1) {
        state.assets[index] = action.payload;
      } else {
        state.assets.push(action.payload);
      }
      
      // Recalculate total value
      let totalAssetValue = 0;
      state.assets.forEach(asset => {
        totalAssetValue += asset.value;
      });
      state.totalValue = totalAssetValue + state.cashBalance;
      
      // Recalculate allocations
      state.assets.forEach(asset => {
        asset.allocation = (asset.value / state.totalValue) * 100;
      });
      
      state.lastUpdated = Date.now();
    },
  },
});

export const {
  fetchPortfolioStart,
  fetchPortfolioSuccess,
  fetchPortfolioFailure,
  fetchPortfolioHistorySuccess,
  updateAsset,
} = portfolioSlice.actions;

// Selectors
export const selectTotalValue = (state: RootState) => state.portfolio.totalValue;
export const selectCashBalance = (state: RootState) => state.portfolio.cashBalance;
export const selectAssets = (state: RootState) => state.portfolio.assets;
export const selectPerformance = (state: RootState) => state.portfolio.performance;
export const selectHistory = (state: RootState) => state.portfolio.history;
export const selectIsLoading = (state: RootState) => state.portfolio.loading;
export const selectLastUpdated = (state: RootState) => state.portfolio.lastUpdated;

export default portfolioSlice.reducer;