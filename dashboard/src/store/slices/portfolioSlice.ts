import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export interface Asset {
  assetId: string;
  symbol: string;
  name: string;
  quantity: number;
  price: number;
  value: number;
  allocation: number;
  dayChange: number;
  dayChangePercent: number;
  weekChange: number;
  weekChangePercent: number;
  monthChange: number;
  monthChangePercent: number;
  costBasis: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
}

export interface PerformancePeriod {
  period: 'day' | 'week' | 'month' | 'quarter' | 'year' | 'all';
  returnValue: number;
  returnPercent: number;
  startDate: string;
  endDate: string;
}

export interface HistoricalValue {
  timestamp: number;
  value: number;
}

interface PortfolioState {
  assets: Asset[];
  totalValue: number;
  cashBalance: number;
  performance: PerformancePeriod[];
  historicalValues: HistoricalValue[];
  loading: boolean;
  lastUpdated: number | null;
}

const initialState: PortfolioState = {
  assets: [
    {
      assetId: '1',
      symbol: 'BTC',
      name: 'Bitcoin',
      quantity: 0.85,
      price: 65800,
      value: 55930,
      allocation: 37.5,
      dayChange: 1650,
      dayChangePercent: 2.8,
      weekChange: 3500,
      weekChangePercent: 6.2,
      monthChange: 8900,
      monthChangePercent: 18.9,
      costBasis: 47000,
      unrealizedPnL: 8930,
      unrealizedPnLPercent: 19.0,
    },
    {
      assetId: '2',
      symbol: 'ETH',
      name: 'Ethereum',
      quantity: 12.5,
      price: 3200,
      value: 40000,
      allocation: 26.8,
      dayChange: -800,
      dayChangePercent: -1.8,
      weekChange: 1200,
      weekChangePercent: 3.1,
      monthChange: 4500,
      monthChangePercent: 12.7,
      costBasis: 32500,
      unrealizedPnL: 7500,
      unrealizedPnLPercent: 23.1,
    },
    {
      assetId: '3',
      symbol: 'SOL',
      name: 'Solana',
      quantity: 220,
      price: 135,
      value: 29700,
      allocation: 19.9,
      dayChange: 650,
      dayChangePercent: 2.2,
      weekChange: 1800,
      weekChangePercent: 6.5,
      monthChange: 3500,
      monthChangePercent: 13.2,
      costBasis: 24200,
      unrealizedPnL: 5500,
      unrealizedPnLPercent: 22.7,
    },
    {
      assetId: '4',
      symbol: 'LINK',
      name: 'Chainlink',
      quantity: 450,
      price: 18.5,
      value: 8325,
      allocation: 5.6,
      dayChange: 225,
      dayChangePercent: 2.7,
      weekChange: 540,
      weekChangePercent: 6.9,
      monthChange: 970,
      monthChangePercent: 13.2,
      costBasis: 6750,
      unrealizedPnL: 1575,
      unrealizedPnLPercent: 23.3,
    },
    {
      assetId: '5',
      symbol: 'MATIC',
      name: 'Polygon',
      quantity: 4500,
      price: 0.95,
      value: 4275,
      allocation: 2.9,
      dayChange: -120,
      dayChangePercent: -2.7,
      weekChange: 275,
      weekChangePercent: 6.9,
      monthChange: 530,
      monthChangePercent: 14.2,
      costBasis: 3600,
      unrealizedPnL: 675,
      unrealizedPnLPercent: 18.8,
    },
    {
      assetId: '6',
      symbol: 'CASH',
      name: 'USD Cash',
      quantity: 11000,
      price: 1,
      value: 11000,
      allocation: 7.4,
      dayChange: 0,
      dayChangePercent: 0,
      weekChange: 0,
      weekChangePercent: 0,
      monthChange: 0,
      monthChangePercent: 0,
      costBasis: 11000,
      unrealizedPnL: 0,
      unrealizedPnLPercent: 0,
    },
  ],
  totalValue: 149230,
  cashBalance: 11000,
  performance: [
    {
      period: 'day',
      returnValue: 1605,
      returnPercent: 1.09,
      startDate: '2025-03-06',
      endDate: '2025-03-07',
    },
    {
      period: 'week',
      returnValue: 7315,
      returnPercent: 5.16,
      startDate: '2025-02-28',
      endDate: '2025-03-07',
    },
    {
      period: 'month',
      returnValue: 18400,
      returnPercent: 14.07,
      startDate: '2025-02-07',
      endDate: '2025-03-07',
    },
    {
      period: 'quarter',
      returnValue: 43750,
      returnPercent: 41.48,
      startDate: '2024-12-07',
      endDate: '2025-03-07',
    },
    {
      period: 'year',
      returnValue: 83520,
      returnPercent: 127.2,
      startDate: '2024-03-07',
      endDate: '2025-03-07',
    },
    {
      period: 'all',
      returnValue: 99230,
      returnPercent: 198.5,
      startDate: '2023-08-15',
      endDate: '2025-03-07',
    },
  ],
  historicalValues: Array.from({ length: 90 }, (_, i) => {
    // Generate fake daily values for 90 days (showing an upward trend with variations)
    const baseValue = 90000; // Starting value 90 days ago
    const trend = i * 700; // Upward trend
    const variation = Math.sin(i * 0.1) * 5000; // Variation
    const randomFactor = (Math.random() * 2 - 1) * 2000; // Random noise
    
    return {
      timestamp: Date.now() - (90 - i) * 24 * 60 * 60 * 1000, // Past 90 days
      value: Math.max(0, baseValue + trend + variation + randomFactor), // Ensure value is positive
    };
  }),
  loading: false,
  lastUpdated: Date.now() - 5 * 60 * 1000, // 5 minutes ago
};

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    fetchPortfolioStart: (state) => {
      state.loading = true;
    },
    fetchPortfolioSuccess: (state, action: PayloadAction<{
      assets: Asset[];
      totalValue: number;
      cashBalance: number;
      performance: PerformancePeriod[];
    }>) => {
      const { assets, totalValue, cashBalance, performance } = action.payload;
      state.assets = assets;
      state.totalValue = totalValue;
      state.cashBalance = cashBalance;
      state.performance = performance;
      state.loading = false;
      state.lastUpdated = Date.now();
    },
    fetchPortfolioFailure: (state) => {
      state.loading = false;
    },
    fetchHistoricalDataSuccess: (state, action: PayloadAction<HistoricalValue[]>) => {
      state.historicalValues = action.payload;
    },
    updateAssetPrice: (state, action: PayloadAction<{
      assetId: string;
      price: number;
    }>) => {
      const asset = state.assets.find(a => a.assetId === action.payload.assetId);
      if (asset) {
        const oldValue = asset.value;
        asset.price = action.payload.price;
        asset.value = asset.quantity * action.payload.price;
        
        // Update day change
        asset.dayChange = asset.value - oldValue + asset.dayChange;
        asset.dayChangePercent = (asset.dayChange / (asset.value - asset.dayChange)) * 100;
        
        // Recalculate total value
        state.totalValue = state.assets.reduce((sum, asset) => sum + asset.value, 0);
        
        // Recalculate allocations
        state.assets.forEach(a => {
          a.allocation = (a.value / state.totalValue) * 100;
        });
      }
    },
    updateUnrealizedPnL: (state) => {
      state.assets.forEach(asset => {
        if (asset.symbol !== 'CASH') {
          asset.unrealizedPnL = asset.value - asset.costBasis;
          asset.unrealizedPnLPercent = (asset.unrealizedPnL / asset.costBasis) * 100;
        }
      });
    },
  },
});

export const {
  fetchPortfolioStart,
  fetchPortfolioSuccess,
  fetchPortfolioFailure,
  fetchHistoricalDataSuccess,
  updateAssetPrice,
  updateUnrealizedPnL,
} = portfolioSlice.actions;

// Selectors
export const selectAssets = (state: RootState) => state.portfolio.assets;
export const selectTotalValue = (state: RootState) => state.portfolio.totalValue;
export const selectCashBalance = (state: RootState) => state.portfolio.cashBalance;
export const selectPerformance = (state: RootState) => state.portfolio.performance;
export const selectHistoricalValues = (state: RootState) => state.portfolio.historicalValues;
export const selectIsLoading = (state: RootState) => state.portfolio.loading;
export const selectLastUpdated = (state: RootState) => state.portfolio.lastUpdated;
export const selectAssetById = (assetId: string) =>
  (state: RootState) => state.portfolio.assets.find(asset => asset.assetId === assetId);

export default portfolioSlice.reducer;