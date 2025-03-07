import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../rootReducer';
import { Asset, PerformancePeriod, HistoricalValue, PortfolioData } from '../../types/portfolio';

interface PortfolioState {
  totalValue: number;
  cashBalance: number;
  assets: Asset[];
  performance: PerformancePeriod[];
  historicalValues: HistoricalValue[];
  lastUpdated: string | null;
  loading: boolean;
  error: string | null;
}

const initialState: PortfolioState = {
  totalValue: 0,
  cashBalance: 0,
  assets: [],
  performance: [],
  historicalValues: [],
  lastUpdated: null,
  loading: false,
  error: null
};

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    fetchPortfolioStart(state) {
      state.loading = true;
      state.error = null;
    },
    fetchPortfolioSuccess(state, action: PayloadAction<PortfolioData>) {
      state.totalValue = action.payload.totalValue;
      state.cashBalance = action.payload.cashBalance;
      state.assets = action.payload.assets;
      state.performance = action.payload.performance;
      state.historicalValues = action.payload.historicalValues;
      state.lastUpdated = action.payload.lastUpdated;
      state.loading = false;
    },
    fetchPortfolioFailure(state, action: PayloadAction<string>) {
      state.loading = false;
      state.error = action.payload;
    },
    updateAssetPrice(state, action: PayloadAction<{ symbol: string; price: number }>) {
      const asset = state.assets.find(a => a.symbol === action.payload.symbol);
      if (asset) {
        const oldValue = asset.price * asset.quantity;
        asset.price = action.payload.price;
        const newValue = asset.price * asset.quantity;
        asset.value = newValue;
        asset.unrealizedPnL = newValue - (asset.costBasis * asset.quantity);
        asset.unrealizedPnLPercent = (asset.unrealizedPnL / (asset.costBasis * asset.quantity)) * 100;
        
        // Update total value
        state.totalValue = state.totalValue - oldValue + newValue;
        
        // Update allocations
        state.assets.forEach(a => {
          a.allocation = (a.value / state.totalValue) * 100;
        });
      }
    },
    addHistoricalValue(state, action: PayloadAction<HistoricalValue>) {
      state.historicalValues.push(action.payload);
      // Sort by timestamp
      state.historicalValues.sort((a, b) => a.timestamp - b.timestamp);
    }
  }
});

export const {
  fetchPortfolioStart,
  fetchPortfolioSuccess,
  fetchPortfolioFailure,
  updateAssetPrice,
  addHistoricalValue
} = portfolioSlice.actions;

// Selectors
export const selectTotalValue = (state: RootState) => state.portfolio.totalValue;
export const selectCashBalance = (state: RootState) => state.portfolio.cashBalance;
export const selectAssets = (state: RootState) => state.portfolio.assets;
export const selectPerformance = (state: RootState) => state.portfolio.performance;
export const selectHistoricalValues = (state: RootState) => state.portfolio.historicalValues;
export const selectLastUpdated = (state: RootState) => state.portfolio.lastUpdated;
export const selectPortfolioLoading = (state: RootState) => state.portfolio.loading;
export const selectPortfolioError = (state: RootState) => state.portfolio.error;
export const selectAssetBySymbol = (symbol: string) => (state: RootState) => 
  state.portfolio.assets.find(asset => asset.symbol === symbol);
export const selectAssetById = (id: string) => (state: RootState) => 
  state.portfolio.assets.find(asset => asset.assetId === id);
export const selectPerformanceByPeriod = (period: string) => (state: RootState) => 
  state.portfolio.performance.find(p => p.period === period);

// Re-export types properly
export type { Asset, PerformancePeriod, HistoricalValue };

export default portfolioSlice.reducer;