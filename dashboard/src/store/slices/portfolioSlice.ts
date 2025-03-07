import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

// Types for portfolio data
export interface PortfolioHistoryPoint {
  date: string;
  value: number;
  benchmark: number;
}

export interface PortfolioAllocationItem {
  asset: string;
  value: number;
  color?: string;
}

export type PositionStatus = 'active' | 'pending' | 'closing' | 'hedged';

export interface PortfolioPosition {
  asset: string;
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnlAmount: number;
  pnlPercentage: number;
  allocation: number;
  status: PositionStatus;
  hedgeRatio?: number;
  stopLoss?: number;
  takeProfit?: number;
  signalConfidence?: number;
  entryDate: string;
}

export interface PortfolioPerformance {
  currentValue: number;
  initialValue: number;
  totalReturn: number;
  daily: number;
  weekly: number;
  monthly: number;
  yearly: number;
  maxDrawdown: number;
  sharpeRatio: number;
  volatility: number;
  startDate: string;
  history: PortfolioHistoryPoint[];
}

interface PortfolioState {
  performance: PortfolioPerformance;
  allocation: PortfolioAllocationItem[];
  positions: PortfolioPosition[];
  rebalancingStatus: 'none' | 'pending' | 'completed';
  lastRebalanced: string | null;
  isLoading: boolean;
  error: string | null;
}

// Initial state with mock data
const initialState: PortfolioState = {
  performance: {
    currentValue: 158462.75,
    initialValue: 100000,
    totalReturn: 58.46,
    daily: 2.34,
    weekly: 5.78,
    monthly: 12.45,
    yearly: 42.18,
    maxDrawdown: 15.7,
    sharpeRatio: 1.87,
    volatility: 22.4,
    startDate: '2023-01-15',
    history: [
      { date: '2023-01-15', value: 100000, benchmark: 100000 },
      { date: '2023-02-01', value: 103500, benchmark: 102000 },
      { date: '2023-03-01', value: 108750, benchmark: 105000 },
      { date: '2023-04-01', value: 112400, benchmark: 107500 },
      { date: '2023-05-01', value: 118600, benchmark: 110000 },
      { date: '2023-06-01', value: 125800, benchmark: 112500 },
      { date: '2023-07-01', value: 131400, benchmark: 115000 },
      { date: '2023-08-01', value: 124300, benchmark: 113000 },
      { date: '2023-09-01', value: 130500, benchmark: 116000 },
      { date: '2023-10-01', value: 138700, benchmark: 119000 },
      { date: '2023-11-01', value: 145200, benchmark: 121500 },
      { date: '2023-12-01', value: 152800, benchmark: 124000 },
      { date: '2024-01-01', value: 158462.75, benchmark: 127500 },
    ],
  },
  allocation: [
    { asset: 'BTC', value: 35.2 },
    { asset: 'ETH', value: 25.8 },
    { asset: 'SOL', value: 12.5 },
    { asset: 'AVAX', value: 8.7 },
    { asset: 'DOT', value: 6.3 },
    { asset: 'LINK', value: 5.4 },
    { asset: 'MATIC', value: 4.1 },
    { asset: 'USD', value: 2.0 },
  ],
  positions: [
    {
      asset: 'BTC',
      size: 1.25,
      entryPrice: 42500,
      currentPrice: 46800,
      pnlAmount: 5375,
      pnlPercentage: 10.12,
      allocation: 35.2,
      status: 'active',
      hedgeRatio: 0.15,
      stopLoss: 38000,
      takeProfit: 52000,
      signalConfidence: 0.82,
      entryDate: '2023-11-15',
    },
    {
      asset: 'ETH',
      size: 12.5,
      entryPrice: 2800,
      currentPrice: 3200,
      pnlAmount: 5000,
      pnlPercentage: 14.29,
      allocation: 25.8,
      status: 'active',
      hedgeRatio: 0.1,
      stopLoss: 2450,
      takeProfit: 3600,
      signalConfidence: 0.75,
      entryDate: '2023-12-02',
    },
    {
      asset: 'SOL',
      size: 350,
      entryPrice: 55,
      currentPrice: 63,
      pnlAmount: 2800,
      pnlPercentage: 14.55,
      allocation: 12.5,
      status: 'active',
      stopLoss: 48,
      takeProfit: 75,
      signalConfidence: 0.68,
      entryDate: '2023-12-10',
    },
    {
      asset: 'AVAX',
      size: 480,
      entryPrice: 28,
      currentPrice: 30,
      pnlAmount: 960,
      pnlPercentage: 7.14,
      allocation: 8.7,
      status: 'hedged',
      hedgeRatio: 0.5,
      stopLoss: 24,
      signalConfidence: 0.62,
      entryDate: '2023-12-18',
    },
    {
      asset: 'DOT',
      size: 1200,
      entryPrice: 7.8,
      currentPrice: 8.25,
      pnlAmount: 540,
      pnlPercentage: 5.77,
      allocation: 6.3,
      status: 'active',
      stopLoss: 6.9,
      takeProfit: 10,
      signalConfidence: 0.58,
      entryDate: '2024-01-05',
    },
    {
      asset: 'LINK',
      size: 850,
      entryPrice: 9.5,
      currentPrice: 10.1,
      pnlAmount: 510,
      pnlPercentage: 6.32,
      allocation: 5.4,
      status: 'active',
      stopLoss: 8.2,
      takeProfit: 12.5,
      signalConfidence: 0.65,
      entryDate: '2024-01-08',
    },
    {
      asset: 'MATIC',
      size: 7500,
      entryPrice: 0.8,
      currentPrice: 0.85,
      pnlAmount: 375,
      pnlPercentage: 6.25,
      allocation: 4.1,
      status: 'pending',
      signalConfidence: 0.55,
      entryDate: '2024-01-12',
    },
  ],
  rebalancingStatus: 'completed',
  lastRebalanced: '2024-01-10T08:30:00Z',
  isLoading: false,
  error: null,
};

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    fetchPortfolioStart: (state) => {
      state.isLoading = true;
      state.error = null;
    },
    fetchPortfolioSuccess: (state, action: PayloadAction<{
      performance: PortfolioPerformance;
      allocation: PortfolioAllocationItem[];
      positions: PortfolioPosition[];
      rebalancingStatus: 'none' | 'pending' | 'completed';
      lastRebalanced: string | null;
    }>) => {
      state.performance = action.payload.performance;
      state.allocation = action.payload.allocation;
      state.positions = action.payload.positions;
      state.rebalancingStatus = action.payload.rebalancingStatus;
      state.lastRebalanced = action.payload.lastRebalanced;
      state.isLoading = false;
    },
    fetchPortfolioFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;
    },
    updatePosition: (state, action: PayloadAction<{
      asset: string;
      updates: Partial<PortfolioPosition>;
    }>) => {
      const position = state.positions.find(p => p.asset === action.payload.asset);
      if (position) {
        Object.assign(position, action.payload.updates);
      }
    },
    rebalancePortfolio: (state) => {
      state.rebalancingStatus = 'pending';
    },
    rebalancePortfolioSuccess: (state, action: PayloadAction<{
      allocation: PortfolioAllocationItem[];
      positions: PortfolioPosition[];
      lastRebalanced: string;
    }>) => {
      state.allocation = action.payload.allocation;
      state.positions = action.payload.positions;
      state.lastRebalanced = action.payload.lastRebalanced;
      state.rebalancingStatus = 'completed';
    },
    rebalancePortfolioFailure: (state, action: PayloadAction<string>) => {
      state.rebalancingStatus = 'none';
      state.error = action.payload;
    },
    addTransaction: (state, action: PayloadAction<{
      asset: string;
      type: 'buy' | 'sell';
      size: number;
      price: number;
      date: string;
    }>) => {
      const { asset, type, size, price } = action.payload;
      const position = state.positions.find(p => p.asset === asset);
      
      if (position) {
        // Update existing position
        if (type === 'buy') {
          const newSize = position.size + size;
          const newEntryPrice = (position.size * position.entryPrice + size * price) / newSize;
          position.size = newSize;
          position.entryPrice = newEntryPrice;
        } else {
          position.size -= size;
          // If sold all, remove position
          if (position.size <= 0) {
            state.positions = state.positions.filter(p => p.asset !== asset);
          }
        }
      } else if (type === 'buy') {
        // Add new position
        state.positions.push({
          asset,
          size,
          entryPrice: price,
          currentPrice: price,
          pnlAmount: 0,
          pnlPercentage: 0,
          allocation: 0, // Will be calculated on next update
          status: 'active',
          entryDate: action.payload.date,
        });
      }
      
      // Update allocations (simplified)
      const totalValue = state.positions.reduce(
        (sum, pos) => sum + pos.size * pos.currentPrice, 
        0
      );
      
      state.positions.forEach(pos => {
        pos.allocation = (pos.size * pos.currentPrice / totalValue) * 100;
      });
      
      state.allocation = state.positions.map(pos => ({
        asset: pos.asset,
        value: pos.allocation,
      }));
    },
  },
});

export const {
  fetchPortfolioStart,
  fetchPortfolioSuccess,
  fetchPortfolioFailure,
  updatePosition,
  rebalancePortfolio,
  rebalancePortfolioSuccess,
  rebalancePortfolioFailure,
  addTransaction,
} = portfolioSlice.actions;

// Selectors
export const selectPortfolioPerformance = (state: RootState) => state.portfolio.performance;
export const selectPortfolioAllocation = (state: RootState) => state.portfolio.allocation;
export const selectPortfolioPositions = (state: RootState) => state.portfolio.positions;
export const selectRebalancingStatus = (state: RootState) => state.portfolio.rebalancingStatus;
export const selectLastRebalanced = (state: RootState) => state.portfolio.lastRebalanced;
export const selectIsLoading = (state: RootState) => state.portfolio.isLoading;
export const selectError = (state: RootState) => state.portfolio.error;

export default portfolioSlice.reducer;