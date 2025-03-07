import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type TradeType = 'buy' | 'sell';
export type TradeStatus = 'pending' | 'completed' | 'failed' | 'canceled';

export interface Trade {
  id: string;
  assetId: string;
  symbol: string;
  type: TradeType;
  price: number;
  quantity: number;
  value: number;
  timestamp: number;
  status: TradeStatus;
  reason?: string;
  agentId?: string;
}

export interface TradeSignal {
  id: string;
  assetId: string;
  symbol: string;
  type: TradeType;
  confidence: number;
  suggestedPrice: number;
  suggestedQuantity: number;
  timestamp: number;
  agentId: string;
  agentName: string;
  rationale: string;
  status: 'new' | 'accepted' | 'rejected' | 'expired';
}

interface TradingState {
  recentTrades: Trade[];
  pendingTrades: Trade[];
  activeSignals: TradeSignal[];
  tradeHistory: Trade[];
  loading: boolean;
  lastUpdated: number | null;
}

const initialState: TradingState = {
  recentTrades: [
    {
      id: 't1',
      assetId: 'btc',
      symbol: 'BTC',
      type: 'buy',
      price: 49500,
      quantity: 0.1,
      value: 4950,
      timestamp: Date.now() - 3600000, // 1 hour ago
      status: 'completed',
      agentId: 'technical-agent',
    },
    {
      id: 't2',
      assetId: 'eth',
      symbol: 'ETH',
      type: 'sell',
      price: 3100,
      quantity: 2,
      value: 6200,
      timestamp: Date.now() - 86400000, // 1 day ago
      status: 'completed',
      agentId: 'value-agent',
    },
    {
      id: 't3',
      assetId: 'sol',
      symbol: 'SOL',
      type: 'buy',
      price: 93,
      quantity: 10,
      value: 930,
      timestamp: Date.now() - 172800000, // 2 days ago
      status: 'completed',
      agentId: 'sentiment-agent',
    },
  ],
  pendingTrades: [
    {
      id: 'p1',
      assetId: 'btc',
      symbol: 'BTC',
      type: 'buy',
      price: 50100,
      quantity: 0.05,
      value: 2505,
      timestamp: Date.now() - 1800000, // 30 minutes ago
      status: 'pending',
      agentId: 'technical-agent',
    },
  ],
  activeSignals: [
    {
      id: 's1',
      assetId: 'eth',
      symbol: 'ETH',
      type: 'buy',
      confidence: 0.78,
      suggestedPrice: 3150,
      suggestedQuantity: 1.5,
      timestamp: Date.now() - 900000, // 15 minutes ago
      agentId: 'technical-agent',
      agentName: 'Technical Analysis Agent',
      rationale: 'Strong bullish pattern detected with positive RSI divergence and increasing volume.',
      status: 'new',
    },
    {
      id: 's2',
      assetId: 'ada',
      symbol: 'ADA',
      type: 'sell',
      confidence: 0.62,
      suggestedPrice: 1.22,
      suggestedQuantity: 1000,
      timestamp: Date.now() - 1200000, // 20 minutes ago
      agentId: 'value-agent',
      agentName: 'Value Analysis Agent',
      rationale: 'Current price exceeds fair value estimation based on network metrics and adoption rates.',
      status: 'new',
    },
  ],
  tradeHistory: [],
  loading: false,
  lastUpdated: Date.now(),
};

const tradingSlice = createSlice({
  name: 'trading',
  initialState,
  reducers: {
    fetchTradesStart: (state) => {
      state.loading = true;
    },
    fetchTradesSuccess: (state, action: PayloadAction<{
      recentTrades: Trade[];
      pendingTrades: Trade[];
    }>) => {
      state.recentTrades = action.payload.recentTrades;
      state.pendingTrades = action.payload.pendingTrades;
      state.loading = false;
      state.lastUpdated = Date.now();
    },
    fetchTradesFailure: (state) => {
      state.loading = false;
    },
    fetchTradeHistorySuccess: (state, action: PayloadAction<Trade[]>) => {
      state.tradeHistory = action.payload;
    },
    fetchSignalsSuccess: (state, action: PayloadAction<TradeSignal[]>) => {
      state.activeSignals = action.payload;
    },
    addTrade: (state, action: PayloadAction<Trade>) => {
      // Add to pending if it's pending, otherwise to recent
      if (action.payload.status === 'pending') {
        state.pendingTrades.unshift(action.payload);
      } else {
        state.recentTrades.unshift(action.payload);
        // Keep recent trades limited to a reasonable number
        if (state.recentTrades.length > 10) {
          state.recentTrades.pop();
        }
      }
      state.lastUpdated = Date.now();
    },
    updateTradeStatus: (state, action: PayloadAction<{
      tradeId: string;
      status: TradeStatus;
      reason?: string;
    }>) => {
      // Look for the trade in pendingTrades first
      const pendingIndex = state.pendingTrades.findIndex(t => t.id === action.payload.tradeId);
      if (pendingIndex !== -1) {
        // Update the trade status
        state.pendingTrades[pendingIndex].status = action.payload.status;
        if (action.payload.reason) {
          state.pendingTrades[pendingIndex].reason = action.payload.reason;
        }
        
        // If it's no longer pending, move it to recentTrades
        if (action.payload.status !== 'pending') {
          const trade = state.pendingTrades[pendingIndex];
          state.pendingTrades.splice(pendingIndex, 1);
          state.recentTrades.unshift(trade);
          // Keep recent trades limited
          if (state.recentTrades.length > 10) {
            state.recentTrades.pop();
          }
        }
      } else {
        // Otherwise, look in recentTrades
        const recentIndex = state.recentTrades.findIndex(t => t.id === action.payload.tradeId);
        if (recentIndex !== -1) {
          state.recentTrades[recentIndex].status = action.payload.status;
          if (action.payload.reason) {
            state.recentTrades[recentIndex].reason = action.payload.reason;
          }
        }
      }
      state.lastUpdated = Date.now();
    },
    updateSignalStatus: (state, action: PayloadAction<{
      signalId: string;
      status: 'accepted' | 'rejected' | 'expired';
    }>) => {
      const index = state.activeSignals.findIndex(s => s.id === action.payload.signalId);
      if (index !== -1) {
        state.activeSignals[index].status = action.payload.status;
        // Remove it from activeSignals (it's no longer "new" after status update)
        state.activeSignals.splice(index, 1);
      }
    },
  },
});

export const {
  fetchTradesStart,
  fetchTradesSuccess,
  fetchTradesFailure,
  fetchTradeHistorySuccess,
  fetchSignalsSuccess,
  addTrade,
  updateTradeStatus,
  updateSignalStatus,
} = tradingSlice.actions;

// Selectors
export const selectRecentTrades = (state: RootState) => state.trading.recentTrades;
export const selectPendingTrades = (state: RootState) => state.trading.pendingTrades;
export const selectActiveSignals = (state: RootState) => state.trading.activeSignals;
export const selectTradeHistory = (state: RootState) => state.trading.tradeHistory;
export const selectIsLoading = (state: RootState) => state.trading.loading;
export const selectLastUpdated = (state: RootState) => state.trading.lastUpdated;

export default tradingSlice.reducer;