import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type SignalDirection = 'buy' | 'sell' | 'hold';
export type SignalStrength = 'weak' | 'moderate' | 'strong';
export type SignalSource = 'technical' | 'fundamental' | 'sentiment' | 'value' | 'ensemble';
export type OrderStatus = 'pending' | 'filled' | 'partial' | 'cancelled' | 'failed';

export interface Signal {
  id: string;
  assetId: string;
  symbol: string;
  name: string;
  direction: SignalDirection;
  strength: SignalStrength;
  confidence: number; // 0-100
  source: SignalSource;
  description: string;
  timestamp: number;
  expiresAt: number;
  suggestedSize: number; // In USD or as percentage of portfolio
  riskScore: number; // 0-100, higher is riskier
  status: 'active' | 'expired' | 'executed';
  indicators?: {
    name: string;
    value: number;
    trend: 'up' | 'down' | 'neutral';
  }[];
}

export interface Trade {
  id: string;
  signalId?: string;
  assetId: string;
  symbol: string;
  direction: 'buy' | 'sell';
  quantity: number;
  price: number;
  total: number;
  timestamp: number;
  status: OrderStatus;
  fees: number;
  pnl?: number;
  pnlPercent?: number;
  executionSpeed?: number; // milliseconds
  notes?: string;
}

interface TradingState {
  activeSignals: Signal[];
  historicalSignals: Signal[];
  recentTrades: Trade[];
  historicalTrades: Trade[];
  pendingOrders: Trade[];
  loading: boolean;
  lastUpdated: number | null;
}

const initialState: TradingState = {
  activeSignals: [
    {
      id: 's1',
      assetId: '1',
      symbol: 'BTC',
      name: 'Bitcoin',
      direction: 'buy',
      strength: 'strong',
      confidence: 85,
      source: 'technical',
      description: 'Bullish breakout pattern confirmed with strong volume support',
      timestamp: Date.now() - 3 * 60 * 60 * 1000, // 3 hours ago
      expiresAt: Date.now() + 21 * 60 * 60 * 1000, // expires in 21 hours
      suggestedSize: 5000,
      riskScore: 45,
      status: 'active',
      indicators: [
        { name: 'RSI', value: 62, trend: 'up' },
        { name: 'MACD', value: 235, trend: 'up' },
        { name: 'MA Cross', value: 1, trend: 'up' },
      ],
    },
    {
      id: 's2',
      assetId: '3',
      symbol: 'SOL',
      name: 'Solana',
      direction: 'buy',
      strength: 'moderate',
      confidence: 68,
      source: 'fundamental',
      description: 'Increasing network adoption and developer activity metrics',
      timestamp: Date.now() - 5 * 60 * 60 * 1000, // 5 hours ago
      expiresAt: Date.now() + 19 * 60 * 60 * 1000, // expires in 19 hours
      suggestedSize: 3500,
      riskScore: 60,
      status: 'active',
      indicators: [
        { name: 'Active Addresses', value: 278500, trend: 'up' },
        { name: 'Transaction Count', value: 4230000, trend: 'up' },
        { name: 'TVL Change', value: 3.8, trend: 'up' },
      ],
    },
    {
      id: 's3',
      assetId: '2',
      symbol: 'ETH',
      name: 'Ethereum',
      direction: 'sell',
      strength: 'weak',
      confidence: 55,
      source: 'sentiment',
      description: 'Increasing negative sentiment in social media metrics',
      timestamp: Date.now() - 8 * 60 * 60 * 1000, // 8 hours ago
      expiresAt: Date.now() + 16 * 60 * 60 * 1000, // expires in 16 hours
      suggestedSize: 2000,
      riskScore: 65,
      status: 'active',
      indicators: [
        { name: 'Social Volume', value: 125400, trend: 'up' },
        { name: 'Sentiment Score', value: -0.12, trend: 'down' },
        { name: 'News Sentiment', value: -0.08, trend: 'down' },
      ],
    },
  ],
  historicalSignals: [
    {
      id: 'hs1',
      assetId: '1',
      symbol: 'BTC',
      name: 'Bitcoin',
      direction: 'buy',
      strength: 'strong',
      confidence: 88,
      source: 'ensemble',
      description: 'Multiple agents confirming bullish trend continuation',
      timestamp: Date.now() - 3 * 24 * 60 * 60 * 1000, // 3 days ago
      expiresAt: Date.now() - 2 * 24 * 60 * 60 * 1000, // expired 2 days ago
      suggestedSize: 8000,
      riskScore: 40,
      status: 'executed',
    },
    {
      id: 'hs2',
      assetId: '4',
      symbol: 'LINK',
      name: 'Chainlink',
      direction: 'buy',
      strength: 'moderate',
      confidence: 72,
      source: 'technical',
      description: 'Cup and handle pattern forming with volume confirmation',
      timestamp: Date.now() - 5 * 24 * 60 * 60 * 1000, // 5 days ago
      expiresAt: Date.now() - 4 * 24 * 60 * 60 * 1000, // expired 4 days ago
      suggestedSize: 2500,
      riskScore: 55,
      status: 'executed',
    },
    {
      id: 'hs3',
      assetId: '5',
      symbol: 'MATIC',
      name: 'Polygon',
      direction: 'sell',
      strength: 'strong',
      confidence: 82,
      source: 'fundamental',
      description: 'Declining network metrics and increased competition',
      timestamp: Date.now() - 8 * 24 * 60 * 60 * 1000, // 8 days ago
      expiresAt: Date.now() - 7 * 24 * 60 * 60 * 1000, // expired 7 days ago
      suggestedSize: 1800,
      riskScore: 35,
      status: 'expired',
    },
  ],
  recentTrades: [
    {
      id: 't1',
      signalId: 'hs1',
      assetId: '1',
      symbol: 'BTC',
      direction: 'buy',
      quantity: 0.12,
      price: 63550,
      total: 7626,
      timestamp: Date.now() - 2.5 * 24 * 60 * 60 * 1000, // 2.5 days ago
      status: 'filled',
      fees: 7.63,
      executionSpeed: 235, // milliseconds
    },
    {
      id: 't2',
      signalId: 'hs2',
      assetId: '4',
      symbol: 'LINK',
      direction: 'buy',
      quantity: 130,
      price: 18.15,
      total: 2359.5,
      timestamp: Date.now() - 4.2 * 24 * 60 * 60 * 1000, // 4.2 days ago
      status: 'filled',
      fees: 2.36,
      executionSpeed: 312, // milliseconds
    },
    {
      id: 't3',
      assetId: '2',
      symbol: 'ETH',
      direction: 'sell',
      quantity: 1.5,
      price: 3150,
      total: 4725,
      timestamp: Date.now() - 6.1 * 24 * 60 * 60 * 1000, // 6.1 days ago
      status: 'filled',
      fees: 4.73,
      pnl: 525,
      pnlPercent: 12.5,
      executionSpeed: 180, // milliseconds
    },
    {
      id: 't4',
      signalId: 'hs3',
      assetId: '5',
      symbol: 'MATIC',
      direction: 'sell',
      quantity: 1800,
      price: 0.98,
      total: 1764,
      timestamp: Date.now() - 7.3 * 24 * 60 * 60 * 1000, // 7.3 days ago
      status: 'filled',
      fees: 1.76,
      pnl: 144,
      pnlPercent: 8.9,
      executionSpeed: 205, // milliseconds
    },
  ],
  historicalTrades: [], // Would be filled with more historical trades
  pendingOrders: [
    {
      id: 'p1',
      assetId: '3',
      symbol: 'SOL',
      direction: 'buy',
      quantity: 25,
      price: 133.5,
      total: 3337.5,
      timestamp: Date.now() - 1 * 60 * 60 * 1000, // 1 hour ago
      status: 'pending',
      fees: 3.34,
    },
  ],
  loading: false,
  lastUpdated: Date.now() - 15 * 60 * 1000, // 15 minutes ago
};

const tradingSlice = createSlice({
  name: 'trading',
  initialState,
  reducers: {
    fetchTradingDataStart: (state) => {
      state.loading = true;
    },
    fetchTradingDataSuccess: (state, action: PayloadAction<{
      activeSignals: Signal[];
      historicalSignals: Signal[];
      recentTrades: Trade[];
      pendingOrders: Trade[];
    }>) => {
      state.activeSignals = action.payload.activeSignals;
      state.historicalSignals = action.payload.historicalSignals;
      state.recentTrades = action.payload.recentTrades;
      state.pendingOrders = action.payload.pendingOrders;
      state.loading = false;
      state.lastUpdated = Date.now();
    },
    fetchTradingDataFailure: (state) => {
      state.loading = false;
    },
    addSignal: (state, action: PayloadAction<Signal>) => {
      state.activeSignals.push(action.payload);
    },
    updateSignal: (state, action: PayloadAction<{
      id: string;
      updates: Partial<Signal>;
    }>) => {
      const signal = state.activeSignals.find(s => s.id === action.payload.id);
      if (signal) {
        Object.assign(signal, action.payload.updates);
      }
    },
    expireSignal: (state, action: PayloadAction<string>) => {
      const signalIndex = state.activeSignals.findIndex(s => s.id === action.payload);
      if (signalIndex !== -1) {
        const signal = state.activeSignals[signalIndex];
        signal.status = 'expired';
        state.historicalSignals.push(signal);
        state.activeSignals.splice(signalIndex, 1);
      }
    },
    executeSignal: (state, action: PayloadAction<{
      signalId: string;
      trade: Trade;
    }>) => {
      const { signalId, trade } = action.payload;
      
      // Find and update the signal
      const signalIndex = state.activeSignals.findIndex(s => s.id === signalId);
      if (signalIndex !== -1) {
        const signal = state.activeSignals[signalIndex];
        signal.status = 'executed';
        state.historicalSignals.push(signal);
        state.activeSignals.splice(signalIndex, 1);
      }
      
      // Add the trade
      state.recentTrades.unshift(trade);
    },
    addTrade: (state, action: PayloadAction<Trade>) => {
      state.recentTrades.unshift(action.payload);
    },
    updateTradeStatus: (state, action: PayloadAction<{
      tradeId: string;
      status: OrderStatus;
      updates?: Partial<Trade>;
    }>) => {
      // Check in pending orders
      const pendingIndex = state.pendingOrders.findIndex(t => t.id === action.payload.tradeId);
      if (pendingIndex !== -1) {
        state.pendingOrders[pendingIndex].status = action.payload.status;
        
        if (action.payload.updates) {
          Object.assign(state.pendingOrders[pendingIndex], action.payload.updates);
        }
        
        // If the order is filled or partially filled, move it to recent trades
        if (action.payload.status === 'filled' || action.payload.status === 'partial') {
          state.recentTrades.unshift(state.pendingOrders[pendingIndex]);
          state.pendingOrders.splice(pendingIndex, 1);
        }
        return;
      }
      
      // Check in recent trades
      const recentIndex = state.recentTrades.findIndex(t => t.id === action.payload.tradeId);
      if (recentIndex !== -1) {
        state.recentTrades[recentIndex].status = action.payload.status;
        
        if (action.payload.updates) {
          Object.assign(state.recentTrades[recentIndex], action.payload.updates);
        }
      }
    },
    archiveTrades: (state, action: PayloadAction<string[]>) => {
      const tradesToArchive = state.recentTrades.filter(t => action.payload.includes(t.id));
      state.historicalTrades.push(...tradesToArchive);
      state.recentTrades = state.recentTrades.filter(t => !action.payload.includes(t.id));
    },
  },
});

export const {
  fetchTradingDataStart,
  fetchTradingDataSuccess,
  fetchTradingDataFailure,
  addSignal,
  updateSignal,
  expireSignal,
  executeSignal,
  addTrade,
  updateTradeStatus,
  archiveTrades,
} = tradingSlice.actions;

// Selectors
export const selectActiveSignals = (state: RootState) => state.trading.activeSignals;
export const selectHistoricalSignals = (state: RootState) => state.trading.historicalSignals;
export const selectRecentTrades = (state: RootState) => state.trading.recentTrades;
export const selectHistoricalTrades = (state: RootState) => state.trading.historicalTrades;
export const selectPendingOrders = (state: RootState) => state.trading.pendingOrders;
export const selectIsLoading = (state: RootState) => state.trading.loading;
export const selectLastUpdated = (state: RootState) => state.trading.lastUpdated;
export const selectSignalById = (id: string) =>
  (state: RootState) => [...state.trading.activeSignals, ...state.trading.historicalSignals].find(s => s.id === id);
export const selectTradeById = (id: string) =>
  (state: RootState) => [...state.trading.recentTrades, ...state.trading.historicalTrades, ...state.trading.pendingOrders].find(t => t.id === id);
export const selectSignalsByAsset = (assetId: string) =>
  (state: RootState) => state.trading.activeSignals.filter(s => s.assetId === assetId);
export const selectTradesByAsset = (assetId: string) =>
  (state: RootState) => state.trading.recentTrades.filter(t => t.assetId === assetId);

export default tradingSlice.reducer;