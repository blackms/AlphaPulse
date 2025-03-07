import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type TradeStatus = 'pending' | 'completed' | 'cancelled' | 'failed';
export type TradeDirection = 'buy' | 'sell';

export interface Trade {
  id: string;
  symbol: string;
  direction: TradeDirection;
  quantity: number;
  price: number;
  value: number;
  status: TradeStatus;
  timestamp: string;
  agent?: string;
  reason?: string;
}

export interface Order {
  id: string;
  symbol: string;
  direction: TradeDirection;
  quantity: number;
  price: number;
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  status: 'open' | 'filled' | 'partial' | 'cancelled';
  created: string;
}

interface TradingState {
  trades: Trade[];
  orders: Order[];
  performance: {
    winRate: number;
    profitFactor: number;
    avgWin: number;
    avgLoss: number;
    largestWin: number;
    largestLoss: number;
  };
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: TradingState = {
  trades: [],
  orders: [],
  performance: {
    winRate: 0,
    profitFactor: 0,
    avgWin: 0,
    avgLoss: 0,
    largestWin: 0,
    largestLoss: 0,
  },
  loading: false,
  error: null,
  lastUpdated: null,
};

const tradingSlice = createSlice({
  name: 'trading',
  initialState,
  reducers: {
    fetchTradesStart(state) {
      state.loading = true;
      state.error = null;
    },
    fetchTradesSuccess(state, action: PayloadAction<Partial<TradingState>>) {
      return {
        ...state,
        ...action.payload,
        loading: false,
        error: null,
        lastUpdated: new Date().toISOString(),
      };
    },
    fetchTradesFailure(state, action: PayloadAction<string>) {
      state.loading = false;
      state.error = action.payload;
    },
    updateTrades(state, action: PayloadAction<Partial<TradingState>>) {
      return {
        ...state,
        ...action.payload,
        lastUpdated: new Date().toISOString(),
      };
    },
    addTrade(state, action: PayloadAction<Trade>) {
      state.trades = [action.payload, ...state.trades].slice(0, 100); // Keep last 100 trades
      state.lastUpdated = new Date().toISOString();
    },
    updateOrder(state, action: PayloadAction<Order>) {
      const index = state.orders.findIndex(order => order.id === action.payload.id);
      if (index >= 0) {
        state.orders[index] = action.payload;
      } else {
        state.orders = [action.payload, ...state.orders];
      }
      state.lastUpdated = new Date().toISOString();
    },
  },
});

export const {
  fetchTradesStart,
  fetchTradesSuccess,
  fetchTradesFailure,
  updateTrades,
  addTrade,
  updateOrder,
} = tradingSlice.actions;

export default tradingSlice.reducer;