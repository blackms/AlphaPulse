import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type SignalDirection = 'buy' | 'sell' | 'neutral';
export type SignalSource = 'technical' | 'fundamental' | 'sentiment' | 'value' | 'activist' | 'combined';
export type SignalTimeframe = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w';
export type OrderType = 'market' | 'limit' | 'stop' | 'stop_limit';
export type OrderStatus = 'pending' | 'open' | 'filled' | 'partially_filled' | 'canceled' | 'rejected' | 'expired';
export type StrategyType = 'trend_following' | 'mean_reversion' | 'breakout' | 'momentum' | 'value' | 'volatility' | 'grid' | 'custom';

export interface TradingSignal {
  id: string;
  timestamp: number;
  asset: string;
  direction: SignalDirection;
  source: SignalSource;
  confidence: number;
  timeframe: SignalTimeframe;
  price: number;
  volume?: number;
  metadata?: Record<string, any>;
  expiration?: number;
  isActive: boolean;
}

export interface TradeOrder {
  id: string;
  timestamp: number;
  asset: string;
  side: 'buy' | 'sell';
  type: OrderType;
  status: OrderStatus;
  quantity: number;
  price: number;
  limitPrice?: number;
  stopPrice?: number;
  filledQuantity: number;
  cost: number;
  fee: number;
  signalId?: string;
  strategyId?: string;
}

export interface TradeExecution {
  id: string;
  orderId: string;
  timestamp: number;
  asset: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  fee: number;
}

export interface TradingStrategy {
  id: string;
  name: string;
  description: string;
  type: StrategyType;
  assets: string[];
  enabled: boolean;
  parameters: Record<string, any>;
  performance: {
    totalTrades: number;
    winRate: number;
    profitFactor: number;
    averageProfit: number;
    sharpeRatio: number;
    maxDrawdown: number;
  };
  createdAt: number;
  updatedAt: number;
}

export interface StrategyBacktest {
  id: string;
  strategyId: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  finalCapital: number;
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  trades: number;
  winRate: number;
  profitFactor: number;
  averageProfit: number;
  averageLoss: number;
  averageHoldingPeriod: number;
  results: {
    dates: string[];
    equity: number[];
    drawdown: number[];
    benchmark?: number[];
  };
  completedAt: number;
}

interface TradingState {
  signals: TradingSignal[];
  orders: TradeOrder[];
  executions: TradeExecution[];
  strategies: TradingStrategy[];
  backtests: StrategyBacktest[];
  activeTrades: number;
  isLoading: boolean;
  error: string | null;
}

const initialState: TradingState = {
  signals: [
    {
      id: 'signal1',
      timestamp: Date.now() - 2 * 60 * 60 * 1000,
      asset: 'BTC',
      direction: 'buy',
      source: 'technical',
      confidence: 0.82,
      timeframe: '4h',
      price: 46750,
      volume: 1.25,
      metadata: {
        pattern: 'breakout',
        indicators: {
          rsi: 65,
          macd: 'bullish',
          ma_cross: true
        }
      },
      isActive: true
    },
    {
      id: 'signal2',
      timestamp: Date.now() - 3 * 60 * 60 * 1000,
      asset: 'ETH',
      direction: 'buy',
      source: 'fundamental',
      confidence: 0.75,
      timeframe: '1d',
      price: 3200,
      volume: 12.5,
      metadata: {
        metrics: {
          network_growth: 'high',
          active_addresses: 'increasing',
          development_activity: 'high'
        }
      },
      isActive: true
    },
    {
      id: 'signal3',
      timestamp: Date.now() - 5 * 60 * 60 * 1000,
      asset: 'SOL',
      direction: 'sell',
      source: 'sentiment',
      confidence: 0.68,
      timeframe: '1d',
      price: 105.50,
      metadata: {
        sentiment: 'bearish',
        social_volume: 'decreasing',
        news_impact: 'negative'
      },
      isActive: false
    },
    {
      id: 'signal4',
      timestamp: Date.now() - 8 * 60 * 60 * 1000,
      asset: 'AVAX',
      direction: 'neutral',
      source: 'value',
      confidence: 0.62,
      timeframe: '1d',
      price: 32.75,
      metadata: {
        metrics: {
          pe_ratio: 'neutral',
          market_cap_to_tvl: 'fair',
          revenue_growth: 'moderate'
        }
      },
      isActive: true
    },
    {
      id: 'signal5',
      timestamp: Date.now() - 1 * 24 * 60 * 60 * 1000,
      asset: 'BTC',
      direction: 'buy',
      source: 'combined',
      confidence: 0.85,
      timeframe: '1d',
      price: 45800,
      metadata: {
        components: {
          technical: 0.78,
          fundamental: 0.82,
          sentiment: 0.62,
          value: 0.75
        }
      },
      isActive: true
    }
  ],
  orders: [
    {
      id: 'order1',
      timestamp: Date.now() - 1 * 60 * 60 * 1000,
      asset: 'BTC',
      side: 'buy',
      type: 'market',
      status: 'filled',
      quantity: 0.25,
      price: 46750,
      filledQuantity: 0.25,
      cost: 11687.50,
      fee: 11.69,
      signalId: 'signal1',
      strategyId: 'strategy1'
    },
    {
      id: 'order2',
      timestamp: Date.now() - 2 * 60 * 60 * 1000,
      asset: 'ETH',
      side: 'buy',
      type: 'limit',
      status: 'open',
      quantity: 5.0,
      price: 3150,
      filledQuantity: 0,
      cost: 0,
      fee: 0,
      signalId: 'signal2',
      strategyId: 'strategy1'
    },
    {
      id: 'order3',
      timestamp: Date.now() - 6 * 60 * 60 * 1000,
      asset: 'SOL',
      side: 'sell',
      type: 'market',
      status: 'filled',
      quantity: 25.0,
      price: 105.50,
      filledQuantity: 25.0,
      cost: 2637.50,
      fee: 2.64,
      signalId: 'signal3',
      strategyId: 'strategy2'
    }
  ],
  executions: [
    {
      id: 'exec1',
      orderId: 'order1',
      timestamp: Date.now() - 1 * 60 * 60 * 1000,
      asset: 'BTC',
      side: 'buy',
      quantity: 0.25,
      price: 46750,
      fee: 11.69
    },
    {
      id: 'exec2',
      orderId: 'order3',
      timestamp: Date.now() - 6 * 60 * 60 * 1000,
      asset: 'SOL',
      side: 'sell',
      quantity: 25.0,
      price: 105.50,
      fee: 2.64
    }
  ],
  strategies: [
    {
      id: 'strategy1',
      name: 'Multi-Agent Alpha',
      description: 'Combined signals from all agents with risk-optimized position sizing',
      type: 'custom',
      assets: ['BTC', 'ETH', 'SOL', 'AVAX', 'DOT', 'LINK', 'MATIC'],
      enabled: true,
      parameters: {
        technical_weight: 0.3,
        fundamental_weight: 0.3,
        sentiment_weight: 0.2,
        value_weight: 0.2,
        min_confidence: 0.65,
        position_size_max: 0.1,
        stop_loss_atr: 2.5
      },
      performance: {
        totalTrades: 87,
        winRate: 0.64,
        profitFactor: 1.85,
        averageProfit: 3.2,
        sharpeRatio: 1.75,
        maxDrawdown: 12.8
      },
      createdAt: Date.now() - 90 * 24 * 60 * 60 * 1000,
      updatedAt: Date.now() - 5 * 24 * 60 * 60 * 1000
    },
    {
      id: 'strategy2',
      name: 'Breakout Momentum',
      description: 'Detects breakouts with volume confirmation and momentum follow-through',
      type: 'breakout',
      assets: ['BTC', 'ETH', 'SOL'],
      enabled: true,
      parameters: {
        lookback_period: 20,
        volume_threshold: 2.0,
        momentum_confirmation: true,
        entry_delay: 1,
        profit_target_atr: 4.0,
        stop_loss_atr: 2.0
      },
      performance: {
        totalTrades: 42,
        winRate: 0.55,
        profitFactor: 1.62,
        averageProfit: 4.5,
        sharpeRatio: 1.48,
        maxDrawdown: 15.2
      },
      createdAt: Date.now() - 60 * 24 * 60 * 60 * 1000,
      updatedAt: Date.now() - 10 * 24 * 60 * 60 * 1000
    }
  ],
  backtests: [
    {
      id: 'backtest1',
      strategyId: 'strategy1',
      startDate: '2022-01-01',
      endDate: '2022-12-31',
      initialCapital: 100000,
      finalCapital: 142500,
      totalReturn: 42.5,
      annualizedReturn: 42.5,
      sharpeRatio: 1.68,
      maxDrawdown: 18.2,
      trades: 124,
      winRate: 0.62,
      profitFactor: 1.78,
      averageProfit: 3.8,
      averageLoss: -2.6,
      averageHoldingPeriod: 3.5,
      results: {
        dates: ['2022-01-01', '2022-04-01', '2022-07-01', '2022-10-01', '2022-12-31'],
        equity: [100000, 112000, 98000, 125000, 142500],
        drawdown: [0, 5.2, 18.2, 8.1, 4.2],
        benchmark: [100000, 95000, 85000, 92000, 105000]
      },
      completedAt: Date.now() - 15 * 24 * 60 * 60 * 1000
    }
  ],
  activeTrades: 2,
  isLoading: false,
  error: null
};

const tradingSlice = createSlice({
  name: 'trading',
  initialState,
  reducers: {
    fetchTradingDataStart: (state) => {
      state.isLoading = true;
      state.error = null;
    },
    fetchTradingDataSuccess: (state, action: PayloadAction<{
      signals: TradingSignal[];
      orders: TradeOrder[];
      executions: TradeExecution[];
      strategies: TradingStrategy[];
      backtests: StrategyBacktest[];
      activeTrades: number;
    }>) => {
      state.signals = action.payload.signals;
      state.orders = action.payload.orders;
      state.executions = action.payload.executions;
      state.strategies = action.payload.strategies;
      state.backtests = action.payload.backtests;
      state.activeTrades = action.payload.activeTrades;
      state.isLoading = false;
    },
    fetchTradingDataFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;
    },
    addSignal: (state, action: PayloadAction<Omit<TradingSignal, 'id'>>) => {
      state.signals.push({
        id: `signal_${Date.now()}`,
        ...action.payload
      });
    },
    updateSignal: (state, action: PayloadAction<{
      id: string;
      updates: Partial<TradingSignal>;
    }>) => {
      const signal = state.signals.find(s => s.id === action.payload.id);
      if (signal) {
        Object.assign(signal, action.payload.updates);
      }
    },
    deactivateSignal: (state, action: PayloadAction<string>) => {
      const signal = state.signals.find(s => s.id === action.payload);
      if (signal) {
        signal.isActive = false;
      }
    },
    addOrder: (state, action: PayloadAction<Omit<TradeOrder, 'id'>>) => {
      state.orders.push({
        id: `order_${Date.now()}`,
        ...action.payload
      });
    },
    updateOrderStatus: (state, action: PayloadAction<{
      id: string;
      status: OrderStatus;
      filledQuantity?: number;
      cost?: number;
      fee?: number;
    }>) => {
      const order = state.orders.find(o => o.id === action.payload.id);
      if (order) {
        order.status = action.payload.status;
        
        if (action.payload.filledQuantity !== undefined) {
          order.filledQuantity = action.payload.filledQuantity;
        }
        
        if (action.payload.cost !== undefined) {
          order.cost = action.payload.cost;
        }
        
        if (action.payload.fee !== undefined) {
          order.fee = action.payload.fee;
        }
      }
    },
    addExecution: (state, action: PayloadAction<Omit<TradeExecution, 'id'>>) => {
      state.executions.push({
        id: `exec_${Date.now()}`,
        ...action.payload
      });
    },
    addStrategy: (state, action: PayloadAction<Omit<TradingStrategy, 'id'>>) => {
      state.strategies.push({
        id: `strategy_${Date.now()}`,
        ...action.payload
      });
    },
    updateStrategy: (state, action: PayloadAction<{
      id: string;
      updates: Partial<TradingStrategy>;
    }>) => {
      const strategy = state.strategies.find(s => s.id === action.payload.id);
      if (strategy) {
        Object.assign(strategy, action.payload.updates);
        strategy.updatedAt = Date.now();
      }
    },
    toggleStrategyEnabled: (state, action: PayloadAction<string>) => {
      const strategy = state.strategies.find(s => s.id === action.payload);
      if (strategy) {
        strategy.enabled = !strategy.enabled;
        strategy.updatedAt = Date.now();
      }
    },
    addBacktest: (state, action: PayloadAction<Omit<StrategyBacktest, 'id'>>) => {
      state.backtests.push({
        id: `backtest_${Date.now()}`,
        ...action.payload
      });
    }
  }
});

export const {
  fetchTradingDataStart,
  fetchTradingDataSuccess,
  fetchTradingDataFailure,
  addSignal,
  updateSignal,
  deactivateSignal,
  addOrder,
  updateOrderStatus,
  addExecution,
  addStrategy,
  updateStrategy,
  toggleStrategyEnabled,
  addBacktest
} = tradingSlice.actions;

// Selectors
export const selectTradingSignals = (state: RootState) => state.trading.signals;
export const selectActiveSignals = (state: RootState) => 
  state.trading.signals.filter(signal => signal.isActive);
export const selectSignalsByAsset = (asset: string) => 
  (state: RootState) => state.trading.signals.filter(signal => signal.asset === asset);
export const selectSignalsBySource = (source: SignalSource) => 
  (state: RootState) => state.trading.signals.filter(signal => signal.source === source);
export const selectTradingOrders = (state: RootState) => state.trading.orders;
export const selectOpenOrders = (state: RootState) => 
  state.trading.orders.filter(order => order.status === 'open' || order.status === 'pending');
export const selectExecutions = (state: RootState) => state.trading.executions;
export const selectStrategies = (state: RootState) => state.trading.strategies;
export const selectEnabledStrategies = (state: RootState) => 
  state.trading.strategies.filter(strategy => strategy.enabled);
export const selectStrategyById = (id: string) => 
  (state: RootState) => state.trading.strategies.find(s => s.id === id);
export const selectBacktests = (state: RootState) => state.trading.backtests;
export const selectBacktestsByStrategyId = (strategyId: string) => 
  (state: RootState) => state.trading.backtests.filter(b => b.strategyId === strategyId);
export const selectActiveTrades = (state: RootState) => state.trading.activeTrades;
export const selectIsLoading = (state: RootState) => state.trading.isLoading;
export const selectError = (state: RootState) => state.trading.error;

export default tradingSlice.reducer;