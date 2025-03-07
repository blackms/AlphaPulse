import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type AssetSymbol = string;
export type SignalDirection = 'buy' | 'sell' | 'hold';
export type OrderStatus = 'pending' | 'filled' | 'partial' | 'canceled' | 'rejected' | 'expired';
export type OrderType = 'market' | 'limit' | 'stop' | 'stop_limit' | 'trailing_stop';
export type OrderSide = 'buy' | 'sell';
export type TimeInForce = 'gtc' | 'ioc' | 'fok' | 'day';
export type AgentType = 'technical' | 'fundamental' | 'sentiment' | 'value' | 'activist';
export type PositionStatus = 'open' | 'closed' | 'partially_closed';

export interface Signal {
  id: string;
  timestamp: number;
  asset: AssetSymbol;
  direction: SignalDirection;
  confidence: number;
  source: AgentType;
  metadata: Record<string, any>;
}

export interface Order {
  id: string;
  timestamp: number;
  asset: AssetSymbol;
  type: OrderType;
  side: OrderSide;
  quantity: number;
  price?: number;
  stopPrice?: number;
  limitPrice?: number;
  timeInForce: TimeInForce;
  status: OrderStatus;
  filledQuantity: number;
  averageFilledPrice?: number;
  fee?: number;
  createdAt: number;
  updatedAt: number;
  metadata: Record<string, any>;
}

export interface Position {
  id: string;
  asset: AssetSymbol;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  openedAt: number;
  updatedAt: number;
  pnl: number;
  pnlPercentage: number;
  status: PositionStatus;
  metadata: Record<string, any>;
}

export interface AssetData {
  symbol: AssetSymbol;
  name: string;
  price: number;
  priceChange24h: number;
  priceChangePercent24h: number;
  volume24h: number;
  marketCap: number;
  updatedAt: number;
}

export interface TradingSettings {
  maxPositionSize: number;
  minConfidenceThreshold: number;
  enableAutomatedTrading: boolean;
  defaultOrderType: OrderType;
  defaultTimeInForce: TimeInForce;
  stoplossPercentage: number;
  takeProfitPercentage: number;
  enableTrailingStopLoss: boolean;
  trailingStopDistance: number;
}

export interface TechnicalAgentSettings {
  enabled: boolean;
  weight: number;
  timeframes: string[];
  indicators: string[];
}

export interface FundamentalAgentSettings {
  enabled: boolean;
  weight: number;
  metrics: string[];
  updateFrequency: string;
}

export interface SentimentAgentSettings {
  enabled: boolean;
  weight: number;
  sources: string[];
  updateFrequency: string;
}

export interface ValueAgentSettings {
  enabled: boolean;
  weight: number;
  metrics: string[];
  updateFrequency: string;
}

export interface ActivistAgentSettings {
  enabled: boolean;
  weight: number;
  trackingItems: string[];
  updateFrequency: string;
}

export interface AgentSettings {
  technical: TechnicalAgentSettings;
  fundamental: FundamentalAgentSettings;
  sentiment: SentimentAgentSettings;
  value: ValueAgentSettings;
  activist: ActivistAgentSettings;
}

interface TradingState {
  signals: Signal[];
  orders: Order[];
  positions: Position[];
  assets: AssetData[];
  tradingSettings: TradingSettings;
  agentSettings: AgentSettings;
  isLoadingSignals: boolean;
  isLoadingOrders: boolean;
  isLoadingPositions: boolean;
  isLoadingAssets: boolean;
  signalsError: string | null;
  ordersError: string | null;
  positionsError: string | null;
  assetsError: string | null;
}

// Initial state with mock data
const initialState: TradingState = {
  signals: [
    {
      id: 'signal1',
      timestamp: Date.now() - 1800000, // 30 minutes ago
      asset: 'BTC',
      direction: 'buy',
      confidence: 0.87,
      source: 'technical',
      metadata: {
        indicators: {
          rsi: 63,
          macd: 'bullish',
          ema_crossover: true
        },
        timeframe: '4h'
      }
    },
    {
      id: 'signal2',
      timestamp: Date.now() - 3600000, // 1 hour ago
      asset: 'ETH',
      direction: 'buy',
      confidence: 0.72,
      source: 'fundamental',
      metadata: {
        metrics: {
          network_activity: 'increasing',
          development_activity: 'high',
          token_velocity: 'moderate'
        },
        analysis: 'Strong ecosystem growth with increasing adoption metrics'
      }
    },
    {
      id: 'signal3',
      timestamp: Date.now() - 7200000, // 2 hours ago
      asset: 'SOL',
      direction: 'sell',
      confidence: 0.68,
      source: 'sentiment',
      metadata: {
        sentiment_score: -0.32,
        social_volume: 'high',
        news_sentiment: 'bearish',
        sources: ['twitter', 'reddit', 'news']
      }
    },
    {
      id: 'signal4',
      timestamp: Date.now() - 14400000, // 4 hours ago
      asset: 'AVAX',
      direction: 'buy',
      confidence: 0.75,
      source: 'value',
      metadata: {
        tvl_ratio: 'undervalued',
        pe_equivalent: 'low',
        revenue_growth: 'high',
        comparison_metrics: {
          peers_average: 'above'
        }
      }
    },
    {
      id: 'signal5',
      timestamp: Date.now() - 86400000, // 24 hours ago
      asset: 'UNI',
      direction: 'buy',
      confidence: 0.82,
      source: 'activist',
      metadata: {
        governance: {
          proposal: 'UIP-123',
          expected_impact: 'positive',
          voting_ends: Date.now() + 172800000 // 48 hours from now
        },
        strategic_developments: {
          partnerships: ['major_defi_protocol'],
          product_launches: ['v4_upgrade']
        }
      }
    }
  ],
  orders: [
    {
      id: 'order1',
      timestamp: Date.now() - 1200000, // 20 minutes ago
      asset: 'BTC',
      type: 'market',
      side: 'buy',
      quantity: 0.5,
      timeInForce: 'gtc',
      status: 'filled',
      filledQuantity: 0.5,
      averageFilledPrice: 48250,
      fee: 12.06,
      createdAt: Date.now() - 1200000,
      updatedAt: Date.now() - 1180000,
      metadata: {
        signal_id: 'signal1',
        exchange: 'binance',
        client_order_id: 'algo_btc_buy_1234'
      }
    },
    {
      id: 'order2',
      timestamp: Date.now() - 3000000, // 50 minutes ago
      asset: 'ETH',
      type: 'limit',
      side: 'buy',
      quantity: 2.5,
      price: 2650,
      timeInForce: 'gtc',
      status: 'pending',
      filledQuantity: 0,
      createdAt: Date.now() - 3000000,
      updatedAt: Date.now() - 3000000,
      metadata: {
        signal_id: 'signal2',
        exchange: 'coinbase',
        client_order_id: 'algo_eth_buy_5678'
      }
    },
    {
      id: 'order3',
      timestamp: Date.now() - 5400000, // 1.5 hours ago
      asset: 'SOL',
      type: 'stop_limit',
      side: 'sell',
      quantity: 25,
      stopPrice: 92,
      limitPrice: 91.5,
      timeInForce: 'gtc',
      status: 'filled',
      filledQuantity: 25,
      averageFilledPrice: 91.8,
      fee: 34.43,
      createdAt: Date.now() - 5400000,
      updatedAt: Date.now() - 5000000,
      metadata: {
        signal_id: 'signal3',
        exchange: 'binance',
        client_order_id: 'algo_sol_sell_9012'
      }
    },
    {
      id: 'order4',
      timestamp: Date.now() - 12600000, // 3.5 hours ago
      asset: 'AVAX',
      type: 'market',
      side: 'buy',
      quantity: 50,
      timeInForce: 'ioc',
      status: 'filled',
      filledQuantity: 50,
      averageFilledPrice: 31.42,
      fee: 23.57,
      createdAt: Date.now() - 12600000,
      updatedAt: Date.now() - 12580000,
      metadata: {
        signal_id: 'signal4',
        exchange: 'kraken',
        client_order_id: 'algo_avax_buy_3456'
      }
    },
    {
      id: 'order5',
      timestamp: Date.now() - 82800000, // 23 hours ago
      asset: 'UNI',
      type: 'limit',
      side: 'buy',
      quantity: 200,
      price: 7.5,
      timeInForce: 'gtc',
      status: 'partial',
      filledQuantity: 120,
      averageFilledPrice: 7.5,
      fee: 13.5,
      createdAt: Date.now() - 82800000,
      updatedAt: Date.now() - 45000000,
      metadata: {
        signal_id: 'signal5',
        exchange: 'coinbase',
        client_order_id: 'algo_uni_buy_7890'
      }
    }
  ],
  positions: [
    {
      id: 'position1',
      asset: 'BTC',
      quantity: 1.2,
      entryPrice: 46800,
      currentPrice: 48250,
      openedAt: Date.now() - 604800000, // 1 week ago
      updatedAt: Date.now() - 600,
      pnl: 1740,
      pnlPercentage: 3.08,
      status: 'open',
      metadata: {
        stopLoss: 42000,
        takeProfit: 56000,
        strategy: 'momentum',
        risk_score: 'medium'
      }
    },
    {
      id: 'position2',
      asset: 'ETH',
      quantity: 10,
      entryPrice: 2500,
      currentPrice: 2650,
      openedAt: Date.now() - 1209600000, // 2 weeks ago
      updatedAt: Date.now() - 600,
      pnl: 1500,
      pnlPercentage: 6.0,
      status: 'open',
      metadata: {
        stopLoss: 2100,
        takeProfit: 3000,
        strategy: 'value',
        risk_score: 'medium'
      }
    },
    {
      id: 'position3',
      asset: 'SOL',
      quantity: 0,
      entryPrice: 105,
      currentPrice: 91.8,
      openedAt: Date.now() - 2592000000, // 30 days ago
      updatedAt: Date.now() - 5000000,
      pnl: -3300,
      pnlPercentage: -12.57,
      status: 'closed',
      metadata: {
        stopLoss: 92,
        takeProfit: 150,
        strategy: 'technical',
        risk_score: 'high',
        closed_reason: 'stop_loss_triggered'
      }
    },
    {
      id: 'position4',
      asset: 'AVAX',
      quantity: 50,
      entryPrice: 31.42,
      currentPrice: 34.18,
      openedAt: Date.now() - 12600000, // 3.5 hours ago
      updatedAt: Date.now() - 600,
      pnl: 138,
      pnlPercentage: 8.78,
      status: 'open',
      metadata: {
        stopLoss: 28,
        takeProfit: 40,
        strategy: 'value',
        risk_score: 'medium'
      }
    },
    {
      id: 'position5',
      asset: 'UNI',
      quantity: 120,
      entryPrice: 7.5,
      currentPrice: 8.2,
      openedAt: Date.now() - 82800000, // 23 hours ago
      updatedAt: Date.now() - 600,
      pnl: 84,
      pnlPercentage: 9.33,
      status: 'open',
      metadata: {
        stopLoss: 6.5,
        takeProfit: 12,
        strategy: 'activist',
        risk_score: 'low'
      }
    }
  ],
  assets: [
    {
      symbol: 'BTC',
      name: 'Bitcoin',
      price: 48250,
      priceChange24h: 1250,
      priceChangePercent24h: 2.66,
      volume24h: 28500000000,
      marketCap: 952000000000,
      updatedAt: Date.now() - 60000
    },
    {
      symbol: 'ETH',
      name: 'Ethereum',
      price: 2650,
      priceChange24h: 120,
      priceChangePercent24h: 4.74,
      volume24h: 18750000000,
      marketCap: 310000000000,
      updatedAt: Date.now() - 60000
    },
    {
      symbol: 'SOL',
      name: 'Solana',
      price: 91.8,
      priceChange24h: -8.5,
      priceChangePercent24h: -8.47,
      volume24h: 4280000000,
      marketCap: 38500000000,
      updatedAt: Date.now() - 60000
    },
    {
      symbol: 'AVAX',
      name: 'Avalanche',
      price: 34.18,
      priceChange24h: 2.85,
      priceChangePercent24h: 9.1,
      volume24h: 1950000000,
      marketCap: 12800000000,
      updatedAt: Date.now() - 60000
    },
    {
      symbol: 'UNI',
      name: 'Uniswap',
      price: 8.2,
      priceChange24h: 0.48,
      priceChangePercent24h: 6.21,
      volume24h: 385000000,
      marketCap: 4150000000,
      updatedAt: Date.now() - 60000
    }
  ],
  tradingSettings: {
    maxPositionSize: 5,
    minConfidenceThreshold: 0.65,
    enableAutomatedTrading: true,
    defaultOrderType: 'market',
    defaultTimeInForce: 'gtc',
    stoplossPercentage: 10,
    takeProfitPercentage: 25,
    enableTrailingStopLoss: true,
    trailingStopDistance: 5
  },
  agentSettings: {
    technical: {
      enabled: true,
      weight: 0.3,
      timeframes: ['1h', '4h', '1d'],
      indicators: ['rsi', 'macd', 'bollinger', 'ema', 'volume']
    },
    fundamental: {
      enabled: true,
      weight: 0.2,
      metrics: ['network_activity', 'development_activity', 'token_velocity', 'user_growth'],
      updateFrequency: 'daily'
    },
    sentiment: {
      enabled: true,
      weight: 0.15,
      sources: ['twitter', 'reddit', 'news', 'telegram', 'discord'],
      updateFrequency: 'hourly'
    },
    value: {
      enabled: true,
      weight: 0.2,
      metrics: ['tvl_ratio', 'pe_equivalent', 'revenue_growth', 'token_economics'],
      updateFrequency: 'daily'
    },
    activist: {
      enabled: true,
      weight: 0.15,
      trackingItems: ['governance', 'partnerships', 'protocol_upgrades', 'token_unlocks'],
      updateFrequency: 'daily'
    }
  },
  isLoadingSignals: false,
  isLoadingOrders: false,
  isLoadingPositions: false,
  isLoadingAssets: false,
  signalsError: null,
  ordersError: null,
  positionsError: null,
  assetsError: null
};

// Type guard to check if an agent type has a specific property
type AgentSettingsPayload = {
  agentType: 'technical';
  settings: Partial<TechnicalAgentSettings>;
} | {
  agentType: 'fundamental';
  settings: Partial<FundamentalAgentSettings>;
} | {
  agentType: 'sentiment';
  settings: Partial<SentimentAgentSettings>;
} | {
  agentType: 'value';
  settings: Partial<ValueAgentSettings>;
} | {
  agentType: 'activist';
  settings: Partial<ActivistAgentSettings>;
};

const tradingSlice = createSlice({
  name: 'trading',
  initialState,
  reducers: {
    // Signal actions
    fetchSignalsStart: (state) => {
      state.isLoadingSignals = true;
      state.signalsError = null;
    },
    fetchSignalsSuccess: (state, action: PayloadAction<Signal[]>) => {
      state.signals = action.payload;
      state.isLoadingSignals = false;
    },
    fetchSignalsFailure: (state, action: PayloadAction<string>) => {
      state.isLoadingSignals = false;
      state.signalsError = action.payload;
    },
    addSignal: (state, action: PayloadAction<Signal>) => {
      state.signals.unshift(action.payload);
    },
    
    // Order actions
    fetchOrdersStart: (state) => {
      state.isLoadingOrders = true;
      state.ordersError = null;
    },
    fetchOrdersSuccess: (state, action: PayloadAction<Order[]>) => {
      state.orders = action.payload;
      state.isLoadingOrders = false;
    },
    fetchOrdersFailure: (state, action: PayloadAction<string>) => {
      state.isLoadingOrders = false;
      state.ordersError = action.payload;
    },
    addOrder: (state, action: PayloadAction<Order>) => {
      state.orders.unshift(action.payload);
    },
    updateOrderStatus: (state, action: PayloadAction<{
      orderId: string;
      status: OrderStatus;
      filledQuantity?: number;
      averageFilledPrice?: number;
      updatedAt: number;
    }>) => {
      const order = state.orders.find(o => o.id === action.payload.orderId);
      if (order) {
        order.status = action.payload.status;
        order.updatedAt = action.payload.updatedAt;
        if (action.payload.filledQuantity !== undefined) {
          order.filledQuantity = action.payload.filledQuantity;
        }
        if (action.payload.averageFilledPrice !== undefined) {
          order.averageFilledPrice = action.payload.averageFilledPrice;
        }
      }
    },
    
    // Position actions
    fetchPositionsStart: (state) => {
      state.isLoadingPositions = true;
      state.positionsError = null;
    },
    fetchPositionsSuccess: (state, action: PayloadAction<Position[]>) => {
      state.positions = action.payload;
      state.isLoadingPositions = false;
    },
    fetchPositionsFailure: (state, action: PayloadAction<string>) => {
      state.isLoadingPositions = false;
      state.positionsError = action.payload;
    },
    updatePosition: (state, action: PayloadAction<Partial<Position> & { id: string }>) => {
      const position = state.positions.find(p => p.id === action.payload.id);
      if (position) {
        Object.assign(position, action.payload);
      }
    },
    
    // Asset actions
    fetchAssetsStart: (state) => {
      state.isLoadingAssets = true;
      state.assetsError = null;
    },
    fetchAssetsSuccess: (state, action: PayloadAction<AssetData[]>) => {
      state.assets = action.payload;
      state.isLoadingAssets = false;
    },
    fetchAssetsFailure: (state, action: PayloadAction<string>) => {
      state.isLoadingAssets = false;
      state.assetsError = action.payload;
    },
    updateAssetPrice: (state, action: PayloadAction<{
      symbol: AssetSymbol;
      price: number;
      priceChange24h: number;
      priceChangePercent24h: number;
      updatedAt: number;
    }>) => {
      const asset = state.assets.find(a => a.symbol === action.payload.symbol);
      if (asset) {
        asset.price = action.payload.price;
        asset.priceChange24h = action.payload.priceChange24h;
        asset.priceChangePercent24h = action.payload.priceChangePercent24h;
        asset.updatedAt = action.payload.updatedAt;
        
        // Update position current prices and PnL
        state.positions.forEach(position => {
          if (position.asset === action.payload.symbol) {
            position.currentPrice = action.payload.price;
            position.pnl = (action.payload.price - position.entryPrice) * position.quantity;
            position.pnlPercentage = ((action.payload.price / position.entryPrice) - 1) * 100;
            position.updatedAt = action.payload.updatedAt;
          }
        });
      }
    },
    
    // Settings actions
    updateTradingSettings: (state, action: PayloadAction<Partial<TradingSettings>>) => {
      state.tradingSettings = {
        ...state.tradingSettings,
        ...action.payload
      };
    },
    updateAgentSettings: (state, action: PayloadAction<Partial<AgentSettings>>) => {
      state.agentSettings = {
        ...state.agentSettings,
        ...action.payload
      };
    },
    updateAgentTypeSettings: (state, action: PayloadAction<AgentSettingsPayload>) => {
      const { agentType, settings } = action.payload;
      
      // Type-safe way to update the specific agent settings
      if (agentType === 'technical') {
        state.agentSettings.technical = {
          ...state.agentSettings.technical,
          ...settings
        };
      } else if (agentType === 'fundamental') {
        state.agentSettings.fundamental = {
          ...state.agentSettings.fundamental,
          ...settings
        };
      } else if (agentType === 'sentiment') {
        state.agentSettings.sentiment = {
          ...state.agentSettings.sentiment,
          ...settings
        };
      } else if (agentType === 'value') {
        state.agentSettings.value = {
          ...state.agentSettings.value,
          ...settings
        };
      } else if (agentType === 'activist') {
        state.agentSettings.activist = {
          ...state.agentSettings.activist,
          ...settings
        };
      }
    }
  }
});

export const {
  fetchSignalsStart,
  fetchSignalsSuccess,
  fetchSignalsFailure,
  addSignal,
  fetchOrdersStart,
  fetchOrdersSuccess,
  fetchOrdersFailure,
  addOrder,
  updateOrderStatus,
  fetchPositionsStart,
  fetchPositionsSuccess,
  fetchPositionsFailure,
  updatePosition,
  fetchAssetsStart,
  fetchAssetsSuccess,
  fetchAssetsFailure,
  updateAssetPrice,
  updateTradingSettings,
  updateAgentSettings,
  updateAgentTypeSettings
} = tradingSlice.actions;

// Selectors
export const selectAllSignals = (state: RootState) => state.trading.signals;
export const selectRecentSignals = (state: RootState) => 
  [...state.trading.signals].sort((a, b) => b.timestamp - a.timestamp).slice(0, 10);
export const selectSignalsByAsset = (asset: AssetSymbol) => 
  (state: RootState) => state.trading.signals.filter(s => s.asset === asset);
export const selectSignalsBySource = (source: AgentType) => 
  (state: RootState) => state.trading.signals.filter(s => s.source === source);

export const selectAllOrders = (state: RootState) => state.trading.orders;
export const selectRecentOrders = (state: RootState) => 
  [...state.trading.orders].sort((a, b) => b.timestamp - a.timestamp).slice(0, 10);
export const selectOrdersByAsset = (asset: AssetSymbol) => 
  (state: RootState) => state.trading.orders.filter(o => o.asset === asset);
export const selectOrdersByStatus = (status: OrderStatus) => 
  (state: RootState) => state.trading.orders.filter(o => o.status === status);

export const selectAllPositions = (state: RootState) => state.trading.positions;
export const selectOpenPositions = (state: RootState) => 
  state.trading.positions.filter(p => p.status === 'open');
export const selectPositionByAsset = (asset: AssetSymbol) => 
  (state: RootState) => state.trading.positions.find(p => p.asset === asset && p.status === 'open');
export const selectPositionsByStatus = (status: PositionStatus) => 
  (state: RootState) => state.trading.positions.filter(p => p.status === status);

export const selectAllAssets = (state: RootState) => state.trading.assets;
export const selectAssetBySymbol = (symbol: AssetSymbol) => 
  (state: RootState) => state.trading.assets.find(a => a.symbol === symbol);

export const selectTradingSettings = (state: RootState) => state.trading.tradingSettings;
export const selectAgentSettings = (state: RootState) => state.trading.agentSettings;
export const selectAgentTypeSettings = (agentType: AgentType) => 
  (state: RootState) => state.trading.agentSettings[agentType];

export const selectLoadingStates = (state: RootState) => ({
  signals: state.trading.isLoadingSignals,
  orders: state.trading.isLoadingOrders,
  positions: state.trading.isLoadingPositions,
  assets: state.trading.isLoadingAssets,
});

export const selectErrorStates = (state: RootState) => ({
  signals: state.trading.signalsError,
  orders: state.trading.ordersError,
  positions: state.trading.positionsError,
  assets: state.trading.assetsError,
});

export default tradingSlice.reducer;