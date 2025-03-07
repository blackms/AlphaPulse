// User related types
export interface User {
  id: string;
  username: string;
  email: string;
  role: string;
  createdAt?: string;
  updatedAt?: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  user: User;
}

// Alert related types
export interface Alert {
  id: string;
  message: string;
  severity: 'info' | 'warning' | 'critical';
  source: string;
  timestamp: string;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
}

export interface AlertFilters {
  severity?: string[];
  acknowledged?: boolean | null;
  timeRange?: {
    start: string | null;
    end: string | null;
  };
  search?: string;
}

// Metric related types
export interface Metric {
  id: string;
  name: string;
  value: number;
  unit: string;
  timestamp: string;
  tags?: Record<string, string>;
}

export interface MetricSeries {
  name: string;
  data: Array<{
    timestamp: string;
    value: number;
  }>;
  unit: string;
}

export interface MetricFilter {
  metricNames?: string[];
  timeRange?: {
    start: string | null;
    end: string | null;
  };
  tags?: Record<string, string>;
  aggregation?: 'avg' | 'sum' | 'min' | 'max' | 'count';
  interval?: string;
}

// Portfolio related types
export interface Position {
  id: string;
  symbol: string;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercentage: number;
  value: number;
  allocation: number;
  tags?: string[];
}

export interface Portfolio {
  id: string;
  name: string;
  totalValue: number;
  cash: number;
  pnl: number;
  pnlPercentage: number;
  positions: Position[];
  updatedAt: string;
}

export interface PortfolioHistory {
  timestamp: string;
  totalValue: number;
  cash: number;
  invested: number;
}

// Trade related types
export interface Trade {
  id: string;
  symbol: string;
  type: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: string;
  status: 'pending' | 'executed' | 'canceled' | 'failed';
  fee?: number;
  total: number;
  notes?: string;
  tags?: string[];
  executedBy: 'system' | 'user';
  signalSource?: string;
}

export interface TradeFilter {
  symbols?: string[];
  types?: ('buy' | 'sell')[];
  status?: ('pending' | 'executed' | 'canceled' | 'failed')[];
  timeRange?: {
    start: string | null;
    end: string | null;
  };
  executedBy?: ('system' | 'user')[];
  signalSource?: string[];
}

// System related types
export interface SystemStatus {
  status: 'operational' | 'degraded' | 'maintenance' | 'outage';
  message?: string;
  components: {
    [key: string]: {
      status: 'operational' | 'degraded' | 'maintenance' | 'outage';
      message?: string;
    };
  };
  updatedAt: string;
}

export interface SystemMetrics {
  cpu: number;
  memory: number;
  disk: number;
  network: {
    in: number;
    out: number;
  };
  processCount: number;
  uptime: number;
}

// API related types
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  errors?: Record<string, string[]>;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

// WebSocket related types
export interface WebSocketMessage<T> {
  type: string;
  data: T;
  timestamp: string;
}

// UI related types
export interface Notification {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  autoHide: boolean;
  duration?: number;
}

export interface ThemeConfig {
  name: string;
  palette: {
    primary: string;
    secondary: string;
    error: string;
    warning: string;
    info: string;
    success: string;
    background: string;
    paper: string;
    text: string;
  };
}