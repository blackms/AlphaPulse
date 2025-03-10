export interface Asset {
  assetId: string;
  symbol: string;
  name: string;
  type: string;
  quantity: number;
  price: number;
  value: number;
  allocation: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  costBasis: number;
  lastUpdated: string;
  change24h: number;
  change24hPercent: number;
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
  change?: number;
  changePercent?: number;
}

export interface PortfolioData {
  totalValue: number;
  cashBalance: number;
  assets: Asset[];
  performance: PerformancePeriod[];
  historicalValues: HistoricalValue[];
  lastUpdated: string;
  assetAllocation: {
    [key: string]: number;
  };
  // Optional error information for frontend error handling
  error?: string;
}