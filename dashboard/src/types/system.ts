export enum SystemStatus {
  OPERATIONAL = 'operational',
  DEGRADED = 'degraded',
  MAINTENANCE = 'maintenance',
  OUTAGE = 'outage'
}

export enum ComponentStatus {
  HEALTHY = 'healthy',
  WARNING = 'warning',
  ERROR = 'error',
  OFFLINE = 'offline'
}

export interface SystemComponent {
  id: string;
  name: string;
  type: string;
  status: string;
  healthScore: number;
  lastUpdated: string;
  description: string;
  version?: string;
  dependencies?: string[];
}

export interface SystemLog {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  source: string;
  message: string;
  details?: string;
  component?: string;
}

export interface SystemMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  timestamp: string;
  change?: number;
  changePercent?: number;
  target?: number;
  status?: 'good' | 'warning' | 'error';
}
