export type AlertSeverity = 'critical' | 'warning' | 'info' | 'success';
export type AlertSource = 'system' | 'trading' | 'portfolio' | 'risk' | 'user';

export interface Alert {
  id: string;
  title: string;
  message: string;
  severity: AlertSeverity;
  source: AlertSource;
  timestamp: string;
  acknowledged: boolean;
  details?: string;
  relatedEntity?: {
    type: string;
    id: string;
    name: string;
  };
}

export interface AlertsState {
  alerts: Alert[];
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
}