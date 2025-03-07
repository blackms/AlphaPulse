export interface Alert {
  id: string;
  title: string;
  message: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  timestamp: string;
  acknowledged: boolean;
  resolved: boolean;
  category: string;
  source: string;
  type: string;
  actions?: AlertAction[];
  metadata?: Record<string, any>;
  read?: boolean;
}

export interface AlertAction {
  label: string;
  action: string;
  color?: 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success';
  variant?: 'text' | 'outlined' | 'contained';
}

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  conditions: AlertCondition[];
  actions: AlertRuleAction[];
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  category: string;
  createdAt: string;
  updatedAt: string;
}

export interface AlertCondition {
  type: string;
  metric: string;
  operator: string;
  value: number | string;
  duration?: number;
}

export interface AlertRuleAction {
  type: string;
  target: string;
  template?: string;
  parameters?: Record<string, any>;
}

export interface AlertPreferences {
  emailNotifications: boolean;
  pushNotifications: boolean;
  soundEnabled: boolean;
  alertThreshold: 'critical' | 'high' | 'medium' | 'low' | 'all';
  channels: {
    email: boolean;
    sms: boolean;
    push: boolean;
    slack: boolean;
  };
  preferences: {
    includeCritical: boolean;
    includeHigh: boolean;
    includeMedium: boolean;
    includeLow: boolean;
  };
}

export enum AlertSeverity {
  CRITICAL = 'critical',
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low',
  INFO = 'info'
}