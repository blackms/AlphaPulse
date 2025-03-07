import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type AlertSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';
export type AlertStatus = 'active' | 'acknowledged' | 'resolved' | 'dismissed';
export type AlertCategory = 'system' | 'trading' | 'risk' | 'portfolio' | 'security' | 'performance';

export interface Alert {
  id: string;
  title: string;
  message: string;
  details?: string;
  severity: AlertSeverity;
  status: AlertStatus;
  category: AlertCategory;
  component?: string;
  timestamp: number;
  acknowledgedAt?: number;
  resolvedAt?: number;
  actions?: string[];
  metadata?: Record<string, any>;
}

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  severity: AlertSeverity;
  category: AlertCategory;
  condition: string;
  parameters: Record<string, any>;
  cooldown: number; // In seconds
  createdAt: number;
  updatedAt: number;
}

export interface AlertPreferences {
  emailNotifications: boolean;
  pushNotifications: boolean;
  smsNotifications: boolean;
  slackNotifications: boolean;
  criticalAlertsOnly: boolean;
  soundEnabled: boolean;
  notificationCooldown: number; // In seconds
  muteStartTime?: string; // Time format: "HH:MM"
  muteEndTime?: string; // Time format: "HH:MM"
  muteDays?: number[]; // 0 = Sunday, 6 = Saturday
}

interface AlertsState {
  alerts: Alert[];
  rules: AlertRule[];
  preferences: AlertPreferences;
  unreadCount: number;
  isLoading: boolean;
  error: string | null;
}

// Initial state with mock data
const initialState: AlertsState = {
  alerts: [
    {
      id: 'alert1',
      title: 'High Portfolio Exposure',
      message: 'Portfolio exposure has reached 85% of the maximum allowed limit',
      details: 'Current exposure is 42.5% with a maximum limit of 50%. Consider reducing position sizes or hedging some positions.',
      severity: 'medium',
      status: 'active',
      category: 'risk',
      component: 'risk-manager',
      timestamp: Date.now() - 25 * 60 * 1000, // 25 minutes ago
      actions: ['view_portfolio', 'adjust_exposure', 'ignore']
    },
    {
      id: 'alert2',
      title: 'API Rate Limit Warning',
      message: 'Sentiment data provider API rate limit at 80%',
      details: 'Current usage is 800 out of 1000 permitted requests per hour. Rate limit resets in 35 minutes.',
      severity: 'low',
      status: 'active',
      category: 'system',
      component: 'sentiment-agent',
      timestamp: Date.now() - 45 * 60 * 1000, // 45 minutes ago
      actions: ['reduce_frequency', 'view_usage', 'ignore']
    },
    {
      id: 'alert3',
      title: 'Stop Loss Triggered',
      message: 'Stop loss triggered for AVAX position',
      details: 'Stop loss at $26.80 has been triggered. Order executed at $26.75 with a 2.1% slippage.',
      severity: 'high',
      status: 'resolved',
      category: 'trading',
      component: 'execution-broker',
      timestamp: Date.now() - 3 * 60 * 60 * 1000, // 3 hours ago
      resolvedAt: Date.now() - 2.9 * 60 * 60 * 1000, // 2.9 hours ago
      actions: ['view_transaction', 'adjust_settings']
    },
    {
      id: 'alert4',
      title: 'Strong Buy Signal',
      message: 'Multiple agents generating strong buy signals for ETH',
      details: 'Technical (0.82), Fundamental (0.78), and Sentiment (0.71) agents all reporting strong buy signals for ETH.',
      severity: 'info',
      status: 'acknowledged',
      category: 'trading',
      timestamp: Date.now() - 5 * 60 * 60 * 1000, // 5 hours ago
      acknowledgedAt: Date.now() - 4.8 * 60 * 60 * 1000, // 4.8 hours ago
      actions: ['place_order', 'view_analysis', 'ignore']
    },
    {
      id: 'alert5',
      title: 'Database Performance Degraded',
      message: 'Database query times increasing beyond threshold',
      details: 'Average query time has increased to 85ms (threshold: 50ms). Database load is at 78%.',
      severity: 'high',
      status: 'active',
      category: 'performance',
      component: 'database',
      timestamp: Date.now() - 30 * 60 * 1000, // 30 minutes ago
      actions: ['optimize_queries', 'increase_resources', 'view_metrics']
    },
    {
      id: 'alert6',
      title: 'Portfolio Rebalancing Required',
      message: 'Portfolio drift exceeds threshold of 5%',
      details: 'Current portfolio allocation has drifted 7.8% from target allocation. Major drifts: BTC +4.2%, ETH -3.5%',
      severity: 'medium',
      status: 'active',
      category: 'portfolio',
      component: 'portfolio-manager',
      timestamp: Date.now() - 12 * 60 * 60 * 1000, // 12 hours ago
      actions: ['rebalance_now', 'view_details', 'schedule_later']
    },
    {
      id: 'alert7',
      title: 'Critical Security Update Required',
      message: 'System security update available for API Gateway',
      details: 'Security patch v2.4.5 addresses critical vulnerability CVE-2024-1234. Immediate update recommended.',
      severity: 'critical',
      status: 'active',
      category: 'security',
      component: 'api-gateway',
      timestamp: Date.now() - 2 * 60 * 60 * 1000, // 2 hours ago
      actions: ['update_now', 'schedule_update', 'view_details']
    }
  ],
  rules: [
    {
      id: 'rule1',
      name: 'Portfolio Exposure Alert',
      description: 'Alerts when portfolio exposure exceeds 85% of maximum limit',
      enabled: true,
      severity: 'medium',
      category: 'risk',
      condition: 'current_exposure > max_exposure * 0.85',
      parameters: {
        max_exposure: 50
      },
      cooldown: 3600, // 1 hour
      createdAt: Date.now() - 30 * 24 * 60 * 60 * 1000, // 30 days ago
      updatedAt: Date.now() - 10 * 24 * 60 * 60 * 1000 // 10 days ago
    },
    {
      id: 'rule2',
      name: 'API Rate Limit Alert',
      description: 'Alerts when API rate limit usage exceeds 80%',
      enabled: true,
      severity: 'low',
      category: 'system',
      condition: 'rate_limit_usage > 0.8',
      parameters: {
        check_interval: 300 // 5 minutes
      },
      cooldown: 1800, // 30 minutes
      createdAt: Date.now() - 45 * 24 * 60 * 60 * 1000, // 45 days ago
      updatedAt: Date.now() - 15 * 24 * 60 * 60 * 1000 // 15 days ago
    },
    {
      id: 'rule3',
      name: 'Stop Loss Triggered Alert',
      description: 'Alerts when a stop loss order is executed',
      enabled: true,
      severity: 'high',
      category: 'trading',
      condition: 'stop_loss_executed == true',
      parameters: {},
      cooldown: 0, // No cooldown, alert immediately
      createdAt: Date.now() - 60 * 24 * 60 * 60 * 1000, // 60 days ago
      updatedAt: Date.now() - 60 * 24 * 60 * 60 * 1000 // 60 days ago
    },
    {
      id: 'rule4',
      name: 'Strong Signal Consensus Alert',
      description: 'Alerts when multiple agents generate strong signals for same asset',
      enabled: true,
      severity: 'info',
      category: 'trading',
      condition: 'count(signals_above_threshold) >= min_agent_count',
      parameters: {
        confidence_threshold: 0.7,
        min_agent_count: 3
      },
      cooldown: 14400, // 4 hours
      createdAt: Date.now() - 20 * 24 * 60 * 60 * 1000, // 20 days ago
      updatedAt: Date.now() - 5 * 24 * 60 * 60 * 1000 // 5 days ago
    },
    {
      id: 'rule5',
      name: 'Database Performance Alert',
      description: 'Alerts when database query times exceed threshold',
      enabled: true,
      severity: 'high',
      category: 'performance',
      condition: 'avg_query_time > threshold',
      parameters: {
        threshold: 50, // milliseconds
        sample_period: 300 // 5 minutes
      },
      cooldown: 1800, // 30 minutes
      createdAt: Date.now() - 90 * 24 * 60 * 60 * 1000, // 90 days ago
      updatedAt: Date.now() - 25 * 24 * 60 * 60 * 1000 // 25 days ago
    },
    {
      id: 'rule6',
      name: 'Portfolio Drift Alert',
      description: 'Alerts when portfolio allocation drifts beyond threshold',
      enabled: true,
      severity: 'medium',
      category: 'portfolio',
      condition: 'max_drift > threshold',
      parameters: {
        threshold: 5, // percentage
        check_interval: 43200 // 12 hours
      },
      cooldown: 43200, // 12 hours
      createdAt: Date.now() - 75 * 24 * 60 * 60 * 1000, // 75 days ago
      updatedAt: Date.now() - 30 * 24 * 60 * 60 * 1000 // 30 days ago
    },
    {
      id: 'rule7',
      name: 'Security Update Alert',
      description: 'Alerts when critical security updates are available',
      enabled: true,
      severity: 'critical',
      category: 'security',
      condition: 'update_severity == "critical"',
      parameters: {
        check_interval: 86400 // 24 hours
      },
      cooldown: 0, // No cooldown for security alerts
      createdAt: Date.now() - 120 * 24 * 60 * 60 * 1000, // 120 days ago
      updatedAt: Date.now() - 45 * 24 * 60 * 60 * 1000 // 45 days ago
    }
  ],
  preferences: {
    emailNotifications: true,
    pushNotifications: true,
    smsNotifications: false,
    slackNotifications: true,
    criticalAlertsOnly: false,
    soundEnabled: true,
    notificationCooldown: 300, // 5 minutes
    muteStartTime: '22:00',
    muteEndTime: '08:00',
    muteDays: [0, 6] // Weekend
  },
  unreadCount: 3,
  isLoading: false,
  error: null
};

const alertsSlice = createSlice({
  name: 'alerts',
  initialState,
  reducers: {
    fetchAlertsStart: (state) => {
      state.isLoading = true;
      state.error = null;
    },
    fetchAlertsSuccess: (state, action: PayloadAction<Alert[]>) => {
      state.alerts = action.payload;
      state.unreadCount = action.payload.filter(alert => 
        alert.status === 'active').length;
      state.isLoading = false;
    },
    fetchAlertsFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;
    },
    fetchRulesSuccess: (state, action: PayloadAction<AlertRule[]>) => {
      state.rules = action.payload;
    },
    fetchPreferencesSuccess: (state, action: PayloadAction<AlertPreferences>) => {
      state.preferences = action.payload;
    },
    addAlert: (state, action: PayloadAction<Omit<Alert, 'id' | 'timestamp' | 'status'>>) => {
      const newAlert: Alert = {
        id: `alert_${Date.now()}`,
        timestamp: Date.now(),
        status: 'active',
        ...action.payload
      };
      
      state.alerts.unshift(newAlert);
      state.unreadCount += 1;
      
      // Sort alerts by severity and timestamp
      state.alerts.sort((a, b) => {
        const severityOrder = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };
        if (a.severity !== b.severity) {
          return severityOrder[a.severity] - severityOrder[b.severity];
        }
        return b.timestamp - a.timestamp;
      });
    },
    updateAlertStatus: (state, action: PayloadAction<{
      id: string;
      status: AlertStatus;
    }>) => {
      const alert = state.alerts.find(a => a.id === action.payload.id);
      if (alert) {
        const oldStatus = alert.status;
        alert.status = action.payload.status;
        
        // Update timestamps based on status change
        if (action.payload.status === 'acknowledged' && oldStatus !== 'acknowledged') {
          alert.acknowledgedAt = Date.now();
        } else if (action.payload.status === 'resolved' && oldStatus !== 'resolved') {
          alert.resolvedAt = Date.now();
        }
        
        // Update unread count
        if (oldStatus === 'active' && action.payload.status !== 'active') {
          state.unreadCount = Math.max(0, state.unreadCount - 1);
        }
      }
    },
    clearAllAlerts: (state) => {
      state.alerts = [];
      state.unreadCount = 0;
    },
    markAllAsAcknowledged: (state) => {
      const now = Date.now();
      state.alerts.forEach(alert => {
        if (alert.status === 'active') {
          alert.status = 'acknowledged';
          alert.acknowledgedAt = now;
        }
      });
      state.unreadCount = 0;
    },
    addRule: (state, action: PayloadAction<Omit<AlertRule, 'id' | 'createdAt' | 'updatedAt'>>) => {
      const newRule: AlertRule = {
        id: `rule_${Date.now()}`,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        ...action.payload
      };
      
      state.rules.push(newRule);
    },
    updateRule: (state, action: PayloadAction<{
      id: string;
      updates: Partial<AlertRule>;
    }>) => {
      const rule = state.rules.find(r => r.id === action.payload.id);
      if (rule) {
        Object.assign(rule, action.payload.updates);
        rule.updatedAt = Date.now();
      }
    },
    deleteRule: (state, action: PayloadAction<string>) => {
      state.rules = state.rules.filter(r => r.id !== action.payload);
    },
    toggleRuleEnabled: (state, action: PayloadAction<string>) => {
      const rule = state.rules.find(r => r.id === action.payload);
      if (rule) {
        rule.enabled = !rule.enabled;
        rule.updatedAt = Date.now();
      }
    },
    updatePreferences: (state, action: PayloadAction<Partial<AlertPreferences>>) => {
      state.preferences = {
        ...state.preferences,
        ...action.payload
      };
    }
  }
});

export const {
  fetchAlertsStart,
  fetchAlertsSuccess,
  fetchAlertsFailure,
  fetchRulesSuccess,
  fetchPreferencesSuccess,
  addAlert,
  updateAlertStatus,
  clearAllAlerts,
  markAllAsAcknowledged,
  addRule,
  updateRule,
  deleteRule,
  toggleRuleEnabled,
  updatePreferences
} = alertsSlice.actions;

// Selectors
export const selectAllAlerts = (state: RootState) => state.alerts.alerts;
export const selectActiveAlerts = (state: RootState) => 
  state.alerts.alerts.filter(alert => alert.status === 'active');
export const selectAlertById = (id: string) => 
  (state: RootState) => state.alerts.alerts.find(alert => alert.id === id);
export const selectAlertsBySeverity = (severity: AlertSeverity) => 
  (state: RootState) => state.alerts.alerts.filter(alert => alert.severity === severity);
export const selectAlertsByCategory = (category: AlertCategory) => 
  (state: RootState) => state.alerts.alerts.filter(alert => alert.category === category);
export const selectAllRules = (state: RootState) => state.alerts.rules;
export const selectEnabledRules = (state: RootState) => 
  state.alerts.rules.filter(rule => rule.enabled);
export const selectRuleById = (id: string) => 
  (state: RootState) => state.alerts.rules.find(rule => rule.id === id);
export const selectAlertPreferences = (state: RootState) => state.alerts.preferences;
export const selectUnreadCount = (state: RootState) => state.alerts.unreadCount;
export const selectAlertsLoading = (state: RootState) => state.alerts.isLoading;
export const selectAlertsError = (state: RootState) => state.alerts.error;

export default alertsSlice.reducer;