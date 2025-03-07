import { Middleware } from 'redux';
import portfolioService from '../../services/api/portfolioService';
import systemService from '../../services/api/systemService';
import alertsService from '../../services/api/alertsService';
import {
  fetchPortfolioStart,
  fetchPortfolioSuccess,
  fetchPortfolioFailure
} from '../slices/portfolioSlice';
import {
  fetchSystemStart,
  fetchSystemStatusSuccess as fetchSystemSuccess,
  fetchSystemStatusFailure as fetchSystemFailure
} from '../slices/systemSlice';
import {
  fetchAlertsStart,
  fetchAlertsSuccess,
  fetchAlertsFailure,
  fetchRulesSuccess,
  fetchPreferencesSuccess,
  updateAlertStatus,
  addRule,
  updateRule,
  deleteRule,
  updateNotificationSettings
} from '../slices/alertsSlice';

/**
 * Middleware to handle API calls
 */
const apiMiddleware: Middleware = ({ dispatch }) => (next) => async (action) => {
  next(action);

  // Handle portfolio data fetching
  if (action.type === fetchPortfolioStart.type) {
    try {
      const portfolioData = await portfolioService.getPortfolio();
      dispatch(fetchPortfolioSuccess(portfolioData));
    } catch (error) {
      dispatch(fetchPortfolioFailure(error instanceof Error ? error.message : 'Unknown error'));
    }
  }

  // Handle system status fetching
  if (action.type === fetchSystemStart.type) {
    try {
      const systemData = await systemService.getSystemStatus();
      dispatch(fetchSystemSuccess({
        status: systemData.status,
        components: systemData.components,
        lastUpdated: systemData.lastUpdated,
        uptime: systemData.uptime || 0
      }));
    } catch (error) {
      dispatch(fetchSystemFailure(error instanceof Error ? error.message : 'Unknown error'));
    }
  }

  // Handle alerts fetching
  if (action.type === fetchAlertsStart.type) {
    try {
      console.log('Fetching alerts data in middleware...');
      const [alertsData, preferencesData] = await Promise.all([
        alertsService.getAlerts(),
        alertsService.getNotificationPreferences()
      ]);
      
      console.log('Alerts data received:', alertsData);
      console.log('Preferences data received:', preferencesData);
      
      // Dispatch each part separately
      dispatch(fetchAlertsSuccess(alertsData.alerts || []));
      dispatch(fetchRulesSuccess(alertsData.rules || []));
      dispatch(fetchPreferencesSuccess(preferencesData));
    } catch (error) {
      console.error('Error in alerts middleware:', error);
      dispatch(fetchAlertsFailure(error instanceof Error ? error.message : 'Unknown error'));
    }
  }

  // Handle alert status updates
  if (action.type === updateAlertStatus.type) {
    try {
      const { id, acknowledged } = action.payload;
      await alertsService.updateAlertStatus(id, { acknowledged });
      // No need to dispatch success as the state is already updated optimistically
    } catch (error) {
      console.error('Error updating alert status:', error);
      // Could dispatch a failure action here if needed
    }
  }

  // Handle alert rule operations
  if (action.type === addRule.type) {
    try {
      await alertsService.createAlertRule(action.payload);
      // State already updated optimistically
    } catch (error) {
      console.error('Error creating alert rule:', error);
    }
  }

  if (action.type === updateRule.type) {
    try {
      await alertsService.updateAlertRule(action.payload.id, action.payload);
      // State already updated optimistically
    } catch (error) {
      console.error('Error updating alert rule:', error);
    }
  }

  if (action.type === deleteRule.type) {
    try {
      await alertsService.deleteAlertRule(action.payload);
      // State already updated optimistically
    } catch (error) {
      console.error('Error deleting alert rule:', error);
    }
  }

  // Handle notification preferences updates
  if (action.type === updateNotificationSettings.type) {
    try {
      await alertsService.updateNotificationPreferences(action.payload);
      // State already updated optimistically
    } catch (error) {
      console.error('Error updating notification preferences:', error);
    }
  }
};

export default apiMiddleware;