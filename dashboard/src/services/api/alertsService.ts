import apiClient from './apiClient';

/**
 * Service for alerts-related API calls
 */
const alertsService = {
  /**
   * Get all alerts
   * @returns Promise with alerts data
   */
  getAlerts: async (): Promise<any> => {
    try {
      const response = await apiClient.get('/api/v1/alerts');
      return response.data;
    } catch (error) {
      console.error('Error fetching alerts:', error);
      throw error;
    }
  },

  /**
   * Get alert rules
   * @returns Promise with alert rules
   */
  getAlertRules: async (): Promise<any> => {
    try {
      // The alerts endpoint might include rules
      const response = await apiClient.get('/api/v1/alerts');
      return response.data.rules || [];
    } catch (error) {
      console.error('Error fetching alert rules:', error);
      throw error;
    }
  },

  /**
   * Create a new alert rule
   * @param rule - Alert rule data
   * @returns Promise with created rule
   */
  createAlertRule: async (rule: any): Promise<any> => {
    try {
      // This might not be supported by the API, but we'll implement it for completeness
      const response = await apiClient.post('/api/v1/alerts/rules', rule);
      return response.data;
    } catch (error) {
      console.error('Error creating alert rule:', error);
      throw error;
    }
  },

  /**
   * Update an existing alert rule
   * @param ruleId - ID of the rule to update
   * @param rule - Updated rule data
   * @returns Promise with updated rule
   */
  updateAlertRule: async (ruleId: string, rule: any): Promise<any> => {
    try {
      // This might not be supported by the API, but we'll implement it for completeness
      const response = await apiClient.put(`/api/v1/alerts/rules/${ruleId}`, rule);
      return response.data;
    } catch (error) {
      console.error('Error updating alert rule:', error);
      throw error;
    }
  },

  /**
   * Delete an alert rule
   * @param ruleId - ID of the rule to delete
   * @returns Promise with deletion status
   */
  deleteAlertRule: async (ruleId: string): Promise<any> => {
    try {
      // This might not be supported by the API, but we'll implement it for completeness
      const response = await apiClient.delete(`/api/v1/alerts/rules/${ruleId}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting alert rule:', error);
      throw error;
    }
  },

  /**
   * Update alert status (e.g., mark as read)
   * @param alertId - ID of the alert to update
   * @param status - New status data
   * @returns Promise with updated alert
   */
  updateAlertStatus: async (alertId: string, status: any): Promise<any> => {
    try {
      // Use the acknowledge endpoint if we're marking as read
      if (status.acknowledged) {
        const response = await apiClient.post(`/api/v1/alerts/${alertId}/acknowledge`);
        return response.data;
      }
      // This might not be supported by the API
      console.warn('API might not support un-acknowledging alerts');
      return { success: false, message: 'Operation not supported' };
    } catch (error) {
      console.error('Error updating alert status:', error);
      throw error;
    }
  },

  /**
   * Get notification preferences
   * @returns Promise with notification preferences
   */
  getNotificationPreferences: async (): Promise<any> => {
    try {
      // This might not be directly available, so we'll mock it
      // In a real implementation, this would come from a user settings endpoint
      return {
        channels: {
          email: true,
          sms: false,
          push: true,
          slack: false
        },
        preferences: {
          includeCritical: true,
          includeHigh: true,
          includeMedium: true,
          includeLow: false,
          muteStartTime: "22:00",
          muteEndTime: "08:00"
        },
        emailNotifications: true,
        pushNotifications: true,
        soundEnabled: true,
        rules: []
      };
    } catch (error) {
      console.error('Error fetching notification preferences:', error);
      throw error;
    }
  },

  /**
   * Update notification preferences
   * @param preferences - Updated preferences
   * @returns Promise with updated preferences
   */
  updateNotificationPreferences: async (preferences: any): Promise<any> => {
    try {
      // This might not be supported by the API, but we'll implement it for completeness
      console.log('Updating notification preferences:', preferences);
      // In a real implementation, this would call a user settings endpoint
      return { ...preferences, success: true };
    } catch (error) {
      console.error('Error updating notification preferences:', error);
      throw error;
    }
  }
};

export default alertsService;