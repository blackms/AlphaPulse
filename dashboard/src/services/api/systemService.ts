import apiClient from './apiClient';

/**
 * Service for system-related API calls
 */
const systemService = {
  /**
   * Get system status
   * @returns Promise with system status data
   */
  getSystemStatus: async (): Promise<any> => {
    try {
      const response = await apiClient.get('/api/v1/system');
      return response.data;
    } catch (error) {
      console.error('Error fetching system status:', error);
      throw error;
    }
  },

  /**
   * Get system logs
   * @param limit - Number of logs to retrieve
   * @param level - Log level filter
   * @returns Promise with system logs
   */
  getSystemLogs: async (limit: number = 50, level?: string): Promise<any> => {
    try {
      // The system endpoint already includes logs
      const response = await apiClient.get('/api/v1/system');
      let logs = response.data.logs || [];
      
      // Apply filters if needed
      if (level) {
        logs = logs.filter((log: any) => log.level === level);
      }
      
      // Apply limit
      logs = logs.slice(0, limit);
      
      return logs;
    } catch (error) {
      console.error('Error fetching system logs:', error);
      throw error;
    }
  },

  /**
   * Get system metrics
   * @returns Promise with system metrics
   */
  getSystemMetrics: async (): Promise<any> => {
    try {
      // First try to get metrics from the metrics endpoint
      try {
        const response = await apiClient.get('/api/v1/metrics/system');
        return response.data;
      } catch (metricsError) {
        // Fallback to the system endpoint which should include metrics
        const response = await apiClient.get('/api/v1/system');
        return response.data.metrics || [];
      }
    } catch (error) {
      console.error('Error fetching system metrics:', error);
      throw error;
    }
  }
};

export default systemService;