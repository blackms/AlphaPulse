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
      const apiData = response.data;
      
      // Transform API data to match frontend expected format
      const transformedData = {
        status: 'operational', // Default to operational
        components: [
          {
            id: 'cpu',
            name: 'CPU',
            type: 'hardware',
            status: apiData.cpu?.usage_percent < 80 ? 'healthy' : 'warning',
            healthScore: 100 - (apiData.cpu?.usage_percent || 0),
            lastUpdated: new Date().toISOString(),
            description: `${apiData.cpu?.cores || 0} cores, ${apiData.cpu?.usage_percent || 0}% usage`
          },
          {
            id: 'memory',
            name: 'Memory',
            type: 'hardware',
            status: apiData.memory?.percent < 80 ? 'healthy' : 'warning',
            healthScore: 100 - (apiData.memory?.percent || 0),
            lastUpdated: new Date().toISOString(),
            description: `${apiData.memory?.used_mb || 0}MB / ${apiData.memory?.total_mb || 0}MB (${apiData.memory?.percent || 0}%)`
          },
          {
            id: 'disk',
            name: 'Disk',
            type: 'hardware',
            status: apiData.disk?.percent < 80 ? 'healthy' : 'warning',
            healthScore: 100 - (apiData.disk?.percent || 0),
            lastUpdated: new Date().toISOString(),
            description: `${apiData.disk?.used_gb || 0}GB / ${apiData.disk?.total_gb || 0}GB (${apiData.disk?.percent || 0}%)`
          },
          {
            id: 'api',
            name: 'API Server',
            type: 'service',
            status: 'healthy',
            healthScore: 100,
            lastUpdated: new Date().toISOString(),
            description: `PID: ${apiData.process?.pid || 0}, Uptime: ${apiData.process?.uptime_seconds || 0}s`
          }
        ],
        logs: [], // No logs in API response
        metrics: [
          {
            id: 'cpu_usage',
            name: 'CPU Usage',
            value: apiData.cpu?.usage_percent || 0,
            unit: '%',
            timestamp: new Date().toISOString(),
            target: 80,
            status: apiData.cpu?.usage_percent < 50 ? 'good' : apiData.cpu?.usage_percent < 80 ? 'warning' : 'error'
          },
          {
            id: 'memory_usage',
            name: 'Memory Usage',
            value: apiData.memory?.percent || 0,
            unit: '%',
            timestamp: new Date().toISOString(),
            target: 80,
            status: apiData.memory?.percent < 50 ? 'good' : apiData.memory?.percent < 80 ? 'warning' : 'error'
          },
          {
            id: 'disk_usage',
            name: 'Disk Usage',
            value: apiData.disk?.percent || 0,
            unit: '%',
            timestamp: new Date().toISOString(),
            target: 80,
            status: apiData.disk?.percent < 50 ? 'good' : apiData.disk?.percent < 80 ? 'warning' : 'error'
          }
        ],
        lastUpdated: new Date().toISOString(),
        uptime: apiData.process?.uptime_seconds || 0
      };
      
      return transformedData;
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