import apiClient from './apiClient';
import { PortfolioData } from '../../types/portfolio';

/**
 * Service for portfolio-related API calls
 */
const portfolioService = {
  /**
   * Get portfolio data
   * @returns Promise with portfolio data
   */
  getPortfolio: async (): Promise<PortfolioData> => {
    try {
      const response = await apiClient.get('/api/v1/portfolio');
      const apiData = response.data;
      
      // Transform API data to match frontend expected format
      const transformedData: PortfolioData = {
        totalValue: apiData.total_value || 0,
        cashBalance: apiData.cash || 0,
        assets: (apiData.positions || []).map((position: any) => ({
          assetId: position.symbol,
          symbol: position.symbol,
          name: position.symbol,
          type: 'crypto',
          quantity: position.quantity || 0,
          price: position.current_price || 0,
          value: position.value || 0,
          allocation: position.value ? (position.value / apiData.total_value) * 100 : 0,
          unrealizedPnL: position.pnl || 0,
          unrealizedPnLPercent: position.pnl_percentage || 0,
          costBasis: position.entry_price || 0,
          lastUpdated: new Date().toISOString(),
          change24h: 0, // Not provided by API
          change24hPercent: 0, // Not provided by API
        })),
        performance: [
          {
            period: 'day',
            returnValue: 0, // Not provided by API
            returnPercent: 0, // Not provided by API
            startDate: new Date(Date.now() - 86400000).toISOString(),
            endDate: new Date().toISOString(),
          },
          {
            period: 'month',
            returnValue: 0, // Not provided by API
            returnPercent: 0, // Not provided by API
            startDate: new Date(Date.now() - 30 * 86400000).toISOString(),
            endDate: new Date().toISOString(),
          },
          {
            period: 'year',
            returnValue: 0, // Not provided by API
            returnPercent: apiData.metrics?.return_since_inception * 100 || 0,
            startDate: new Date(Date.now() - 365 * 86400000).toISOString(),
            endDate: new Date().toISOString(),
          },
        ],
        historicalValues: [], // Not provided by API
        lastUpdated: new Date().toISOString(),
        assetAllocation: {}, // Not provided by API
      };
      
      return transformedData;
    } catch (error) {
      console.error('Error fetching portfolio data:', error);
      throw error;
    }
  },

  /**
   * Get portfolio history
   * @param period - Time period for history (day, week, month, year, all)
   * @returns Promise with portfolio history data
   */
  getPortfolioHistory: async (period: string): Promise<any> => {
    try {
      const response = await apiClient.get(`/api/v1/portfolio?include_history=true`);
      // Extract the historical data for the requested period
      return response.data.history || [];
    } catch (error) {
      console.error('Error fetching portfolio history:', error);
      throw error;
    }
  },

  /**
   * Get asset details
   * @param assetId - ID of the asset
   * @returns Promise with asset details
   */
  getAssetDetails: async (assetId: string): Promise<any> => {
    try {
      // Get the full portfolio and filter for the specific asset
      const response = await apiClient.get('/api/v1/portfolio');
      const asset = response.data.assets.find((a: any) => a.assetId === assetId);
      return asset || null;
    } catch (error) {
      console.error('Error fetching asset details:', error);
      throw error;
    }
  }
};

export default portfolioService;