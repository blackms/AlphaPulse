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
      return response.data;
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