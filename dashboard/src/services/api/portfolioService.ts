import apiClient from './apiClient';
import { PortfolioData, Asset, PerformancePeriod, HistoricalValue } from '../../types/portfolio';
import { AxiosError, AxiosResponse } from 'axios';

/**
 * Portfolio service for accessing portfolio data from the API
 * with proper error handling and data transformations
 */

// Configuration
const MAX_RETRIES = 3;
const RETRY_DELAY_BASE = 1000; // 1 second base delay (will be multiplied by 2^attempt for exponential backoff)

// Type definitions
interface ApiErrorResponse {
  error?: string;
  message?: string;
  status?: number;
}

// Transform API data to frontend format
const transformApiData = (apiData: any): PortfolioData => {
  console.log('Transforming API data to frontend format');
  
  // Calculate total values with safe defaults if data is incomplete
  const totalValue = apiData.total_value || 0;
  const cashBalance = apiData.cash || 0;

  // No longer creating a synthetic cash asset

  const positions = Array.isArray(apiData.positions) ? apiData.positions : [];
  
  // Transform API data to match frontend expected format
  const transformedData: PortfolioData = {
    totalValue: totalValue,
    cashBalance: cashBalance,
    assets: [
      ...positions.map((position: any) => {
        // Safely extract values with fallbacks
        const symbol = position.symbol || 'UNKNOWN';
        const quantity = position.quantity || 0;
        const price = position.current_price || 0;
        const value = position.value || (quantity * price);
        
        return {
          assetId: symbol,
          symbol: symbol.split('-')[0], // Remove the -USD suffix
          name: symbol.split('-')[0], // Use symbol as name
          type: 'crypto',
          quantity: quantity,
          price: price,
          value: value,
          allocation: totalValue > 0 ? (value / totalValue) * 100 : 0,
          unrealizedPnL: position.pnl || 0,
          unrealizedPnLPercent: position.pnl_percentage || 0,
          costBasis: position.entry_price || 0,
          lastUpdated: new Date().toISOString(),
          change24h: position.pnl || 0,
          change24hPercent: position.pnl_percentage || 0,
        };
      })
    ],
    performance: [],
    historicalValues: [],
    lastUpdated: new Date().toISOString(),
    assetAllocation: {
      crypto: totalValue > 0 ? (totalValue - cashBalance) / totalValue * 100 : 0,
      fiat: totalValue > 0 ? cashBalance / totalValue * 100 : 0,
    },
    error: apiData.error,
  };
  
  return transformedData;
};

/**
 * Portfolio service for API calls with error handling
 * Updated to ensure compatibility with backend API expectations
 */
const portfolioService = {
  /**
   * Get portfolio data with retry logic and fallbacks
   * IMPORTANT: This call avoids sending any exchange_id parameter
   * to match the PortfolioService.__init__() requirement
   * @returns Promise with portfolio data
   */
  getPortfolio: async (retryCount = 0): Promise<PortfolioData> => {
    const fetchWithRetry = async (attempt = 0): Promise<PortfolioData> => {
      try {
        console.log(`Fetching portfolio data (attempt ${attempt + 1}/${MAX_RETRIES + 1})...`);
        
        // Make API call with ONLY the include_history parameter - NO exchange_id!
        // This matches what the backend PortfolioService constructor expects
        const response = await apiClient.get('/api/v1/portfolio', {
          params: {
            include_history: false
          }
        });
        
        const apiData = response.data;
        console.log('Portfolio data received');

        if (apiData.error) {
          console.warn('API returned error structure:', apiData.error);

          // If we have at least some data, we can transform it despite the error
          if (apiData.positions || apiData.total_value) {
            console.log('Returning partial data despite error');
            return transformApiData(apiData);
          } 
        }
        
        return transformApiData(apiData);
      } catch (error) {
        const axiosError = error as AxiosError;
        const errorData = axiosError.response?.data as ApiErrorResponse;
        
        // Log detailed error information for debugging
        console.error('Portfolio API error:', {
          status: axiosError.response?.status,
          statusText: axiosError.response?.statusText,
          errorMessage: errorData?.error || errorData?.message || axiosError.message,
          url: axiosError.config?.url
        });

        // Try to retry if we haven't hit the maximum retries
        if (attempt < MAX_RETRIES) {
          const delay = RETRY_DELAY_BASE * Math.pow(2, attempt);
          console.log(`Retrying in ${delay}ms (${attempt + 1}/${MAX_RETRIES})...`);
          await new Promise(resolve => setTimeout(resolve, delay));
          return fetchWithRetry(attempt + 1);
        }
        
        // Return empty data structure with error information
        return {
          totalValue: 0,
          cashBalance: 0,
          assets: [],
          performance: [],
          historicalValues: [],
          lastUpdated: new Date().toISOString(),
          assetAllocation: { crypto: 0, fiat: 0 },
          error: errorData?.error || axiosError.message || 'Unknown error fetching portfolio data'
        };
      }
    };
    
    return fetchWithRetry();
  },

  /**
   * Get portfolio history
   * IMPORTANT: This call avoids sending any exchange_id parameter
   * to match the PortfolioService.__init__() requirement
   * @param period - Time period for history (day, week, month, year, all)
   * @returns Promise with portfolio history data
   */
  getPortfolioHistory: async (period: string): Promise<HistoricalValue[]> => {
    try {
      // Make API call with ONLY the include_history and period parameters - NO exchange_id!
      const response = await apiClient.get('/api/v1/portfolio', {
        params: {
          include_history: true,
          period: period
        }
      });

      // Return just the historical data if available
      if (response.data.history && Array.isArray(response.data.history)) {
        return response.data.history.map((entry: any) => ({
          timestamp: new Date(entry.timestamp).getTime(),
          value: entry.total_value || 0,
          change: 0,
          changePercent: 0
        }));
      }
      
      // Return empty data if history not available
      return [];
    } catch (error) {
      console.error('Error fetching portfolio history:', error);
      return [];
    }
  },

  /**
   * Get asset details
   * @param assetId - ID of the asset
   * @returns Promise with asset details
   */
  getAssetDetails: async (assetId: string): Promise<Asset | null> => {
    try {
      // Get the full portfolio and filter for the specific asset
      // This is the most efficient approach as the API doesn't support 
      // direct fetching of a single asset
      const portfolioData = await portfolioService.getPortfolio();
      return portfolioData.assets.find(a => a.assetId === assetId) || null;
    } catch (error) {
      console.error('Error fetching asset details:', error);
      throw error;
    }
  },
  
  /**
   * Reload exchange data (will NOT pass exchange_id to match backend expectations)
   * @returns Promise with the reload result
   */
  reloadData: async (): Promise<any> => {
    try {
      // Call the reload endpoint WITHOUT any exchange_id parameter
      const response = await apiClient.post('/api/v1/portfolio/reload');
      return response.data;
    } catch (error) {
      console.error('Error reloading portfolio data:', error);
      throw error;
    }
  }
};

export default portfolioService;