import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import apiClient from '../apiClient';
import portfolioService from '../portfolioService';

// Create a mock for axios
const mockAxios = new MockAdapter(apiClient);

describe('Portfolio Service', () => {
  // Reset mock between tests
  beforeEach(() => {
    mockAxios.reset();
  });

  describe('getPortfolio', () => {
    it('should transform API data correctly', async () => {
      // Mock a successful response
      mockAxios.onGet('/api/v1/portfolio').reply(200, {
        total_value: 100000,
        cash: 25000,
        positions: [
          {
            symbol: 'BTC-USD',
            quantity: 1.5,
            current_price: 45000,
            value: 67500,
            pnl: 3000,
            pnl_percentage: 6.67,
            entry_price: 43000
          }
        ]
      });

      // Call the service
      const result = await portfolioService.getPortfolio();

      // Check the transformed data
      expect(result.totalValue).toEqual(100000);
      expect(result.cashBalance).toEqual(25000);
      expect(result.assets.length).toEqual(2); // Cash + BTC
      expect(result.assets[0].assetId).toEqual('CASH-USD');
      expect(result.assets[1].assetId).toEqual('BTC-USD');
      expect(result.assets[1].value).toEqual(67500);
    });

    it('should handle API errors gracefully', async () => {
      // Mock a server error
      mockAxios.onGet('/api/v1/portfolio').reply(500, {
        error: 'Internal server error'
      });

      // Call the service - it should not throw
      const result = await portfolioService.getPortfolio();

      // Check the error is captured
      expect(result.error).toBeTruthy();
      expect(result.totalValue).toEqual(0);
      expect(result.assets.length).toEqual(0);
    });

    it('should handle network errors gracefully', async () => {
      // Mock a network error
      mockAxios.onGet('/api/v1/portfolio').networkError();

      // Call the service - it should not throw
      const result = await portfolioService.getPortfolio();

      // Check the error is captured
      expect(result.error).toBeTruthy();
      expect(result.totalValue).toEqual(0);
      expect(result.assets.length).toEqual(0);
    });

    it('should handle the specific PortfolioService init error', async () => {
      // Mock the specific error that causes issues in backend
      mockAxios.onGet('/api/v1/portfolio').reply(500, {
        error: "PortfolioService.__init__() takes 1 positional argument but 2 were given"
      });

      // Call the service - it should not throw
      const result = await portfolioService.getPortfolio();

      // Check the error is captured but we still have a usable structure
      expect(result.error).toContain('PortfolioService.__init__()');
      expect(result.totalValue).toEqual(0);
    });

    it('should retry failed requests up to MAX_RETRIES times', async () => {
      // First two calls fail, third succeeds
      mockAxios.onGet('/api/v1/portfolio')
        .replyOnce(503) // First attempt fails
        .replyOnce(503) // Second attempt fails
        .replyOnce(200, { // Third attempt succeeds
          total_value: 100000,
          cash: 25000,
          positions: []
        });

      // Call the service
      const result = await portfolioService.getPortfolio();

      // Check the successful result after retries
      expect(result.totalValue).toEqual(100000);
      expect(result.error).toBeFalsy();
    });
  });

  describe('getPortfolioHistory', () => {
    it('should transform historical data correctly', async () => {
      // Mock a successful response
      mockAxios.onGet('/api/v1/portfolio').reply(200, {
        history: [
          {
            timestamp: '2023-01-01T00:00:00Z',
            total_value: 90000
          },
          {
            timestamp: '2023-01-02T00:00:00Z',
            total_value: 92000
          }
        ]
      });

      // Call the service
      const result = await portfolioService.getPortfolioHistory('week');

      // Check the transformed data
      expect(result.length).toEqual(2);
      expect(result[0].value).toEqual(90000);
      expect(result[1].value).toEqual(92000);
    });

    it('should return empty array on error', async () => {
      // Mock an error response
      mockAxios.onGet('/api/v1/portfolio').reply(500);

      // Call the service
      const result = await portfolioService.getPortfolioHistory('week');

      // Check empty array is returned
      expect(result).toEqual([]);
    });
  });

  describe('getAssetDetails', () => {
    it('should return correct asset details', async () => {
      // Mock portfolio response with assets
      mockAxios.onGet('/api/v1/portfolio').reply(200, {
        total_value: 100000,
        cash: 25000,
        positions: [
          {
            symbol: 'BTC-USD',
            quantity: 1.5,
            current_price: 45000,
            value: 67500,
            pnl: 3000,
            pnl_percentage: 6.67,
            entry_price: 43000
          },
          {
            symbol: 'ETH-USD',
            quantity: 10,
            current_price: 2500,
            value: 25000,
            pnl: 5000,
            pnl_percentage: 25,
            entry_price: 2000
          }
        ]
      });

      // Call the service for a specific asset
      const result = await portfolioService.getAssetDetails('ETH-USD');

      // Check the asset details
      expect(result).toBeTruthy();
      expect(result?.assetId).toEqual('ETH-USD');
      expect(result?.quantity).toEqual(10);
      expect(result?.value).toEqual(25000);
    });

    it('should return null for non-existent assets', async () => {
      // Mock portfolio response with assets
      mockAxios.onGet('/api/v1/portfolio').reply(200, {
        total_value: 100000,
        cash: 25000,
        positions: [
          {
            symbol: 'BTC-USD',
            quantity: 1.5,
            current_price: 45000,
            value: 67500,
            pnl: 3000,
            pnl_percentage: 6.67,
            entry_price: 43000
          }
        ]
      });

      // Call the service for a non-existent asset
      const result = await portfolioService.getAssetDetails('SOL-USD');

      // Check that null is returned
      expect(result).toBeNull();
    });

    it('should propagate errors for asset details', async () => {
      // Mock an error response
      mockAxios.onGet('/api/v1/portfolio').reply(500);

      // Call the service and expect an error
      await expect(portfolioService.getAssetDetails('BTC-USD')).rejects.toThrow();
    });
  });
});