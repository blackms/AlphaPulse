import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Paper, CircularProgress, Alert } from '@mui/material';
import portfolioService from '../services/api/portfolioService';
import { PortfolioData } from '../types/portfolio';

/**
 * Component to test the portfolio API connection
 * This component makes calls to the portfolio API and displays the results
 * to verify that the backend fix for PortfolioService.__init__() is working
 */
const PortfolioApiTester: React.FC = () => {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const fetchPortfolio = async (refresh: boolean = false) => {
    setLoading(true);
    setError(null);
    setSuccess(false);
    
    try {
      // Use the portfolioService to fetch data
      let data;
      if (refresh) {
        // Call reload endpoint first, then get the updated portfolio data
        await portfolioService.reloadData();
        data = await portfolioService.getPortfolio();
      } else {
        data = await portfolioService.getPortfolio();
      }
      
      if (data.error) {
        setError(data.error);
      } else {
        setPortfolioData(data);
        setSuccess(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch on component mount
  useEffect(() => {
    fetchPortfolio();
  }, []);

  return (
    <Paper sx={{ p: 3, my: 2 }}>
      <Typography variant="h5" gutterBottom>
        Portfolio API Connection Test
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={() => fetchPortfolio()}
          disabled={loading}
        >
          Fetch Portfolio
        </Button>
        
        <Button 
          variant="outlined" 
          color="secondary" 
          onClick={() => fetchPortfolio(true)}
          disabled={loading}
        >
          Refresh from Exchange
        </Button>
      </Box>
      
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
          <CircularProgress />
        </Box>
      )}
      
      {error && (
        <Alert severity="error" sx={{ my: 2 }}>
          Error: {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" sx={{ my: 2 }}>
          API call successful!
        </Alert>
      )}
      
      {portfolioData && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6">Portfolio Data:</Typography>
          <Box component="pre" sx={{ 
            backgroundColor: '#f5f5f5', 
            p: 2, 
            borderRadius: 1,
            overflow: 'auto',
            maxHeight: '400px'
          }}>
            {JSON.stringify(portfolioData, null, 2)}
          </Box>
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1">
              Total Value: ${portfolioData.totalValue?.toLocaleString() || 'N/A'}
            </Typography>
            <Typography variant="subtitle1">
              Cash: ${portfolioData.cashBalance?.toLocaleString() || 'N/A'}
            </Typography>
            <Typography variant="subtitle1">
              Assets: {portfolioData.assets?.length || 0}
            </Typography>
          </Box>
        </Box> 
      )}
    </Paper>
  );
};

export default PortfolioApiTester;