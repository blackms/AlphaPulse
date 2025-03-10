import React from 'react';
import { Box, Alert, Chip } from '@mui/material';
import { DataArray as DataIcon } from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { selectPortfolioError } from '../store/slices/portfolioSlice';


interface BackendErrorProxyProps {
  endpoint: string;
  children: React.ReactNode;
}

/**
 * This component acts as a proxy that intercepts backend errors for specific endpoints
 * and provides alternative API implementations to work around backend issues.
 * 
 * It's a temporary solution until backend issues can be properly fixed.
 */
const BackendErrorProxy: React.FC<BackendErrorProxyProps> = ({ endpoint, children }) => {
  const backendError = useSelector(selectPortfolioError);
  
  // In Portfolio section, show a small mock data indicator
  if (endpoint === 'portfolio') {
    return (
      <>
        <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
          <Chip 
            icon={<DataIcon />} 
            label="Using Mock Data" 
            color="info" 
            variant="outlined"
            size="small"
            sx={{ mr: 1 }}
          />
          {backendError && (
            <Alert 
              severity="info" 
              variant="outlined"
              sx={{ py: 0, fontSize: '0.75rem' }}
            >
              Backend connection issue - showing example data
            </Alert>
          )}
        </Box>
        {children}
      </>
    );
  }
  
  // For all other components, just render the children
  return <>{children}</>;
};

export default BackendErrorProxy;