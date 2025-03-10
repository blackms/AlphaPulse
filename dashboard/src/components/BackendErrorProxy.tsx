import React from 'react';
import { 
  Box, 
  Alert, 
  AlertTitle, 
  Button, 
  Typography, 
  Collapse, 
  IconButton,
  CircularProgress
} from '@mui/material';
import { 
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  ErrorOutline as ErrorIcon
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { selectPortfolioError, fetchPortfolioStart } from '../store/slices/portfolioSlice';

interface BackendErrorProxyProps {
  endpoint: string;
  children: React.ReactNode;
}

/**
 * BackendErrorProxy - A component that handles API errors gracefully
 * 
 * This component displays appropriate error messages and recovery options
 * when backend API calls fail, while still allowing the UI to function.
 */
const BackendErrorProxy: React.FC<BackendErrorProxyProps> = ({ endpoint, children }) => {
  const [loading, setLoading] = React.useState(false);
  const [detailsOpen, setDetailsOpen] = React.useState(false);
  const backendError = useSelector(selectPortfolioError);
  const dispatch = useDispatch();

  // Handle retry logic
  const handleRetry = () => {
    setLoading(true);
    
    // Dispatch appropriate action based on endpoint
    if (endpoint === 'portfolio') {
      dispatch(fetchPortfolioStart());
    }
    
    // Reset loading after a delay to give API time to respond
    setTimeout(() => {
      setLoading(false);
    }, 2000);
  };
  
  // If there's no error, just render the children
  if (!backendError) {
    return <>{children}</>;
  }
  
  // Show a spinner while loading
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4, flexDirection: 'column', alignItems: 'center' }}>
        <CircularProgress size={40} sx={{ mb: 2 }} />
        <Typography variant="body2">Reconnecting to backend services...</Typography>
      </Box>
    );
  }
  
  // Create a user-friendly error message
  const getUserFriendlyError = (error: string) => {
    // Check for specific errors and provide friendly messages
    if (error.includes('PortfolioService.__init__()')) {
      return 'The portfolio service is temporarily unavailable due to a parameter mismatch.';
    }
    
    if (error.includes('timeout') || error.includes('network')) {
      return 'There was a network issue connecting to the server.';
    }
    
    if (error.includes('401') || error.includes('403')) {
      return 'Your session may have expired. Please try refreshing the page.';
    }
    
    // Default message for unknown errors
    return 'There was an issue retrieving data from the server.';
  };

  // Determine the error severity
  const getErrorSeverity = (error: string) => {
    if (error.includes('404') || error.includes('not found')) {
      return 'info';
    }
    if (error.includes('401') || error.includes('403')) {
      return 'warning';
    }
    return 'error';
  };
  
  // Return an error UI that allows graceful degradation
  return (
    <>
      <Alert 
        severity={getErrorSeverity(backendError)}
        sx={{ mb: 2 }}
        action={
          <Button 
            color="inherit" 
            size="small" 
            startIcon={<RefreshIcon />}
            onClick={handleRetry}
          >
            Retry
          </Button>
        }
      >
        <AlertTitle>Connection Issue</AlertTitle>
        {getUserFriendlyError(backendError)}
        <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Technical details
          </Typography>
          <IconButton 
            size="small" 
            onClick={() => setDetailsOpen(!detailsOpen)}
            sx={{ ml: 1 }}
          >
            {detailsOpen ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
          </IconButton>
        </Box>
        <Collapse in={detailsOpen}>
          <Box 
            sx={{ 
              mt: 1, 
              p: 1, 
              bgcolor: 'background.paper', 
              borderRadius: 1,
              fontSize: '0.75rem',
              fontFamily: 'monospace'
            }}
          >
            {backendError}
          </Box>
        </Collapse>
      </Alert>
      
      {/* Render children components to allow partial functionality */}
      {children}
    </>
  );
};

export default BackendErrorProxy;