import React from 'react';
import { Box, Typography, Button, Paper, Alert } from '@mui/material';
import { Refresh as RefreshIcon, BugReport as BugIcon } from '@mui/icons-material';

interface ErrorFallbackProps {
  error: string;
  resetErrorBoundary?: () => void;
  retry?: () => void;
  showDetails?: boolean;
}

/**
 * A reusable error fallback component that displays a friendly error message
 * with options to retry or view more details.
 */
const ErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  resetErrorBoundary,
  retry,
  showDetails = false,
}) => {
  const [showErrorDetails, setShowErrorDetails] = React.useState(showDetails);

  return (
    <Paper
      elevation={2}
      sx={{
        p: 4,
        m: 2,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
      }}
    >
      <BugIcon color="error" sx={{ fontSize: 60, mb: 2 }} />
      
      <Typography variant="h5" gutterBottom>
        Something went wrong
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3, maxWidth: '80%' }}>
        We encountered an issue while loading your data. This could be due to a temporary 
        connection problem or a backend service issue.
      </Typography>

      {showErrorDetails && (
        <Alert severity="error" sx={{ mb: 3, width: '100%', maxWidth: 600 }}>
          <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {error}
          </Typography>
        </Alert>
      )}

      <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
        {retry && (
          <Button
            variant="contained"
            color="primary"
            startIcon={<RefreshIcon />}
            onClick={() => {
              retry();
            }}
          >
            Retry
          </Button>
        )}
        
        {resetErrorBoundary && (
          <Button 
            variant="outlined" 
            onClick={resetErrorBoundary}
          >
            Reset
          </Button>
        )}

        <Button
          variant="text"
          onClick={() => setShowErrorDetails(!showErrorDetails)}
        >
          {showErrorDetails ? 'Hide Details' : 'Show Details'}
        </Button>
      </Box>
    </Paper>
  );
};

export default ErrorFallback;