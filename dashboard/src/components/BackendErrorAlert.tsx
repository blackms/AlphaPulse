import React, { useState } from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Collapse,
  Snackbar,
  TextField,
  Typography,
  Paper,
} from '@mui/material';
import {
  Code as CodeIcon,
  ContentCopy as CopyIcon,
  Check as CheckIcon,
} from '@mui/icons-material';

interface BackendErrorAlertProps {
  error: string;
  onDismiss?: () => void;
}

/**
 * A specialized error alert that provides detailed information about backend errors
 * and suggests fixes for common issues.
 */
const BackendErrorAlert: React.FC<BackendErrorAlertProps> = ({ error, onDismiss }) => {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);

  // Check if this is the PortfolioService initialization error
  const isPortfolioServiceError = error.includes('PortfolioService.__init__() takes 1 positional argument but 2 were given');

  // Handle copying fix code to clipboard
  const handleCopyFix = () => {
    if (isPortfolioServiceError) {
      const fixCode = `# In src/alpha_pulse/exchange_sync/portfolio_service.py
# Update the PortfolioService class's __init__ method to accept exchange_id

class PortfolioService:
    """
    Service for synchronizing portfolio data from exchanges.
    
    This class handles the synchronization of portfolio data from exchanges,
    storing it in the database, and providing access to the data.
    """
    
    def __init__(self, exchange_id=None):  # Add exchange_id parameter with default None
        """Initialize the portfolio service."""
        self.repository = PortfolioRepository()
        self.exchange_id = exchange_id  # Store the exchange_id
`;
      
      navigator.clipboard.writeText(fixCode);
      setCopied(true);
      setSnackbarOpen(true);
      
      setTimeout(() => {
        setCopied(false);
      }, 2000);
    }
  };

  if (!error) return null;

  return (
    <>
      <Alert 
        severity="warning"
        sx={{ mb: 2 }}
        action={
          <Button 
            color="inherit" 
            size="small" 
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Hide Details' : 'Show Details'}
          </Button>
        }
      >
        <AlertTitle>Backend Error Detected</AlertTitle>
        <Typography variant="body2">
          {error}
        </Typography>
      </Alert>
      
      <Collapse in={expanded}>
        <Paper sx={{ p: 2, mb: 2, bgcolor: '#f8f9fa' }}>
          <Typography variant="h6" gutterBottom>
            <CodeIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Diagnostic Information
          </Typography>
          
          {isPortfolioServiceError ? (
            <>
              <Typography variant="body1" paragraph>
                <strong>Issue:</strong> The PortfolioService class is not accepting the exchange_id parameter in its constructor.
              </Typography>
              
              <Typography variant="body1" paragraph>
                <strong>Location:</strong> src/alpha_pulse/exchange_sync/portfolio_service.py
              </Typography>
              
              <Typography variant="body1" paragraph>
                <strong>Fix:</strong> Update the PortfolioService.__init__() method to accept the exchange_id parameter.
              </Typography>
              
              <Box sx={{ p: 2, bgcolor: '#2b2b2b', color: '#eee', borderRadius: 1, mb: 2, fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
                <code>{`def __init__(self, exchange_id=None):
    """Initialize the portfolio service."""
    self.repository = PortfolioRepository()
    self.exchange_id = exchange_id`}</code>
              </Box>
              
              <Button
                variant="contained"
                color={copied ? "success" : "primary"}
                startIcon={copied ? <CheckIcon /> : <CopyIcon />}
                onClick={handleCopyFix}
                sx={{ mb: 1 }}
              >
                {copied ? "Copied!" : "Copy Fix"}
              </Button>
            </>
          ) : (
            <Typography variant="body1">
              This is an unknown backend error. Please check the server logs for more information.
            </Typography>
          )}
        </Paper>
      </Collapse>
      
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={3000}
        onClose={() => setSnackbarOpen(false)}
        message="Fix code copied to clipboard"
      />
    </>
  );
};

export default BackendErrorAlert;