import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Divider,
  Button,
  TextField,
  Card,
  CardContent,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Code as CodeIcon,
  BugReport as BugIcon,
  ContentCopy as CopyIcon,
  Check as CheckIcon,
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { selectPortfolioError } from '../../store/slices/portfolioSlice';
import BackendErrorAlert from '../../components/BackendErrorAlert';

/**
 * Diagnostic page for backend developers
 * This page provides detailed information about backend errors
 * and suggestions for fixes
 */
const DiagnosticPage: React.FC = () => {
  const portfolioError = useSelector(selectPortfolioError);
  const [copied, setCopied] = useState(false);

  const isPortfolioServiceError = portfolioError && 
    portfolioError.includes('PortfolioService.__init__() takes 1 positional argument but 2 were given');

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
      
      setTimeout(() => {
        setCopied(false);
      }, 2000);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        <BugIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
        System Diagnostics
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        This page provides diagnostic information about the system and backend services.
        It's intended for developers and administrators to help identify and fix issues.
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Errors
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              {portfolioError ? (
                <BackendErrorAlert error={portfolioError} />
              ) : (
                <Alert severity="success">
                  No portfolio errors detected
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {isPortfolioServiceError && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <CodeIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Fix for PortfolioService.__init__ Error
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <Typography variant="body1" paragraph>
                  The error occurs because the PortfolioService class doesn't accept an exchange_id parameter in its constructor, 
                  but the portfolio.py code is trying to pass one.
                </Typography>
                
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Problem in portfolio_service.py</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Paper sx={{ p: 2, bgcolor: '#f5f5f5' }}>
                      <Typography variant="body2" component="pre" sx={{ fontFamily: 'monospace' }}>
                        {`class PortfolioService:
    def __init__(self):  # Missing exchange_id parameter
        self.repository = PortfolioRepository()`}
                      </Typography>
                    </Paper>
                  </AccordionDetails>
                </Accordion>
                
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Problem in portfolio.py</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Paper sx={{ p: 2, bgcolor: '#f5f5f5' }}>
                      <Typography variant="body2" component="pre" sx={{ fontFamily: 'monospace' }}>
                        {`if self._exchange_sync_service is None:
    self._exchange_sync_service = PortfolioService(self._exchange_id)  # Passing exchange_id
    await self._exchange_sync_service.initialize()`}
                      </Typography>
                    </Paper>
                  </AccordionDetails>
                </Accordion>
                
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Recommended Fix
                  </Typography>
                  
                  <Paper sx={{ p: 2, bgcolor: '#2b2b2b', color: '#eee', fontFamily: 'monospace', mb: 2 }}>
                    <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                      {`# In src/alpha_pulse/exchange_sync/portfolio_service.py
# Update the PortfolioService class's __init__ method

def __init__(self, exchange_id=None):  # Add exchange_id parameter with default None
    """Initialize the portfolio service."""
    self.repository = PortfolioRepository()
    self.exchange_id = exchange_id  # Store the exchange_id`}
                    </Typography>
                  </Paper>
                  
                  <Button
                    variant="contained"
                    startIcon={copied ? <CheckIcon /> : <CopyIcon />}
                    color={copied ? "success" : "primary"}
                    onClick={handleCopyFix}
                  >
                    {copied ? "Copied!" : "Copy Fix to Clipboard"}
                  </Button>
                </Box>
                
                <Box sx={{ mt: 3 }}>
                  <Alert severity="info">
                    <Typography variant="body2">
                      After applying this fix, restart the API server for the changes to take effect.
                    </Typography>
                  </Alert>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
        
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="body1">
                    <strong>Frontend Version:</strong> 1.0.0
                  </Typography>
                  <Typography variant="body1">
                    <strong>API Status:</strong> {portfolioError ? "Partial Issues" : "Operational"}
                  </Typography>
                  <Typography variant="body1">
                    <strong>Database:</strong> Connected
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body1">
                    <strong>Last API Check:</strong> {new Date().toLocaleTimeString()}
                  </Typography>
                  <Typography variant="body1">
                    <strong>Environment:</strong> Development
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DiagnosticPage;