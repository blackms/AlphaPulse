import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Divider,
  Tabs,
  Tab,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Button,
  IconButton,
  Paper,
  CircularProgress,
  Alert,
  Switch,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
  Analytics as AnalyticsIcon,
  BugReport as BugIcon,
  Warning as WarningIcon,
  // Removed unused SyncIcon import
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import useDataRefresh from '../../hooks/useDataRefresh';
import {
  selectAssets,
  selectTotalValue,
  selectCashBalance,
  selectPerformance,
  selectHistoricalValues,
  selectPortfolioLoading,
  selectPortfolioError,
  selectLastUpdated,
  fetchPortfolioStart,
  // Asset, // Unused import
} from '../../store/slices/portfolioSlice';
import ErrorFallback from '../../components/ErrorFallback';
import BackendErrorAlert from '../../components/BackendErrorAlert';
// Removed unused BackendErrorProxy import
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`portfolio-tabpanel-${index}`}
      aria-labelledby={`portfolio-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
};

const PortfolioPage: React.FC = () => {
  const dispatch = useDispatch();
  const assets = useSelector(selectAssets);
  const totalValue = useSelector(selectTotalValue);
  const cashBalance = useSelector(selectCashBalance);
  const performance = useSelector(selectPerformance);
  const historicalValues = useSelector(selectHistoricalValues);
  const isLoading = useSelector(selectPortfolioLoading);
  const error = useSelector(selectPortfolioError);
  const lastUpdated = useSelector(selectLastUpdated);
  
  const [tabValue, setTabValue] = useState(0);
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(true);
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Set up the data refresh functionality with the hook
  const { 
    refresh, 
    isRefreshing
  } = useDataRefresh({
    refreshFn: () => handleRefresh(),
    interval: 30000, // 30 seconds
    autoRefresh: autoRefreshEnabled,
    onError: (err) => {
      console.error("Error refreshing portfolio data:", err);
      // We don't need to do anything here as the error will be handled by the Redux store
    }
  });
  
  const handleRefresh = () => {
    try {
      dispatch(fetchPortfolioStart());
    } catch (err) {
      console.error("Error during portfolio data refresh:", err);
    }
  };
  
  // Error message for user-friendly display
  const getUserFriendlyErrorMessage = () => {
    return "We're experiencing difficulties communicating with our servers. " +
           "Please try again later.";
  };
  
  // Helper functions
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };
  
  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const formatLastUpdated = (timestamp: string | null) => {
    if (!timestamp) return 'Never updated';
    
    const diffMinutes = Math.round((Date.now() - new Date(timestamp).getTime()) / 60000);
    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes === 1) return '1 minute ago';
    return `${diffMinutes} minutes ago`;
  };
  
  // Chart data preparation
  const prepareChartData = () => {
    if (!historicalValues || historicalValues.length === 0) {
      return {
        labels: [],
        datasets: [
          {
            label: 'Portfolio Value',
            data: [],
            fill: false,
            backgroundColor: 'rgba(75, 192, 192, 0.4)',
            borderColor: 'rgba(75, 192, 192, 1)',
            tension: 0.1,
          },
        ],
      };
    }
    
    const sortedData = [...historicalValues].sort((a, b) => a.timestamp - b.timestamp);
    
    return {
      labels: sortedData.map(d => new Date(d.timestamp).toLocaleDateString()),
      datasets: [
        {
          label: 'Portfolio Value',
          data: sortedData.map(d => d.value),
          fill: false,
          backgroundColor: 'rgba(75, 192, 192, 0.4)',
          borderColor: 'rgba(75, 192, 192, 1)',
          tension: 0.1,
        },
      ],
    };
  };
  
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            return `Value: ${formatCurrency(context.raw)}`;
          },
        },
      },
    },
    scales: {
      y: {
        ticks: {
          callback: function(value: any) {
            return formatCurrency(value);
          },
        },
      },
    },
  };

  // Loading state - show a loading indicator
  if (isLoading && !assets.length) {
    return (
      <Box sx={{ p: 3, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
        <CircularProgress size={60} thickness={4} sx={{ mb: 3 }} />
        <Typography variant="h6" color="text.secondary" align="center">
          Loading portfolio data...
        </Typography>
      </Box>
    );
  };

  // Check for specific PortfolioService error
  const isPortfolioServiceError = error && error.includes('PortfolioService.__init__() takes 1 positional argument but 2 were given');

  // If there's an error and no data, show the error component
  if (error && !assets.length) {
    return isPortfolioServiceError ? (
      <Box sx={{ p: 3 }}>
        <BackendErrorAlert error={error} />
        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Button href="/dashboard/system/diagnostics" variant="contained" startIcon={<BugIcon />}>View Diagnostic Tools</Button>
        </Box>
      </Box>
    ) : (
      <Box sx={{ p: 3, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
        <Alert 
          severity="error" 
          sx={{ width: '100%', maxWidth: 600, mb: 3 }}
          variant="filled"
        >
          <Typography variant="h6" gutterBottom>Connection Error</Typography>
          <Typography>{getUserFriendlyErrorMessage()}</Typography>
        </Alert>
        <Button variant="contained" onClick={handleRefresh} startIcon={isRefreshing ? <CircularProgress size={20} color="inherit" /> : <RefreshIcon />}>
          {isRefreshing ? "Trying to reconnect..." : "Try Again"}
        </Button>
      </Box>
    );
  }
  
  return (
    <Box sx={{ p: 3 }}>
      {/* Show error alert if needed */}
      {error && (
        <Alert 
          severity="error" 
          sx={{ mb: 3 }}
        >
          <Typography>{getUserFriendlyErrorMessage()}</Typography>
        </Alert>
      )}
      
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Portfolio
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <Tooltip title={`Auto-refresh ${autoRefreshEnabled ? 'enabled' : 'disabled'}`}>
            <Box display="flex" alignItems="center">
              <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
                Auto refresh
              </Typography>
              <Switch 
                checked={autoRefreshEnabled}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setAutoRefreshEnabled(e.target.checked)}
              />
            </Box>
          </Tooltip>
          {isPortfolioServiceError ? (
            <Button
              variant="outlined"
              color="warning"
              component="a"
              href="/dashboard/system/diagnostics"
              startIcon={<BugIcon />}
              sx={{ mr: 2 }}
              size="small"
            >
              Diagnostic Tools
            </Button>
          ) : (
            <>
              {/* No specific error handling needed here due to our improved error UI */}
            </>
          )}
          <Button
            variant="contained"
            startIcon={isRefreshing ? <CircularProgress size={18} color="inherit" /> : <RefreshIcon />}
            onClick={refresh}
          >
            {isRefreshing ? "Refreshing..." : "Refresh"}
          </Button>
        </Box>
      </Box>
      
      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Total Value</Typography>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {isLoading && (
                  <CircularProgress size={16} thickness={4} sx={{ mr: 1 }} />
                )}
                {formatCurrency(totalValue)}
              </Typography>
              <Box display="flex" alignItems="center">
                {performance && performance.length > 0 && performance.find(p => p.period === 'day')?.returnPercent !== undefined && (
                  <Chip
                    icon={performance.find(p => p.period === 'day')!.returnPercent >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                    label={formatPercentage(performance.find(p => p.period === 'day')!.returnPercent)}
                    color={performance.find(p => p.period === 'day')!.returnPercent >= 0 ? 'success' : 'error'}
                    size="small"
                    sx={{ mr: 1 }}
                  />
                )}
                <Typography variant="body2">Today</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Only show cash balance if it represents real data */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Cash Balance
                <Tooltip title="Note: Cash balance is not retrieved from the exchange directly and may not represent your actual cash holdings.">
                  <IconButton size="small" sx={{ ml: 1, mt: -1 }}>
                    <InfoIcon fontSize="small" color="action" />
                  </IconButton>
                </Tooltip>
              </Typography>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {isLoading ? (
                  <CircularProgress size={16} thickness={4} sx={{ mr: 1 }} />
                ) : totalValue > 0 && Math.abs((cashBalance / totalValue) - 0.2) < 0.001 ? (
                  <Typography variant="body2" color="text.secondary" component="span">
                    Value estimated (not actual exchange data)
                  </Typography>
                ) : (
                  formatCurrency(cashBalance)
                )}
              </Typography>
              
              
              <Box display="flex" alignItems="center">
                <Typography variant="body2">
                  {((cashBalance / totalValue) * 100).toFixed(1)}% of portfolio
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Monthly Return</Typography>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {formatPercentage(performance && performance.length > 0 ?
                  performance.find(p => p.period === 'month')?.returnPercent || 0 : 0)}
              </Typography>
              <Box display="flex" alignItems="center">
                <Typography variant="body2">
                  {formatCurrency(performance && performance.length > 0 ?
                    performance.find(p => p.period === 'month')?.returnValue || 0 : 0)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>YTD Return</Typography>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {formatPercentage(performance && performance.length > 0 ?
                  performance.find(p => p.period === 'year')?.returnPercent || 0 : 0)}
              </Typography>
              <Box display="flex" alignItems="center">
                <Typography variant="body2">
                  {formatCurrency(performance && performance.length > 0 ?
                    performance.find(p => p.period === 'year')?.returnValue || 0 : 0)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Main Content with Tabs */}
      <Card>
        <CardHeader 
          title="Portfolio Performance"
          action={
            <Box>
              {(isRefreshing || lastUpdated) && (
                <Box sx={{ display: 'inline-flex', alignItems: 'center', mr: 2 }}>
                  {isRefreshing && (
                    <CircularProgress size={16} sx={{ mr: 1 }} />
                  )}
                  <Typography 
                    variant="caption" 
                    color="text.secondary"
                    sx={{ 
                      display: 'inline-block',
                      verticalAlign: 'middle' 
                    }}
                  >
                    {isRefreshing ? 'Refreshing...' : `Last updated: ${formatLastUpdated(lastUpdated)}`}
                  </Typography>
                </Box>
               )}
              <Button 
                startIcon={<AnalyticsIcon />}
                sx={{ mr: 1 }}
                disabled={isLoading || !!error}
              >
                Rebalance
              </Button>
            </Box>
          }
        />
        <Divider />
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="portfolio tabs">
            <Tab label="Overview" />
            <Tab label="Assets" />
            <Tab label="Performance" />
            <Tab label="Allocation" />
          </Tabs>
        </Box>
        
        <CardContent>
          {/* Overview Tab */}
          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>Portfolio Value History</Typography>
                  <Box height={300}>
                    <Line data={prepareChartData()} options={chartOptions} />
                  </Box>
                </Paper>
              </Grid>
              
              <Grid item xs={12}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>Portfolio Summary</Typography>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body1" gutterBottom>
                        <strong>Total Assets:</strong> {assets && assets.length ? assets.length : 0}
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        <strong>Portfolio Value:</strong> {formatCurrency(totalValue)}
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        <strong>Cash Balance:</strong>{' '}
                        {totalValue > 0 && Math.abs((cashBalance / totalValue) - 0.2) < 0.001 ? (
                          <span>Value estimated (not actual exchange data)</span>
                        ) : (
                          formatCurrency(cashBalance)
                        )
                        }
                        
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body1" gutterBottom>
                        <strong>Last Updated:</strong> {lastUpdated ? 
                          new Date(lastUpdated).toLocaleString() : 
                          'Never updated'}
                      </Typography>
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>
            </Grid>
          </TabPanel>
          
          {/* Assets Tab */}
          <TabPanel value={tabValue} index={1}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Asset</TableCell>
                  <TableCell align="right">Price</TableCell>
                  <TableCell align="right">Quantity</TableCell>
                  <TableCell align="right">Value</TableCell>
                  <TableCell align="right">Allocation</TableCell>
                  <TableCell align="right">Day Change</TableCell>
                  <TableCell align="right">Unrealized P&L</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {assets && assets.length > 0 ? assets.map((asset) => (
                  <TableRow key={asset.assetId}>
                    <TableCell component="th" scope="row">
                      <Box display="flex" alignItems="center">
                        <Typography variant="body1">
                          <strong>{asset.symbol}</strong>
                        </Typography>
                        <Typography variant="body2" sx={{ ml: 1, color: 'text.secondary' }}>
                          {asset.name}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="right">{formatCurrency(asset.price)}</TableCell>
                    <TableCell align="right">{asset.quantity.toLocaleString()}</TableCell>
                    <TableCell align="right">{formatCurrency(asset.value)}</TableCell>
                    <TableCell align="right">{asset.allocation.toFixed(1)}%</TableCell>
                    <TableCell align="right">
                      <Box display="flex" alignItems="center" justifyContent="flex-end">
                        {asset.change24hPercent >= 0 ?
                          <TrendingUpIcon fontSize="small" color="success" /> :
                          <TrendingDownIcon fontSize="small" color="error" />
                        }
                        <Typography
                          variant="body2"
                          color={asset.change24hPercent >= 0 ? 'success.main' : 'error.main'}
                          sx={{ ml: 0.5 }}
                        >
                          {formatPercentage(asset.change24hPercent)}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Typography
                        variant="body2"
                        color={asset.unrealizedPnL >= 0 ? 'success.main' : 'error.main'}
                      >
                        {formatCurrency(asset.unrealizedPnL)} ({formatPercentage(asset.unrealizedPnLPercent)})
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                )) : (
                  <TableRow>
                    <TableCell colSpan={8} align="center">No assets available</TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TabPanel>
          
          {/* Performance Tab */}
          <TabPanel value={tabValue} index={2}>    
            {performance && performance.length > 0 ? (
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>Performance Summary</Typography>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Period</TableCell>
                          <TableCell align="right">Return (%)</TableCell>
                          <TableCell align="right">Return ($)</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {performance.map((period) => (
                          <TableRow key={period.period}>
                            <TableCell component="th" scope="row">
                              {period.period.charAt(0).toUpperCase() + period.period.slice(1)}
                            </TableCell>
                            <TableCell align="right">
                              <Typography
                                variant="body2"
                                color={period.returnPercent >= 0 ? 'success.main' : 'error.main'}
                              >
                                {formatPercentage(period.returnPercent)}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography
                                variant="body2"
                                color={period.returnValue >= 0 ? 'success.main' : 'error.main'}
                              >
                                {formatCurrency(period.returnValue)}
                              </Typography>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>Top Performers</Typography>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Asset</TableCell>
                        <TableCell align="right">Return (%)</TableCell>
                        <TableCell align="right">Contribution</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {assets && assets.length > 0 ?
                        assets
                          .sort((a, b) => b.unrealizedPnLPercent - a.unrealizedPnLPercent)
                          .slice(0, 5)
                          .map((asset) => (
                            <TableRow key={asset.assetId}>
                              <TableCell component="th" scope="row">
                                {asset.symbol}
                              </TableCell>
                              <TableCell align="right">
                                <Typography
                                  variant="body2"
                                  color={asset.unrealizedPnLPercent >= 0 ? 'success.main' : 'error.main'}
                                >
                                  {formatPercentage(asset.unrealizedPnLPercent)}
                                </Typography>
                              </TableCell>
                              <TableCell align="right">
                                <Typography
                                  variant="body2"
                                  color={asset.unrealizedPnL >= 0 ? 'success.main' : 'error.main'}
                                >
                                  {formatCurrency(asset.unrealizedPnL)}
                                </Typography>
                              </TableCell>
                            </TableRow>
                          ))
                        : (
                          <TableRow>
                            <TableCell colSpan={3} align="center">No asset data available</TableCell>
                          </TableRow>
                        )
                      }
                    </TableBody>
                  </Table>
                </Paper>
              </Grid>
              </Grid>
            ) : (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 4 }}>
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography>No performance data available at this time.</Typography>
                </Alert>
                <Typography variant="body2" color="text.secondary">
                  Performance metrics will be displayed once they become available.
                </Typography>
              </Box>
            )}
          </TabPanel>
          
          {/* Allocation Tab */}
          <TabPanel value={tabValue} index={3}>
            {assets && assets.length > 0 ? (
              <Box sx={{ py: 2, display: 'flex', justifyContent: 'center' }}>
                <Paper sx={{ p: 3, maxWidth: '800px', width: '100%' }}>
                  <Typography variant="h6" gutterBottom align="center">Asset Allocation</Typography>
                  <Alert severity="info" sx={{ mb: 3 }}>
                    <Typography>
                      Asset allocation data is based on current portfolio composition.
                      Target allocations will be available in future updates.
                    </Typography>
                  </Alert>
                  {/* Display a simple allocation table */}
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Asset</TableCell>
                        <TableCell align="right">Current Allocation</TableCell>
                        <TableCell align="right">Value</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {assets.map((asset) => (
                        <TableRow key={asset.assetId}>
                          <TableCell>{asset.symbol}</TableCell>
                          <TableCell align="right">{asset.allocation.toFixed(1)}%</TableCell>
                          <TableCell align="right">{formatCurrency(asset.value)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Paper>
              </Box>
            ) : (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 4 }}>
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography>No allocation data available at this time.</Typography>
                </Alert>
                <Typography variant="body2" color="text.secondary">
                  Asset allocation will be displayed once portfolio data is available.
                </Typography>
              </Box>
            )}
          </TabPanel>
        </CardContent>
      </Card>
    </Box>
  );
};

export default PortfolioPage;