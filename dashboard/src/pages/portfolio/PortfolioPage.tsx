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
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
  Analytics as AnalyticsIcon,
  SwapHoriz as RebalanceIcon,
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import {
  selectAssets,
  selectTotalValue,
  selectCashBalance,
  selectPerformance,
  selectHistoricalValues,
  fetchPortfolioStart,
  Asset,
} from '../../store/slices/portfolioSlice';
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
  
  const [tabValue, setTabValue] = useState(0);
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  const handleRefresh = () => {
    dispatch(fetchPortfolioStart());
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
  
  // Chart data preparation
  const prepareChartData = () => {
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
  
  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Portfolio
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
        >
          Refresh
        </Button>
      </Box>
      
      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Total Value</Typography>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {formatCurrency(totalValue)}
              </Typography>
              <Box display="flex" alignItems="center">
                {performance.find(p => p.period === 'day')?.returnPercent !== undefined && (
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
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Cash Balance</Typography>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {formatCurrency(cashBalance)}
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
                {formatPercentage(performance.find(p => p.period === 'month')?.returnPercent || 0)}
              </Typography>
              <Box display="flex" alignItems="center">
                <Typography variant="body2">
                  {formatCurrency(performance.find(p => p.period === 'month')?.returnValue || 0)}
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
                {formatPercentage(performance.find(p => p.period === 'year')?.returnPercent || 0)}
              </Typography>
              <Box display="flex" alignItems="center">
                <Typography variant="body2">
                  {formatCurrency(performance.find(p => p.period === 'year')?.returnValue || 0)}
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
              <Button 
                startIcon={<AnalyticsIcon />}
                sx={{ mr: 1 }}
              >
                Analytics
              </Button>
              <Button 
                startIcon={<RebalanceIcon />}
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
                        <strong>Assets:</strong> {assets.length}
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        <strong>Diversification Score:</strong> 78%
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        <strong>Risk Level:</strong> Medium
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body1" gutterBottom>
                        <strong>Sharpe Ratio:</strong> 1.8
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        <strong>Volatility:</strong> 12.3%
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        <strong>Max Drawdown:</strong> 15.2%
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
                {assets.map((asset) => (
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
                ))}
              </TableBody>
            </Table>
          </TabPanel>
          
          {/* Performance Tab */}
          <TabPanel value={tabValue} index={2}>
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
                      {assets
                        .filter(asset => asset.symbol !== 'CASH')
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
                        ))}
                    </TableBody>
                  </Table>
                </Paper>
              </Grid>
            </Grid>
          </TabPanel>
          
          {/* Allocation Tab */}
          <TabPanel value={tabValue} index={3}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>Current Allocation</Typography>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Asset</TableCell>
                        <TableCell align="right">Current</TableCell>
                        <TableCell align="right">Target</TableCell>
                        <TableCell align="right">Difference</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {assets.map((asset) => {
                        // Mock target allocations for demo purposes
                        const targetAllocation = 
                          asset.symbol === 'BTC' ? 35.0 :
                          asset.symbol === 'ETH' ? 25.0 :
                          asset.symbol === 'SOL' ? 20.0 :
                          asset.symbol === 'LINK' ? 5.0 :
                          asset.symbol === 'MATIC' ? 5.0 :
                          asset.symbol === 'CASH' ? 10.0 : 0;
                          
                        const difference = asset.allocation - targetAllocation;
                        
                        return (
                          <TableRow key={asset.assetId}>
                            <TableCell component="th" scope="row">
                              {asset.symbol}
                            </TableCell>
                            <TableCell align="right">{asset.allocation.toFixed(1)}%</TableCell>
                            <TableCell align="right">{targetAllocation.toFixed(1)}%</TableCell>
                            <TableCell align="right">
                              <Typography 
                                variant="body2"
                                color={Math.abs(difference) > 2 ? 'warning.main' : 'text.primary'}
                              >
                                {difference >= 0 ? '+' : ''}{difference.toFixed(1)}%
                              </Typography>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>Rebalancing Recommendations</Typography>
                  <Box>
                    <Typography variant="body1" paragraph>
                      Based on the current market conditions and portfolio allocation, the following adjustments are recommended:
                    </Typography>
                    
                    <Box component="ul">
                      {assets.map((asset) => {
                        // Mock target allocations for demo purposes
                        const targetAllocation = 
                          asset.symbol === 'BTC' ? 35.0 :
                          asset.symbol === 'ETH' ? 25.0 :
                          asset.symbol === 'SOL' ? 20.0 :
                          asset.symbol === 'LINK' ? 5.0 :
                          asset.symbol === 'MATIC' ? 5.0 :
                          asset.symbol === 'CASH' ? 10.0 : 0;
                          
                        const difference = asset.allocation - targetAllocation;
                        
                        if (Math.abs(difference) > 2) {
                          const action = difference > 0 ? 'Reduce' : 'Increase';
                          const amountToAdjust = Math.abs(difference) * totalValue / 100;
                          
                          return (
                            <Box component="li" key={asset.assetId} sx={{ mb: 1 }}>
                              <Typography variant="body2">
                                <strong>{action} {asset.symbol}:</strong> {formatCurrency(amountToAdjust)} ({Math.abs(difference).toFixed(1)}%)
                              </Typography>
                            </Box>
                          );
                        }
                        return null;
                      }).filter(Boolean)}
                    </Box>
                    
                    <Box mt={3}>
                      <Button 
                        variant="contained" 
                        color="primary"
                        startIcon={<RebalanceIcon />}
                      >
                        Execute Rebalancing
                      </Button>
                    </Box>
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          </TabPanel>
        </CardContent>
      </Card>
    </Box>
  );
};

export default PortfolioPage;