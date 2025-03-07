import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Chip,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import {
  selectTotalValue,
  selectCashBalance,
  selectAssets,
  selectPerformance,
  selectHistory,
  selectLastUpdated,
} from '../../store/slices/portfolioSlice';

const PortfolioPage: React.FC = () => {
  const totalValue = useSelector(selectTotalValue);
  const cashBalance = useSelector(selectCashBalance);
  const assets = useSelector(selectAssets);
  const performance = useSelector(selectPerformance);
  const lastUpdated = useSelector(selectLastUpdated);

  // Helper functions for formatting
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const getDailyPerformance = () => {
    return performance.find(p => p.period === 'day') || { returnValue: 0, returnPercent: 0 };
  };

  const getWeeklyPerformance = () => {
    return performance.find(p => p.period === 'week') || { returnValue: 0, returnPercent: 0 };
  };

  const getMonthlyPerformance = () => {
    return performance.find(p => p.period === 'month') || { returnValue: 0, returnPercent: 0 };
  };

  const getYearlyPerformance = () => {
    return performance.find(p => p.period === 'year') || { returnValue: 0, returnPercent: 0 };
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Portfolio
      </Typography>

      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Portfolio Value" />
            <Divider />
            <CardContent>
              <Typography variant="h4">{formatCurrency(totalValue)}</Typography>
              <Box display="flex" alignItems="center" mt={1}>
                <Chip
                  icon={getDailyPerformance().returnPercent >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                  label={formatPercentage(getDailyPerformance().returnPercent)}
                  color={getDailyPerformance().returnPercent >= 0 ? 'success' : 'error'}
                  size="small"
                />
                <Typography variant="body2" sx={{ ml: 1 }}>
                  Today ({formatCurrency(getDailyPerformance().returnValue)})
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Cash Balance" />
            <Divider />
            <CardContent>
              <Typography variant="h4">{formatCurrency(cashBalance)}</Typography>
              <Box display="flex" alignItems="center" mt={1}>
                <Typography variant="body2">
                  {((cashBalance / totalValue) * 100).toFixed(2)}% of portfolio
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Monthly Performance" />
            <Divider />
            <CardContent>
              <Typography variant="h4">{formatPercentage(getMonthlyPerformance().returnPercent)}</Typography>
              <Box display="flex" alignItems="center" mt={1}>
                <Chip
                  icon={getMonthlyPerformance().returnPercent >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                  label={formatCurrency(getMonthlyPerformance().returnValue)}
                  color={getMonthlyPerformance().returnPercent >= 0 ? 'success' : 'error'}
                  size="small"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Overview */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Performance Overview" />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle1" gutterBottom>
                      Daily
                    </Typography>
                    <Typography 
                      variant="h5" 
                      color={getDailyPerformance().returnPercent >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercentage(getDailyPerformance().returnPercent)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {formatCurrency(getDailyPerformance().returnValue)}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle1" gutterBottom>
                      Weekly
                    </Typography>
                    <Typography 
                      variant="h5" 
                      color={getWeeklyPerformance().returnPercent >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercentage(getWeeklyPerformance().returnPercent)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {formatCurrency(getWeeklyPerformance().returnValue)}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle1" gutterBottom>
                      Monthly
                    </Typography>
                    <Typography 
                      variant="h5" 
                      color={getMonthlyPerformance().returnPercent >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercentage(getMonthlyPerformance().returnPercent)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {formatCurrency(getMonthlyPerformance().returnValue)}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle1" gutterBottom>
                      Yearly
                    </Typography>
                    <Typography 
                      variant="h5" 
                      color={getYearlyPerformance().returnPercent >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercentage(getYearlyPerformance().returnPercent)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {formatCurrency(getYearlyPerformance().returnValue)}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Asset Allocation */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Asset Allocation" />
            <Divider />
            <CardContent>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Asset</TableCell>
                      <TableCell align="right">Price</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Value</TableCell>
                      <TableCell align="right">Allocation</TableCell>
                      <TableCell align="right">24h Change</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {assets.map((asset) => (
                      <TableRow key={asset.assetId}>
                        <TableCell>
                          <Box display="flex" alignItems="center">
                            <Typography variant="body1">{asset.symbol}</Typography>
                            <Typography variant="body2" color="textSecondary" sx={{ ml: 1 }}>
                              {asset.name}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell align="right">{formatCurrency(asset.price)}</TableCell>
                        <TableCell align="right">{asset.quantity.toLocaleString()}</TableCell>
                        <TableCell align="right">{formatCurrency(asset.value)}</TableCell>
                        <TableCell align="right">{asset.allocation.toFixed(2)}%</TableCell>
                        <TableCell 
                          align="right"
                          sx={{ 
                            color: asset.dayChangePercent >= 0 ? 'success.main' : 'error.main'
                          }}
                        >
                          {formatPercentage(asset.dayChangePercent)}
                        </TableCell>
                      </TableRow>
                    ))}
                    {/* Cash Row */}
                    <TableRow>
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          <Typography variant="body1">USD</Typography>
                          <Typography variant="body2" color="textSecondary" sx={{ ml: 1 }}>
                            Cash
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="right">-</TableCell>
                      <TableCell align="right">-</TableCell>
                      <TableCell align="right">{formatCurrency(cashBalance)}</TableCell>
                      <TableCell align="right">
                        {((cashBalance / totalValue) * 100).toFixed(2)}%
                      </TableCell>
                      <TableCell align="right">-</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
              <Typography variant="caption" display="block" mt={2} color="textSecondary">
                Last updated: {lastUpdated ? new Date(lastUpdated).toLocaleString() : 'Never'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PortfolioPage;