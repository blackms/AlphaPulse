import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Divider,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { RootState } from '../../store/store';
import { PortfolioPosition } from '../../store/slices/portfolioSlice';

// Placeholder for chart components
const PieChart = ({ data }: { data: any }) => (
  <Box height={300} display="flex" alignItems="center" justifyContent="center">
    <Typography>Asset Allocation Chart (Placeholder)</Typography>
  </Box>
);

const LineChart = ({ data }: { data: any }) => (
  <Box height={300} display="flex" alignItems="center" justifyContent="center">
    <Typography>Performance Chart (Placeholder)</Typography>
  </Box>
);

// This is a placeholder component until the full implementation is complete
const PortfolioPage: React.FC = () => {
  const dispatch = useDispatch();
  const { totalValue, cashBalance, positions, performance, riskMetrics, loading, error } = useSelector(
    (state: RootState) => state.portfolio
  );

  useEffect(() => {
    // Placeholder for fetching portfolio data
    // dispatch(fetchPortfolio());
  }, [dispatch]);

  const handleRefresh = () => {
    // Placeholder for refreshing portfolio data
    // dispatch(fetchPortfolio());
  };

  // Calculate asset allocation for pie chart
  const getAllocationData = () => {
    if (!positions || positions.length === 0) return [];
    return positions.map(position => ({
      name: position.symbol,
      value: position.allocation,
    }));
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Portfolio
        </Typography>
        <IconButton onClick={handleRefresh} disabled={loading}>
          <RefreshIcon />
        </IconButton>
      </Box>

      {loading ? (
        <Typography>Loading portfolio data...</Typography>
      ) : error ? (
        <Paper sx={{ p: 2, bgcolor: 'error.light' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {/* Summary Cards */}
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Total Value
                </Typography>
                <Typography variant="h4">{formatCurrency(totalValue)}</Typography>
                <Box display="flex" alignItems="center" mt={1}>
                  {performance.daily >= 0 ? (
                    <TrendingUpIcon color="success" fontSize="small" sx={{ mr: 0.5 }} />
                  ) : (
                    <TrendingDownIcon color="error" fontSize="small" sx={{ mr: 0.5 }} />
                  )}
                  <Typography
                    variant="body2"
                    color={performance.daily >= 0 ? 'success.main' : 'error.main'}
                  >
                    {formatPercentage(performance.daily)} Today
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Cash Balance
                </Typography>
                <Typography variant="h4">{formatCurrency(cashBalance)}</Typography>
                <Typography variant="body2" color="text.secondary" mt={1}>
                  {((cashBalance / totalValue) * 100).toFixed(2)}% of Portfolio
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Monthly Performance
                </Typography>
                <Typography variant="h4">{formatPercentage(performance.monthly)}</Typography>
                <Typography variant="body2" color="text.secondary" mt={1}>
                  Previous: {formatPercentage(performance.monthly - 1.2)} {/* Placeholder */}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Volatility
                </Typography>
                <Typography variant="h4">{riskMetrics.volatility.toFixed(2)}%</Typography>
                <Typography variant="body2" color="text.secondary" mt={1}>
                  Sharpe Ratio: {riskMetrics.sharpeRatio.toFixed(2)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Charts */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Asset Allocation" />
              <Divider />
              <CardContent>
                <PieChart data={getAllocationData()} />
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Performance History" />
              <Divider />
              <CardContent>
                <LineChart data={[]} /> {/* Placeholder */}
              </CardContent>
            </Card>
          </Grid>

          {/* Positions Table */}
          <Grid item xs={12}>
            <Card>
              <CardHeader title="Positions" />
              <Divider />
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Entry Price</TableCell>
                      <TableCell align="right">Current Price</TableCell>
                      <TableCell align="right">Value</TableCell>
                      <TableCell align="right">P&L</TableCell>
                      <TableCell align="right">P&L %</TableCell>
                      <TableCell align="right">Allocation</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {positions && positions.length > 0 ? (
                      positions.map((position: PortfolioPosition) => (
                        <TableRow key={position.symbol}>
                          <TableCell component="th" scope="row">
                            {position.symbol}
                          </TableCell>
                          <TableCell align="right">{position.quantity}</TableCell>
                          <TableCell align="right">{formatCurrency(position.averageEntryPrice)}</TableCell>
                          <TableCell align="right">{formatCurrency(position.currentPrice)}</TableCell>
                          <TableCell align="right">{formatCurrency(position.value)}</TableCell>
                          <TableCell
                            align="right"
                            sx={{ color: position.pnl >= 0 ? 'success.main' : 'error.main' }}
                          >
                            {formatCurrency(position.pnl)}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{ color: position.pnlPercentage >= 0 ? 'success.main' : 'error.main' }}
                          >
                            {formatPercentage(position.pnlPercentage)}
                          </TableCell>
                          <TableCell align="right">{position.allocation.toFixed(2)}%</TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={8} align="center">
                          No positions found
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default PortfolioPage;