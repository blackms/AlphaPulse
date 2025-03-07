import React, { useEffect, useState } from 'react';
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
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { RootState } from '../../store/store';
import { Trade, Order, TradeStatus, TradeDirection } from '../../store/slices/tradingSlice';

// Interface for TabPanel
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

// TabPanel component
const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
};

// This is a placeholder component until the full implementation is complete
const TradingPage: React.FC = () => {
  const dispatch = useDispatch();
  const { trades, orders, performance, loading, error } = useSelector(
    (state: RootState) => state.trading
  );
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    // Placeholder for fetching trading data
    // dispatch(fetchTrades());
  }, [dispatch]);

  const handleRefresh = () => {
    // Placeholder for refreshing trading data
    // dispatch(fetchTrades());
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getStatusColor = (status: TradeStatus | string) => {
    switch (status) {
      case 'completed':
      case 'filled':
        return 'success';
      case 'pending':
      case 'open':
      case 'partial':
        return 'info';
      case 'cancelled':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const getDirectionColor = (direction: TradeDirection) => {
    return direction === 'buy' ? 'success' : 'error';
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Trading
        </Typography>
        <IconButton onClick={handleRefresh} disabled={loading}>
          <RefreshIcon />
        </IconButton>
      </Box>

      {loading ? (
        <Typography>Loading trading data...</Typography>
      ) : error ? (
        <Paper sx={{ p: 2, bgcolor: 'error.light' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {/* Trading Performance Cards */}
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Win Rate
                </Typography>
                <Typography variant="h4">{performance.winRate.toFixed(2)}%</Typography>
                <Typography variant="body2" color="text.secondary" mt={1}>
                  Based on last 100 trades
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Profit Factor
                </Typography>
                <Typography variant="h4">{performance.profitFactor.toFixed(2)}</Typography>
                <Typography variant="body2" color="text.secondary" mt={1}>
                  Gross Profit / Gross Loss
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Average Win
                </Typography>
                <Typography variant="h4">{formatCurrency(performance.avgWin)}</Typography>
                <Box display="flex" alignItems="center" mt={1}>
                  <TrendingUpIcon color="success" fontSize="small" sx={{ mr: 0.5 }} />
                  <Typography variant="body2" color="success.main">
                    {formatCurrency(performance.largestWin)} Largest
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Average Loss
                </Typography>
                <Typography variant="h4">{formatCurrency(performance.avgLoss)}</Typography>
                <Box display="flex" alignItems="center" mt={1}>
                  <TrendingDownIcon color="error" fontSize="small" sx={{ mr: 0.5 }} />
                  <Typography variant="body2" color="error.main">
                    {formatCurrency(performance.largestLoss)} Largest
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Trading Tabs: Trades & Orders */}
          <Grid item xs={12}>
            <Card>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={tabValue} onChange={handleTabChange} aria-label="trading tabs">
                  <Tab label="Recent Trades" />
                  <Tab label="Open Orders" />
                </Tabs>
              </Box>

              {/* Trades Tab */}
              <TabPanel value={tabValue} index={0}>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Time</TableCell>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Direction</TableCell>
                        <TableCell align="right">Quantity</TableCell>
                        <TableCell align="right">Price</TableCell>
                        <TableCell align="right">Value</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Agent</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {trades && trades.length > 0 ? (
                        trades.map((trade: Trade) => (
                          <TableRow key={trade.id}>
                            <TableCell>{formatTimestamp(trade.timestamp)}</TableCell>
                            <TableCell>{trade.symbol}</TableCell>
                            <TableCell>
                              <Chip
                                label={trade.direction.toUpperCase()}
                                color={getDirectionColor(trade.direction) as any}
                                size="small"
                              />
                            </TableCell>
                            <TableCell align="right">{trade.quantity}</TableCell>
                            <TableCell align="right">{formatCurrency(trade.price)}</TableCell>
                            <TableCell align="right">{formatCurrency(trade.value)}</TableCell>
                            <TableCell>
                              <Chip
                                label={trade.status}
                                color={getStatusColor(trade.status) as any}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>{trade.agent || '-'}</TableCell>
                          </TableRow>
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={8} align="center">
                            No trades found
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </TabPanel>

              {/* Orders Tab */}
              <TabPanel value={tabValue} index={1}>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Time</TableCell>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Direction</TableCell>
                        <TableCell align="right">Quantity</TableCell>
                        <TableCell align="right">Price</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {orders && orders.length > 0 ? (
                        orders.map((order: Order) => (
                          <TableRow key={order.id}>
                            <TableCell>{formatTimestamp(order.created)}</TableCell>
                            <TableCell>{order.symbol}</TableCell>
                            <TableCell>{order.type.replace('_', ' ').toUpperCase()}</TableCell>
                            <TableCell>
                              <Chip
                                label={order.direction.toUpperCase()}
                                color={getDirectionColor(order.direction) as any}
                                size="small"
                              />
                            </TableCell>
                            <TableCell align="right">{order.quantity}</TableCell>
                            <TableCell align="right">{formatCurrency(order.price)}</TableCell>
                            <TableCell>
                              <Chip
                                label={order.status}
                                color={getStatusColor(order.status) as any}
                                size="small"
                              />
                            </TableCell>
                          </TableRow>
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={7} align="center">
                            No open orders
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </TabPanel>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default TradingPage;