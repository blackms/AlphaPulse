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
  Button,
  Paper,
} from '@mui/material';
import {
  ArrowUpward as BuyIcon,
  ArrowDownward as SellIcon,
  CheckCircle as CompletedIcon,
  Pending as PendingIcon,
  Error as FailedIcon,
  Cancel as CanceledIcon,
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import {
  selectRecentTrades,
  selectPendingTrades,
  selectActiveSignals,
  updateSignalStatus,
  TradeType,
  TradeStatus,
} from '../../store/slices/tradingSlice';

const TradingPage: React.FC = () => {
  const dispatch = useDispatch();
  const recentTrades = useSelector(selectRecentTrades);
  const pendingTrades = useSelector(selectPendingTrades);
  const activeSignals = useSelector(selectActiveSignals);

  // Helper functions for formatting
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const getTradeTypeIcon = (type: TradeType) => {
    return type === 'buy' ? <BuyIcon /> : <SellIcon />;
  };

  const getTradeTypeColor = (type: TradeType) => {
    return type === 'buy' ? 'success' : 'error';
  };

  const getStatusIcon = (status: TradeStatus) => {
    switch (status) {
      case 'completed':
        return <CompletedIcon />;
      case 'pending':
        return <PendingIcon />;
      case 'failed':
        return <FailedIcon />;
      case 'canceled':
        return <CanceledIcon />;
      default:
        return <PendingIcon />; // Default icon
    }
  };

  const getStatusColor = (status: TradeStatus) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'pending':
        return 'warning';
      case 'failed':
        return 'error';
      case 'canceled':
        return 'default';
      default:
        return 'default';
    }
  };

  const handleAcceptSignal = (signalId: string) => {
    dispatch(updateSignalStatus({ signalId, status: 'accepted' }));
  };

  const handleRejectSignal = (signalId: string) => {
    dispatch(updateSignalStatus({ signalId, status: 'rejected' }));
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Trading
      </Typography>

      <Grid container spacing={3}>
        {/* Trade Signals */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Active Trading Signals" />
            <Divider />
            <CardContent>
              {activeSignals.length === 0 ? (
                <Typography variant="body1" textAlign="center" py={2}>
                  No active trading signals available.
                </Typography>
              ) : (
                <>
                  {activeSignals.map((signal) => (
                    <Paper key={signal.id} variant="outlined" sx={{ p: 2, mb: 2 }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                        <Box display="flex" alignItems="center">
                          <Chip
                            label={signal.type.toUpperCase()}
                            color={getTradeTypeColor(signal.type) as any}
                            icon={getTradeTypeIcon(signal.type)}
                            size="small"
                            sx={{ mr: 1 }}
                          />
                          <Typography variant="h6">
                            {signal.symbol} ({formatCurrency(signal.suggestedPrice)})
                          </Typography>
                        </Box>
                        <Chip
                          label={`Confidence: ${(signal.confidence * 100).toFixed(0)}%`}
                          color={signal.confidence > 0.7 ? 'success' : 'warning'}
                          size="small"
                        />
                      </Box>
                      <Typography variant="body1" mb={1}>
                        Suggested Quantity: {signal.suggestedQuantity.toLocaleString()}
                      </Typography>
                      <Typography variant="body1" mb={1}>
                        Total Value: {formatCurrency(signal.suggestedPrice * signal.suggestedQuantity)}
                      </Typography>
                      <Box mb={2}>
                        <Typography variant="subtitle2" gutterBottom>
                          Rationale:
                        </Typography>
                        <Typography variant="body2">
                          {signal.rationale}
                        </Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography variant="caption" color="textSecondary">
                          From {signal.agentName} • {formatTimestamp(signal.timestamp)}
                        </Typography>
                        <Box>
                          <Button
                            variant="outlined"
                            color="error"
                            size="small"
                            onClick={() => handleRejectSignal(signal.id)}
                            sx={{ mr: 1 }}
                          >
                            Reject
                          </Button>
                          <Button
                            variant="contained"
                            color="primary"
                            size="small"
                            onClick={() => handleAcceptSignal(signal.id)}
                          >
                            Accept
                          </Button>
                        </Box>
                      </Box>
                    </Paper>
                  ))}
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Pending Trades */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Pending Trades" />
            <Divider />
            <CardContent>
              {pendingTrades.length === 0 ? (
                <Typography variant="body1" textAlign="center" py={2}>
                  No pending trades.
                </Typography>
              ) : (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Type</TableCell>
                        <TableCell>Asset</TableCell>
                        <TableCell align="right">Price</TableCell>
                        <TableCell align="right">Quantity</TableCell>
                        <TableCell align="right">Value</TableCell>
                        <TableCell align="right">Time</TableCell>
                        <TableCell align="right">Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {pendingTrades.map((trade) => (
                        <TableRow key={trade.id}>
                          <TableCell>
                            <Chip
                              label={trade.type.toUpperCase()}
                              color={getTradeTypeColor(trade.type) as any}
                              icon={getTradeTypeIcon(trade.type)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{trade.symbol}</TableCell>
                          <TableCell align="right">{formatCurrency(trade.price)}</TableCell>
                          <TableCell align="right">{trade.quantity.toLocaleString()}</TableCell>
                          <TableCell align="right">{formatCurrency(trade.value)}</TableCell>
                          <TableCell align="right">{formatTimestamp(trade.timestamp)}</TableCell>
                          <TableCell align="right">
                            <Chip
                              label={trade.status.toUpperCase()}
                              color={getStatusColor(trade.status) as any}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Trades */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Recent Trades" />
            <Divider />
            <CardContent>
              {recentTrades.length === 0 ? (
                <Typography variant="body1" textAlign="center" py={2}>
                  No recent trades.
                </Typography>
              ) : (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Type</TableCell>
                        <TableCell>Asset</TableCell>
                        <TableCell align="right">Price</TableCell>
                        <TableCell align="right">Quantity</TableCell>
                        <TableCell align="right">Value</TableCell>
                        <TableCell align="right">Time</TableCell>
                        <TableCell align="right">Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {recentTrades.map((trade) => (
                        <TableRow key={trade.id}>
                          <TableCell>
                            <Chip
                              label={trade.type.toUpperCase()}
                              color={getTradeTypeColor(trade.type) as any}
                              icon={getTradeTypeIcon(trade.type)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{trade.symbol}</TableCell>
                          <TableCell align="right">{formatCurrency(trade.price)}</TableCell>
                          <TableCell align="right">{trade.quantity.toLocaleString()}</TableCell>
                          <TableCell align="right">{formatCurrency(trade.value)}</TableCell>
                          <TableCell align="right">{formatTimestamp(trade.timestamp)}</TableCell>
                          <TableCell align="right">
                            <Chip
                              label={trade.status.toUpperCase()}
                              color={getStatusColor(trade.status) as any}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TradingPage;