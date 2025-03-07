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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  CheckCircle as SuccessIcon,
  Cancel as CancelIcon,
  Info as InfoIcon,
  PlayArrow as ExecuteIcon,
  Close as CloseIcon,
  Warning as WarningIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import {
  selectActiveSignals,
  selectHistoricalSignals,
  selectRecentTrades,
  selectPendingOrders,
  fetchTradingDataStart,
  updateSignal,
  expireSignal,
  executeSignal,
  updateTradeStatus,
  Signal,
  Trade,
  OrderStatus,
  SignalDirection,
  SignalStrength,
} from '../../store/slices/tradingSlice';

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
      id={`trading-tabpanel-${index}`}
      aria-labelledby={`trading-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
};

const TradingPage: React.FC = () => {
  const dispatch = useDispatch();
  const activeSignals = useSelector(selectActiveSignals);
  const historicalSignals = useSelector(selectHistoricalSignals);
  const recentTrades = useSelector(selectRecentTrades);
  const pendingOrders = useSelector(selectPendingOrders);
  
  const [tabValue, setTabValue] = useState(0);
  const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);
  const [signalDetailsOpen, setSignalDetailsOpen] = useState(false);
  const [selectedTrade, setSelectedTrade] = useState<Trade | null>(null);
  const [tradeDetailsOpen, setTradeDetailsOpen] = useState(false);
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  const handleRefresh = () => {
    dispatch(fetchTradingDataStart());
  };
  
  const handleOpenSignalDetails = (signal: Signal) => {
    setSelectedSignal(signal);
    setSignalDetailsOpen(true);
  };
  
  const handleCloseSignalDetails = () => {
    setSignalDetailsOpen(false);
  };
  
  const handleOpenTradeDetails = (trade: Trade) => {
    setSelectedTrade(trade);
    setTradeDetailsOpen(true);
  };
  
  const handleCloseTradeDetails = () => {
    setTradeDetailsOpen(false);
  };
  
  const handleExecuteSignal = (signal: Signal) => {
    // In a real app, we would dispatch an action to execute the signal
    // For this example, we'll mock it with a simple trade
    
    // Skip 'hold' directions since they don't result in trades
    if (signal.direction === 'hold') {
      return;
    }
    
    // Using type assertion to ensure the direction is 'buy' or 'sell'
    const tradeDirection: 'buy' | 'sell' = signal.direction;
    
    const mockTrade: Trade = {
      id: `t${Date.now()}`,
      signalId: signal.id,
      assetId: signal.assetId,
      symbol: signal.symbol,
      direction: tradeDirection,
      quantity: tradeDirection === 'buy' ? signal.suggestedSize / 65000 : 0.1, // Mock calculation
      price: tradeDirection === 'buy' ? 65000 : 64000, // Mock price
      total: tradeDirection === 'buy' ? signal.suggestedSize : 6400, // Mock total
      timestamp: Date.now(),
      status: 'filled',
      fees: tradeDirection === 'buy' ? signal.suggestedSize * 0.001 : 6.4, // Mock fees
    };
    
    dispatch(executeSignal({
      signalId: signal.id,
      trade: mockTrade,
    }));
  };
  
  const handleCancelSignal = (signalId: string) => {
    dispatch(expireSignal(signalId));
  };
  
  const handleCancelOrder = (tradeId: string) => {
    dispatch(updateTradeStatus({
      tradeId,
      status: 'cancelled',
    }));
  };
  
  // Helper functions
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };
  
  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };
  
  const getDirectionColor = (direction: SignalDirection) => {
    switch (direction) {
      case 'buy':
        return 'success';
      case 'sell':
        return 'error';
      default:
        return 'default';
    }
  };
  
  const getStrengthColor = (strength: SignalStrength) => {
    switch (strength) {
      case 'strong':
        return 'success';
      case 'moderate':
        return 'warning';
      case 'weak':
        return 'error';
      default:
        return 'default';
    }
  };
  
  const getStatusColor = (status: OrderStatus) => {
    switch (status) {
      case 'filled':
        return 'success';
      case 'partial':
        return 'warning';
      case 'cancelled':
        return 'error';
      case 'pending':
        return 'info';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };
  
  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Trading
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
        >
          Refresh
        </Button>
      </Box>
      
      {/* Main Content with Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="trading tabs">
            <Tab label="Active Signals" />
            <Tab label="Recent Trades" />
            <Tab label="Pending Orders" />
            <Tab label="Historical Signals" />
          </Tabs>
        </Box>
        
        <CardContent>
          {/* Active Signals Tab */}
          <TabPanel value={tabValue} index={0}>
            {activeSignals.length === 0 ? (
              <Box textAlign="center" py={4}>
                <Typography variant="body1" color="textSecondary">
                  No active signals
                </Typography>
              </Box>
            ) : (
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Asset</TableCell>
                    <TableCell>Direction</TableCell>
                    <TableCell>Strength</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Source</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {activeSignals.map((signal) => (
                    <TableRow key={signal.id}>
                      <TableCell>{signal.symbol}</TableCell>
                      <TableCell>
                        <Chip
                          label={signal.direction.toUpperCase()}
                          color={getDirectionColor(signal.direction) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={signal.strength.charAt(0).toUpperCase() + signal.strength.slice(1)}
                          color={getStrengthColor(signal.strength) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{signal.confidence}%</TableCell>
                      <TableCell>{signal.source.charAt(0).toUpperCase() + signal.source.slice(1)}</TableCell>
                      <TableCell>{formatCurrency(signal.suggestedSize)}</TableCell>
                      <TableCell>{formatTimestamp(signal.timestamp)}</TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => handleOpenSignalDetails(signal)}
                          sx={{ mr: 1 }}
                        >
                          <InfoIcon fontSize="small" />
                        </IconButton>
                        {signal.direction !== 'hold' && (
                          <IconButton
                            size="small"
                            color="success"
                            onClick={() => handleExecuteSignal(signal)}
                            sx={{ mr: 1 }}
                          >
                            <ExecuteIcon fontSize="small" />
                          </IconButton>
                        )}
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleCancelSignal(signal.id)}
                        >
                          <CancelIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </TabPanel>
          
          {/* Recent Trades Tab */}
          <TabPanel value={tabValue} index={1}>
            {recentTrades.length === 0 ? (
              <Box textAlign="center" py={4}>
                <Typography variant="body1" color="textSecondary">
                  No recent trades
                </Typography>
              </Box>
            ) : (
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Asset</TableCell>
                    <TableCell>Direction</TableCell>
                    <TableCell>Quantity</TableCell>
                    <TableCell>Price</TableCell>
                    <TableCell>Total</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {recentTrades.map((trade) => (
                    <TableRow key={trade.id}>
                      <TableCell>{trade.symbol}</TableCell>
                      <TableCell>
                        <Chip
                          label={trade.direction.toUpperCase()}
                          color={trade.direction === 'buy' ? 'success' : 'error'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{trade.quantity.toLocaleString()}</TableCell>
                      <TableCell>{formatCurrency(trade.price)}</TableCell>
                      <TableCell>{formatCurrency(trade.total)}</TableCell>
                      <TableCell>
                        <Chip
                          label={trade.status.charAt(0).toUpperCase() + trade.status.slice(1)}
                          color={getStatusColor(trade.status) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{formatTimestamp(trade.timestamp)}</TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => handleOpenTradeDetails(trade)}
                        >
                          <InfoIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </TabPanel>
          
          {/* Pending Orders Tab */}
          <TabPanel value={tabValue} index={2}>
            {pendingOrders.length === 0 ? (
              <Box textAlign="center" py={4}>
                <Typography variant="body1" color="textSecondary">
                  No pending orders
                </Typography>
              </Box>
            ) : (
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Asset</TableCell>
                    <TableCell>Direction</TableCell>
                    <TableCell>Quantity</TableCell>
                    <TableCell>Price</TableCell>
                    <TableCell>Total</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {pendingOrders.map((order) => (
                    <TableRow key={order.id}>
                      <TableCell>{order.symbol}</TableCell>
                      <TableCell>
                        <Chip
                          label={order.direction.toUpperCase()}
                          color={order.direction === 'buy' ? 'success' : 'error'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{order.quantity.toLocaleString()}</TableCell>
                      <TableCell>{formatCurrency(order.price)}</TableCell>
                      <TableCell>{formatCurrency(order.total)}</TableCell>
                      <TableCell>
                        <Chip
                          label="Pending"
                          color="info"
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{formatTimestamp(order.timestamp)}</TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleCancelOrder(order.id)}
                        >
                          <CancelIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </TabPanel>
          
          {/* Historical Signals Tab */}
          <TabPanel value={tabValue} index={3}>
            {historicalSignals.length === 0 ? (
              <Box textAlign="center" py={4}>
                <Typography variant="body1" color="textSecondary">
                  No historical signals
                </Typography>
              </Box>
            ) : (
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Asset</TableCell>
                    <TableCell>Direction</TableCell>
                    <TableCell>Strength</TableCell>
                    <TableCell>Source</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {historicalSignals.map((signal) => (
                    <TableRow key={signal.id}>
                      <TableCell>{signal.symbol}</TableCell>
                      <TableCell>
                        <Chip
                          label={signal.direction.toUpperCase()}
                          color={getDirectionColor(signal.direction) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={signal.strength.charAt(0).toUpperCase() + signal.strength.slice(1)}
                          color={getStrengthColor(signal.strength) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{signal.source.charAt(0).toUpperCase() + signal.source.slice(1)}</TableCell>
                      <TableCell>
                        <Chip
                          label={signal.status.charAt(0).toUpperCase() + signal.status.slice(1)}
                          color={signal.status === 'executed' ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{formatTimestamp(signal.timestamp)}</TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => handleOpenSignalDetails(signal)}
                        >
                          <InfoIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </TabPanel>
        </CardContent>
      </Card>
      
      {/* Signal Details Dialog */}
      <Dialog
        open={signalDetailsOpen}
        onClose={handleCloseSignalDetails}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">
              Signal Details
            </Typography>
            <IconButton onClick={handleCloseSignalDetails}>
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          {selectedSignal && (
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Asset:</strong> {selectedSignal.symbol} ({selectedSignal.name})
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Direction:</strong>{' '}
                  <Chip
                    label={selectedSignal.direction.toUpperCase()}
                    color={getDirectionColor(selectedSignal.direction) as any}
                    size="small"
                  />
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Strength:</strong>{' '}
                  <Chip
                    label={selectedSignal.strength.charAt(0).toUpperCase() + selectedSignal.strength.slice(1)}
                    color={getStrengthColor(selectedSignal.strength) as any}
                    size="small"
                  />
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Confidence:</strong> {selectedSignal.confidence}%
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Source:</strong> {selectedSignal.source.charAt(0).toUpperCase() + selectedSignal.source.slice(1)}
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Risk Score:</strong> {selectedSignal.riskScore}/100
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Created:</strong> {formatTimestamp(selectedSignal.timestamp)}
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Expires:</strong> {formatTimestamp(selectedSignal.expiresAt)}
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Status:</strong>{' '}
                  <Chip
                    label={selectedSignal.status.charAt(0).toUpperCase() + selectedSignal.status.slice(1)}
                    color={selectedSignal.status === 'active' ? 'success' : 
                           selectedSignal.status === 'executed' ? 'info' : 'default'}
                    size="small"
                  />
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Suggested Size:</strong> {formatCurrency(selectedSignal.suggestedSize)}
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Description
                </Typography>
                <Typography variant="body1" paragraph>
                  {selectedSignal.description}
                </Typography>
              </Grid>
              {selectedSignal.indicators && (
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Technical Indicators
                  </Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Indicator</TableCell>
                        <TableCell align="right">Value</TableCell>
                        <TableCell align="right">Trend</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {selectedSignal.indicators.map((indicator, index) => (
                        <TableRow key={index}>
                          <TableCell>{indicator.name}</TableCell>
                          <TableCell align="right">{indicator.value}</TableCell>
                          <TableCell align="right">
                            <Chip
                              label={indicator.trend.toUpperCase()}
                              color={indicator.trend === 'up' ? 'success' : 
                                     indicator.trend === 'down' ? 'error' : 'default'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          {selectedSignal && selectedSignal.status === 'active' && (
            <>
              {selectedSignal.direction !== 'hold' && (
                <Button
                  variant="contained"
                  color="success"
                  onClick={() => {
                    if (selectedSignal) {
                      handleExecuteSignal(selectedSignal);
                      handleCloseSignalDetails();
                    }
                  }}
                  startIcon={<ExecuteIcon />}
                >
                  Execute Signal
                </Button>
              )}
              <Button
                variant="outlined"
                color="error"
                onClick={() => {
                  if (selectedSignal) {
                    handleCancelSignal(selectedSignal.id);
                    handleCloseSignalDetails();
                  }
                }}
                startIcon={<CancelIcon />}
              >
                Cancel Signal
              </Button>
            </>
          )}
          <Button onClick={handleCloseSignalDetails}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Trade Details Dialog */}
      <Dialog
        open={tradeDetailsOpen}
        onClose={handleCloseTradeDetails}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">
              Trade Details
            </Typography>
            <IconButton onClick={handleCloseTradeDetails}>
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          {selectedTrade && (
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Asset:</strong> {selectedTrade.symbol}
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Direction:</strong>{' '}
                  <Chip
                    label={selectedTrade.direction.toUpperCase()}
                    color={selectedTrade.direction === 'buy' ? 'success' : 'error'}
                    size="small"
                  />
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Quantity:</strong> {selectedTrade.quantity.toLocaleString()}
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Price:</strong> {formatCurrency(selectedTrade.price)}
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Total:</strong> {formatCurrency(selectedTrade.total)}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Status:</strong>{' '}
                  <Chip
                    label={selectedTrade.status.charAt(0).toUpperCase() + selectedTrade.status.slice(1)}
                    color={getStatusColor(selectedTrade.status) as any}
                    size="small"
                  />
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Time:</strong> {formatTimestamp(selectedTrade.timestamp)}
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                  <strong>Fees:</strong> {formatCurrency(selectedTrade.fees)}
                </Typography>
                {selectedTrade.executionSpeed && (
                  <Typography variant="subtitle1" gutterBottom>
                    <strong>Execution Speed:</strong> {selectedTrade.executionSpeed}ms
                  </Typography>
                )}
                {selectedTrade.pnl !== undefined && (
                  <Typography variant="subtitle1" gutterBottom>
                    <strong>Profit/Loss:</strong>{' '}
                    <Typography
                      component="span"
                      color={selectedTrade.pnl >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatCurrency(selectedTrade.pnl)} ({selectedTrade.pnlPercent ? `${selectedTrade.pnlPercent >= 0 ? '+' : ''}${selectedTrade.pnlPercent.toFixed(2)}%` : ''})
                    </Typography>
                  </Typography>
                )}
              </Grid>
              {selectedTrade.notes && (
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    Notes
                  </Typography>
                  <Typography variant="body1">
                    {selectedTrade.notes}
                  </Typography>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseTradeDetails}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TradingPage;