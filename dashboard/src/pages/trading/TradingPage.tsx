import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  Button,
  Chip,
  Divider,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import AutorenewIcon from '@mui/icons-material/Autorenew';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';
import SyncAltIcon from '@mui/icons-material/SyncAlt';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import PendingOutlinedIcon from '@mui/icons-material/PendingOutlined';

// Redux
import {
  fetchSignalsStart,
  fetchOrdersStart,
  selectAllSignals,
  selectRecentSignals,
  selectRecentOrders,
  selectAgentSettings,
  updateOrderStatus,
  AgentType,
  Signal,
  Order,
  OrderStatus
} from '../../store/slices/tradingSlice';

// Placeholder components
const SignalTablePlaceholder = () => (
  <Paper sx={{ p: 2, height: '100%' }}>
    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
      <Typography variant="h6">Recent Signals</Typography>
      <Button startIcon={<RefreshIcon />} size="small">Refresh</Button>
    </Box>
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Time</TableCell>
            <TableCell>Asset</TableCell>
            <TableCell>Direction</TableCell>
            <TableCell>Confidence</TableCell>
            <TableCell>Source</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          <TableRow>
            <TableCell colSpan={6} align="center">
              <CircularProgress size={24} sx={{ my: 2 }} />
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </TableContainer>
  </Paper>
);

const OrderTablePlaceholder = () => (
  <Paper sx={{ p: 2, height: '100%' }}>
    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
      <Typography variant="h6">Recent Orders</Typography>
      <Button startIcon={<RefreshIcon />} size="small">Refresh</Button>
    </Box>
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Time</TableCell>
            <TableCell>Asset</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Side</TableCell>
            <TableCell>Quantity</TableCell>
            <TableCell>Price</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          <TableRow>
            <TableCell colSpan={7} align="center">
              <CircularProgress size={24} sx={{ my: 2 }} />
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </TableContainer>
  </Paper>
);

const TradingActivityWidgetPlaceholder = () => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Typography variant="h6" gutterBottom>Trading Activity</Typography>
      <Box sx={{ height: 200, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <CircularProgress />
      </Box>
    </CardContent>
  </Card>
);

const AssetPerformanceWidgetPlaceholder = () => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Typography variant="h6" gutterBottom>Asset Performance</Typography>
      <Box sx={{ height: 200, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <CircularProgress />
      </Box>
    </CardContent>
  </Card>
);

const AgentSettingsPanelPlaceholder = () => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Typography variant="h6" gutterBottom>Agent Settings</Typography>
      <Divider sx={{ my: 2 }} />
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {['Technical', 'Fundamental', 'Sentiment', 'Value', 'Activist'].map((agent) => (
          <Box key={agent} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography>{agent} Agent</Typography>
            <Chip label="Enabled" color="success" size="small" />
          </Box>
        ))}
      </Box>
    </CardContent>
  </Card>
);

// Helper component for progress bars
const LinearProgressWithLabel = (props: { value: number }) => {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <Box sx={{ width: '100%', mr: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
        <Box
          sx={{
            height: 10,
            borderRadius: 1,
            bgcolor: 
              props.value < 30 ? 'error.main' :
              props.value < 70 ? 'warning.main' : 'success.main',
            width: `${props.value}%`,
          }}
        />
      </Box>
      <Box sx={{ minWidth: 35 }}>
        <Typography variant="body2" color="text.secondary">
          {`${Math.round(props.value)}%`}
        </Typography>
      </Box>
    </Box>
  );
};

const TradingPage: React.FC = () => {
  const dispatch = useDispatch();
  
  // Get trading data from Redux store
  const recentSignals = useSelector(selectRecentSignals);
  const allSignals = useSelector(selectAllSignals);
  const recentOrders = useSelector(selectRecentOrders);
  const agentSettings = useSelector(selectAgentSettings);
  
  // Local state
  const [tabValue, setTabValue] = useState(0);
  const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // Fetch data on component mount
  useEffect(() => {
    dispatch(fetchSignalsStart());
    dispatch(fetchOrdersStart());
  }, [dispatch]);
  
  // Tab change handler
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Signal selection handler
  const handleSignalSelect = (signal: Signal) => {
    setSelectedSignal(signal);
  };
  
  // Order status update handler
  const handleOrderStatusUpdate = (orderId: string, newStatus: OrderStatus) => {
    dispatch(updateOrderStatus({
      orderId,
      status: newStatus,
      updatedAt: Date.now()
    }));
  };
  
  return (
    <Box sx={{ p: 0 }}>
      <Grid container spacing={3}>
        {/* Header and tabs */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h5">Trading Dashboard</Typography>
                <Box>
                  <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    sx={{ mr: 2 }}
                  >
                    Refresh Data
                  </Button>
                  <Button
                    variant="contained"
                    startIcon={<AutorenewIcon />}
                    color="primary"
                  >
                    Auto Trading: ON
                  </Button>
                </Box>
              </Box>
              
              <Tabs
                value={tabValue}
                onChange={handleTabChange}
                variant="scrollable"
                scrollButtons="auto"
                aria-label="trading dashboard tabs"
              >
                <Tab label="Overview" />
                <Tab label="Signals" />
                <Tab label="Orders" />
                <Tab label="Positions" />
                <Tab label="Settings" />
              </Tabs>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Overview tab */}
        {tabValue === 0 && (
          <>
            {/* Activity widgets */}
            <Grid item xs={12} md={8}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <TradingActivityWidgetPlaceholder />
                </Grid>
                <Grid item xs={12}>
                  <AssetPerformanceWidgetPlaceholder />
                </Grid>
              </Grid>
            </Grid>
            
            {/* Agent settings and recent signals */}
            <Grid item xs={12} md={4}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <AgentSettingsPanelPlaceholder />
                </Grid>
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>Signal Summary</Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="h4" color="success.main">
                            {recentSignals.filter(s => s.direction === 'buy').length}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">Buy</Typography>
                        </Box>
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="h4" color="error.main">
                            {recentSignals.filter(s => s.direction === 'sell').length}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">Sell</Typography>
                        </Box>
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="h4" color="text.secondary">
                            {recentSignals.filter(s => s.direction === 'hold').length}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">Hold</Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Grid>
            
            {/* Recent signals and orders */}
            <Grid item xs={12} md={6}>
              <SignalTablePlaceholder />
            </Grid>
            <Grid item xs={12} md={6}>
              <OrderTablePlaceholder />
            </Grid>
          </>
        )}
        
        {/* Signals tab */}
        {tabValue === 1 && (
          <>
            <Grid item xs={12} md={selectedSignal ? 8 : 12}>
              <SignalTablePlaceholder />
            </Grid>
            
            {selectedSignal && (
              <Grid item xs={12} md={4}>
                <Paper sx={{ p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>Signal Details</Typography>
                  <Divider sx={{ mb: 2 }} />
                  
                  <Typography variant="subtitle1">
                    {selectedSignal.asset} {selectedSignal.direction.toUpperCase()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {new Date(selectedSignal.timestamp).toLocaleString()}
                  </Typography>
                  
                  <Box sx={{ my: 2 }}>
                    <Typography variant="subtitle2">Confidence</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                      <Box sx={{ width: '100%', mr: 1 }}>
                        <LinearProgressWithLabel value={selectedSignal.confidence * 100} />
                      </Box>
                    </Box>
                  </Box>
                  
                  <Box sx={{ my: 2 }}>
                    <Typography variant="subtitle2">Source</Typography>
                    <Chip
                      label={selectedSignal.source}
                      color={
                        selectedSignal.source === 'technical' ? 'primary' :
                        selectedSignal.source === 'fundamental' ? 'secondary' :
                        selectedSignal.source === 'sentiment' ? 'info' :
                        selectedSignal.source === 'value' ? 'success' : 'warning'
                      }
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  </Box>
                  
                  <Box sx={{ my: 2 }}>
                    <Typography variant="subtitle2">Metadata</Typography>
                    <pre style={{ overflow: 'auto', maxHeight: 200 }}>
                      {JSON.stringify(selectedSignal.metadata, null, 2)}
                    </pre>
                  </Box>
                  
                  <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                    <Button variant="outlined" onClick={() => setSelectedSignal(null)}>
                      Close
                    </Button>
                  </Box>
                </Paper>
              </Grid>
            )}
          </>
        )}
        
        {/* Orders tab */}
        {tabValue === 2 && (
          <Grid item xs={12}>
            <OrderTablePlaceholder />
          </Grid>
        )}
        
        {/* Positions tab */}
        {tabValue === 3 && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>Open Positions</Typography>
              <Alert severity="info" sx={{ mb: 2 }}>
                This feature is under development.
              </Alert>
            </Paper>
          </Grid>
        )}
        
        {/* Settings tab */}
        {tabValue === 4 && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>Trading Settings</Typography>
              <Alert severity="info" sx={{ mb: 2 }}>
                This feature is under development.
              </Alert>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default TradingPage;