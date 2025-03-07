import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  CardHeader,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  CircularProgress,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  ErrorOutline as ErrorIcon,
  WarningAmber as WarningIcon,
  Info as InfoIcon,
  CheckCircle as CheckIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreVertIcon,
} from '@mui/icons-material';
import { 
  selectAlerts, 
  selectUnreadAlertCount,
  Alert,
  AlertSeverity
} from '../../store/slices/alertsSlice';
import {
  selectPerformance,
  selectAssets,
  selectHistoricalValues,
  selectTotalValue,
  fetchPortfolioStart,
} from '../../store/slices/portfolioSlice';
import {
  selectSystemMetrics,
  selectSystemComponents,
  fetchSystemStart,
} from '../../store/slices/systemSlice';
import {
  LineChart,
  ResponsiveContainer,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A569BD', '#5DADE2', '#48C9B0', '#F4D03F'];

const DashboardPage: React.FC = () => {
  const dispatch = useDispatch();
  const alerts = useSelector(selectAlerts);
  const unreadAlertCount = useSelector(selectUnreadAlertCount);
  const performance = useSelector(selectPerformance);
  const assets = useSelector(selectAssets);
  const historicalValues = useSelector(selectHistoricalValues);
  const totalValue = useSelector(selectTotalValue);
  
  // Derived data structures
  const portfolioPerformance = {
    currentValue: totalValue,
    totalReturn: performance.find(p => p.period === 'all')?.returnPercent || 0,
    startDate: performance.find(p => p.period === 'all')?.startDate || 'N/A',
    daily: performance.find(p => p.period === 'day')?.returnPercent || 0,
    history: historicalValues.map(h => ({
      date: new Date(h.timestamp).toLocaleDateString(),
      value: h.value,
      benchmark: h.value * 0.9 // Mock benchmark data
    }))
  };
  
  const portfolioAllocation = assets.map(asset => ({
    asset: asset.symbol,
    value: asset.allocation
  }));
  
  const portfolioPositions = assets.map(asset => ({
    asset: asset.symbol,
    size: asset.quantity,
    entryPrice: asset.costBasis,
    currentPrice: asset.price,
    pnlPercentage: asset.unrealizedPnLPercent,
    allocation: asset.allocation,
    status: asset.unrealizedPnLPercent >= 0 ? 'active' : 'warning'
  }));
  const systemMetrics = useSelector(selectSystemMetrics);
  const systemComponents = useSelector(selectSystemComponents);

  useEffect(() => {
    dispatch(fetchPortfolioStart());
    dispatch(fetchSystemStart());
    
    // Set up polling every 30 seconds
    const interval = setInterval(() => {
      dispatch(fetchPortfolioStart());
      dispatch(fetchSystemStart());
    }, 30000);
    
    return () => clearInterval(interval);
  }, [dispatch]);

  const refreshData = () => {
    dispatch(fetchPortfolioStart());
    dispatch(fetchSystemStart());
  };

  // Filter recent alerts (last 24 hours)
  const recentAlerts = alerts
    .filter(alert => !alert.read && new Date(alert.timestamp).getTime() > Date.now() - 24 * 60 * 60 * 1000)
    .slice(0, 5);

  // Get overall system health percentage
  const systemHealth = systemComponents.length > 0
    ? Math.floor(systemComponents.reduce((sum, comp) => sum + comp.healthScore, 0) / systemComponents.length)
    : 0;

  // Get daily portfolio performance
  const portfolioDaily = portfolioPerformance?.daily || 0;

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <ErrorIcon color="error" />;
      case 'high':
        return <ErrorIcon color="error" />;
      case 'medium':
        return <WarningIcon color="warning" />;
      case 'low':
        return <InfoIcon color="info" />;
      default:
        return <InfoIcon />;
    }
  };
  
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'error';
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Dashboard
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={refreshData}
        >
          Refresh
        </Button>
      </Box>

      {/* Key Performance Indicators */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Portfolio Value
            </Typography>
            <Typography variant="h4">
              ${portfolioPerformance?.currentValue.toLocaleString()}
            </Typography>
            <Box display="flex" alignItems="center" mt={1}>
              <Typography 
                variant="body2" 
                color={portfolioDaily >= 0 ? 'success.main' : 'error.main'}
                sx={{ display: 'flex', alignItems: 'center' }}
              >
                {portfolioDaily >= 0 ? <TrendingUpIcon fontSize="small" sx={{ mr: 0.5 }} /> : <TrendingDownIcon fontSize="small" sx={{ mr: 0.5 }} />}
                {portfolioDaily >= 0 ? '+' : ''}{portfolioDaily.toFixed(2)}% Today
              </Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Total Return
            </Typography>
            <Typography variant="h4">
              {portfolioPerformance?.totalReturn >= 0 ? '+' : ''}{portfolioPerformance?.totalReturn.toFixed(2)}%
            </Typography>
            <Box display="flex" alignItems="center" mt={1}>
              <Typography 
                variant="body2" 
                color="text.secondary"
              >
                Since inception ({portfolioPerformance?.startDate})
              </Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              System Health
            </Typography>
            <Typography variant="h4">
              {systemHealth}%
            </Typography>
            <Box display="flex" alignItems="center" mt={1}>
              <LinearProgress 
                variant="determinate" 
                value={systemHealth} 
                color={systemHealth > 90 ? 'success' : systemHealth > 75 ? 'info' : systemHealth > 50 ? 'warning' : 'error'} 
                sx={{ width: '100%', height: 8, borderRadius: 5 }} 
              />
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Active Alerts
            </Typography>
            <Typography variant="h4">
              {unreadAlertCount}
            </Typography>
            <Box display="flex" alignItems="center" mt={1}>
              <Typography 
                variant="body2" 
                color={unreadAlertCount > 5 ? 'error.main' : unreadAlertCount > 2 ? 'warning.main' : 'info.main'}
              >
                {unreadAlertCount === 0 ? 'All clear' : unreadAlertCount > 5 ? 'Critical attention needed' : 'Requires attention'}
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Portfolio Performance Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardHeader 
              title="Portfolio Performance" 
              action={
                <IconButton>
                  <MoreVertIcon />
                </IconButton>
              } 
            />
            <Divider />
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={portfolioPerformance?.history || []}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <RechartsTooltip />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#8884d8" 
                    activeDot={{ r: 8 }} 
                    name="Portfolio Value"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="benchmark" 
                    stroke="#82ca9d"
                    name="Benchmark"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Alerts */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader 
              title="Recent Alerts" 
              action={
                <Button size="small" href="/alerts">
                  View All
                </Button>
              } 
            />
            <Divider />
            <CardContent sx={{ p: 0 }}>
              <List>
                {recentAlerts.length === 0 ? (
                  <ListItem>
                    <ListItemText 
                      primary="No recent alerts" 
                      secondary="All systems operating normally"
                    />
                  </ListItem>
                ) : (
                  recentAlerts.map((alert) => (
                    <React.Fragment key={alert.id}>
                      <ListItem 
                        alignItems="flex-start"
                        secondaryAction={
                          <Chip 
                            label={alert.severity.toUpperCase()} 
                            size="small" 
                            color={getSeverityColor(alert.severity) as any}
                          />
                        }
                      >
                        <ListItemIcon>
                          {getSeverityIcon(alert.severity)}
                        </ListItemIcon>
                        <ListItemText
                          primary={alert.title}
                          secondary={
                            <React.Fragment>
                              <Typography
                                sx={{ display: 'inline' }}
                                component="span"
                                variant="body2"
                                color="text.primary"
                              >
                                {alert.type}
                              </Typography>
                              {` — ${alert.message}`}
                            </React.Fragment>
                          }
                        />
                      </ListItem>
                      <Divider variant="inset" component="li" />
                    </React.Fragment>
                  ))
                )}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Portfolio Allocation */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Portfolio Allocation" />
            <Divider />
            <CardContent>
              <Box display="flex" justifyContent="center">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={portfolioAllocation || []}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                      nameKey="asset"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {(portfolioAllocation || []).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip formatter={(value, name) => [`${value}%`, name]} />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* System Metrics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="System Metrics" />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                {systemMetrics.slice(0, 6).map((metric) => (
                  <Grid item xs={12} sm={6} key={metric.id}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                      <Typography variant="body2" color="text.secondary">
                        {metric.name}
                      </Typography>
                      <Chip
                        label={metric.status}
                        size="small"
                        color={
                          metric.status === 'good' ? 'success' :
                          metric.status === 'warning' ? 'warning' : 'error'
                        }
                      />
                    </Box>
                    <Typography variant="h6">
                      {typeof metric.value === 'number' 
                        ? metric.value.toLocaleString()
                        : metric.value} {metric.unit}
                    </Typography>
                    {metric.change !== undefined && (
                      <Typography 
                        variant="caption" 
                        color={metric.change >= 0 ? 'success.main' : 'error.main'}
                        display="block"
                      >
                        {metric.change >= 0 ? '+' : ''}{metric.changePercent?.toFixed(1)}% from previous
                      </Typography>
                    )}
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Open Positions */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Current Positions" />
            <Divider />
            <Box sx={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Asset</th>
                    <th style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Position Size</th>
                    <th style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Entry Price</th>
                    <th style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Current Price</th>
                    <th style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>P&L</th>
                    <th style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Allocation</th>
                    <th style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(portfolioPositions || []).map((position) => (
                    <tr key={position.asset}>
                      <td style={{ padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                        <Box display="flex" alignItems="center">
                          <Typography variant="body1">{position.asset}</Typography>
                        </Box>
                      </td>
                      <td style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                        {position.size.toFixed(4)}
                      </td>
                      <td style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                        ${position.entryPrice.toLocaleString()}
                      </td>
                      <td style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                        ${position.currentPrice.toLocaleString()}
                      </td>
                      <td 
                        style={{ 
                          textAlign: 'right', 
                          padding: '12px 16px', 
                          borderBottom: '1px solid rgba(224, 224, 224, 1)',
                          color: position.pnlPercentage >= 0 ? '#2e7d32' : '#d32f2f',
                        }}
                      >
                        {position.pnlPercentage >= 0 ? '+' : ''}{position.pnlPercentage.toFixed(2)}%
                      </td>
                      <td style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                        {position.allocation.toFixed(2)}%
                      </td>
                      <td style={{ textAlign: 'right', padding: '12px 16px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                        <Chip
                          label={position.status}
                          size="small"
                          color={
                            position.status === 'active' ? 'success' :
                            position.status === 'pending' ? 'warning' : 'error'
                          }
                        />
                      </td>
                    </tr>
                  ))}
                  {(!portfolioPositions || portfolioPositions.length === 0) && (
                    <tr>
                      <td colSpan={7} style={{ textAlign: 'center', padding: '24px 16px' }}>
                        <Typography variant="body1" color="text.secondary">
                          No open positions
                        </Typography>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </Box>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;