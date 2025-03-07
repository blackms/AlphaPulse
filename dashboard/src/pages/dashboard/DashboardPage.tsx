import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Divider,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { RootState } from '../../store/store';
import { Trade } from '../../store/slices/tradingSlice';
import { Alert } from '../../types/alerts';

// Placeholder chart component
const Chart = ({ type, data }: { type: string; data: any }) => (
  <Box height={200} display="flex" alignItems="center" justifyContent="center">
    <Typography>{type} Chart (Placeholder)</Typography>
  </Box>
);

const DashboardPage: React.FC = () => {
  const dispatch = useDispatch();
  
  // Get state from Redux store
  const { performance } = useSelector((state: RootState) => state.metrics);
  const portfolio = useSelector((state: RootState) => state.portfolio);
  const { trades } = useSelector((state: RootState) => state.trading);
  const systemStatus = useSelector((state: RootState) => state.system.status);
  const { alerts } = useSelector((state: RootState) => state.alerts);

  useEffect(() => {
    // Placeholder for fetching dashboard data
    // dispatch(fetchDashboardData());
  }, [dispatch]);

  const handleRefresh = () => {
    // Placeholder for refreshing dashboard data
    // dispatch(fetchDashboardData());
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

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'operational':
        return <CheckCircleIcon color="success" />;
      case 'degraded':
        return <WarningIcon color="warning" />;
      case 'maintenance':
        return <InfoIcon color="info" />;
      case 'outage':
        return <ErrorIcon color="error" />;
      default:
        return <InfoIcon />;
    }
  };

  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'info':
        return <InfoIcon color="info" />;
      case 'success':
        return <CheckCircleIcon color="success" />;
      default:
        return <InfoIcon />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Dashboard
        </Typography>
        <IconButton onClick={handleRefresh}>
          <RefreshIcon />
        </IconButton>
      </Box>

      <Grid container spacing={3}>
        {/* Performance Summary */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardHeader title="Portfolio Performance" />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                      Daily
                    </Typography>
                    <Typography
                      variant="h5"
                      color={performance.daily >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercentage(performance.daily)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                      Weekly
                    </Typography>
                    <Typography
                      variant="h5"
                      color={performance.weekly >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercentage(performance.weekly)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                      Monthly
                    </Typography>
                    <Typography
                      variant="h5"
                      color={performance.monthly >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercentage(performance.monthly)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                      Yearly
                    </Typography>
                    <Typography
                      variant="h5"
                      color={performance.yearly >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercentage(performance.yearly)}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
              
              <Box mt={3}>
                <Chart type="Performance" data={[]} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* System Status */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardHeader title="System Status" />
            <Divider />
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Box mr={1}>{getStatusIcon(systemStatus.overall)}</Box>
                <Typography variant="h6">
                  System is {systemStatus.overall}
                </Typography>
              </Box>
              {systemStatus.message && (
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  {systemStatus.message}
                </Typography>
              )}
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" gutterBottom>
                Resources
              </Typography>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2">CPU</Typography>
                <Typography variant="body2">
                  {portfolio.totalValue > 0 
                    ? formatPercentage(Math.random() * 30 + 10) 
                    : '10.5%'}
                </Typography>
              </Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2">Memory</Typography>
                <Typography variant="body2">
                  {portfolio.totalValue > 0 
                    ? formatPercentage(Math.random() * 40 + 20) 
                    : '24.2%'}
                </Typography>
              </Box>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2">Disk</Typography>
                <Typography variant="body2">
                  {portfolio.totalValue > 0 
                    ? formatPercentage(Math.random() * 20 + 5) 
                    : '8.3%'}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Portfolio Summary */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Portfolio Summary" />
            <Divider />
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="subtitle1">Total Value</Typography>
                <Typography variant="h6">{formatCurrency(portfolio.totalValue)}</Typography>
              </Box>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="subtitle1">Cash Balance</Typography>
                <Typography variant="h6">{formatCurrency(portfolio.cashBalance)}</Typography>
              </Box>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="subtitle1">Positions</Typography>
                <Typography variant="h6">{portfolio.positions?.length || 0}</Typography>
              </Box>
              <Box mt={2}>
                <Chart type="Allocation" data={[]} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Trades */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Recent Trades" />
            <Divider />
            <CardContent>
              {trades && trades.length > 0 ? (
                <List dense>
                  {trades.slice(0, 5).map((trade: Trade) => (
                    <ListItem key={trade.id} divider>
                      <ListItemText
                        primary={`${trade.symbol} - ${formatCurrency(trade.price)}`}
                        secondary={new Date(trade.timestamp).toLocaleString()}
                      />
                      <Chip
                        label={trade.direction.toUpperCase()}
                        color={trade.direction === 'buy' ? 'success' : 'error'}
                        size="small"
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography align="center">No recent trades</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Alerts */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Recent Alerts" />
            <Divider />
            <CardContent>
              {alerts && alerts.length > 0 ? (
                <List dense>
                  {alerts.slice(0, 5).map((alert: Alert) => (
                    <ListItem key={alert.id}>
                      <ListItemIcon>{getAlertIcon(alert.severity)}</ListItemIcon>
                      <ListItemText
                        primary={alert.title}
                        secondary={new Date(alert.timestamp).toLocaleString()}
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography align="center">No alerts</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;