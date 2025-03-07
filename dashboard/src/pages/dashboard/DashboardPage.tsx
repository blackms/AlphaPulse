import React, { useEffect, useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  Divider,
  CircularProgress,
  Alert,
} from '@mui/material';
import { useSocket } from '../../hooks/useSocket';
import { Metric, Portfolio, Alert as AlertType } from '../../types';

const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [alerts, setAlerts] = useState<AlertType[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { isConnected, addListener } = useSocket();
  
  useEffect(() => {
    // Fetch initial data
    const fetchData = async () => {
      try {
        setIsLoading(true);
        
        // In a real implementation, we would fetch data from the API
        // For now, we'll just simulate a delay
        await new Promise((resolve) => setTimeout(resolve, 1000));
        
        // Mock data
        setMetrics([
          {
            id: '1',
            name: 'Portfolio Value',
            value: 1250000,
            unit: 'USD',
            timestamp: new Date().toISOString(),
          },
          {
            id: '2',
            name: 'Daily P&L',
            value: 12500,
            unit: 'USD',
            timestamp: new Date().toISOString(),
          },
          {
            id: '3',
            name: 'Monthly Return',
            value: 3.2,
            unit: '%',
            timestamp: new Date().toISOString(),
          },
          {
            id: '4',
            name: 'Sharpe Ratio',
            value: 1.8,
            unit: '',
            timestamp: new Date().toISOString(),
          },
        ]);
        
        setPortfolio({
          id: '1',
          name: 'Main Portfolio',
          totalValue: 1250000,
          cash: 250000,
          pnl: 12500,
          pnlPercentage: 1.01,
          positions: [
            {
              id: '1',
              symbol: 'BTC',
              quantity: 5,
              entryPrice: 50000,
              currentPrice: 52000,
              pnl: 10000,
              pnlPercentage: 4,
              value: 260000,
              allocation: 20.8,
            },
            {
              id: '2',
              symbol: 'ETH',
              quantity: 50,
              entryPrice: 3000,
              currentPrice: 3050,
              pnl: 2500,
              pnlPercentage: 1.67,
              value: 152500,
              allocation: 12.2,
            },
          ],
          updatedAt: new Date().toISOString(),
        });
        
        setAlerts([
          {
            id: '1',
            message: 'Portfolio value increased by 5%',
            severity: 'info',
            source: 'portfolio',
            timestamp: new Date().toISOString(),
            acknowledged: false,
          },
          {
            id: '2',
            message: 'BTC position approaching stop loss',
            severity: 'warning',
            source: 'risk',
            timestamp: new Date().toISOString(),
            acknowledged: false,
          },
        ]);
        
        setError(null);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch dashboard data');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
    
    // Set up WebSocket listeners
    if (isConnected) {
      const metricsCleanup = addListener('metrics', (data: Metric) => {
        setMetrics((prevMetrics) => {
          const index = prevMetrics.findIndex((m) => m.id === data.id);
          if (index !== -1) {
            const newMetrics = [...prevMetrics];
            newMetrics[index] = data;
            return newMetrics;
          }
          return [...prevMetrics, data];
        });
      });
      
      const portfolioCleanup = addListener('portfolio', (data: Portfolio) => {
        setPortfolio(data);
      });
      
      const alertsCleanup = addListener('alerts', (data: AlertType) => {
        setAlerts((prevAlerts) => [data, ...prevAlerts]);
      });
      
      return () => {
        metricsCleanup();
        portfolioCleanup();
        alertsCleanup();
      };
    }
  }, [isConnected, addListener]);
  
  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }
  
  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {metrics.map((metric) => (
          <Grid item xs={12} sm={6} md={3} key={metric.id}>
            <Paper
              sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                height: 140,
              }}
            >
              <Typography variant="h6" color="text.secondary">
                {metric.name}
              </Typography>
              <Typography
                component="p"
                variant="h4"
                sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}
              >
                {metric.value.toLocaleString()}
                {metric.unit && (
                  <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                    {metric.unit}
                  </Typography>
                )}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Last updated: {new Date(metric.timestamp).toLocaleTimeString()}
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>
      
      {/* Portfolio Summary */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardHeader title="Portfolio Summary" />
            <Divider />
            <CardContent>
              {portfolio ? (
                <Box>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Total Value
                      </Typography>
                      <Typography variant="h6">
                        ${portfolio.totalValue.toLocaleString()}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Cash
                      </Typography>
                      <Typography variant="h6">
                        ${portfolio.cash.toLocaleString()}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        P&L
                      </Typography>
                      <Typography
                        variant="h6"
                        color={portfolio.pnl >= 0 ? 'success.main' : 'error.main'}
                      >
                        ${portfolio.pnl.toLocaleString()} ({portfolio.pnlPercentage.toFixed(2)}%)
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Positions
                      </Typography>
                      <Typography variant="h6">{portfolio.positions.length}</Typography>
                    </Grid>
                  </Grid>
                </Box>
              ) : (
                <Typography>No portfolio data available</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Recent Alerts */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardHeader title="Recent Alerts" />
            <Divider />
            <CardContent>
              {alerts.length > 0 ? (
                <Box>
                  {alerts.slice(0, 3).map((alert) => (
                    <Box key={alert.id} sx={{ mb: 2 }}>
                      <Alert
                        severity={
                          alert.severity === 'critical'
                            ? 'error'
                            : alert.severity === 'warning'
                            ? 'warning'
                            : 'info'
                        }
                        sx={{ mb: 1 }}
                      >
                        {alert.message}
                      </Alert>
                      <Typography variant="caption" color="text.secondary">
                        {new Date(alert.timestamp).toLocaleString()}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              ) : (
                <Typography>No recent alerts</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;