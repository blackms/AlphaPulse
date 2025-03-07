import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Grid,
  Typography,
  Box,
  Container,
  Paper,
  CircularProgress,
  Alert,
  useTheme,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  AccountBalance as AccountBalanceIcon,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { useSocket } from '../../hooks/useSocket';
import { 
  MetricCard, 
  AlertsWidget, 
  PortfolioSummaryWidget, 
  TradingActivityWidget, 
  SystemStatusWidget 
} from '../../components/widgets';
import { LineChart } from '../../components/charts';
import { 
  Metric, 
  Portfolio, 
  Alert as AlertType, 
  Trade, 
  SystemStatus, 
  SystemMetrics 
} from '../../types';

const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  
  // State for data
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [alerts, setAlerts] = useState<AlertType[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  
  // State for UI
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // WebSocket connection
  const { isConnected, addListener } = useSocket();
  
  // Performance data for chart
  const [performanceData, setPerformanceData] = useState<{ timestamp: string; value: number }[]>([]);
  
  useEffect(() => {
    // Fetch initial data
    const fetchData = async () => {
      try {
        setIsLoading(true);
        
        // In a real implementation, we would fetch data from the API
        // For now, we'll just simulate a delay
        await new Promise((resolve) => setTimeout(resolve, 1000));
        
        // Mock metrics data
        setMetrics([
          {
            id: '1',
            name: 'Portfolio Value',
            value: 1250000,
            unit: 'USD',
            timestamp: new Date().toISOString(),
            tags: { type: 'portfolio' },
          },
          {
            id: '2',
            name: 'Daily P&L',
            value: 12500,
            unit: 'USD',
            timestamp: new Date().toISOString(),
            tags: { type: 'portfolio' },
          },
          {
            id: '3',
            name: 'Monthly Return',
            value: 3.2,
            unit: '%',
            timestamp: new Date().toISOString(),
            tags: { type: 'performance' },
          },
          {
            id: '4',
            name: 'Sharpe Ratio',
            value: 1.8,
            unit: '',
            timestamp: new Date().toISOString(),
            tags: { type: 'risk' },
          },
        ]);
        
        // Mock portfolio data
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
              tags: ['crypto'],
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
              tags: ['crypto'],
            },
            {
              id: '3',
              symbol: 'SOL',
              quantity: 1000,
              entryPrice: 100,
              currentPrice: 98,
              pnl: -2000,
              pnlPercentage: -2,
              value: 98000,
              allocation: 7.84,
              tags: ['crypto'],
            },
            {
              id: '4',
              symbol: 'AAPL',
              quantity: 200,
              entryPrice: 150,
              currentPrice: 155,
              pnl: 1000,
              pnlPercentage: 3.33,
              value: 31000,
              allocation: 2.48,
              tags: ['stock'],
            },
          ],
          updatedAt: new Date().toISOString(),
        });
        
        // Mock alerts data
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
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            acknowledged: false,
          },
          {
            id: '3',
            message: 'System CPU usage above 90%',
            severity: 'critical',
            source: 'system',
            timestamp: new Date(Date.now() - 7200000).toISOString(),
            acknowledged: true,
            acknowledgedBy: 'admin',
            acknowledgedAt: new Date(Date.now() - 7000000).toISOString(),
          },
        ]);
        
        // Mock trades data
        setTrades([
          {
            id: '1',
            symbol: 'BTC',
            type: 'buy',
            quantity: 0.5,
            price: 52000,
            timestamp: new Date(Date.now() - 1800000).toISOString(),
            status: 'executed',
            total: 26000,
            executedBy: 'system',
            signalSource: 'technical',
          },
          {
            id: '2',
            symbol: 'ETH',
            type: 'sell',
            quantity: 5,
            price: 3050,
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            status: 'executed',
            total: 15250,
            executedBy: 'system',
            signalSource: 'rebalance',
          },
          {
            id: '3',
            symbol: 'SOL',
            type: 'buy',
            quantity: 100,
            price: 98,
            timestamp: new Date(Date.now() - 5400000).toISOString(),
            status: 'pending',
            total: 9800,
            executedBy: 'system',
            signalSource: 'value',
          },
        ]);
        
        // Mock system status data
        setSystemStatus({
          status: 'operational',
          message: 'All systems operational',
          components: {
            'Data Pipeline': {
              status: 'operational',
              message: 'Processing data normally',
            },
            'Trading Engine': {
              status: 'operational',
              message: 'Executing trades normally',
            },
            'Risk Management': {
              status: 'operational',
              message: 'Monitoring positions',
            },
            'API Services': {
              status: 'degraded',
              message: 'Experiencing higher latency',
            },
          },
          updatedAt: new Date().toISOString(),
        });
        
        // Mock system metrics data
        setSystemMetrics({
          cpu: 65,
          memory: 72,
          disk: 48,
          network: {
            in: 2.5,
            out: 1.8,
          },
          processCount: 42,
          uptime: 345600, // 4 days in seconds
        });
        
        // Mock performance data for chart
        const now = new Date();
        const performanceHistory = Array.from({ length: 30 }, (_, i) => {
          const date = new Date(now);
          date.setDate(date.getDate() - (29 - i));
          
          // Generate a somewhat realistic performance curve
          const baseValue = 1200000;
          const dayFactor = i / 29; // 0 to 1
          const trendValue = baseValue * (1 + dayFactor * 0.05); // 5% increase over period
          
          // Add some randomness
          const randomFactor = 0.98 + Math.random() * 0.04; // ±2%
          const value = trendValue * randomFactor;
          
          return {
            timestamp: date.toISOString(),
            value: Math.round(value),
          };
        });
        
        setPerformanceData(performanceHistory);
        
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
      
      const tradesCleanup = addListener('trades', (data: Trade) => {
        setTrades((prevTrades) => [data, ...prevTrades]);
      });
      
      const systemCleanup = addListener('system', (data: any) => {
        if (data.type === 'status') {
          setSystemStatus(data.data);
        } else if (data.type === 'metrics') {
          setSystemMetrics(data.data);
        }
      });
      
      return () => {
        metricsCleanup();
        portfolioCleanup();
        alertsCleanup();
        tradesCleanup();
        systemCleanup();
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
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }
  
  // Extract key metrics for the top cards
  const portfolioValueMetric = metrics.find((m) => m.name === 'Portfolio Value');
  const dailyPnLMetric = metrics.find((m) => m.name === 'Daily P&L');
  const monthlyReturnMetric = metrics.find((m) => m.name === 'Monthly Return');
  const sharpeRatioMetric = metrics.find((m) => m.name === 'Sharpe Ratio');
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Portfolio Value"
            value={portfolioValueMetric?.value || 0}
            unit="USD"
            percentChange={portfolio?.pnlPercentage || 0}
            icon={<AccountBalanceIcon />}
            onClick={() => navigate('/portfolio')}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Daily P&L"
            value={dailyPnLMetric?.value || 0}
            unit="USD"
            percentChange={dailyPnLMetric ? (dailyPnLMetric.value / (portfolio?.totalValue || 1)) * 100 : 0}
            icon={<TrendingUpIcon />}
            onClick={() => navigate('/portfolio')}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Monthly Return"
            value={monthlyReturnMetric?.value || 0}
            unit="%"
            icon={<TimelineIcon />}
            onClick={() => navigate('/portfolio')}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Sharpe Ratio"
            value={sharpeRatioMetric?.value || 0}
            icon={<SpeedIcon />}
            onClick={() => navigate('/portfolio')}
          />
        </Grid>
      </Grid>
      
      {/* Portfolio Performance Chart */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Portfolio Performance
            </Typography>
            <Box height={300}>
              <LineChart
                data={performanceData}
                title="Portfolio Value"
                yAxisLabel="Value (USD)"
                color={theme.palette.primary.main}
                showGrid={true}
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>
      
      {/* Portfolio and Alerts */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <PortfolioSummaryWidget
            portfolio={portfolio}
            onViewDetails={() => navigate('/portfolio')}
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <AlertsWidget
            alerts={alerts}
            onViewAll={() => navigate('/alerts')}
            onAcknowledge={(alertId) => {
              // In a real app, this would call an API
              setAlerts((prevAlerts) =>
                prevAlerts.map((alert) =>
                  alert.id === alertId
                    ? {
                        ...alert,
                        acknowledged: true,
                        acknowledgedBy: 'current_user',
                        acknowledgedAt: new Date().toISOString(),
                      }
                    : alert
                )
              );
            }}
            onAlertClick={(alert) => navigate(`/alerts/${alert.id}`)}
          />
        </Grid>
      </Grid>
      
      {/* Trading Activity and System Status */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <TradingActivityWidget
            trades={trades}
            onViewAll={() => navigate('/trading')}
            onTradeClick={(trade) => navigate(`/trading/${trade.id}`)}
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <SystemStatusWidget
            status={systemStatus}
            metrics={systemMetrics}
            onViewDetails={() => navigate('/system')}
          />
        </Grid>
      </Grid>
    </Container>
  );
};

export default DashboardPage;