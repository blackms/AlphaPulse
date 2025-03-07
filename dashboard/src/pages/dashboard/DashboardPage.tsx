import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Divider,
  Chip,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  AttachMoney as MoneyIcon,
  Timeline as TimelineIcon,
  Notifications as NotificationsIcon,
  ShowChart as ChartIcon,
  Memory as AgentIcon,
  Security as RiskIcon,
  AccountBalance as PortfolioIcon,
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { selectTotalValue, selectAssets, selectPerformance } from '../../store/slices/portfolioSlice';
import { selectActiveSignals, selectRecentTrades } from '../../store/slices/tradingSlice';
import { selectSystemStatus, selectSystemComponents } from '../../store/slices/systemSlice';
import { selectAlerts, selectUnreadCount } from '../../store/slices/alertsSlice';

const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  
  // Portfolio data
  const portfolioValue = useSelector(selectTotalValue);
  const assets = useSelector(selectAssets);
  const performance = useSelector(selectPerformance);
  
  // Trading data
  const activeSignals = useSelector(selectActiveSignals);
  const recentTrades = useSelector(selectRecentTrades);
  
  // System data
  const systemStatus = useSelector(selectSystemStatus);
  const systemComponents = useSelector(selectSystemComponents);
  
  // Alerts data
  const alerts = useSelector(selectAlerts);
  const unreadAlerts = useSelector(selectUnreadCount);
  
  // Helper functions
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
  
  const getMonthlyPerformance = () => {
    return performance.find(p => p.period === 'month') || { returnValue: 0, returnPercent: 0 };
  };
  
  // Navigation handlers
  const navigateToPortfolio = () => navigate('/dashboard/portfolio');
  const navigateToTrading = () => navigate('/dashboard/trading');
  const navigateToAlerts = () => navigate('/dashboard/alerts');
  const navigateToSystem = () => navigate('/dashboard/system');
  
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        AI Hedge Fund Dashboard
      </Typography>
      
      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <MoneyIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6">Portfolio Value</Typography>
              </Box>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {formatCurrency(portfolioValue)}
              </Typography>
              <Box display="flex" alignItems="center">
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
              <Button 
                variant="text" 
                size="small" 
                sx={{ mt: 1 }}
                onClick={navigateToPortfolio}
              >
                View Portfolio
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <TimelineIcon sx={{ mr: 1, color: 'info.main' }} />
                <Typography variant="h6">Trading</Typography>
              </Box>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {activeSignals.length} Signals
              </Typography>
              <Box display="flex" alignItems="center">
                <Chip
                  label={`${recentTrades.length} Recent Trades`}
                  color="info"
                  size="small"
                />
              </Box>
              <Button 
                variant="text" 
                size="small" 
                sx={{ mt: 1 }}
                onClick={navigateToTrading}
              >
                View Trading
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <NotificationsIcon sx={{ mr: 1, color: 'warning.main' }} />
                <Typography variant="h6">Alerts</Typography>
              </Box>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {unreadAlerts} Unread
              </Typography>
              <Box display="flex" alignItems="center">
                <Chip
                  label={`${alerts.length} Total Alerts`}
                  color="warning"
                  size="small"
                />
              </Box>
              <Button 
                variant="text" 
                size="small" 
                sx={{ mt: 1 }}
                onClick={navigateToAlerts}
              >
                View Alerts
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <ChartIcon sx={{ mr: 1, color: systemStatus === 'operational' ? 'success.main' : 'error.main' }} />
                <Typography variant="h6">System Status</Typography>
              </Box>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {systemStatus.charAt(0).toUpperCase() + systemStatus.slice(1)}
              </Typography>
              <Box display="flex" alignItems="center">
                <Chip
                  label={`${systemComponents.filter(c => c.status === 'operational').length}/${systemComponents.length} Components Online`}
                  color={systemStatus === 'operational' ? 'success' : 'warning'}
                  size="small"
                />
              </Box>
              <Button 
                variant="text" 
                size="small" 
                sx={{ mt: 1 }}
                onClick={navigateToSystem}
              >
                View System
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Main Dashboard Content */}
      <Grid container spacing={3}>
        {/* Asset Allocation */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Asset Allocation" />
            <Divider />
            <CardContent>
              <List>
                {assets.slice(0, 5).map((asset) => (
                  <ListItem key={asset.assetId}>
                    <ListItemAvatar>
                      <Avatar>{asset.symbol.charAt(0)}</Avatar>
                    </ListItemAvatar>
                    <ListItemText 
                      primary={`${asset.symbol} - ${asset.name}`}
                      secondary={`${formatCurrency(asset.value)} (${asset.allocation.toFixed(1)}%)`}
                    />
                    <Box display="flex" flexDirection="column" alignItems="flex-end">
                      <Typography 
                        variant="body2"
                        color={asset.dayChangePercent >= 0 ? 'success.main' : 'error.main'}
                        sx={{ display: 'flex', alignItems: 'center' }}
                      >
                        {asset.dayChangePercent >= 0 ? <TrendingUpIcon fontSize="small" /> : <TrendingDownIcon fontSize="small" />}
                        {formatPercentage(asset.dayChangePercent)}
                      </Typography>
                    </Box>
                  </ListItem>
                ))}
              </List>
              <Button 
                variant="outlined" 
                fullWidth 
                sx={{ mt: 2 }}
                onClick={navigateToPortfolio}
              >
                View All Assets
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Recent Alerts */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Recent Alerts" />
            <Divider />
            <CardContent>
              <List>
                {alerts.slice(0, 5).map((alert) => (
                  <ListItem key={alert.id}>
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: alert.severity === 'error' ? 'error.main' : 
                                           alert.severity === 'warning' ? 'warning.main' :
                                           alert.severity === 'success' ? 'success.main' : 'info.main' }}>
                        {alert.category.charAt(0).toUpperCase()}
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText 
                      primary={alert.title}
                      secondary={alert.message.length > 60 ? alert.message.substring(0, 60) + '...' : alert.message}
                    />
                    {!alert.read && (
                      <Chip label="New" size="small" color="primary" />
                    )}
                  </ListItem>
                ))}
              </List>
              <Button 
                variant="outlined" 
                fullWidth 
                sx={{ mt: 2 }}
                onClick={navigateToAlerts}
              >
                View All Alerts
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        {/* AI Agent Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="AI Agent System" />
            <Divider />
            <CardContent>
              <List>
                <ListItem>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'primary.main' }}>
                      <AgentIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText 
                    primary="Technical Agent"
                    secondary="Analyzing market patterns and momentum"
                  />
                  <LinearProgress 
                    variant="determinate" 
                    value={90} 
                    sx={{ width: 100, mr: 1 }}
                  />
                </ListItem>
                <ListItem>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'primary.main' }}>
                      <AgentIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText 
                    primary="Fundamental Agent"
                    secondary="Evaluating on-chain metrics and fundamentals"
                  />
                  <LinearProgress 
                    variant="determinate" 
                    value={85} 
                    sx={{ width: 100, mr: 1 }}
                  />
                </ListItem>
                <ListItem>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'primary.main' }}>
                      <AgentIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText 
                    primary="Sentiment Agent"
                    secondary="Processing market sentiment data"
                  />
                  <LinearProgress 
                    variant="determinate" 
                    value={75} 
                    sx={{ width: 100, mr: 1 }}
                  />
                </ListItem>
                <ListItem>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'primary.main' }}>
                      <AgentIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText 
                    primary="Value Agent"
                    secondary="Calculating intrinsic value metrics"
                  />
                  <LinearProgress 
                    variant="determinate" 
                    value={80} 
                    sx={{ width: 100, mr: 1 }}
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Risk Management */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Risk Management" />
            <Divider />
            <CardContent>
              <List>
                <ListItem>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'success.main' }}>
                      <RiskIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText 
                    primary="Portfolio Exposure"
                    secondary="Current leverage: 1.1x (Max: 1.5x)"
                  />
                  <LinearProgress 
                    variant="determinate" 
                    value={(1.1 / 1.5) * 100} 
                    sx={{ width: 100, mr: 1 }}
                    color="success"
                  />
                </ListItem>
                <ListItem>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'warning.main' }}>
                      <RiskIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText 
                    primary="Volatility"
                    secondary="Market volatility: Medium"
                  />
                  <LinearProgress 
                    variant="determinate" 
                    value={60} 
                    sx={{ width: 100, mr: 1 }}
                    color="warning"
                  />
                </ListItem>
                <ListItem>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'info.main' }}>
                      <PortfolioIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText 
                    primary="Diversification"
                    secondary="Current: 78% (Target: 85%)"
                  />
                  <LinearProgress 
                    variant="determinate" 
                    value={(78 / 85) * 100} 
                    sx={{ width: 100, mr: 1 }}
                    color="info"
                  />
                </ListItem>
                <ListItem>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'success.main' }}>
                      <PortfolioIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText 
                    primary="Monthly Performance"
                    secondary={`${formatPercentage(getMonthlyPerformance().returnPercent)} (${formatCurrency(getMonthlyPerformance().returnValue)})`}
                  />
                  <Chip
                    label={getMonthlyPerformance().returnPercent >= 0 ? "Positive" : "Negative"}
                    color={getMonthlyPerformance().returnPercent >= 0 ? "success" : "error"}
                    size="small"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;