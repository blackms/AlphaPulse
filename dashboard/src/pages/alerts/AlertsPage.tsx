import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  ListItemSecondaryAction,
  IconButton,
  Button,
  Chip,
  Tabs,
  Tab,
  Tooltip,
  Badge,
} from '@mui/material';
import {
  Notifications as AlertIcon,
  Check as MarkReadIcon,
  Delete as DeleteIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Error as ErrorIcon,
  CheckCircle as SuccessIcon,
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import {
  selectAlerts,
  selectUnreadCount,
  markAlertAsRead,
  markAllAlertsAsRead,
  removeAlert,
  clearAlerts,
  AlertSeverity,
  AlertCategory,
} from '../../store/slices/alertsSlice';

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
      id={`alerts-tabpanel-${index}`}
      aria-labelledby={`alerts-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
};

const AlertsPage: React.FC = () => {
  const dispatch = useDispatch();
  const alerts = useSelector(selectAlerts);
  const unreadCount = useSelector(selectUnreadCount);
  
  const [tabValue, setTabValue] = useState(0);
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  const handleMarkAsRead = (alertId: string) => {
    dispatch(markAlertAsRead(alertId));
  };
  
  const handleDeleteAlert = (alertId: string) => {
    dispatch(removeAlert(alertId));
  };
  
  const handleMarkAllRead = () => {
    dispatch(markAllAlertsAsRead());
  };
  
  const handleClearAll = () => {
    dispatch(clearAlerts());
  };
  
  const getSeverityIcon = (severity: AlertSeverity) => {
    switch (severity) {
      case 'error':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'success':
        return <SuccessIcon color="success" />;
      case 'info':
      default:
        return <InfoIcon color="info" />;
    }
  };
  
  const getSeverityColor = (severity: AlertSeverity) => {
    switch (severity) {
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'success':
        return 'success';
      case 'info':
      default:
        return 'info';
    }
  };
  
  const getCategoryLabel = (category: AlertCategory) => {
    switch (category) {
      case 'system':
        return 'System';
      case 'trading':
        return 'Trading';
      case 'portfolio':
        return 'Portfolio';
      case 'security':
        return 'Security';
      case 'market':
        return 'Market';
      default:
        return category;
    }
  };
  
  const formatTimestamp = (timestamp: number) => {
    const now = new Date().getTime();
    const diff = now - timestamp;
    
    // Less than a minute
    if (diff < 60 * 1000) {
      return 'Just now';
    }
    
    // Less than an hour
    if (diff < 60 * 60 * 1000) {
      const minutes = Math.floor(diff / (60 * 1000));
      return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
    }
    
    // Less than a day
    if (diff < 24 * 60 * 60 * 1000) {
      const hours = Math.floor(diff / (60 * 60 * 1000));
      return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
    }
    
    // Format as date
    return new Date(timestamp).toLocaleString();
  };
  
  const filterAlertsByTab = (tabIndex: number) => {
    switch (tabIndex) {
      case 0: // All
        return alerts;
      case 1: // Unread
        return alerts.filter(alert => !alert.read);
      case 2: // Trading
        return alerts.filter(alert => alert.category === 'trading');
      case 3: // System
        return alerts.filter(alert => alert.category === 'system' || alert.category === 'security');
      case 4: // Market
        return alerts.filter(alert => alert.category === 'market' || alert.category === 'portfolio');
      default:
        return alerts;
    }
  };
  
  const filteredAlerts = filterAlertsByTab(tabValue);
  
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Alerts
      </Typography>
      
      <Card>
        <CardHeader 
          title={
            <Box display="flex" alignItems="center">
              <AlertIcon sx={{ mr: 1 }} />
              <Typography variant="h6">
                Notifications & Alerts
              </Typography>
              {unreadCount > 0 && (
                <Badge 
                  badgeContent={unreadCount} 
                  color="error"
                  sx={{ ml: 1 }}
                />
              )}
            </Box>
          }
          action={
            <Box>
              <Button 
                color="primary" 
                onClick={handleMarkAllRead}
                disabled={unreadCount === 0}
                sx={{ mr: 1 }}
              >
                Mark All Read
              </Button>
              <Button 
                color="error" 
                onClick={handleClearAll}
                disabled={alerts.length === 0}
              >
                Clear All
              </Button>
            </Box>
          }
        />
        <Divider />
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="alert tabs">
            <Tab label="All" />
            <Tab label={
              <Badge badgeContent={unreadCount} color="error">
                Unread
              </Badge>
            } />
            <Tab label="Trading" />
            <Tab label="System" />
            <Tab label="Market" />
          </Tabs>
        </Box>
        <CardContent>
          <TabPanel value={tabValue} index={0}>
            {renderAlertsList(filteredAlerts)}
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            {renderAlertsList(filteredAlerts)}
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            {renderAlertsList(filteredAlerts)}
          </TabPanel>
          <TabPanel value={tabValue} index={3}>
            {renderAlertsList(filteredAlerts)}
          </TabPanel>
          <TabPanel value={tabValue} index={4}>
            {renderAlertsList(filteredAlerts)}
          </TabPanel>
        </CardContent>
      </Card>
    </Box>
  );
  
  function renderAlertsList(alerts: ReturnType<typeof selectAlerts>) {
    if (alerts.length === 0) {
      return (
        <Box textAlign="center" py={4}>
          <Typography variant="body1" color="textSecondary">
            No alerts to display
          </Typography>
        </Box>
      );
    }
    
    return (
      <List sx={{ width: '100%' }}>
        {alerts.map((alert) => (
          <ListItem
            key={alert.id}
            alignItems="flex-start"
            sx={{
              bgcolor: alert.read ? 'transparent' : 'action.hover',
              mb: 1,
              borderRadius: 1,
            }}
          >
            <ListItemAvatar>
              <Avatar sx={{ bgcolor: getSeverityColor(alert.severity) as any }}>
                {getSeverityIcon(alert.severity)}
              </Avatar>
            </ListItemAvatar>
            <ListItemText
              primary={
                <Box display="flex" alignItems="center">
                  <Typography
                    variant="subtitle1"
                    component="div"
                    sx={{ fontWeight: alert.read ? 'normal' : 'bold' }}
                  >
                    {alert.title}
                  </Typography>
                  <Chip
                    size="small"
                    label={getCategoryLabel(alert.category)}
                    sx={{ ml: 1 }}
                  />
                  {alert.actionRequired && (
                    <Chip
                      size="small"
                      label="Action Required"
                      color="warning"
                      sx={{ ml: 1 }}
                    />
                  )}
                </Box>
              }
              secondary={
                <>
                  <Typography
                    component="span"
                    variant="body2"
                    color="textPrimary"
                    sx={{ display: 'block', mb: 0.5 }}
                  >
                    {alert.message}
                  </Typography>
                  
                  <Typography
                    component="span"
                    variant="caption"
                    color="textSecondary"
                  >
                    {formatTimestamp(alert.timestamp)}
                  </Typography>
                  
                  {alert.actionRequired && alert.actionLink && (
                    <Button
                      size="small"
                      variant="outlined"
                      color="primary"
                      href={alert.actionLink}
                      sx={{ ml: 2 }}
                    >
                      {alert.actionText || 'View'}
                    </Button>
                  )}
                </>
              }
            />
            <ListItemSecondaryAction>
              {!alert.read && (
                <Tooltip title="Mark as read">
                  <IconButton 
                    edge="end" 
                    onClick={() => handleMarkAsRead(alert.id)}
                    aria-label="mark as read"
                    sx={{ mr: 1 }}
                  >
                    <MarkReadIcon />
                  </IconButton>
                </Tooltip>
              )}
              <Tooltip title="Delete">
                <IconButton 
                  edge="end" 
                  onClick={() => handleDeleteAlert(alert.id)}
                  aria-label="delete"
                >
                  <DeleteIcon />
                </IconButton>
              </Tooltip>
            </ListItemSecondaryAction>
          </ListItem>
        ))}
      </List>
    );
  }
};

export default AlertsPage;