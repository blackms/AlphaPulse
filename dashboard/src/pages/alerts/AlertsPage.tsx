import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Chip,
  Button,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Switch,
  FormControlLabel,
  Tooltip,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Notifications as NotificationsIcon,
  NotificationsActive as NotificationsActiveIcon,
  NotificationsOff as NotificationsOffIcon,
  ErrorOutline as ErrorIcon,
  WarningAmber as WarningIcon,
  Info as InfoIcon,
  Check as CheckIcon,
  Add as AddIcon,
  Close as CloseIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material';
import {
  selectAlerts,
  selectNotificationSettings,
  selectLoading,
  fetchAlertsStart,
  markAlertAsRead,
  deleteAlert,
  updateNotificationSettings,
  addAlertRule,
  deleteAlertRule,
  updateAlertRule,
  Alert,
  AlertSeverity,
  AlertType,
  AlertRule,
  NotificationChannel,
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
      id={`alert-tabpanel-${index}`}
      aria-labelledby={`alert-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
};

const AlertsPage: React.FC = () => {
  const dispatch = useDispatch();
  const alerts = useSelector(selectAlerts);
  const notificationSettings = useSelector(selectNotificationSettings);
  const loading = useSelector(selectLoading);
  
  const [tabValue, setTabValue] = useState(0);
  const [filterSeverity, setFilterSeverity] = useState<AlertSeverity | 'all'>('all');
  const [filterType, setFilterType] = useState<AlertType | 'all'>('all');
  const [editRuleDialogOpen, setEditRuleDialogOpen] = useState(false);
  const [currentRule, setCurrentRule] = useState<AlertRule | null>(null);
  const [notificationDialogOpen, setNotificationDialogOpen] = useState(false);
  
  useEffect(() => {
    dispatch(fetchAlertsStart());
    
    // Set up polling every 30 seconds
    const interval = setInterval(() => {
      dispatch(fetchAlertsStart());
    }, 30000);
    
    return () => clearInterval(interval);
  }, [dispatch]);
  
  const handleRefresh = () => {
    dispatch(fetchAlertsStart());
  };
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  const handleMarkAsRead = (alertId: string) => {
    dispatch(markAlertAsRead(alertId));
  };
  
  const handleDeleteAlert = (alertId: string) => {
    dispatch(deleteAlert(alertId));
  };
  
  const handleAddRule = () => {
    setCurrentRule({
      id: '',
      name: '',
      description: '',
      type: 'portfolio',
      condition: '',
      threshold: 0,
      severity: 'medium',
      enabled: true,
      createdAt: Date.now(),
    });
    setEditRuleDialogOpen(true);
  };
  
  const handleEditRule = (rule: AlertRule) => {
    setCurrentRule(rule);
    setEditRuleDialogOpen(true);
  };
  
  const handleSaveRule = () => {
    if (currentRule) {
      if (currentRule.id) {
        dispatch(updateAlertRule(currentRule));
      } else {
        dispatch(addAlertRule({
          ...currentRule,
          id: `rule_${Date.now()}`,
        }));
      }
    }
    setEditRuleDialogOpen(false);
  };
  
  const handleDeleteRule = (ruleId: string) => {
    dispatch(deleteAlertRule(ruleId));
  };
  
  const handleOpenNotificationSettings = () => {
    setNotificationDialogOpen(true);
  };
  
  const handleSaveNotificationSettings = () => {
    dispatch(updateNotificationSettings(notificationSettings));
    setNotificationDialogOpen(false);
  };
  
  const handleToggleNotificationChannel = (channel: NotificationChannel) => {
    dispatch(updateNotificationSettings({
      ...notificationSettings,
      channels: {
        ...notificationSettings.channels,
        [channel]: !notificationSettings.channels[channel],
      }
    }));
  };
  
  const filteredAlerts = alerts.filter(alert => {
    if (filterSeverity !== 'all' && alert.severity !== filterSeverity) return false;
    if (filterType !== 'all' && alert.type !== filterType) return false;
    return true;
  });
  
  const getSeverityIcon = (severity: AlertSeverity) => {
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
  
  const getSeverityColor = (severity: AlertSeverity) => {
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
  
  const formatDateTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };
  
  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Alerts & Notifications
        </Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<NotificationsIcon />}
            onClick={handleOpenNotificationSettings}
            sx={{ mr: 2 }}
          >
            Notification Settings
          </Button>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
      </Box>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="alert tabs">
          <Tab label="Active Alerts" />
          <Tab label="Alert Rules" />
          <Tab label="Alert History" />
        </Tabs>
      </Box>
      
      {/* Active Alerts Tab */}
      <TabPanel value={tabValue} index={0}>
        <Box mb={3} display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center">
            <FormControl sx={{ mr: 2, minWidth: 120 }} size="small">
              <InputLabel id="severity-filter-label">Severity</InputLabel>
              <Select
                labelId="severity-filter-label"
                value={filterSeverity}
                label="Severity"
                onChange={(e) => setFilterSeverity(e.target.value as AlertSeverity | 'all')}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="low">Low</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl sx={{ minWidth: 120 }} size="small">
              <InputLabel id="type-filter-label">Type</InputLabel>
              <Select
                labelId="type-filter-label"
                value={filterType}
                label="Type"
                onChange={(e) => setFilterType(e.target.value as AlertType | 'all')}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="portfolio">Portfolio</MenuItem>
                <MenuItem value="trading">Trading</MenuItem>
                <MenuItem value="system">System</MenuItem>
                <MenuItem value="security">Security</MenuItem>
                <MenuItem value="market">Market</MenuItem>
              </Select>
            </FormControl>
          </Box>
          
          <Typography variant="body2" color="text.secondary">
            Showing {filteredAlerts.length} of {alerts.length} alerts
          </Typography>
        </Box>
        
        {filteredAlerts.length === 0 ? (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <NotificationsOffIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              No alerts found
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {filterSeverity !== 'all' || filterType !== 'all' 
                ? 'Try changing your filters to see more results' 
                : 'You are all caught up!'}
            </Typography>
          </Paper>
        ) : (
          <Grid container spacing={2}>
            {filteredAlerts.map((alert) => (
              <Grid item xs={12} key={alert.id}>
                <Paper sx={{ 
                  p: 2, 
                  borderLeft: 4, 
                  borderColor: `${getSeverityColor(alert.severity)}.main`,
                  bgcolor: alert.read ? 'background.paper' : 'action.hover'
                }}>
                  <Box display="flex" justifyContent="space-between" alignItems="flex-start">
                    <Box display="flex" alignItems="center">
                      {getSeverityIcon(alert.severity)}
                      <Box ml={1}>
                        <Typography variant="h6" fontWeight={alert.read ? 'normal' : 'bold'}>
                          {alert.title}
                        </Typography>
                        <Box display="flex" alignItems="center" mt={0.5}>
                          <Chip 
                            label={alert.severity.toUpperCase()} 
                            size="small" 
                            color={getSeverityColor(alert.severity) as any}
                            sx={{ mr: 1 }}
                          />
                          <Chip 
                            label={alert.type.charAt(0).toUpperCase() + alert.type.slice(1)} 
                            size="small"
                            sx={{ mr: 1 }}
                          />
                          <Typography variant="caption" color="text.secondary">
                            {formatDateTime(alert.timestamp)}
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                    <Box>
                      {!alert.read && (
                        <Tooltip title="Mark as read">
                          <IconButton onClick={() => handleMarkAsRead(alert.id)} size="small" sx={{ mr: 1 }}>
                            <CheckIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                      <Tooltip title="Delete">
                        <IconButton onClick={() => handleDeleteAlert(alert.id)} size="small">
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </Box>
                  <Typography variant="body1" sx={{ mt: 1 }}>
                    {alert.message}
                  </Typography>
                  {alert.details && (
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      {alert.details}
                    </Typography>
                  )}
                  {alert.actions && alert.actions.length > 0 && (
                    <Box mt={2}>
                      {alert.actions.map((action, index) => (
                        <Button
                          key={index}
                          variant={index === 0 ? "contained" : "outlined"}
                          size="small"
                          onClick={action.onClick}
                          sx={{ mr: 1 }}
                        >
                          {action.label}
                        </Button>
                      ))}
                    </Box>
                  )}
                </Paper>
              </Grid>
            ))}
          </Grid>
        )}
      </TabPanel>
      
      {/* Alert Rules Tab */}
      <TabPanel value={tabValue} index={1}>
        <Box mb={3} display="flex" justifyContent="flex-end">
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleAddRule}
          >
            Add Rule
          </Button>
        </Box>
        
        <Card>
          <CardHeader title="Alert Rules" />
          <Divider />
          <CardContent>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Rule</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Condition</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell width="120">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {notificationSettings.rules.map((rule) => (
                  <TableRow key={rule.id}>
                    <TableCell>
                      <Typography variant="body1">{rule.name}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {rule.description}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={rule.type.charAt(0).toUpperCase() + rule.type.slice(1)} 
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {rule.condition} {rule.threshold}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={rule.severity.toUpperCase()} 
                        size="small" 
                        color={getSeverityColor(rule.severity) as any}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={rule.enabled ? 'Enabled' : 'Disabled'} 
                        color={rule.enabled ? 'success' : 'default'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Tooltip title="Edit">
                        <IconButton onClick={() => handleEditRule(rule)} size="small" sx={{ mr: 1 }}>
                          <EditIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton onClick={() => handleDeleteRule(rule.id)} size="small">
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
                {notificationSettings.rules.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={6} sx={{ textAlign: 'center', py: 3 }}>
                      <Typography variant="body1" color="text.secondary">
                        No alert rules defined
                      </Typography>
                      <Button 
                        variant="outlined" 
                        startIcon={<AddIcon />} 
                        onClick={handleAddRule}
                        sx={{ mt: 1 }}
                      >
                        Add Rule
                      </Button>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </TabPanel>
      
      {/* Alert History Tab */}
      <TabPanel value={tabValue} index={2}>
        <Card>
          <CardHeader title="Alert History" />
          <Divider />
          <CardContent>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Alert</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Time</TableCell>
                  <TableCell>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {alerts.filter(alert => alert.read).map((alert) => (
                  <TableRow key={alert.id}>
                    <TableCell>
                      <Typography variant="body1">{alert.title}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {alert.message}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={alert.type.charAt(0).toUpperCase() + alert.type.slice(1)} 
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={alert.severity.toUpperCase()} 
                        size="small" 
                        color={getSeverityColor(alert.severity) as any}
                      />
                    </TableCell>
                    <TableCell>
                      {formatDateTime(alert.timestamp)}
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label="Resolved" 
                        color="success"
                        size="small"
                      />
                    </TableCell>
                  </TableRow>
                ))}
                {alerts.filter(alert => alert.read).length === 0 && (
                  <TableRow>
                    <TableCell colSpan={5} sx={{ textAlign: 'center', py: 3 }}>
                      <Typography variant="body1" color="text.secondary">
                        No alert history available
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </TabPanel>
      
      {/* Edit Rule Dialog */}
      <Dialog 
        open={editRuleDialogOpen} 
        onClose={() => setEditRuleDialogOpen(false)}
        fullWidth
        maxWidth="md"
      >
        <DialogTitle>
          {currentRule?.id ? 'Edit Alert Rule' : 'New Alert Rule'}
        </DialogTitle>
        <DialogContent>
          {currentRule && (
            <Grid container spacing={3} sx={{ mt: 0 }}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Rule Name"
                  value={currentRule.name}
                  onChange={(e) => setCurrentRule({...currentRule, name: e.target.value})}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Alert Type</InputLabel>
                  <Select
                    value={currentRule.type}
                    label="Alert Type"
                    onChange={(e) => setCurrentRule({...currentRule, type: e.target.value as AlertType})}
                  >
                    <MenuItem value="portfolio">Portfolio</MenuItem>
                    <MenuItem value="trading">Trading</MenuItem>
                    <MenuItem value="system">System</MenuItem>
                    <MenuItem value="security">Security</MenuItem>
                    <MenuItem value="market">Market</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Description"
                  value={currentRule.description}
                  onChange={(e) => setCurrentRule({...currentRule, description: e.target.value})}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Condition"
                  value={currentRule.condition}
                  onChange={(e) => setCurrentRule({...currentRule, condition: e.target.value})}
                  margin="normal"
                  placeholder="e.g. 'Portfolio drawdown exceeds'"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Threshold"
                  type="number"
                  value={currentRule.threshold}
                  onChange={(e) => setCurrentRule({...currentRule, threshold: parseFloat(e.target.value)})}
                  margin="normal"
                  placeholder="e.g. 5 (for 5%)"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Severity</InputLabel>
                  <Select
                    value={currentRule.severity}
                    label="Severity"
                    onChange={(e) => setCurrentRule({...currentRule, severity: e.target.value as AlertSeverity})}
                  >
                    <MenuItem value="critical">Critical</MenuItem>
                    <MenuItem value="high">High</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="low">Low</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={currentRule.enabled}
                      onChange={(e) => setCurrentRule({...currentRule, enabled: e.target.checked})}
                    />
                  }
                  label="Enabled"
                  sx={{ mt: 2 }}
                />
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditRuleDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSaveRule} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>
      
      {/* Notification Settings Dialog */}
      <Dialog 
        open={notificationDialogOpen} 
        onClose={() => setNotificationDialogOpen(false)}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>
          Notification Settings
        </DialogTitle>
        <DialogContent>
          <Typography variant="subtitle2" gutterBottom sx={{ mt: 1 }}>
            Notification Channels
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={notificationSettings.channels.email}
                onChange={() => handleToggleNotificationChannel('email')}
              />
            }
            label="Email"
          />
          <FormControlLabel
            control={
              <Switch
                checked={notificationSettings.channels.sms}
                onChange={() => handleToggleNotificationChannel('sms')}
              />
            }
            label="SMS"
          />
          <FormControlLabel
            control={
              <Switch
                checked={notificationSettings.channels.push}
                onChange={() => handleToggleNotificationChannel('push')}
              />
            }
            label="Push Notifications"
          />
          <FormControlLabel
            control={
              <Switch
                checked={notificationSettings.channels.slack}
                onChange={() => handleToggleNotificationChannel('slack')}
              />
            }
            label="Slack"
          />
          
          <Divider sx={{ my: 2 }} />
          
          <Typography variant="subtitle2" gutterBottom>
            Notification Preferences
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={notificationSettings.preferences.includeCritical}
                onChange={(e) => dispatch(updateNotificationSettings({
                  ...notificationSettings,
                  preferences: {
                    ...notificationSettings.preferences,
                    includeCritical: e.target.checked,
                  }
                }))}
              />
            }
            label="Critical Alerts"
          />
          <FormControlLabel
            control={
              <Switch
                checked={notificationSettings.preferences.includeHigh}
                onChange={(e) => dispatch(updateNotificationSettings({
                  ...notificationSettings,
                  preferences: {
                    ...notificationSettings.preferences,
                    includeHigh: e.target.checked,
                  }
                }))}
              />
            }
            label="High Alerts"
          />
          <FormControlLabel
            control={
              <Switch
                checked={notificationSettings.preferences.includeMedium}
                onChange={(e) => dispatch(updateNotificationSettings({
                  ...notificationSettings,
                  preferences: {
                    ...notificationSettings.preferences,
                    includeMedium: e.target.checked,
                  }
                }))}
              />
            }
            label="Medium Alerts"
          />
          <FormControlLabel
            control={
              <Switch
                checked={notificationSettings.preferences.includeLow}
                onChange={(e) => dispatch(updateNotificationSettings({
                  ...notificationSettings,
                  preferences: {
                    ...notificationSettings.preferences,
                    includeLow: e.target.checked,
                  }
                }))}
              />
            }
            label="Low Alerts"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNotificationDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSaveNotificationSettings} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AlertsPage;