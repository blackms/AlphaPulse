import React, { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Divider,
  Card,
  CardContent,
  CardHeader,
  Button,
  Switch,
  FormControl,
  FormControlLabel,
  TextField,
  InputLabel,
  MenuItem,
  Select,
  SelectChangeEvent,
  Alert,
  Snackbar,
} from '@mui/material';
import { Save as SaveIcon } from '@mui/icons-material';
import { RootState } from '../../store/store';

interface SettingsState {
  theme: 'light' | 'dark' | 'system';
  notifications: {
    alerts: boolean;
    trades: boolean;
    performance: boolean;
    system: boolean;
    email: boolean;
  };
  refreshInterval: number;
  apiEndpoint: string;
  chartSettings: {
    showAnimations: boolean;
    detailedTooltips: boolean;
    defaultTimeframe: string;
  };
}

// This is a placeholder component until the full implementation is complete
const SettingsPage: React.FC = () => {
  const dispatch = useDispatch();
  
  // In a real implementation, these settings would be stored in Redux
  const [settings, setSettings] = useState<SettingsState>({
    theme: 'system',
    notifications: {
      alerts: true,
      trades: true,
      performance: false,
      system: true,
      email: false,
    },
    refreshInterval: 30,
    apiEndpoint: 'http://localhost:8000',
    chartSettings: {
      showAnimations: true,
      detailedTooltips: true,
      defaultTimeframe: '1d',
    },
  });
  
  const [saveStatus, setSaveStatus] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error';
  }>({
    open: false,
    message: '',
    severity: 'success',
  });

  const handleThemeChange = (event: SelectChangeEvent) => {
    setSettings({
      ...settings,
      theme: event.target.value as 'light' | 'dark' | 'system',
    });
  };

  const handleNotificationChange = (setting: keyof typeof settings.notifications) => {
    setSettings({
      ...settings,
      notifications: {
        ...settings.notifications,
        [setting]: !settings.notifications[setting],
      },
    });
  };

  const handleRefreshIntervalChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(event.target.value, 10);
    if (!isNaN(value) && value > 0) {
      setSettings({
        ...settings,
        refreshInterval: value,
      });
    }
  };

  const handleApiEndpointChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSettings({
      ...settings,
      apiEndpoint: event.target.value,
    });
  };

  const handleChartSettingChange = (setting: keyof typeof settings.chartSettings, value: any) => {
    setSettings({
      ...settings,
      chartSettings: {
        ...settings.chartSettings,
        [setting]: value,
      },
    });
  };

  const handleSaveSettings = () => {
    // In a real implementation, this would dispatch an action to update Redux state
    // dispatch(updateSettings(settings));
    
    // Show success message
    setSaveStatus({
      open: true,
      message: 'Settings saved successfully',
      severity: 'success',
    });
  };

  const handleCloseSnackbar = () => {
    setSaveStatus({
      ...saveStatus,
      open: false,
    });
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Settings
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<SaveIcon />}
          onClick={handleSaveSettings}
        >
          Save Settings
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* Appearance Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Appearance" />
            <Divider />
            <CardContent>
              <FormControl fullWidth margin="normal">
                <InputLabel id="theme-select-label">Theme</InputLabel>
                <Select
                  labelId="theme-select-label"
                  id="theme-select"
                  value={settings.theme}
                  label="Theme"
                  onChange={handleThemeChange}
                >
                  <MenuItem value="light">Light</MenuItem>
                  <MenuItem value="dark">Dark</MenuItem>
                  <MenuItem value="system">System Default</MenuItem>
                </Select>
              </FormControl>

              <Box mt={2}>
                <Typography variant="subtitle1" gutterBottom>
                  Chart Settings
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.chartSettings.showAnimations}
                      onChange={(e) => handleChartSettingChange('showAnimations', e.target.checked)}
                    />
                  }
                  label="Enable Chart Animations"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.chartSettings.detailedTooltips}
                      onChange={(e) => handleChartSettingChange('detailedTooltips', e.target.checked)}
                    />
                  }
                  label="Detailed Chart Tooltips"
                />
                <FormControl fullWidth margin="normal">
                  <InputLabel id="timeframe-select-label">Default Timeframe</InputLabel>
                  <Select
                    labelId="timeframe-select-label"
                    id="timeframe-select"
                    value={settings.chartSettings.defaultTimeframe}
                    label="Default Timeframe"
                    onChange={(e) => handleChartSettingChange('defaultTimeframe', e.target.value)}
                  >
                    <MenuItem value="1h">1 Hour</MenuItem>
                    <MenuItem value="4h">4 Hours</MenuItem>
                    <MenuItem value="1d">1 Day</MenuItem>
                    <MenuItem value="1w">1 Week</MenuItem>
                    <MenuItem value="1m">1 Month</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Notification Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Notifications" />
            <Divider />
            <CardContent>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.notifications.alerts}
                    onChange={() => handleNotificationChange('alerts')}
                  />
                }
                label="Alert Notifications"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.notifications.trades}
                    onChange={() => handleNotificationChange('trades')}
                  />
                }
                label="Trade Notifications"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.notifications.performance}
                    onChange={() => handleNotificationChange('performance')}
                  />
                }
                label="Performance Updates"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.notifications.system}
                    onChange={() => handleNotificationChange('system')}
                  />
                }
                label="System Notifications"
              />
              <Divider sx={{ my: 2 }} />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.notifications.email}
                    onChange={() => handleNotificationChange('email')}
                  />
                }
                label="Email Notifications"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Data & API Settings */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Data & API" />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Refresh Interval (seconds)"
                    type="number"
                    value={settings.refreshInterval}
                    onChange={handleRefreshIntervalChange}
                    inputProps={{ min: 5 }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="API Endpoint"
                    value={settings.apiEndpoint}
                    onChange={handleApiEndpointChange}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Success/Error Notification */}
      <Snackbar
        open={saveStatus.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={saveStatus.severity}
          sx={{ width: '100%' }}
        >
          {saveStatus.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default SettingsPage;