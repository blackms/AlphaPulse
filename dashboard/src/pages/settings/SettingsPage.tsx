import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Divider,
  FormControl,
  FormControlLabel,
  FormGroup,
  FormLabel,
  Switch,
  TextField,
  Select,
  MenuItem,
  InputLabel,
  Button,
  Snackbar,
  Alert,
  SelectChangeEvent,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import {
  selectTheme,
  selectCompactMode,
  setTheme,
  setCompactMode,
} from '../../store/slices/uiSlice';

const SettingsPage: React.FC = () => {
  const dispatch = useDispatch();
  const currentTheme = useSelector(selectTheme);
  const compactMode = useSelector(selectCompactMode);
  
  // Local state for form values
  const [tradingEnabled, setTradingEnabled] = useState<boolean>(false);
  const [maxTradeSize, setMaxTradeSize] = useState<string>('5');
  const [riskLevel, setRiskLevel] = useState<string>('medium');
  const [alertFrequency, setAlertFrequency] = useState<string>('important');
  const [emailNotifications, setEmailNotifications] = useState<boolean>(true);
  
  // Snackbar state
  const [snackbarOpen, setSnackbarOpen] = useState<boolean>(false);
  const [snackbarMessage, setSnackbarMessage] = useState<string>('');
  
  const handleThemeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    dispatch(setTheme(event.target.checked ? 'dark' : 'light'));
  };
  
  const handleCompactModeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    dispatch(setCompactMode(event.target.checked));
  };
  
  const handleTradingEnabledChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTradingEnabled(event.target.checked);
  };
  
  const handleMaxTradeSizeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setMaxTradeSize(event.target.value);
  };
  
  const handleRiskLevelChange = (event: SelectChangeEvent) => {
    setRiskLevel(event.target.value);
  };
  
  const handleAlertFrequencyChange = (event: SelectChangeEvent) => {
    setAlertFrequency(event.target.value);
  };
  
  const handleEmailNotificationsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setEmailNotifications(event.target.checked);
  };
  
  const handleSaveSettings = () => {
    // In a real app, this would save to backend
    // For now, we just show a success message
    setSnackbarMessage('Settings saved successfully');
    setSnackbarOpen(true);
  };
  
  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
  };
  
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Settings
      </Typography>
      
      <Grid container spacing={3}>
        {/* Appearance Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Appearance Settings" />
            <Divider />
            <CardContent>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Switch
                      checked={currentTheme === 'dark'}
                      onChange={handleThemeChange}
                    />
                  }
                  label="Dark Theme"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={compactMode}
                      onChange={handleCompactModeChange}
                    />
                  }
                  label="Compact Mode"
                />
              </FormGroup>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Trading Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Trading Settings" />
            <Divider />
            <CardContent>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Switch
                      checked={tradingEnabled}
                      onChange={handleTradingEnabledChange}
                    />
                  }
                  label="Enable Automated Trading"
                />
                
                <Box mt={2} mb={2}>
                  <TextField
                    label="Maximum Trade Size (%)"
                    type="number"
                    value={maxTradeSize}
                    onChange={handleMaxTradeSizeChange}
                    InputProps={{ inputProps: { min: 1, max: 100 } }}
                    fullWidth
                    disabled={!tradingEnabled}
                  />
                </Box>
                
                <FormControl fullWidth sx={{ mt: 2 }} disabled={!tradingEnabled}>
                  <InputLabel id="risk-level-label">Risk Level</InputLabel>
                  <Select
                    labelId="risk-level-label"
                    value={riskLevel}
                    label="Risk Level"
                    onChange={handleRiskLevelChange}
                  >
                    <MenuItem value="low">Low</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="high">High</MenuItem>
                  </Select>
                </FormControl>
              </FormGroup>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Notification Settings */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Notification Settings" />
            <Divider />
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel id="alert-frequency-label">Alert Frequency</InputLabel>
                    <Select
                      labelId="alert-frequency-label"
                      value={alertFrequency}
                      label="Alert Frequency"
                      onChange={handleAlertFrequencyChange}
                    >
                      <MenuItem value="all">All Alerts</MenuItem>
                      <MenuItem value="important">Important Only</MenuItem>
                      <MenuItem value="critical">Critical Only</MenuItem>
                      <MenuItem value="none">None</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <FormGroup>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={emailNotifications}
                          onChange={handleEmailNotificationsChange}
                        />
                      }
                      label="Email Notifications"
                    />
                  </FormGroup>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Save Button */}
        <Grid item xs={12}>
          <Box display="flex" justifyContent="flex-end">
            <Button
              variant="contained"
              color="primary"
              size="large"
              onClick={handleSaveSettings}
            >
              Save Settings
            </Button>
          </Box>
        </Grid>
      </Grid>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseSnackbar} severity="success">
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default SettingsPage;