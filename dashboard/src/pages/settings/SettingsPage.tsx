import React, { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  // Divider, // Unused import
  Switch,
  FormControlLabel,
  FormGroup,
  Tab,
  Tabs,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  TextField,
  Paper,
  Alert,
  SelectChangeEvent,
} from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import NotificationsIcon from '@mui/icons-material/Notifications';
import PaletteIcon from '@mui/icons-material/Palette';
import SecurityIcon from '@mui/icons-material/Security';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import SettingsIcon from '@mui/icons-material/Settings';

// Redux
import { 
  selectThemeMode, 
  selectDisplayDensity, 
  setThemeMode, 
  setDisplayDensity,
  selectChartStyle,
  setChartStyle,
  selectAnimationsEnabled,
  toggleAnimations,
  ThemeMode,
  DisplayDensity,
  ChartStyle
} from '../../store/slices/uiSlice';

import {
  selectAlertPreferences,
  updatePreferences,
  AlertPreferences
} from '../../store/slices/alertsSlice';

import {
  selectUser,
  updateUserStart,
  updateUserSuccess,
  updatePreferences as updateUserPreferences,
  // User, // Unused import
  // UserPreferences // Unused import
} from '../../store/slices/authSlice';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const a11yProps = (index: number) => {
  return {
    id: `settings-tab-${index}`,
    'aria-controls': `settings-tabpanel-${index}`,
  };
};

const SettingsPage: React.FC = () => {
  const dispatch = useDispatch();
  
  // UI settings
  const themeMode = useSelector(selectThemeMode);
  const displayDensity = useSelector(selectDisplayDensity);
  const chartStyle = useSelector(selectChartStyle);
  const animationsEnabled = useSelector(selectAnimationsEnabled);
  
  // Alert preferences
  const alertPreferences = useSelector(selectAlertPreferences);
  
  // User settings
  const user = useSelector(selectUser);
  
  // Local state
  const [tabValue, setTabValue] = useState(0);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [quietHours, setQuietHours] = useState({
    muteStartTime: "22:00",
    muteEndTime: "08:00"
  });
  
  // Local form state
  const [userForm, setUserForm] = useState({
    firstName: user?.firstName || '',
    lastName: user?.lastName || '',
    email: user?.email || '',
    timezone: user?.preferences.timezone || 'UTC',
    language: user?.preferences.language || 'en',
  });
  
  // Tab change handler
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // UI settings handlers
  const handleThemeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    dispatch(setThemeMode(event.target.checked ? ThemeMode.DARK : ThemeMode.LIGHT));
  };
  
  const handleDensityChange = (event: SelectChangeEvent<DisplayDensity>) => {
    dispatch(setDisplayDensity(event.target.value as DisplayDensity));
  };
  
  const handleChartStyleChange = (event: SelectChangeEvent<ChartStyle>) => {
    dispatch(setChartStyle(event.target.value as ChartStyle));
  };
  
  const handleAnimationsToggle = () => {
    dispatch(toggleAnimations());
  };
  
  // Alert preferences handlers
  const handleAlertPreferenceChange = (key: keyof AlertPreferences) => 
    (event: React.ChangeEvent<HTMLInputElement>) => {
    dispatch(updatePreferences({ [key]: event.target.checked }));
  };
  
  // User form handlers
  const handleUserFormChange = (key: string) => 
    (event: React.ChangeEvent<HTMLInputElement>) => {
    setUserForm({
      ...userForm,
      [key]: event.target.value
    });
  };
  
  // Save user profile
  const handleSaveUserProfile = () => {
    // Manually call updateUserStart action
    dispatch(updateUserStart());
    
    // Then manually update user with data
    dispatch(updateUserSuccess({
      firstName: userForm.firstName,
      lastName: userForm.lastName,
      email: userForm.email
    }));
    
    dispatch(updateUserPreferences({
      timezone: userForm.timezone,
      language: userForm.language
    }));
    
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };
  
  return (
    <Box sx={{ width: '100%' }}>
      <Paper sx={{ width: '100%', mb: 2 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="settings tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab icon={<PaletteIcon />} label="Appearance" {...a11yProps(0)} />
            <Tab icon={<NotificationsIcon />} label="Notifications" {...a11yProps(1)} />
            <Tab icon={<AccountCircleIcon />} label="Profile" {...a11yProps(2)} />
            <Tab icon={<SecurityIcon />} label="Security" {...a11yProps(3)} />
            <Tab icon={<SettingsIcon />} label="Advanced" {...a11yProps(4)} />
          </Tabs>
        </Box>
        
        {/* Appearance Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Interface Settings
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <FormGroup>
                    <FormControlLabel
                      control={
                        <Switch 
                          checked={themeMode === 'dark'} 
                          onChange={handleThemeChange} 
                        />
                      }
                      label="Dark Mode"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch 
                          checked={animationsEnabled} 
                          onChange={handleAnimationsToggle} 
                        />
                      }
                      label="Enable Animations"
                    />
                  </FormGroup>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel id="density-select-label">Display Density</InputLabel>
                    <Select
                      labelId="density-select-label"
                      value={displayDensity}
                      label="Display Density"
                      onChange={handleDensityChange}
                    >
                      <MenuItem value="comfortable">Comfortable</MenuItem>
                      <MenuItem value="compact">Compact</MenuItem>
                      <MenuItem value="spacious">Spacious</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <FormControl fullWidth>
                    <InputLabel id="chart-style-select-label">Chart Style</InputLabel>
                    <Select
                      labelId="chart-style-select-label"
                      value={chartStyle}
                      label="Chart Style"
                      onChange={handleChartStyleChange}
                    >
                      <MenuItem value="candle">Candlestick</MenuItem>
                      <MenuItem value="line">Line</MenuItem>
                      <MenuItem value="area">Area</MenuItem>
                      <MenuItem value="bar">Bar</MenuItem>
                    </Select>
                  </FormControl>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Notifications Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Alert & Notification Settings
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Notification Channels
                  </Typography>
                  <FormGroup>
                    <FormControlLabel
                      control={
                        <Switch 
                          checked={alertPreferences?.emailNotifications} 
                          onChange={handleAlertPreferenceChange('emailNotifications')} 
                        />
                      }
                      label="Email Notifications"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch 
                          checked={alertPreferences?.pushNotifications} 
                          onChange={handleAlertPreferenceChange('pushNotifications')} 
                        />
                      }
                      label="Push Notifications"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch 
                          checked={alertPreferences?.channels.sms}
                          onChange={(e) => dispatch(updatePreferences({
                            channels: { ...alertPreferences?.channels, sms: e.target.checked }
                          }))}
                        />
                      }
                      label="SMS Notifications"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch 
                          checked={alertPreferences?.channels.slack}
                          onChange={(e) => dispatch(updatePreferences({
                            channels: { ...alertPreferences?.channels, slack: e.target.checked }
                          }))}
                        />
                      }
                      label="Slack Notifications"
                    />
                  </FormGroup>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Notification Settings
                  </Typography>
                  <FormGroup>
                    <FormControlLabel
                      control={
                        <Switch 
                          checked={alertPreferences?.preferences.includeCritical &&
                                  !alertPreferences?.preferences.includeHigh &&
                                  !alertPreferences?.preferences.includeMedium &&
                                  !alertPreferences?.preferences.includeLow}
                          onChange={(e) => dispatch(updatePreferences({
                            preferences: {
                              ...alertPreferences?.preferences,
                              includeCritical: e.target.checked,
                              includeHigh: !e.target.checked,
                              includeMedium: !e.target.checked,
                              includeLow: !e.target.checked
                            }
                          }))}
                        />
                      }
                      label="Critical Alerts Only"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch 
                          checked={alertPreferences?.soundEnabled} 
                          onChange={handleAlertPreferenceChange('soundEnabled')} 
                        />
                      }
                      label="Sound Alerts"
                    />
                  </FormGroup>
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Quiet Hours
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <TextField
                          label="Start Time"
                          type="time"
                          value={quietHours.muteStartTime}
                          onChange={(e) => setQuietHours({...quietHours, muteStartTime: e.target.value})}
                          InputLabelProps={{
                            shrink: true,
                          }}
                          fullWidth
                        />
                      </Grid>
                      <Grid item xs={6}>
                        <TextField
                          label="End Time"
                          type="time"
                          value={quietHours.muteEndTime}
                          onChange={(e) => setQuietHours({...quietHours, muteEndTime: e.target.value})}
                          InputLabelProps={{
                            shrink: true,
                          }}
                          fullWidth
                        />
                      </Grid>
                    </Grid>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Profile Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                User Profile
              </Typography>
              {saveSuccess && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  Profile updated successfully!
                </Alert>
              )}
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Personal Information
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="First Name"
                        value={userForm.firstName}
                        onChange={handleUserFormChange('firstName')}
                        fullWidth
                        margin="normal"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Last Name"
                        value={userForm.lastName}
                        onChange={handleUserFormChange('lastName')}
                        fullWidth
                        margin="normal"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        label="Email"
                        type="email"
                        value={userForm.email}
                        onChange={handleUserFormChange('email')}
                        fullWidth
                        margin="normal"
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Regional Settings
                  </Typography>
                  <FormControl fullWidth sx={{ mb: 2, mt: 1 }}>
                    <InputLabel id="timezone-select-label">Timezone</InputLabel>
                    <Select
                      labelId="timezone-select-label"
                      value={userForm.timezone}
                      label="Timezone"
                      onChange={(e: SelectChangeEvent) => setUserForm({ ...userForm, timezone: e.target.value })}
                    >
                      <MenuItem value="UTC">UTC</MenuItem>
                      <MenuItem value="America/New_York">Eastern Time (ET)</MenuItem>
                      <MenuItem value="America/Chicago">Central Time (CT)</MenuItem>
                      <MenuItem value="America/Denver">Mountain Time (MT)</MenuItem>
                      <MenuItem value="America/Los_Angeles">Pacific Time (PT)</MenuItem>
                      <MenuItem value="Europe/London">London (GMT)</MenuItem>
                      <MenuItem value="Europe/Paris">Central European Time (CET)</MenuItem>
                      <MenuItem value="Asia/Tokyo">Japan (JST)</MenuItem>
                      <MenuItem value="Asia/Hong_Kong">Hong Kong</MenuItem>
                      <MenuItem value="Australia/Sydney">Sydney</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel id="language-select-label">Language</InputLabel>
                    <Select
                      labelId="language-select-label"
                      value={userForm.language}
                      label="Language"
                      onChange={(e: SelectChangeEvent) => setUserForm({ ...userForm, language: e.target.value })}
                    >
                      <MenuItem value="en">English</MenuItem>
                      <MenuItem value="es">Spanish</MenuItem>
                      <MenuItem value="fr">French</MenuItem>
                      <MenuItem value="de">German</MenuItem>
                      <MenuItem value="ja">Japanese</MenuItem>
                      <MenuItem value="zh">Chinese</MenuItem>
                    </Select>
                  </FormControl>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={handleSaveUserProfile}
                >
                  Save Changes
                </Button>
              </Box>
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Security Tab */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Security Settings
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Change Password
                  </Typography>
                  <TextField
                    label="Current Password"
                    type="password"
                    fullWidth
                    margin="normal"
                  />
                  <TextField
                    label="New Password"
                    type="password"
                    fullWidth
                    margin="normal"
                  />
                  <TextField
                    label="Confirm New Password"
                    type="password"
                    fullWidth
                    margin="normal"
                  />
                  <Box sx={{ mt: 2 }}>
                    <Button variant="contained">Update Password</Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Two-Factor Authentication
                  </Typography>
                  <FormGroup>
                    <FormControlLabel
                      control={<Switch />}
                      label="Enable Two-Factor Authentication"
                    />
                  </FormGroup>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    Two-factor authentication adds an additional layer of security to your account by requiring more than just a password to log in.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Advanced Tab */}
        <TabPanel value={tabValue} index={4}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Advanced Settings
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Data & Privacy
                  </Typography>
                  <Button variant="outlined" color="primary" sx={{ mr: 2, mb: 2 }}>
                    Export My Data
                  </Button>
                  <Button variant="outlined" color="error" sx={{ mb: 2 }}>
                    Delete Account
                  </Button>
                  <Typography variant="body2" color="text.secondary">
                    Deleting your account will remove all your data and cannot be undone.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    API Access
                  </Typography>
                  <TextField
                    label="API Key"
                    value="••••••••••••••••••••••••••••••"
                    InputProps={{
                      readOnly: true,
                    }}
                    fullWidth
                    margin="normal"
                  />
                  <Box sx={{ mt: 2 }}>
                    <Button variant="outlined" sx={{ mr: 2 }}>
                      Regenerate API Key
                    </Button>
                    <Button variant="outlined">
                      View API Documentation
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default SettingsPage;