import React, { useEffect } from 'react';
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
  IconButton,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
  Router as RouterIcon,
} from '@mui/icons-material';
import { RootState } from '../../store/store';
import { ComponentStatus, SystemComponent, SystemStatus } from '../../store/slices/systemSlice';

// This is a placeholder component until the full implementation is complete
const SystemStatusPage: React.FC = () => {
  const dispatch = useDispatch();
  const { status, components, resources, api, lastErrors, loading, error } = useSelector(
    (state: RootState) => state.system
  );

  useEffect(() => {
    // Placeholder for fetching system status
    // dispatch(fetchSystemStatus());
  }, [dispatch]);

  const handleRefresh = () => {
    // Placeholder for refreshing system status
    // dispatch(fetchSystemStatus());
  };

  const getStatusIcon = (status: SystemStatus | ComponentStatus) => {
    switch (status) {
      case 'operational':
        return <CheckCircleIcon />;
      case 'degraded':
        return <WarningIcon />;
      case 'maintenance':
        return <InfoIcon />;
      case 'outage':
      case 'down':
        return <ErrorIcon />;
      default:
        return <InfoIcon />;
    }
  };

  const getStatusColor = (status: SystemStatus | ComponentStatus) => {
    switch (status) {
      case 'operational':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'maintenance':
        return 'info';
      case 'outage':
      case 'down':
        return 'error';
      default:
        return 'info';
    }
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getResourceIcon = (resourceType: string) => {
    switch (resourceType) {
      case 'cpu':
        return <SpeedIcon />;
      case 'memory':
        return <MemoryIcon />;
      case 'disk':
        return <StorageIcon />;
      case 'network':
        return <RouterIcon />;
      default:
        return <InfoIcon />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          System Status
        </Typography>
        <IconButton onClick={handleRefresh} disabled={loading}>
          <RefreshIcon />
        </IconButton>
      </Box>

      {loading ? (
        <Typography>Loading system status...</Typography>
      ) : error ? (
        <Paper sx={{ p: 2, bgcolor: 'error.light' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {/* Overall System Status */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center">
                  <Chip
                    label={status.overall}
                    color={getStatusColor(status.overall) as any}
                    icon={getStatusIcon(status.overall)}
                    sx={{ mr: 2 }}
                  />
                  {status.message && (
                    <Typography variant="body1">{status.message}</Typography>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* System Resources */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="System Resources" />
              <Divider />
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Box display="flex" alignItems="center" mb={1}>
                      <SpeedIcon sx={{ mr: 1 }} />
                      <Typography variant="body1" sx={{ minWidth: 100 }}>
                        CPU
                      </Typography>
                      <Box sx={{ width: '100%', mr: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={resources.cpu}
                          color={resources.cpu > 80 ? 'error' : resources.cpu > 60 ? 'warning' : 'success'}
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {formatPercentage(resources.cpu)}
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12}>
                    <Box display="flex" alignItems="center" mb={1}>
                      <MemoryIcon sx={{ mr: 1 }} />
                      <Typography variant="body1" sx={{ minWidth: 100 }}>
                        Memory
                      </Typography>
                      <Box sx={{ width: '100%', mr: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={resources.memory}
                          color={resources.memory > 80 ? 'error' : resources.memory > 60 ? 'warning' : 'success'}
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {formatPercentage(resources.memory)}
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12}>
                    <Box display="flex" alignItems="center" mb={1}>
                      <StorageIcon sx={{ mr: 1 }} />
                      <Typography variant="body1" sx={{ minWidth: 100 }}>
                        Disk
                      </Typography>
                      <Box sx={{ width: '100%', mr: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={resources.disk}
                          color={resources.disk > 80 ? 'error' : resources.disk > 60 ? 'warning' : 'success'}
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {formatPercentage(resources.disk)}
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12}>
                    <Box display="flex" alignItems="center">
                      <RouterIcon sx={{ mr: 1 }} />
                      <Typography variant="body1" sx={{ minWidth: 100 }}>
                        Network
                      </Typography>
                      <Box sx={{ width: '100%' }}>
                        <Typography variant="body2" color="text.secondary">
                          In: {resources.network.in.toFixed(2)} Mbps / Out: {resources.network.out.toFixed(2)} Mbps
                        </Typography>
                      </Box>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* API Status */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="API Status" />
              <Divider />
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="body1">Response Time</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {api.responseTime.toFixed(2)} ms
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12}>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="body1">Requests (Total)</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {api.requests.total}
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12}>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="body1">Success Rate</Typography>
                      <Typography
                        variant="body2"
                        color={api.requests.error / api.requests.total > 0.1 ? 'error.main' : 'success.main'}
                      >
                        {api.requests.total > 0
                          ? formatPercentage((api.requests.success / api.requests.total) * 100)
                          : '100%'}
                      </Typography>
                    </Box>
                  </Grid>

                  {api.lastError && (
                    <Grid item xs={12}>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography variant="body1">Last Error</Typography>
                        <Typography variant="body2" color="error.main">
                          {api.lastError}
                        </Typography>
                      </Box>
                    </Grid>
                  )}
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Component Status */}
          <Grid item xs={12}>
            <Card>
              <CardHeader title="Component Status" />
              <Divider />
              <CardContent>
                <Grid container spacing={2}>
                  {components && components.length > 0 ? (
                    components.map((component: SystemComponent) => (
                      <Grid item xs={12} sm={6} md={4} key={component.id}>
                        <Box display="flex" alignItems="center" p={1}>
                          <Typography variant="body1" sx={{ flexGrow: 1 }}>
                            {component.name}
                          </Typography>
                          <Box display="flex" alignItems="center">
                            <Chip
                              size="small"
                              color={getStatusColor(component.status) as any}
                              icon={getStatusIcon(component.status)}
                            />
                          </Box>
                        </Box>
                      </Grid>
                    ))
                  ) : (
                    <Grid item xs={12}>
                      <Typography align="center">No components found</Typography>
                    </Grid>
                  )}
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Recent Errors */}
          <Grid item xs={12}>
            <Card>
              <CardHeader title="Recent Errors" />
              <Divider />
              <CardContent>
                {lastErrors && lastErrors.length > 0 ? (
                  <List>
                    {lastErrors.slice(0, 5).map((error, index) => (
                      <ListItem key={index} divider={index !== lastErrors.slice(0, 5).length - 1}>
                        <ListItemIcon>
                          {error.level === 'critical' ? (
                            <ErrorIcon color="error" />
                          ) : error.level === 'error' ? (
                            <ErrorIcon color="error" />
                          ) : error.level === 'warning' ? (
                            <WarningIcon color="warning" />
                          ) : (
                            <InfoIcon color="info" />
                          )}
                        </ListItemIcon>
                        <ListItemText
                          primary={error.message}
                          secondary={`${error.component} - ${formatTimestamp(error.timestamp)}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                ) : (
                  <Typography align="center">No recent errors</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default SystemStatusPage;