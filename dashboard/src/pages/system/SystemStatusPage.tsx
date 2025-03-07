import React, { useEffect } from 'react';
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
  LinearProgress,
  Alert,
  IconButton,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  CheckCircle as OperationalIcon,
  Warning as DegradedIcon,
  Error as DownIcon,
  Build as MaintenanceIcon,
  AccessTime as TimeIcon,
  Help as UnknownIcon,
} from '@mui/icons-material';
import {
  selectSystemStatus,
  selectSystemComponents,
  selectSystemLogs,
  selectSystemMetrics,
  selectSystemLastUpdated,
  selectSystemLoading,
  fetchSystemStart,
  SystemStatus,
  ComponentStatus,
} from '../../store/slices/systemSlice';

const SystemStatusPage: React.FC = () => {
  const dispatch = useDispatch();
  const status = useSelector(selectSystemStatus);
  const components = useSelector(selectSystemComponents);
  const logs = useSelector(selectSystemLogs);
  const metrics = useSelector(selectSystemMetrics);
  const lastUpdated = useSelector(selectSystemLastUpdated);
  const isLoading = useSelector(selectSystemLoading);

  useEffect(() => {
    dispatch(fetchSystemStart());
    
    // Set up polling every 30 seconds
    const interval = setInterval(() => {
      dispatch(fetchSystemStart());
    }, 30000);
    
    return () => clearInterval(interval);
  }, [dispatch]);

  const handleRefresh = () => {
    dispatch(fetchSystemStart());
  };

  const getStatusColor = (status: SystemStatus | ComponentStatus) => {
    switch (status) {
      case 'operational':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'down':
        return 'error';
      case 'maintenance':
        return 'info';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: SystemStatus | ComponentStatus) => {
    switch (status) {
      case 'operational':
        return <OperationalIcon color="success" />;
      case 'degraded':
        return <DegradedIcon color="warning" />;
      case 'down':
        return <DownIcon color="error" />;
      case 'maintenance':
        return <MaintenanceIcon color="info" />;
      default:
        return <UnknownIcon color="disabled" />;
    }
  };

  const formatDateTime = (timestamp: number | null) => {
    if (!timestamp) return 'Never';
    
    return new Date(timestamp).toLocaleString();
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          System Status
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          disabled={isLoading}
        >
          Refresh
        </Button>
      </Box>

      {/* Overall Status Card */}
      <Paper sx={{ p: 3, mb: 3, position: 'relative' }}>
        {isLoading && (
          <LinearProgress sx={{ position: 'absolute', top: 0, left: 0, right: 0 }} />
        )}
        <Box display="flex" alignItems="center" mb={2}>
          {getStatusIcon(status)}
          <Typography variant="h5" sx={{ ml: 1 }}>
            AI Hedge Fund System is currently{' '}
            <Box component="span" fontWeight="bold" color={`${getStatusColor(status)}.main`}>
              {status.toUpperCase()}
            </Box>
          </Typography>
        </Box>
        <Typography variant="body1" color="text.secondary">
          Last updated: {formatDateTime(lastUpdated)}
        </Typography>
      </Paper>

      {/* Component Status Section */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Component Status" />
            <Divider />
            <CardContent>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Component</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Health</TableCell>
                    <TableCell>Last Updated</TableCell>
                    <TableCell>Description</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {components.map((component) => (
                    <TableRow key={component.id}>
                      <TableCell>
                        <Typography variant="body1" fontWeight="medium">
                          {component.name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={component.type.charAt(0).toUpperCase() + component.type.slice(1)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          icon={getStatusIcon(component.status)}
                          label={component.status.charAt(0).toUpperCase() + component.status.slice(1)}
                          color={getStatusColor(component.status) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Box width="100%" mr={1}>
                            <LinearProgress
                              variant="determinate"
                              value={component.healthScore}
                              color={
                                component.healthScore > 80 ? 'success' :
                                component.healthScore > 50 ? 'warning' : 'error'
                              }
                              sx={{ height: 8, borderRadius: 5 }}
                            />
                          </Box>
                          <Box minWidth={35}>
                            <Typography variant="body2" color="text.secondary">
                              {component.healthScore}%
                            </Typography>
                          </Box>
                        </Box>
                      </TableCell>
                      <TableCell>
                        {formatDateTime(component.lastUpdated)}
                      </TableCell>
                      <TableCell>
                        {component.description}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* System Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12}>
          <Card>
            <CardHeader title="System Metrics" />
            <Divider />
            <CardContent>
              <Grid container spacing={3}>
                {metrics.map((metric) => (
                  <Grid item xs={12} sm={6} md={4} key={metric.id}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        {metric.name}
                      </Typography>
                      <Box display="flex" alignItems="center" mt={1}>
                        <Typography variant="h5" fontWeight="medium">
                          {metric.value.toLocaleString()} {metric.unit}
                        </Typography>
                        {metric.change !== undefined && (
                          <Chip
                            size="small"
                            label={`${metric.change >= 0 ? '+' : ''}${metric.changePercent?.toFixed(1)}%`}
                            color={metric.change >= 0 ? 'success' : 'error'}
                            sx={{ ml: 1 }}
                          />
                        )}
                      </Box>
                      {metric.target !== undefined && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            Target: {metric.target} {metric.unit}
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={Math.min((metric.value / metric.target) * 100, 100)}
                            color={
                              metric.status === 'good' ? 'success' :
                              metric.status === 'warning' ? 'warning' : 'error'
                            }
                            sx={{ height: 4, borderRadius: 5, mt: 0.5 }}
                          />
                        </Box>
                      )}
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* System Logs */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardHeader title="System Logs" />
            <Divider />
            <CardContent>
              {logs.length === 0 ? (
                <Typography variant="body1" color="text.secondary" textAlign="center" py={3}>
                  No system logs available
                </Typography>
              ) : (
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Time</TableCell>
                      <TableCell>Level</TableCell>
                      <TableCell>Component</TableCell>
                      <TableCell>Message</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {logs.map((log) => (
                      <TableRow key={log.id}>
                        <TableCell>
                          <Box display="flex" alignItems="center">
                            <TimeIcon fontSize="small" sx={{ mr: 0.5 }} />
                            {new Date(log.timestamp).toLocaleString()}
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={log.level.toUpperCase()}
                            size="small"
                            color={
                              log.level === 'error' || log.level === 'critical' ? 'error' :
                              log.level === 'warning' ? 'warning' :
                              log.level === 'info' ? 'info' : 'default'
                            }
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {log.component}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {log.message}
                            {log.details && (
                              <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5 }}>
                                {log.details}
                              </Typography>
                            )}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SystemStatusPage;