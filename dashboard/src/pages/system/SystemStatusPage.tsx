import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  Paper,
  Button,
} from '@mui/material';
import {
  CheckCircle as OperationalIcon,
  Warning as DegradedIcon,
  Info as MaintenanceIcon,
  Error as OutageIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import {
  selectSystemStatus,
  selectSystemMessage,
  selectSystemComponents,
  selectLastChecked,
  selectIsLoading,
  fetchSystemStatusStart,
  SystemStatus,
} from '../../store/slices/systemSlice';

const SystemStatusPage: React.FC = () => {
  const dispatch = useDispatch();
  const status = useSelector(selectSystemStatus);
  const message = useSelector(selectSystemMessage);
  const components = useSelector(selectSystemComponents);
  const lastChecked = useSelector(selectLastChecked);
  const loading = useSelector(selectIsLoading);

  const getStatusColor = (status: SystemStatus) => {
    switch (status) {
      case 'operational':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'maintenance':
        return 'info';
      case 'outage':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: SystemStatus) => {
    switch (status) {
      case 'operational':
        return <OperationalIcon />;
      case 'degraded':
        return <DegradedIcon />;
      case 'maintenance':
        return <MaintenanceIcon />;
      case 'outage':
        return <OutageIcon />;
      default:
        return <MaintenanceIcon />; // Using MaintenanceIcon instead of InfoIcon
    }
  };

  const formatTimestamp = (timestamp: number | null) => {
    if (!timestamp) return 'Never';
    return new Date(timestamp).toLocaleString();
  };

  const handleRefresh = () => {
    dispatch(fetchSystemStatusStart());
    // This would typically trigger an API call
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
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* System Status Overview */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="System Overview" />
            <Divider />
            <CardContent>
              <Box display="flex" alignItems="center">
                <Chip
                  label={status.toUpperCase()}
                  color={getStatusColor(status) as any}
                  icon={getStatusIcon(status)}
                  sx={{ mr: 2 }}
                />
                {message && (
                  <Typography variant="body1">{message}</Typography>
                )}
              </Box>
              <Typography variant="caption" display="block" mt={2} color="textSecondary">
                Last updated: {formatTimestamp(lastChecked)}
              </Typography>
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
                {components.map((component) => (
                  <Grid item xs={12} sm={6} md={4} key={component.id}>
                    <Box
                      component={Paper}
                      variant="outlined"
                      sx={{ p: 2, height: '100%' }}
                    >
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography variant="h6">{component.name}</Typography>
                        <Chip
                          size="small"
                          color={getStatusColor(component.status) as any}
                          icon={getStatusIcon(component.status)}
                        />
                      </Box>
                      {component.message && (
                        <Typography variant="body2" mt={1}>
                          {component.message}
                        </Typography>
                      )}
                      <Typography variant="caption" display="block" mt={1} color="textSecondary">
                        Last updated: {formatTimestamp(component.lastUpdated)}
                      </Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SystemStatusPage;