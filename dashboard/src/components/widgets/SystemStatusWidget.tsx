import React from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Typography,
  Divider,
  Grid,
  Box,
  Chip,
  LinearProgress,
  IconButton,
  CircularProgress,
  useTheme,
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Build as BuildIcon,
} from '@mui/icons-material';
import { SystemStatus, SystemMetrics } from '../../types';

interface SystemStatusWidgetProps {
  status: SystemStatus | null;
  metrics: SystemMetrics | null;
  isLoading?: boolean;
  error?: string | null;
  onViewDetails?: () => void;
}

const SystemStatusWidget: React.FC<SystemStatusWidgetProps> = ({
  status,
  metrics,
  isLoading = false,
  error = null,
  onViewDetails,
}) => {
  const theme = useTheme();
  
  // Get status icon
  const getStatusIcon = (statusType: string): React.ReactElement => {
    switch (statusType) {
      case 'operational':
        return <CheckCircleIcon fontSize="small" />;
      case 'degraded':
        return <WarningIcon fontSize="small" />;
      case 'maintenance':
        return <BuildIcon fontSize="small" />;
      case 'outage':
        return <ErrorIcon fontSize="small" />;
      default:
        return <CheckCircleIcon fontSize="small" color="disabled" />;
    }
  };
  
  // Get status color
  const getStatusColor = (statusType: string) => {
    switch (statusType) {
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
  
  // Format uptime
  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / (3600 * 24));
    const hours = Math.floor((seconds % (3600 * 24)) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  };
  
  // Get color for resource usage
  const getResourceColor = (usage: number) => {
    if (usage > 90) {
      return theme.palette.error.main;
    } else if (usage > 70) {
      return theme.palette.warning.main;
    } else {
      return theme.palette.success.main;
    }
  };
  
  return (
    <Card>
      <CardHeader
        title="System Status"
        action={
          <IconButton aria-label="settings">
            <MoreVertIcon />
          </IconButton>
        }
      />
      <Divider />
      <CardContent>
        {isLoading ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={200}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={200}>
            <Typography color="error">{error}</Typography>
          </Box>
        ) : !status ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={200}>
            <Typography color="textSecondary">No system status available</Typography>
          </Box>
        ) : (
          <Grid container spacing={3}>
            {/* Overall System Status */}
            <Grid item xs={12}>
              <Box display="flex" alignItems="center" mb={2}>
                <Chip
                  label={status.status}
                  color={getStatusColor(status.status) as any}
                  icon={getStatusIcon(status.status)}
                  sx={{ mr: 2 }}
                />
                {status.message && (
                  <Typography variant="body2" color="textSecondary">
                    {status.message}
                  </Typography>
                )}
              </Box>
              <Typography variant="caption" color="textSecondary">
                Last updated: {new Date(status.updatedAt).toLocaleString()}
              </Typography>
            </Grid>
            
            {/* Component Status */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Component Status
              </Typography>
              <Grid container spacing={1}>
                {Object.entries(status.components).map(([name, component]) => (
                  <Grid item xs={12} sm={6} key={name}>
                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        p: 1,
                        borderRadius: 1,
                        bgcolor: `${getStatusColor(component.status)}.light`,
                        mb: 1,
                      }}
                    >
                      <Typography variant="body2">{name}</Typography>
                      <Chip
                        label={component.status}
                        size="small"
                        color={getStatusColor(component.status) as any}
                        icon={getStatusIcon(component.status)}
                      />
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Grid>
            
            {/* System Metrics */}
            {metrics && (
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  System Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">
                      CPU Usage
                    </Typography>
                    <Box display="flex" alignItems="center">
                      <Box width="100%" mr={1}>
                        <LinearProgress
                          variant="determinate"
                          value={metrics.cpu}
                          sx={{
                            height: 8,
                            borderRadius: 5,
                            bgcolor: theme.palette.grey[200],
                            '& .MuiLinearProgress-bar': {
                              bgcolor: getResourceColor(metrics.cpu),
                            },
                          }}
                        />
                      </Box>
                      <Typography variant="body2" color="textSecondary">
                        {metrics.cpu}%
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">
                      Memory Usage
                    </Typography>
                    <Box display="flex" alignItems="center">
                      <Box width="100%" mr={1}>
                        <LinearProgress
                          variant="determinate"
                          value={metrics.memory}
                          sx={{
                            height: 8,
                            borderRadius: 5,
                            bgcolor: theme.palette.grey[200],
                            '& .MuiLinearProgress-bar': {
                              bgcolor: getResourceColor(metrics.memory),
                            },
                          }}
                        />
                      </Box>
                      <Typography variant="body2" color="textSecondary">
                        {metrics.memory}%
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">
                      Disk Usage
                    </Typography>
                    <Box display="flex" alignItems="center">
                      <Box width="100%" mr={1}>
                        <LinearProgress
                          variant="determinate"
                          value={metrics.disk}
                          sx={{
                            height: 8,
                            borderRadius: 5,
                            bgcolor: theme.palette.grey[200],
                            '& .MuiLinearProgress-bar': {
                              bgcolor: getResourceColor(metrics.disk),
                            },
                          }}
                        />
                      </Box>
                      <Typography variant="body2" color="textSecondary">
                        {metrics.disk}%
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Box>
                      <Typography variant="body2" color="textSecondary">
                        Network I/O
                      </Typography>
                      <Typography variant="body1">
                        {metrics.network.in.toFixed(2)} MB/s in, {metrics.network.out.toFixed(2)} MB/s out
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box>
                      <Typography variant="body2" color="textSecondary">
                        Processes
                      </Typography>
                      <Typography variant="body1">{metrics.processCount}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box>
                      <Typography variant="body2" color="textSecondary">
                        Uptime
                      </Typography>
                      <Typography variant="body1">{formatUptime(metrics.uptime)}</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Grid>
            )}
            
            {onViewDetails && (
              <Grid item xs={12}>
                <Box display="flex" justifyContent="flex-end">
                  <Typography
                    variant="body2"
                    color="primary"
                    sx={{ cursor: 'pointer' }}
                    onClick={onViewDetails}
                  >
                    View System Details
                  </Typography>
                </Box>
              </Grid>
            )}
          </Grid>
        )}
      </CardContent>
    </Card>
  );
};

export default SystemStatusWidget;