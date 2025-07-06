import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Chip,
  Grid,
  LinearProgress,
  Alert,
  Tooltip,
  IconButton,
  Divider
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Thermostat as ThermostatIcon,
  Bolt as BoltIcon,
  Refresh as RefreshIcon,
  Computer as ComputerIcon
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface GPUDevice {
  id: string;
  name: string;
  utilization: number;
  memory_usage: number;
  temperature: number;
  status: string;
  memory_total?: number;
  memory_used?: number;
}

interface GPUMetrics {
  gpu_available: boolean;
  service_status: string;
  devices: Record<string, GPUDevice>;
  performance_summary: {
    total_devices: number;
    devices_available: number;
    avg_utilization: number;
    avg_memory_usage: number;
    total_models: number;
  };
  models: Record<string, any>;
}

const GPUMonitoringWidget: React.FC = () => {
  const theme = useTheme();
  const [gpuMetrics, setGpuMetrics] = useState<GPUMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchGPUMetrics = async () => {
    try {
      const response = await fetch('/api/v1/gpu/metrics');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setGpuMetrics(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch GPU metrics');
      console.error('Error fetching GPU metrics:', err);
    } finally {
      setLoading(false);
      setLastUpdate(new Date());
    }
  };

  useEffect(() => {
    fetchGPUMetrics();
    
    // Set up periodic updates every 30 seconds
    const interval = setInterval(fetchGPUMetrics, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running':
      case 'active':
        return 'success';
      case 'idle':
        return 'warning';
      case 'error':
      case 'unavailable':
        return 'error';
      default:
        return 'default';
    }
  };

  const getUtilizationColor = (utilization: number) => {
    if (utilization > 80) return theme.palette.error.main;
    if (utilization > 60) return theme.palette.warning.main;
    return theme.palette.success.main;
  };

  const getTemperatureColor = (temp: number) => {
    if (temp > 80) return theme.palette.error.main;
    if (temp > 70) return theme.palette.warning.main;
    return theme.palette.info.main;
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (loading) {
    return (
      <Card>
        <CardHeader 
          title="GPU Monitoring" 
          avatar={<ComputerIcon />}
        />
        <CardContent>
          <Box display="flex" justifyContent="center" p={2}>
            <Typography>Loading GPU metrics...</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error || !gpuMetrics?.gpu_available) {
    return (
      <Card>
        <CardHeader 
          title="GPU Monitoring" 
          avatar={<ComputerIcon />}
          action={
            <IconButton onClick={fetchGPUMetrics} size="small">
              <RefreshIcon />
            </IconButton>
          }
        />
        <CardContent>
          <Alert severity="warning">
            {error || 'GPU not available - running in CPU fallback mode'}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const devices = Object.entries(gpuMetrics.devices);
  const summary = gpuMetrics.performance_summary;

  return (
    <Card>
      <CardHeader 
        title="GPU Monitoring"
        avatar={<ComputerIcon />}
        action={
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={gpuMetrics.service_status} 
              color={getStatusColor(gpuMetrics.service_status)} 
              size="small" 
            />
            <IconButton onClick={fetchGPUMetrics} size="small">
              <RefreshIcon />
            </IconButton>
          </Box>
        }
        subheader={`Last updated: ${lastUpdate.toLocaleTimeString()}`}
      />
      <CardContent>
        {/* Performance Summary */}
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={3}>
            <Box textAlign="center">
              <Typography variant="h6" color="primary">
                {summary.total_devices}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Total GPUs
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={3}>
            <Box textAlign="center">
              <Typography variant="h6" color="success.main">
                {summary.devices_available}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Available
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={3}>
            <Box textAlign="center">
              <Typography variant="h6">
                {(summary.avg_utilization * 100).toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Avg Usage
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={3}>
            <Box textAlign="center">
              <Typography variant="h6" color="info.main">
                {summary.total_models}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Models
              </Typography>
            </Box>
          </Grid>
        </Grid>

        <Divider sx={{ my: 2 }} />

        {/* Individual GPU Devices */}
        {devices.map(([deviceId, device]) => (
          <Box key={deviceId} sx={{ mb: 2, p: 1, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <Typography variant="subtitle2" fontWeight="bold">
                GPU {deviceId}: {device.name}
              </Typography>
              <Chip 
                label={device.status} 
                color={getStatusColor(device.status)} 
                size="small" 
              />
            </Box>

            <Grid container spacing={1}>
              {/* GPU Utilization */}
              <Grid item xs={12} sm={6} md={3}>
                <Box display="flex" alignItems="center" gap={1}>
                  <SpeedIcon fontSize="small" />
                  <Box flex={1}>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="caption">Utilization</Typography>
                      <Typography variant="caption" fontWeight="bold">
                        {(device.utilization * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={device.utilization * 100}
                      sx={{ 
                        '& .MuiLinearProgress-bar': { 
                          backgroundColor: getUtilizationColor(device.utilization * 100) 
                        }
                      }}
                    />
                  </Box>
                </Box>
              </Grid>

              {/* Memory Usage */}
              <Grid item xs={12} sm={6} md={3}>
                <Box display="flex" alignItems="center" gap={1}>
                  <MemoryIcon fontSize="small" />
                  <Box flex={1}>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="caption">Memory</Typography>
                      <Typography variant="caption" fontWeight="bold">
                        {(device.memory_usage * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={device.memory_usage * 100}
                      color={device.memory_usage > 0.8 ? 'error' : device.memory_usage > 0.6 ? 'warning' : 'primary'}
                    />
                  </Box>
                </Box>
              </Grid>

              {/* Temperature */}
              <Grid item xs={12} sm={6} md={3}>
                <Tooltip title={`GPU Temperature: ${device.temperature}°C`}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <ThermostatIcon 
                      fontSize="small" 
                      sx={{ color: getTemperatureColor(device.temperature) }}
                    />
                    <Typography variant="body2">
                      {device.temperature}°C
                    </Typography>
                  </Box>
                </Tooltip>
              </Grid>

              {/* Performance Indicator */}
              <Grid item xs={12} sm={6} md={3}>
                <Box display="flex" alignItems="center" gap={1}>
                  <BoltIcon fontSize="small" color="warning" />
                  <Typography variant="body2">
                    {device.utilization > 0.7 ? 'High Load' : 
                     device.utilization > 0.3 ? 'Active' : 'Idle'}
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Box>
        ))}

        {/* Models Information */}
        {Object.keys(gpuMetrics.models).length > 0 && (
          <Box mt={2}>
            <Typography variant="subtitle2" gutterBottom>
              Loaded Models ({Object.keys(gpuMetrics.models).length})
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {Object.entries(gpuMetrics.models).map(([modelName, modelInfo]) => (
                <Chip 
                  key={modelName}
                  label={`${modelName} (${(modelInfo as any).type || 'unknown'})`}
                  size="small"
                  variant="outlined"
                  color="primary"
                />
              ))}
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default GPUMonitoringWidget;