import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Grid,
  LinearProgress,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  DataUsage as DataUsageIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Timeline as TimelineIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Speed as SpeedIcon,
  Shield as ShieldIcon
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface QualityMetrics {
  overall_score: number;
  completeness: number;
  accuracy: number;
  consistency: number;
  timeliness: number;
  validity: number;
  uniqueness: number;
  anomaly_rate: number;
  data_volume: number;
  source_reliability: number;
}

interface QualityAlert {
  id: string;
  symbol: string;
  metric: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  timestamp: string;
  threshold_violated: number;
  current_value: number;
}

interface DataSourceStatus {
  source_id: string;
  source_name: string;
  status: 'active' | 'degraded' | 'failed';
  reliability_score: number;
  last_update: string;
  quality_score: number;
  data_volume: number;
  error_count: number;
}

interface QuarantinedData {
  id: string;
  symbol: string;
  reason: string;
  quarantine_time: string;
  quality_score: number;
  data_points: number;
}

const DataQualityWidget: React.FC = () => {
  const theme = useTheme();
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics | null>(null);
  const [qualityAlerts, setQualityAlerts] = useState<QualityAlert[]>([]);
  const [dataSources, setDataSources] = useState<DataSourceStatus[]>([]);
  const [quarantinedData, setQuarantinedData] = useState<QuarantinedData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('ALL');
  const [configDialogOpen, setConfigDialogOpen] = useState(false);

  const symbols = ['ALL', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']; // In real app, fetch from API

  const fetchQualityData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch quality metrics
      const metricsResponse = await fetch(`/api/v1/data-quality/metrics?symbol=${selectedSymbol}`);
      if (!metricsResponse.ok) {
        throw new Error(`HTTP ${metricsResponse.status}: ${metricsResponse.statusText}`);
      }
      const metrics = await metricsResponse.json();
      setQualityMetrics(metrics);

      // Fetch quality alerts
      const alertsResponse = await fetch(`/api/v1/data-quality/alerts?limit=20&symbol=${selectedSymbol}`);
      if (alertsResponse.ok) {
        const alerts = await alertsResponse.json();
        setQualityAlerts(alerts);
      }

      // Fetch data source status
      const sourcesResponse = await fetch('/api/v1/data-quality/sources');
      if (sourcesResponse.ok) {
        const sources = await sourcesResponse.json();
        setDataSources(sources);
      }

      // Fetch quarantined data
      const quarantineResponse = await fetch('/api/v1/data-quality/quarantine?limit=10');
      if (quarantineResponse.ok) {
        const quarantine = await quarantineResponse.json();
        setQuarantinedData(quarantine);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data quality metrics');
      console.error('Error fetching data quality data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQualityData();
    
    // Set up periodic updates every 30 seconds
    const interval = setInterval(fetchQualityData, 30000);
    
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  const getQualityColor = (score: number) => {
    if (score >= 0.9) return theme.palette.success.main;
    if (score >= 0.7) return theme.palette.warning.main;
    return theme.palette.error.main;
  };

  const getQualityLabel = (score: number) => {
    if (score >= 0.9) return 'Excellent';
    if (score >= 0.8) return 'Good';
    if (score >= 0.7) return 'Fair';
    if (score >= 0.6) return 'Poor';
    return 'Critical';
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <ErrorIcon color="error" />;
      case 'warning': return <WarningIcon color="warning" />;
      case 'info': return <InfoIcon color="info" />;
      default: return <InfoIcon />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircleIcon color="success" />;
      case 'degraded': return <WarningIcon color="warning" />;
      case 'failed': return <ErrorIcon color="error" />;
      default: return <InfoIcon />;
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader 
          title="Data Quality Monitoring" 
          avatar={<DataUsageIcon />}
        />
        <CardContent>
          <Box display="flex" justifyContent="center" p={2}>
            <Typography>Loading data quality metrics...</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error || !qualityMetrics) {
    return (
      <Card>
        <CardHeader 
          title="Data Quality Monitoring" 
          avatar={<DataUsageIcon />}
          action={
            <IconButton onClick={fetchQualityData}>
              <RefreshIcon />
            </IconButton>
          }
        />
        <CardContent>
          <Alert severity="error">
            {error || 'Unable to load data quality metrics'}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        title="Data Quality Monitoring"
        avatar={<DataUsageIcon />}
        action={
          <Box display="flex" gap={1} alignItems="center">
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Symbol</InputLabel>
              <Select
                value={selectedSymbol}
                label="Symbol"
                onChange={(e) => setSelectedSymbol(e.target.value)}
              >
                {symbols.map(symbol => (
                  <MenuItem key={symbol} value={symbol}>{symbol}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <IconButton onClick={() => setConfigDialogOpen(true)}>
              <SettingsIcon />
            </IconButton>
            <IconButton onClick={fetchQualityData}>
              <RefreshIcon />
            </IconButton>
          </Box>
        }
      />
      <CardContent>
        {/* Overall Quality Score */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ 
                  color: getQualityColor(qualityMetrics.overall_score),
                  fontWeight: 'bold' 
                }}>
                  {(qualityMetrics.overall_score * 100).toFixed(1)}%
                </Typography>
                <Typography variant="h6" color="textSecondary">
                  Overall Quality
                </Typography>
                <Chip 
                  label={getQualityLabel(qualityMetrics.overall_score)}
                  sx={{ 
                    backgroundColor: getQualityColor(qualityMetrics.overall_score),
                    color: 'white',
                    mt: 1
                  }}
                />
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ 
                  color: qualityMetrics.anomaly_rate < 0.02 ? theme.palette.success.main : theme.palette.error.main,
                  fontWeight: 'bold' 
                }}>
                  {(qualityMetrics.anomaly_rate * 100).toFixed(2)}%
                </Typography>
                <Typography variant="h6" color="textSecondary">
                  Anomaly Rate
                </Typography>
                <Box display="flex" alignItems="center" justifyContent="center" mt={1}>
                  <ShieldIcon fontSize="small" sx={{ mr: 0.5 }} />
                  <Typography variant="caption">
                    {qualityMetrics.anomaly_rate < 0.02 ? 'Normal' : 'Elevated'}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ 
                  color: theme.palette.primary.main,
                  fontWeight: 'bold' 
                }}>
                  {qualityMetrics.data_volume.toLocaleString()}
                </Typography>
                <Typography variant="h6" color="textSecondary">
                  Data Points
                </Typography>
                <Box display="flex" alignItems="center" justifyContent="center" mt={1}>
                  <TimelineIcon fontSize="small" sx={{ mr: 0.5 }} />
                  <Typography variant="caption">
                    Last 24h
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Quality Dimensions */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Quality Dimensions</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              {[
                { label: 'Completeness', value: qualityMetrics.completeness, icon: <CheckCircleIcon /> },
                { label: 'Accuracy', value: qualityMetrics.accuracy, icon: <SpeedIcon /> },
                { label: 'Consistency', value: qualityMetrics.consistency, icon: <ShieldIcon /> },
                { label: 'Timeliness', value: qualityMetrics.timeliness, icon: <TimelineIcon /> },
                { label: 'Validity', value: qualityMetrics.validity, icon: <CheckCircleIcon /> },
                { label: 'Uniqueness', value: qualityMetrics.uniqueness, icon: <DataUsageIcon /> }
              ].map((dimension, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                    <Box display="flex" alignItems="center" mb={1}>
                      {dimension.icon}
                      <Typography variant="subtitle2" sx={{ ml: 1, fontWeight: 'bold' }}>
                        {dimension.label}
                      </Typography>
                    </Box>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="h6" sx={{ color: getQualityColor(dimension.value) }}>
                        {(dimension.value * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={dimension.value * 100}
                      sx={{
                        height: 8,
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: getQualityColor(dimension.value)
                        }
                      }}
                    />
                  </Box>
                </Grid>
              ))}
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* Quality Alerts */}
        {qualityAlerts.length > 0 && (
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">
                Quality Alerts ({qualityAlerts.length})
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <List>
                {qualityAlerts.slice(0, 5).map((alert, index) => (
                  <ListItem key={index} divider>
                    <ListItemIcon>
                      {getSeverityIcon(alert.severity)}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" justifyContent="space-between">
                          <Typography variant="body1">{alert.message}</Typography>
                          <Chip 
                            label={alert.severity.toUpperCase()}
                            size="small"
                            color={
                              alert.severity === 'critical' ? 'error' :
                              alert.severity === 'warning' ? 'warning' : 'info'
                            }
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="caption" color="textSecondary">
                            Symbol: {alert.symbol} | Metric: {alert.metric} | 
                            Value: {alert.current_value} | Threshold: {alert.threshold_violated}
                          </Typography>
                          <br />
                          <Typography variant="caption" color="textSecondary">
                            {new Date(alert.timestamp).toLocaleString()}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Data Sources Status */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Data Sources ({dataSources.length})</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Source</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Reliability</TableCell>
                  <TableCell>Quality</TableCell>
                  <TableCell>Volume</TableCell>
                  <TableCell>Errors</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {dataSources.map((source, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        {getStatusIcon(source.status)}
                        <Typography variant="body2" sx={{ ml: 1 }}>
                          {source.source_name}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={source.status}
                        size="small"
                        color={
                          source.status === 'active' ? 'success' :
                          source.status === 'degraded' ? 'warning' : 'error'
                        }
                      />
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          {(source.reliability_score * 100).toFixed(0)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={source.reliability_score * 100}
                          sx={{ width: 60, height: 4 }}
                          color={source.reliability_score > 0.8 ? 'success' : 'warning'}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ 
                        color: getQualityColor(source.quality_score) 
                      }}>
                        {(source.quality_score * 100).toFixed(0)}%
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {source.data_volume.toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color={source.error_count > 0 ? 'error' : 'textSecondary'}>
                        {source.error_count}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </AccordionDetails>
        </Accordion>

        {/* Quarantined Data */}
        {quarantinedData.length > 0 && (
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">
                Quarantined Data ({quarantinedData.length})
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Reason</TableCell>
                    <TableCell>Quality Score</TableCell>
                    <TableCell>Data Points</TableCell>
                    <TableCell>Quarantine Time</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {quarantinedData.map((item, index) => (
                    <TableRow key={index}>
                      <TableCell>{item.symbol}</TableCell>
                      <TableCell>{item.reason}</TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ 
                          color: getQualityColor(item.quality_score) 
                        }}>
                          {(item.quality_score * 100).toFixed(1)}%
                        </Typography>
                      </TableCell>
                      <TableCell>{item.data_points}</TableCell>
                      <TableCell>
                        <Typography variant="caption">
                          {new Date(item.quarantine_time).toLocaleString()}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </AccordionDetails>
          </Accordion>
        )}
      </CardContent>

      {/* Configuration Dialog */}
      <Dialog open={configDialogOpen} onClose={() => setConfigDialogOpen(false)}>
        <DialogTitle>Data Quality Configuration</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
            Configure data quality thresholds and monitoring settings.
          </Typography>
          {/* Add configuration options here */}
          <Alert severity="info">
            Configuration interface coming soon. Current settings are managed automatically.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfigDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
};

export default DataQualityWidget;