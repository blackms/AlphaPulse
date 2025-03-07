import React, { useEffect, useState } from 'react';
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
  CardActions,
  Avatar,
  IconButton,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  CheckCircle as SuccessIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material';
import { RootState } from '../../store/store';
import { Alert } from '../../types/alerts';

// This is a placeholder component until the full implementation is complete
const AlertsPage: React.FC = () => {
  const dispatch = useDispatch();
  const { alerts, loading, error } = useSelector((state: RootState) => state.alerts);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    // Placeholder for fetching alerts
    // dispatch(fetchAlerts());
  }, [dispatch]);

  const handleRefresh = () => {
    // Placeholder for refreshing alerts
    // dispatch(fetchAlerts());
  };

  const handleFilterChange = (event: SelectChangeEvent) => {
    setFilter(event.target.value);
  };

  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'info':
        return <InfoIcon color="info" />;
      case 'success':
        return <SuccessIcon color="success" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  const getAlertColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      case 'success':
        return 'success';
      default:
        return 'info';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Alerts
        </Typography>
        <Box display="flex" alignItems="center">
          <FormControl variant="outlined" size="small" sx={{ minWidth: 120, mr: 2 }}>
            <InputLabel id="alert-filter-label">Filter</InputLabel>
            <Select
              labelId="alert-filter-label"
              id="alert-filter"
              value={filter}
              onChange={handleFilterChange}
              label="Filter"
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="critical">Critical</MenuItem>
              <MenuItem value="warning">Warning</MenuItem>
              <MenuItem value="info">Info</MenuItem>
              <MenuItem value="success">Success</MenuItem>
            </Select>
          </FormControl>
          <IconButton onClick={handleRefresh} disabled={loading}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {loading ? (
        <Typography>Loading alerts...</Typography>
      ) : error ? (
        <Paper sx={{ p: 2, bgcolor: 'error.light' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      ) : (
        <Grid container spacing={2}>
          {alerts && alerts.length > 0 ? (
            alerts.map((alert: Alert) => (
              <Grid item xs={12} key={alert.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={1}>
                      <Avatar sx={{ mr: 2, bgcolor: `${getAlertColor(alert.severity)}.main` }}>
                        {getAlertIcon(alert.severity)}
                      </Avatar>
                      <Box>
                        <Typography variant="h6">{alert.title}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {new Date(alert.timestamp).toLocaleString()}
                        </Typography>
                      </Box>
                      <Box flexGrow={1} />
                      <Chip
                        label={alert.severity}
                        color={getAlertColor(alert.severity) as any}
                        size="small"
                      />
                    </Box>
                    <Typography variant="body1">{alert.message}</Typography>
                    {alert.details && (
                      <Typography variant="body2" color="text.secondary" mt={1}>
                        {alert.details}
                      </Typography>
                    )}
                  </CardContent>
                  <CardActions>
                    <Button size="small" color="primary">
                      View Details
                    </Button>
                    {!alert.acknowledged && (
                      <Button size="small" color="secondary">
                        Acknowledge
                      </Button>
                    )}
                  </CardActions>
                </Card>
              </Grid>
            ))
          ) : (
            <Grid item xs={12}>
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <Typography>No alerts found.</Typography>
              </Paper>
            </Grid>
          )}
        </Grid>
      )}
    </Box>
  );
};

export default AlertsPage;