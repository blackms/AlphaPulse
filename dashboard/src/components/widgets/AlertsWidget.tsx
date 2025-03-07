import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Typography,
  Divider,
  Button,
  Box,
  Chip,
  IconButton,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  MoreVert as MoreVertIcon,
} from '@mui/icons-material';
import { Alert } from '../../types';
import { formatDistanceToNow } from 'date-fns';

interface AlertsWidgetProps {
  alerts: Alert[];
  isLoading?: boolean;
  error?: string | null;
  maxItems?: number;
  onViewAll?: () => void;
  onAcknowledge?: (alertId: string) => void;
  onAlertClick?: (alert: Alert) => void;
}

const AlertsWidget: React.FC<AlertsWidgetProps> = ({
  alerts,
  isLoading = false,
  error = null,
  maxItems = 5,
  onViewAll,
  onAcknowledge,
  onAlertClick,
}) => {
  const [expandedAlert, setExpandedAlert] = useState<string | null>(null);
  
  const toggleExpand = (alertId: string) => {
    setExpandedAlert(expandedAlert === alertId ? null : alertId);
  };
  
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'info':
      default:
        return <InfoIcon color="info" />;
    }
  };
  
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
      default:
        return 'info';
    }
  };
  
  return (
    <Card>
      <CardHeader
        title="Recent Alerts"
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
        ) : alerts.length === 0 ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={200}>
            <Typography color="textSecondary">No alerts to display</Typography>
          </Box>
        ) : (
          <List disablePadding>
            {alerts.slice(0, maxItems).map((alert) => (
              <React.Fragment key={alert.id}>
                <ListItem
                  alignItems="flex-start"
                  button
                  onClick={() => onAlertClick && onAlertClick(alert)}
                  sx={{ px: 1 }}
                >
                  <ListItemIcon sx={{ minWidth: 40 }}>
                    {getSeverityIcon(alert.severity)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography
                          variant="subtitle2"
                          sx={{
                            fontWeight: alert.acknowledged ? 'normal' : 'bold',
                            color: alert.acknowledged ? 'text.secondary' : 'text.primary',
                          }}
                        >
                          {alert.message}
                        </Typography>
                        <Chip
                          label={alert.severity}
                          size="small"
                          color={getSeverityColor(alert.severity) as any}
                          sx={{ ml: 1 }}
                        />
                      </Box>
                    }
                    secondary={
                      <Box mt={0.5}>
                        <Typography variant="caption" color="textSecondary" component="span">
                          {formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })}
                          {' · '}
                          {alert.source}
                        </Typography>
                        
                        {expandedAlert === alert.id && (
                          <Box mt={1}>
                            {alert.acknowledged ? (
                              <Box display="flex" alignItems="center">
                                <CheckCircleIcon color="success" fontSize="small" sx={{ mr: 0.5 }} />
                                <Typography variant="caption">
                                  Acknowledged by {alert.acknowledgedBy} at{' '}
                                  {new Date(alert.acknowledgedAt || '').toLocaleString()}
                                </Typography>
                              </Box>
                            ) : (
                              onAcknowledge && (
                                <Button
                                  size="small"
                                  variant="outlined"
                                  color="primary"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onAcknowledge(alert.id);
                                  }}
                                >
                                  Acknowledge
                                </Button>
                              )
                            )}
                          </Box>
                        )}
                      </Box>
                    }
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleExpand(alert.id);
                    }}
                  />
                </ListItem>
                {alerts.indexOf(alert) < Math.min(alerts.length, maxItems) - 1 && <Divider component="li" />}
              </React.Fragment>
            ))}
          </List>
        )}
        
        {!isLoading && !error && alerts.length > 0 && onViewAll && (
          <Box display="flex" justifyContent="center" mt={2}>
            <Button color="primary" onClick={onViewAll}>
              View All Alerts
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default AlertsWidget;