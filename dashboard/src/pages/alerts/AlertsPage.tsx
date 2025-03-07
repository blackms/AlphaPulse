import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  IconButton,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Button,
} from '@mui/material';
import {
  Notifications as AlertsIcon,
  CheckCircle as MarkReadIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import {
  selectAlerts,
  markAlertAsRead,
  markAllAlertsAsRead,
  deleteAlert,
  clearAllAlerts,
} from '../../store/slices/alertsSlice';
import { RootState } from '../../store/store';

const AlertsPage: React.FC = () => {
  const dispatch = useDispatch();
  const alerts = useSelector(selectAlerts);
  
  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const getAlertTypeColor = (type: string) => {
    switch (type) {
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'success':
        return 'success';
      case 'info':
      default:
        return 'info';
    }
  };

  const handleMarkAsRead = (id: string) => {
    dispatch(markAlertAsRead(id));
  };

  const handleMarkAllAsRead = () => {
    dispatch(markAllAlertsAsRead());
  };

  const handleDeleteAlert = (id: string) => {
    dispatch(deleteAlert(id));
  };

  const handleClearAllAlerts = () => {
    dispatch(clearAllAlerts());
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Alerts & Notifications
        </Typography>
        <Box>
          <Button
            variant="outlined"
            color="primary"
            onClick={handleMarkAllAsRead}
            sx={{ mr: 1 }}
          >
            Mark All Read
          </Button>
          <Button
            variant="outlined"
            color="secondary"
            onClick={handleClearAllAlerts}
          >
            Clear All
          </Button>
        </Box>
      </Box>

      <Card>
        <CardContent>
          {alerts.length === 0 ? (
            <Box textAlign="center" py={3}>
              <AlertsIcon sx={{ fontSize: 48, color: 'action.disabled', mb: 2 }} />
              <Typography variant="h6" color="textSecondary">
                No alerts to display
              </Typography>
            </Box>
          ) : (
            <List>
              {alerts.map((alert) => (
                <React.Fragment key={alert.id}>
                  <ListItem
                    alignItems="flex-start"
                    sx={{
                      bgcolor: alert.read ? 'inherit' : 'action.hover',
                      py: 1.5,
                    }}
                  >
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center">
                          <Typography variant="subtitle1" component="span" fontWeight={alert.read ? 400 : 600}>
                            {alert.title || 'System Alert'}
                          </Typography>
                          <Chip
                            label={alert.type}
                            color={getAlertTypeColor(alert.type) as any}
                            size="small"
                            sx={{ ml: 1 }}
                          />
                        </Box>
                      }
                      secondary={
                        <>
                          <Typography
                            component="span"
                            variant="body2"
                            color="textPrimary"
                            display="block"
                            mt={1}
                          >
                            {alert.message}
                          </Typography>
                          <Typography
                            component="span"
                            variant="caption"
                            color="textSecondary"
                            display="block"
                            mt={1}
                          >
                            {formatTimestamp(alert.timestamp)}
                          </Typography>
                        </>
                      }
                    />
                    <ListItemSecondaryAction>
                      {!alert.read && (
                        <IconButton
                          edge="end"
                          aria-label="mark as read"
                          onClick={() => handleMarkAsRead(alert.id)}
                          sx={{ mr: 1 }}
                        >
                          <MarkReadIcon />
                        </IconButton>
                      )}
                      <IconButton
                        edge="end"
                        aria-label="delete"
                        onClick={() => handleDeleteAlert(alert.id)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                  <Divider component="li" />
                </React.Fragment>
              ))}
            </List>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default AlertsPage;