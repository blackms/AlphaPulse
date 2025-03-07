import React from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Typography,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Box,
  IconButton,
  CircularProgress,
  useTheme,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  MoreVert as MoreVertIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Pending as PendingIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { Trade } from '../../types';
import { formatDistanceToNow } from 'date-fns';

interface TradingActivityWidgetProps {
  trades: Trade[];
  isLoading?: boolean;
  error?: string | null;
  maxItems?: number;
  onViewAll?: () => void;
  onTradeClick?: (trade: Trade) => void;
}

const TradingActivityWidget: React.FC<TradingActivityWidgetProps> = ({
  trades,
  isLoading = false,
  error = null,
  maxItems = 5,
  onViewAll,
  onTradeClick,
}) => {
  const theme = useTheme();
  
  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };
  
  // Get trade icon
  const getTradeIcon = (trade: Trade) => {
    if (trade.type === 'buy') {
      return <TrendingUpIcon style={{ color: theme.palette.success.main }} />;
    } else {
      return <TrendingDownIcon style={{ color: theme.palette.error.main }} />;
    }
  };
  
  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'executed':
        return <CheckCircleIcon fontSize="small" color="success" />;
      case 'pending':
        return <PendingIcon fontSize="small" color="warning" />;
      case 'canceled':
        return <CancelIcon fontSize="small" color="error" />;
      case 'failed':
        return <ErrorIcon fontSize="small" color="error" />;
      default:
        return null;
    }
  };
  
  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'executed':
        return 'success';
      case 'pending':
        return 'warning';
      case 'canceled':
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };
  
  return (
    <Card>
      <CardHeader
        title="Recent Trades"
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
        ) : trades.length === 0 ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={200}>
            <Typography color="textSecondary">No recent trades</Typography>
          </Box>
        ) : (
          <List disablePadding>
            {trades.slice(0, maxItems).map((trade) => (
              <React.Fragment key={trade.id}>
                <ListItem
                  alignItems="flex-start"
                  button={!!onTradeClick}
                  onClick={() => onTradeClick && onTradeClick(trade)}
                  sx={{ px: 1 }}
                >
                  <ListItemIcon sx={{ minWidth: 40 }}>
                    {getTradeIcon(trade)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography variant="subtitle2">
                          {trade.type.toUpperCase()} {trade.quantity} {trade.symbol}
                        </Typography>
                        <Chip
                          label={trade.status}
                          size="small"
                          color={getStatusColor(trade.status) as any}
                          icon={getStatusIcon(trade.status)}
                          sx={{ ml: 1 }}
                        />
                      </Box>
                    }
                    secondary={
                      <Box mt={0.5}>
                        <Typography variant="body2" component="span">
                          {formatCurrency(trade.price)} per unit
                        </Typography>
                        <Typography variant="body2" component="span" sx={{ mx: 1 }}>
                          •
                        </Typography>
                        <Typography variant="body2" component="span" fontWeight="bold">
                          Total: {formatCurrency(trade.total)}
                        </Typography>
                        <Box mt={0.5}>
                          <Typography variant="caption" color="textSecondary">
                            {formatDistanceToNow(new Date(trade.timestamp), { addSuffix: true })}
                            {trade.signalSource && (
                              <>
                                {' · '}
                                Source: {trade.signalSource}
                              </>
                            )}
                          </Typography>
                        </Box>
                      </Box>
                    }
                  />
                </ListItem>
                {trades.indexOf(trade) < Math.min(trades.length, maxItems) - 1 && <Divider component="li" />}
              </React.Fragment>
            ))}
          </List>
        )}
        
        {!isLoading && !error && trades.length > 0 && onViewAll && (
          <Box display="flex" justifyContent="center" mt={2}>
            <Typography
              variant="body2"
              color="primary"
              sx={{ cursor: 'pointer' }}
              onClick={onViewAll}
            >
              View All Trades
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default TradingActivityWidget;