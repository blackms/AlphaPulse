import React from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Typography,
  Divider,
  Grid,
  Box,
  LinearProgress,
  Tooltip,
  IconButton,
  CircularProgress,
  useTheme,
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  ArrowUpward as ArrowUpwardIcon,
  ArrowDownward as ArrowDownwardIcon,
} from '@mui/icons-material';
import { Portfolio } from '../../types';
import { PieChart } from '../charts';

interface PortfolioSummaryWidgetProps {
  portfolio: Portfolio | null;
  isLoading?: boolean;
  error?: string | null;
  onViewDetails?: () => void;
}

const PortfolioSummaryWidget: React.FC<PortfolioSummaryWidgetProps> = ({
  portfolio,
  isLoading = false,
  error = null,
  onViewDetails,
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
  
  // Format percentage
  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };
  
  // Get color based on value
  const getValueColor = (value: number) => {
    return value > 0 ? theme.palette.success.main : value < 0 ? theme.palette.error.main : theme.palette.text.secondary;
  };
  
  // Prepare data for pie chart
  const getPieChartData = () => {
    if (!portfolio || !portfolio.positions || portfolio.positions.length === 0) {
      return { labels: ['No Data'], data: [1] };
    }
    
    // Sort positions by value
    const sortedPositions = [...portfolio.positions].sort((a, b) => b.value - a.value);
    
    // Take top 5 positions and group the rest as "Others"
    const topPositions = sortedPositions.slice(0, 5);
    const otherPositions = sortedPositions.slice(5);
    
    const labels = topPositions.map((pos) => pos.symbol);
    const data = topPositions.map((pos) => pos.value);
    
    // Add "Others" category if there are more than 5 positions
    if (otherPositions.length > 0) {
      const otherValue = otherPositions.reduce((sum, pos) => sum + pos.value, 0);
      labels.push('Others');
      data.push(otherValue);
    }
    
    // Add cash
    labels.push('Cash');
    data.push(portfolio.cash);
    
    return { labels, data };
  };
  
  return (
    <Card>
      <CardHeader
        title="Portfolio Summary"
        action={
          <IconButton aria-label="settings">
            <MoreVertIcon />
          </IconButton>
        }
      />
      <Divider />
      <CardContent>
        {isLoading ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={300}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={300}>
            <Typography color="error">{error}</Typography>
          </Box>
        ) : !portfolio ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={300}>
            <Typography color="textSecondary">No portfolio data available</Typography>
          </Box>
        ) : (
          <Grid container spacing={3}>
            {/* Portfolio Value and Performance */}
            <Grid item xs={12} md={6}>
              <Box mb={3}>
                <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                  Total Value
                </Typography>
                <Typography variant="h4">{formatCurrency(portfolio.totalValue)}</Typography>
                
                <Box display="flex" alignItems="center" mt={0.5}>
                  {portfolio.pnl >= 0 ? (
                    <ArrowUpwardIcon fontSize="small" style={{ color: theme.palette.success.main }} />
                  ) : (
                    <ArrowDownwardIcon fontSize="small" style={{ color: theme.palette.error.main }} />
                  )}
                  <Typography
                    variant="body2"
                    style={{ color: getValueColor(portfolio.pnl) }}
                    sx={{ fontWeight: 'bold', ml: 0.5 }}
                  >
                    {formatCurrency(portfolio.pnl)} ({formatPercentage(portfolio.pnlPercentage)})
                  </Typography>
                </Box>
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                    Cash
                  </Typography>
                  <Typography variant="h6">{formatCurrency(portfolio.cash)}</Typography>
                  <Typography variant="body2" color="textSecondary">
                    {((portfolio.cash / portfolio.totalValue) * 100).toFixed(1)}% of portfolio
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                    Invested
                  </Typography>
                  <Typography variant="h6">
                    {formatCurrency(portfolio.totalValue - portfolio.cash)}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    {(((portfolio.totalValue - portfolio.cash) / portfolio.totalValue) * 100).toFixed(1)}% of portfolio
                  </Typography>
                </Grid>
              </Grid>
              
              <Box mt={3}>
                <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                  Positions
                </Typography>
                <Typography variant="h6">{portfolio.positions.length}</Typography>
              </Box>
            </Grid>
            
            {/* Asset Allocation Chart */}
            <Grid item xs={12} md={6}>
              <Box height={250}>
                <PieChart
                  {...getPieChartData()}
                  title="Asset Allocation"
                  doughnut={true}
                  cutout="60%"
                />
              </Box>
            </Grid>
            
            {/* Top Positions */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                Top Positions
              </Typography>
              
              {portfolio.positions.length === 0 ? (
                <Typography color="textSecondary">No positions</Typography>
              ) : (
                <Box>
                  {portfolio.positions.slice(0, 3).map((position) => (
                    <Box key={position.id} mb={1}>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2" fontWeight="bold">
                          {position.symbol}
                        </Typography>
                        <Typography variant="body2">{formatCurrency(position.value)}</Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography variant="caption" color="textSecondary">
                          {position.quantity} @ {formatCurrency(position.entryPrice)}
                        </Typography>
                        <Typography
                          variant="caption"
                          style={{ color: getValueColor(position.pnlPercentage) }}
                        >
                          {formatPercentage(position.pnlPercentage)}
                        </Typography>
                      </Box>
                      <Tooltip title={`${position.allocation.toFixed(1)}% of portfolio`}>
                        <LinearProgress
                          variant="determinate"
                          value={position.allocation}
                          sx={{ mt: 0.5, height: 4, borderRadius: 2 }}
                        />
                      </Tooltip>
                    </Box>
                  ))}
                </Box>
              )}
              
              {portfolio.positions.length > 3 && onViewDetails && (
                <Box display="flex" justifyContent="flex-end" mt={1}>
                  <Typography
                    variant="body2"
                    color="primary"
                    sx={{ cursor: 'pointer' }}
                    onClick={onViewDetails}
                  >
                    View All Positions
                  </Typography>
                </Box>
              )}
            </Grid>
          </Grid>
        )}
      </CardContent>
    </Card>
  );
};

export default PortfolioSummaryWidget;