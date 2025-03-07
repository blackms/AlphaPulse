import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Divider,
} from '@mui/material';

const DashboardPage: React.FC = () => {
  // Placeholder functions for formatting
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Performance Summary */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Portfolio Performance" />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                      Daily
                    </Typography>
                    <Typography
                      variant="h5"
                      color="success.main"
                    >
                      {formatPercentage(1.2)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                      Weekly
                    </Typography>
                    <Typography
                      variant="h5"
                      color="success.main"
                    >
                      {formatPercentage(3.7)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                      Monthly
                    </Typography>
                    <Typography
                      variant="h5"
                      color="success.main"
                    >
                      {formatPercentage(8.4)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box textAlign="center">
                    <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                      Yearly
                    </Typography>
                    <Typography
                      variant="h5"
                      color="error.main"
                    >
                      {formatPercentage(-2.1)}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Portfolio Summary */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Portfolio Summary" />
            <Divider />
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="subtitle1">Total Value</Typography>
                <Typography variant="h6">{formatCurrency(125000)}</Typography>
              </Box>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="subtitle1">Cash Balance</Typography>
                <Typography variant="h6">{formatCurrency(25000)}</Typography>
              </Box>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="subtitle1">Positions</Typography>
                <Typography variant="h6">7</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* System Status */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="System Status" />
            <Divider />
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System is operational
              </Typography>
              <Typography variant="body2" color="textSecondary">
                All components are running properly. Last check: {new Date().toLocaleString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;