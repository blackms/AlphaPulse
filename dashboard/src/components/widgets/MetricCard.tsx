import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Tooltip,
  CircularProgress,
  useTheme,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
} from '@mui/icons-material';

interface MetricCardProps {
  title: string;
  value: number | string;
  unit?: string;
  previousValue?: number;
  percentChange?: number;
  isLoading?: boolean;
  icon?: React.ReactNode;
  tooltip?: string;
  precision?: number;
  showTrend?: boolean;
  trendThreshold?: number;
  onClick?: () => void;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit = '',
  previousValue,
  percentChange,
  isLoading = false,
  icon,
  tooltip,
  precision = 2,
  showTrend = true,
  trendThreshold = 0.1,
  onClick,
}) => {
  const theme = useTheme();
  
  // Format the value
  const formattedValue = typeof value === 'number' ? value.toLocaleString(undefined, {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  }) : value;
  
  // Determine trend
  let trendIcon: React.ReactElement | undefined = undefined;
  let trendColor = theme.palette.text.secondary;
  
  if (showTrend && percentChange !== undefined) {
    if (percentChange > trendThreshold) {
      trendIcon = <TrendingUpIcon fontSize="small" />;
      trendColor = theme.palette.success.main;
    } else if (percentChange < -trendThreshold) {
      trendIcon = <TrendingDownIcon fontSize="small" />;
      trendColor = theme.palette.error.main;
    } else {
      trendIcon = <TrendingFlatIcon fontSize="small" />;
      trendColor = theme.palette.warning.main;
    }
  }
  
  const cardContent = (
    <CardContent>
      <Typography variant="subtitle2" color="textSecondary" gutterBottom>
        {title}
      </Typography>
      
      {isLoading ? (
        <Box display="flex" justifyContent="center" alignItems="center" height={60}>
          <CircularProgress size={24} />
        </Box>
      ) : (
        <>
          <Typography variant="h4" component="div">
            {formattedValue}
            {unit && (
              <Typography variant="body2" component="span" color="textSecondary" sx={{ ml: 0.5 }}>
                {unit}
              </Typography>
            )}
          </Typography>
          
          {percentChange !== undefined && (
            <Box display="flex" alignItems="center" mt={1}>
              <Chip
                icon={trendIcon}
                label={`${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(2)}%`}
                size="small"
                sx={{
                  backgroundColor: `${trendColor}20`,
                  color: trendColor,
                  fontWeight: 'bold',
                }}
              />
            </Box>
          )}
        </>
      )}
    </CardContent>
  );
  
  const card = (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'transform 0.2s, box-shadow 0.2s',
        '&:hover': onClick ? {
          transform: 'translateY(-4px)',
          boxShadow: 4,
        } : {},
      }}
      onClick={onClick}
    >
      {icon && (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'flex-end',
            p: 1,
            color: theme.palette.primary.main,
          }}
        >
          {icon}
        </Box>
      )}
      {cardContent}
    </Card>
  );
  
  return tooltip ? (
    <Tooltip title={tooltip} arrow>
      {card}
    </Tooltip>
  ) : (
    card
  );
};

export default MetricCard;