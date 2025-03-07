import React, { useRef, useEffect } from 'react';
import { Chart, ChartConfiguration, ChartOptions } from 'chart.js/auto';
import { Box, useTheme } from '@mui/material';

interface DataPoint {
  timestamp: string;
  value: number;
}

interface LineChartProps {
  data: DataPoint[];
  title?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  color?: string;
  height?: number;
  width?: string;
  showGrid?: boolean;
  timeFormat?: string;
  tooltipFormat?: string;
  animate?: boolean;
  showLegend?: boolean;
}

const LineChart: React.FC<LineChartProps> = ({
  data,
  title = '',
  xAxisLabel = '',
  yAxisLabel = '',
  color,
  height = 300,
  width = '100%',
  showGrid = true,
  timeFormat = 'MM/dd HH:mm',
  tooltipFormat = 'MMM dd, yyyy HH:mm',
  animate = true,
  showLegend = false,
}) => {
  const chartRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstance = useRef<Chart | null>(null);
  const theme = useTheme();
  
  useEffect(() => {
    if (!chartRef.current) return;
    
    // Destroy existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }
    
    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;
    
    // Prepare data
    const labels = data.map((point) => new Date(point.timestamp));
    const values = data.map((point) => point.value);
    
    // Chart options
    const options: ChartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      animation: animate ? undefined : false,
      plugins: {
        legend: {
          display: showLegend,
          position: 'top',
          labels: {
            color: theme.palette.text.primary,
          },
        },
        title: {
          display: !!title,
          text: title,
          color: theme.palette.text.primary,
          font: {
            size: 16,
            weight: 'bold',
          },
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: (context) => {
              return `${context.dataset.label || ''}: ${context.parsed.y}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'hour',
            displayFormats: {
              hour: timeFormat,
            },
            tooltipFormat: tooltipFormat,
          },
          title: {
            display: !!xAxisLabel,
            text: xAxisLabel,
            color: theme.palette.text.secondary,
          },
          grid: {
            display: showGrid,
            color: theme.palette.divider,
          },
          ticks: {
            color: theme.palette.text.secondary,
          },
        },
        y: {
          title: {
            display: !!yAxisLabel,
            text: yAxisLabel,
            color: theme.palette.text.secondary,
          },
          grid: {
            display: showGrid,
            color: theme.palette.divider,
          },
          ticks: {
            color: theme.palette.text.secondary,
          },
        },
      },
    };
    
    // Chart configuration
    const config: ChartConfiguration = {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: title,
            data: values,
            borderColor: color || theme.palette.primary.main,
            backgroundColor: color
              ? `${color}33` // Add transparency
              : `${theme.palette.primary.main}33`,
            borderWidth: 2,
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.2, // Slight curve
            fill: true,
          },
        ],
      },
      options,
    };
    
    // Create chart
    chartInstance.current = new Chart(ctx, config);
    
    // Cleanup on unmount
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [
    data,
    title,
    xAxisLabel,
    yAxisLabel,
    color,
    showGrid,
    timeFormat,
    tooltipFormat,
    animate,
    showLegend,
    theme,
  ]);
  
  return (
    <Box sx={{ height, width, position: 'relative' }}>
      <canvas ref={chartRef} />
    </Box>
  );
};

export default LineChart;