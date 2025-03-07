import React, { useRef, useEffect } from 'react';
import { Chart, ChartConfiguration, ChartOptions } from 'chart.js/auto';
import { Box, useTheme } from '@mui/material';

interface BarChartProps {
  labels: string[];
  data: number[];
  title?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  colors?: string[];
  height?: number;
  width?: string;
  showGrid?: boolean;
  animate?: boolean;
  showLegend?: boolean;
  horizontal?: boolean;
}

const BarChart: React.FC<BarChartProps> = ({
  labels,
  data,
  title = '',
  xAxisLabel = '',
  yAxisLabel = '',
  colors,
  height = 300,
  width = '100%',
  showGrid = true,
  animate = true,
  showLegend = false,
  horizontal = false,
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
    
    // Generate colors if not provided
    const defaultColors = colors || [
      theme.palette.primary.main,
      theme.palette.secondary.main,
      theme.palette.error.main,
      theme.palette.warning.main,
      theme.palette.info.main,
      theme.palette.success.main,
    ];
    
    // Ensure we have enough colors
    const chartColors = data.map((_, i) => defaultColors[i % defaultColors.length]);
    
    // Chart options
    const options: ChartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: horizontal ? 'y' : 'x',
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
        },
      },
      scales: {
        x: {
          title: {
            display: !!xAxisLabel,
            text: xAxisLabel,
            color: theme.palette.text.secondary,
          },
          grid: {
            display: showGrid && !horizontal,
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
            display: showGrid && horizontal,
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
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: title,
            data,
            backgroundColor: chartColors,
            borderColor: chartColors.map((color) => color),
            borderWidth: 1,
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
    labels,
    data,
    title,
    xAxisLabel,
    yAxisLabel,
    colors,
    showGrid,
    animate,
    showLegend,
    horizontal,
    theme,
  ]);
  
  return (
    <Box sx={{ height, width, position: 'relative' }}>
      <canvas ref={chartRef} />
    </Box>
  );
};

export default BarChart;