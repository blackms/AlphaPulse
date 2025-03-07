import React, { useRef, useEffect } from 'react';
import { Chart, ChartConfiguration, ChartOptions } from 'chart.js/auto';
import { Box, useTheme } from '@mui/material';

interface PieChartProps {
  labels: string[];
  data: number[];
  title?: string;
  colors?: string[];
  height?: number;
  width?: string;
  animate?: boolean;
  showLegend?: boolean;
  doughnut?: boolean;
  cutout?: string;
  showPercentage?: boolean;
}

const PieChart: React.FC<PieChartProps> = ({
  labels,
  data,
  title = '',
  colors,
  height = 300,
  width = '100%',
  animate = true,
  showLegend = true,
  doughnut = false,
  cutout = '50%',
  showPercentage = true,
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
      '#8884d8',
      '#82ca9d',
      '#ffc658',
      '#8dd1e1',
    ];
    
    // Ensure we have enough colors
    const chartColors = data.map((_, i) => defaultColors[i % defaultColors.length]);
    
    // Calculate total for percentage
    const total = data.reduce((sum, value) => sum + value, 0);
    
    // Chart options
    const options: ChartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      animation: animate ? undefined : false,
      // @ts-ignore - cutout is valid for doughnut/pie charts but not in the type definition
      cutout: doughnut ? cutout : 0,
      plugins: {
        legend: {
          display: showLegend,
          position: 'right',
          labels: {
            color: theme.palette.text.primary,
            generateLabels: (chart) => {
              const datasets = chart.data.datasets;
              return chart.data.labels?.map((label, i) => {
                const meta = chart.getDatasetMeta(0);
                // @ts-ignore - Chart.js v3 API expects 1 argument, but TypeScript definition expects 2
                const style = meta.controller.getStyle(i);
                
                return {
                  text: showPercentage
                    ? `${label}: ${((data[i] / total) * 100).toFixed(1)}%`
                    : `${label}`,
                  fillStyle: style.backgroundColor,
                  strokeStyle: style.borderColor,
                  lineWidth: style.borderWidth,
                  hidden: !chart.getDataVisibility(i),
                  index: i,
                };
              }) || [];
            },
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
          callbacks: {
            label: (context) => {
              const value = context.parsed;
              const percentage = ((value / total) * 100).toFixed(1);
              return showPercentage
                ? `${context.label}: ${value} (${percentage}%)`
                : `${context.label}: ${value}`;
            },
          },
        },
      },
    };
    
    // Chart configuration
    const config: ChartConfiguration = {
      type: doughnut ? 'doughnut' : 'pie',
      data: {
        labels,
        datasets: [
          {
            data,
            backgroundColor: chartColors,
            borderColor: theme.palette.background.paper,
            borderWidth: 2,
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
    colors,
    animate,
    showLegend,
    doughnut,
    cutout,
    showPercentage,
    theme,
  ]);
  
  return (
    <Box sx={{ height, width, position: 'relative' }}>
      <canvas ref={chartRef} />
    </Box>
  );
};

export default PieChart;