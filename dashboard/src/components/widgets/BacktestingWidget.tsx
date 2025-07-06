import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Grid,
  Button,
  TextField,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tabs,
  Tab,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Compare as CompareIcon,
  Timeline as TimelineIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface BacktestResult {
  symbol: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  benchmark_return: number;
  alpha: number;
  beta: number;
  execution_time_ms: number;
}

interface BacktestResponse {
  request_id: string;
  status: string;
  results: Record<string, BacktestResult>;
  data_source_used: string;
  execution_time_ms: number;
  metadata: any;
}

interface DataSourceComparison {
  comparison_results: Record<string, any>;
  recommendation: string;
  performance_summary: Record<string, any>;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`backtest-tabpanel-${index}`}
      aria-labelledby={`backtest-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

const BacktestingWidget: React.FC = () => {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backtestResults, setBacktestResults] = useState<BacktestResponse | null>(null);
  const [comparison, setComparison] = useState<DataSourceComparison | null>(null);
  const [dataLakeStatus, setDataLakeStatus] = useState<any>(null);
  
  // Backtest form state
  const [symbols, setSymbols] = useState('AAPL,GOOGL,MSFT');
  const [timeframe, setTimeframe] = useState('1d');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [strategyType, setStrategyType] = useState('simple_ma');
  const [dataSource, setDataSource] = useState('auto');
  const [initialCapital, setInitialCapital] = useState(100000);
  const [commission, setCommission] = useState(0.002);
  
  // Strategy parameters
  const [shortWindow, setShortWindow] = useState(20);
  const [longWindow, setLongWindow] = useState(50);
  const [rsiPeriod, setRsiPeriod] = useState(14);
  const [oversold, setOversold] = useState(30);
  const [overbought, setOverbought] = useState(70);

  useEffect(() => {
    fetchDataLakeStatus();
  }, []);

  const fetchDataLakeStatus = async () => {
    try {
      const response = await fetch('/api/v1/backtesting/data-lake/status');
      if (response.ok) {
        const status = await response.json();
        setDataLakeStatus(status);
      }
    } catch (err) {
      console.error('Failed to fetch data lake status:', err);
    }
  };

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const strategyParams = strategyType === 'simple_ma' 
        ? { short_window: shortWindow, long_window: longWindow }
        : { rsi_period: rsiPeriod, oversold: oversold, overbought: overbought };

      const request = {
        symbols: symbols.split(',').map(s => s.trim()),
        timeframe,
        start_date: new Date(startDate).toISOString(),
        end_date: new Date(endDate).toISOString(),
        strategy_type: strategyType,
        strategy_params: strategyParams,
        initial_capital: initialCapital,
        commission: commission / 100, // Convert percentage to decimal
        data_source: dataSource
      };

      const response = await fetch('/api/v1/backtesting/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      setBacktestResults(result);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Backtest failed');
    } finally {
      setLoading(false);
    }
  };

  const compareDataSources = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const request = {
        symbols: symbols.split(',').map(s => s.trim()),
        timeframe,
        start_date: new Date(startDate).toISOString(),
        end_date: new Date(endDate).toISOString()
      };

      const response = await fetch('/api/v1/backtesting/compare-sources', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      setComparison(result);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed');
    } finally {
      setLoading(false);
    }
  };

  const getReturnColor = (value: number) => {
    if (value > 0) return theme.palette.success.main;
    if (value < 0) return theme.palette.error.main;
    return theme.palette.text.secondary;
  };

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatNumber = (value: number) => value.toFixed(2);

  return (
    <Card>
      <CardHeader
        title="Enhanced Backtesting with Data Lake"
        avatar={<AssessmentIcon />}
        action={
          <Box display="flex" gap={1}>
            <Tooltip title="Data Lake Status">
              <IconButton>
                <StorageIcon color={dataLakeStatus?.data_lake_available ? 'success' : 'error'} />
              </IconButton>
            </Tooltip>
            <IconButton onClick={fetchDataLakeStatus}>
              <RefreshIcon />
            </IconButton>
          </Box>
        }
      />
      <CardContent>
        <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
          <Tab label="Backtest" />
          <Tab label="Data Source Comparison" />
          <Tab label="Results" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            {/* Configuration */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader title="Backtest Configuration" />
                <CardContent>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Symbols (comma-separated)"
                        value={symbols}
                        onChange={(e) => setSymbols(e.target.value)}
                        helperText="e.g., AAPL,GOOGL,MSFT"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <FormControl fullWidth>
                        <InputLabel>Timeframe</InputLabel>
                        <Select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
                          <MenuItem value="1d">1 Day</MenuItem>
                          <MenuItem value="1h">1 Hour</MenuItem>
                          <MenuItem value="4h">4 Hour</MenuItem>
                          <MenuItem value="1w">1 Week</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={6}>
                      <FormControl fullWidth>
                        <InputLabel>Data Source</InputLabel>
                        <Select value={dataSource} onChange={(e) => setDataSource(e.target.value)}>
                          <MenuItem value="auto">Auto (Recommended)</MenuItem>
                          <MenuItem value="data_lake">Data Lake</MenuItem>
                          <MenuItem value="database">Database</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        type="date"
                        label="Start Date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        type="date"
                        label="End Date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Initial Capital"
                        value={initialCapital}
                        onChange={(e) => setInitialCapital(Number(e.target.value))}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Commission (%)"
                        value={commission * 100}
                        onChange={(e) => setCommission(Number(e.target.value) / 100)}
                        inputProps={{ step: 0.01 }}
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Strategy Parameters */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader title="Strategy Configuration" />
                <CardContent>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <FormControl fullWidth>
                        <InputLabel>Strategy</InputLabel>
                        <Select value={strategyType} onChange={(e) => setStrategyType(e.target.value)}>
                          <MenuItem value="simple_ma">Moving Average Crossover</MenuItem>
                          <MenuItem value="rsi">RSI Strategy</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    
                    {strategyType === 'simple_ma' && (
                      <>
                        <Grid item xs={6}>
                          <TextField
                            fullWidth
                            type="number"
                            label="Short Window"
                            value={shortWindow}
                            onChange={(e) => setShortWindow(Number(e.target.value))}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            fullWidth
                            type="number"
                            label="Long Window"
                            value={longWindow}
                            onChange={(e) => setLongWindow(Number(e.target.value))}
                          />
                        </Grid>
                      </>
                    )}
                    
                    {strategyType === 'rsi' && (
                      <>
                        <Grid item xs={4}>
                          <TextField
                            fullWidth
                            type="number"
                            label="RSI Period"
                            value={rsiPeriod}
                            onChange={(e) => setRsiPeriod(Number(e.target.value))}
                          />
                        </Grid>
                        <Grid item xs={4}>
                          <TextField
                            fullWidth
                            type="number"
                            label="Oversold"
                            value={oversold}
                            onChange={(e) => setOversold(Number(e.target.value))}
                          />
                        </Grid>
                        <Grid item xs={4}>
                          <TextField
                            fullWidth
                            type="number"
                            label="Overbought"
                            value={overbought}
                            onChange={(e) => setOverbought(Number(e.target.value))}
                          />
                        </Grid>
                      </>
                    )}
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Actions */}
            <Grid item xs={12}>
              <Box display="flex" gap={2} justifyContent="center">
                <Button
                  variant="contained"
                  startIcon={<PlayIcon />}
                  onClick={runBacktest}
                  disabled={loading}
                  size="large"
                >
                  Run Backtest
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<CompareIcon />}
                  onClick={compareDataSources}
                  disabled={loading}
                >
                  Compare Data Sources
                </Button>
              </Box>
            </Grid>

            {loading && (
              <Grid item xs={12}>
                <Box sx={{ width: '100%' }}>
                  <LinearProgress />
                  <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
                    Running backtest...
                  </Typography>
                </Box>
              </Grid>
            )}

            {error && (
              <Grid item xs={12}>
                <Alert severity="error">{error}</Alert>
              </Grid>
            )}
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          {comparison ? (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Alert severity="info" icon={<SpeedIcon />}>
                  <strong>Recommendation:</strong> {comparison.recommendation}
                </Alert>
              </Grid>
              
              <Grid item xs={12}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Data Source</TableCell>
                      <TableCell>Available</TableCell>
                      <TableCell>Load Time (s)</TableCell>
                      <TableCell>Records</TableCell>
                      <TableCell>Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(comparison.performance_summary).map(([source, data]: [string, any]) => (
                      <TableRow key={source}>
                        <TableCell>
                          <Box display="flex" alignItems="center">
                            <StorageIcon sx={{ mr: 1 }} />
                            {source.replace('_', ' ').toUpperCase()}
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={data.available ? 'Yes' : 'No'} 
                            color={data.available ? 'success' : 'error'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{data.load_time?.toFixed(3) || 'N/A'}</TableCell>
                        <TableCell>{data.record_count?.toLocaleString() || 'N/A'}</TableCell>
                        <TableCell>
                          <Chip 
                            label={data.available ? 'Ready' : 'Unavailable'} 
                            color={data.available ? 'success' : 'default'}
                            size="small"
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Grid>
            </Grid>
          ) : (
            <Box textAlign="center" py={4}>
              <Typography variant="body1" color="textSecondary">
                Click "Compare Data Sources" to analyze performance
              </Typography>
            </Box>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          {backtestResults ? (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Alert severity="success">
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <span>
                      Backtest completed using <strong>{backtestResults.data_source_used}</strong> data source
                    </span>
                    <Chip 
                      label={`${backtestResults.execution_time_ms.toFixed(0)}ms`} 
                      size="small" 
                      icon={<SpeedIcon />}
                    />
                  </Box>
                </Alert>
              </Grid>

              <Grid item xs={12}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Total Return</TableCell>
                      <TableCell>Sharpe Ratio</TableCell>
                      <TableCell>Max Drawdown</TableCell>
                      <TableCell>Trades</TableCell>
                      <TableCell>Win Rate</TableCell>
                      <TableCell>Alpha</TableCell>
                      <TableCell>Beta</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.values(backtestResults.results).map((result) => (
                      <TableRow key={result.symbol}>
                        <TableCell>
                          <Box display="flex" alignItems="center">
                            <TimelineIcon sx={{ mr: 1 }} />
                            <strong>{result.symbol}</strong>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Box display="flex" alignItems="center">
                            {result.total_return >= 0 ? <TrendingUpIcon color="success" /> : <TrendingDownIcon color="error" />}
                            <Typography 
                              variant="body2" 
                              sx={{ ml: 0.5, color: getReturnColor(result.total_return) }}
                            >
                              {formatPercent(result.total_return)}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>{formatNumber(result.sharpe_ratio)}</TableCell>
                        <TableCell sx={{ color: theme.palette.error.main }}>
                          {formatPercent(result.max_drawdown)}
                        </TableCell>
                        <TableCell>{result.total_trades}</TableCell>
                        <TableCell>{formatPercent(result.win_rate)}</TableCell>
                        <TableCell sx={{ color: getReturnColor(result.alpha) }}>
                          {formatPercent(result.alpha)}
                        </TableCell>
                        <TableCell>{formatNumber(result.beta)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Grid>
            </Grid>
          ) : (
            <Box textAlign="center" py={4}>
              <Typography variant="body1" color="textSecondary">
                Run a backtest to see results
              </Typography>
            </Box>
          )}
        </TabPanel>
      </CardContent>
    </Card>
  );
};

export default BacktestingWidget;