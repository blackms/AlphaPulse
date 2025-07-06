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
  Tooltip,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  TableContainer,
  TablePagination
} from '@mui/material';
import {
  Search as SearchIcon,
  Storage as StorageIcon,
  QueryStats as QueryIcon,
  Assessment as AssessmentIcon,
  Visibility as ViewIcon,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  DataObject as DataIcon,
  Schema as SchemaIcon,
  TableChart as TableIcon,
  BarChart as ChartIcon,
  Download as DownloadIcon,
  PlayArrow as PlayIcon,
  Clear as ClearIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface Dataset {
  id: string;
  name: string;
  description: string;
  layer: string;
  dataset_type: string;
  owner: string;
  created_at: string;
  updated_at: string;
  schema: any;
  size_bytes: number;
  record_count: number;
  partition_keys: string[];
  quality_score: number;
  tags: string[];
  lineage: any;
}

interface QueryResult {
  query_id: string;
  sql: string;
  status: string;
  execution_time_ms: number;
  record_count: number;
  columns: string[];
  data: any[];
  metadata: any;
}

interface DataProfile {
  dataset_id: string;
  profile_timestamp: string;
  record_count: number;
  column_count: number;
  column_profiles: any;
  quality_metrics: any;
  correlations?: any;
  recommendations: string[];
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
      id={`explorer-tabpanel-${index}`}
      aria-labelledby={`explorer-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

const DataLakeExplorer: React.FC = () => {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Datasets tab state
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [layerFilter, setLayerFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [datasetDetails, setDatasetDetails] = useState<any>(null);
  const [sampleData, setSampleData] = useState<any>(null);
  const [dataProfile, setDataProfile] = useState<DataProfile | null>(null);
  
  // Query tab state
  const [sqlQuery, setSqlQuery] = useState('SELECT * FROM market_data_ohlcv LIMIT 100');
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [queryHistory, setQueryHistory] = useState<QueryResult[]>([]);
  
  // Statistics state
  const [dataLakeStats, setDataLakeStats] = useState<any>(null);
  const [healthStatus, setHealthStatus] = useState<any>(null);
  
  // Pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);

  useEffect(() => {
    fetchDatasets();
    fetchDataLakeStats();
    fetchHealthStatus();
  }, []);

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (searchQuery) params.append('query', searchQuery);
      if (layerFilter) params.append('layer', layerFilter);
      if (typeFilter) params.append('dataset_type', typeFilter);
      params.append('limit', '50');
      params.append('offset', '0');

      const response = await fetch(`/api/v1/datalake/datasets?${params}`);
      if (response.ok) {
        const data = await response.json();
        setDatasets(data);
      } else {
        setError('Failed to fetch datasets');
      }
    } catch (err) {
      setError('Error fetching datasets');
    } finally {
      setLoading(false);
    }
  };

  const fetchDataLakeStats = async () => {
    try {
      const response = await fetch('/api/v1/datalake/statistics');
      if (response.ok) {
        const stats = await response.json();
        setDataLakeStats(stats);
      }
    } catch (err) {
      console.error('Failed to fetch data lake statistics:', err);
    }
  };

  const fetchHealthStatus = async () => {
    try {
      const response = await fetch('/api/v1/datalake/health');
      if (response.ok) {
        const health = await response.json();
        setHealthStatus(health);
      }
    } catch (err) {
      console.error('Failed to fetch health status:', err);
    }
  };

  const viewDatasetDetails = async (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setLoading(true);
    
    try {
      // Fetch detailed dataset info
      const response = await fetch(`/api/v1/datalake/datasets/${dataset.id}`);
      if (response.ok) {
        const details = await response.json();
        setDatasetDetails(details);
      }
    } catch (err) {
      setError('Failed to fetch dataset details');
    } finally {
      setLoading(false);
    }
  };

  const sampleDataset = async (dataset: Dataset) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/datalake/datasets/${dataset.id}/sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 100 })
      });
      
      if (response.ok) {
        const sample = await response.json();
        setSampleData(sample);
      } else {
        setError('Failed to sample dataset');
      }
    } catch (err) {
      setError('Error sampling dataset');
    } finally {
      setLoading(false);
    }
  };

  const profileDataset = async (dataset: Dataset) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/datalake/datasets/${dataset.id}/profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          include_histogram: true,
          include_correlations: true,
          max_categorical_values: 20
        })
      });
      
      if (response.ok) {
        const profile = await response.json();
        setDataProfile(profile);
      } else {
        setError('Failed to profile dataset');
      }
    } catch (err) {
      setError('Error profiling dataset');
    } finally {
      setLoading(false);
    }
  };

  const executeQuery = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v1/datalake/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sql: sqlQuery,
          limit: 1000,
          timeout_seconds: 30,
          cache_enabled: true
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        setQueryResult(result);
        
        // Add to history if successful
        if (result.status === 'completed') {
          setQueryHistory(prev => [result, ...prev.slice(0, 9)]); // Keep last 10 queries
        }
      } else {
        setError('Query execution failed');
      }
    } catch (err) {
      setError('Error executing query');
    } finally {
      setLoading(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatNumber = (num: number) => {
    return num.toLocaleString();
  };

  const getLayerColor = (layer: string) => {
    switch (layer.toLowerCase()) {
      case 'bronze': return theme.palette.warning.main;
      case 'silver': return theme.palette.info.main;
      case 'gold': return theme.palette.success.main;
      default: return theme.palette.grey[500];
    }
  };

  const getQualityColor = (score: number) => {
    if (score >= 0.8) return theme.palette.success.main;
    if (score >= 0.6) return theme.palette.warning.main;
    return theme.palette.error.main;
  };

  return (
    <Card>
      <CardHeader
        title="Data Lake Explorer"
        avatar={<StorageIcon />}
        action={
          <Box display="flex" gap={1}>
            <Tooltip title="Refresh">
              <IconButton onClick={() => {
                fetchDatasets();
                fetchDataLakeStats();
                fetchHealthStatus();
              }}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            {healthStatus && (
              <Tooltip title={`Data Lake Status: ${healthStatus.status}`}>
                <IconButton>
                  {healthStatus.status === 'healthy' ? 
                    <SuccessIcon color="success" /> : 
                    <ErrorIcon color="error" />
                  }
                </IconButton>
              </Tooltip>
            )}
          </Box>
        }
      />
      <CardContent>
        <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
          <Tab label="Datasets" icon={<TableIcon />} />
          <Tab label="Query" icon={<QueryIcon />} />
          <Tab label="Analytics" icon={<AssessmentIcon />} />
          <Tab label="Health" icon={<SpeedIcon />} />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          {/* Dataset Browser */}
          <Grid container spacing={3}>
            {/* Search and Filters */}
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} md={4}>
                    <TextField
                      fullWidth
                      label="Search datasets"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      InputProps={{
                        startAdornment: <SearchIcon sx={{ mr: 1 }} />
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <FormControl fullWidth>
                      <InputLabel>Layer</InputLabel>
                      <Select value={layerFilter} onChange={(e) => setLayerFilter(e.target.value)}>
                        <MenuItem value="">All Layers</MenuItem>
                        <MenuItem value="bronze">Bronze</MenuItem>
                        <MenuItem value="silver">Silver</MenuItem>
                        <MenuItem value="gold">Gold</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <FormControl fullWidth>
                      <InputLabel>Type</InputLabel>
                      <Select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
                        <MenuItem value="">All Types</MenuItem>
                        <MenuItem value="RAW">Raw</MenuItem>
                        <MenuItem value="PROCESSED">Processed</MenuItem>
                        <MenuItem value="BUSINESS">Business</MenuItem>
                        <MenuItem value="FEATURE">Feature</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <Button 
                      variant="contained" 
                      onClick={fetchDatasets}
                      startIcon={<SearchIcon />}
                      fullWidth
                    >
                      Search
                    </Button>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>

            {/* Dataset List */}
            <Grid item xs={12} md={8}>
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Dataset</TableCell>
                      <TableCell>Layer</TableCell>
                      <TableCell>Records</TableCell>
                      <TableCell>Size</TableCell>
                      <TableCell>Quality</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {datasets.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((dataset) => (
                      <TableRow key={dataset.id}>
                        <TableCell>
                          <Box>
                            <Typography variant="subtitle2">{dataset.name}</Typography>
                            <Typography variant="caption" color="textSecondary">
                              {dataset.description}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={dataset.layer} 
                            size="small" 
                            sx={{ backgroundColor: getLayerColor(dataset.layer), color: 'white' }}
                          />
                        </TableCell>
                        <TableCell>{formatNumber(dataset.record_count)}</TableCell>
                        <TableCell>{formatBytes(dataset.size_bytes)}</TableCell>
                        <TableCell>
                          <Chip 
                            label={`${(dataset.quality_score * 100).toFixed(0)}%`}
                            size="small"
                            sx={{ backgroundColor: getQualityColor(dataset.quality_score), color: 'white' }}
                          />
                        </TableCell>
                        <TableCell>
                          <Box display="flex" gap={1}>
                            <Tooltip title="View Details">
                              <IconButton size="small" onClick={() => viewDatasetDetails(dataset)}>
                                <ViewIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Sample Data">
                              <IconButton size="small" onClick={() => sampleDataset(dataset)}>
                                <DataIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Profile Data">
                              <IconButton size="small" onClick={() => profileDataset(dataset)}>
                                <ChartIcon />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <TablePagination
                  component="div"
                  count={datasets.length}
                  page={page}
                  onPageChange={(e, newPage) => setPage(newPage)}
                  rowsPerPage={rowsPerPage}
                  onRowsPerPageChange={(e) => setRowsPerPage(parseInt(e.target.value, 10))}
                />
              </TableContainer>
            </Grid>

            {/* Dataset Details Panel */}
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, height: '600px', overflow: 'auto' }}>
                {selectedDataset ? (
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      {selectedDataset.name}
                    </Typography>
                    
                    {datasetDetails && (
                      <Box>
                        <Typography variant="body2" paragraph>
                          {datasetDetails.description}
                        </Typography>
                        
                        <Accordion>
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography>Schema</Typography>
                          </AccordionSummary>
                          <AccordionDetails>
                            <List dense>
                              {Object.entries(datasetDetails.schema.columns || {}).map(([col, type]: [string, any]) => (
                                <ListItem key={col}>
                                  <ListItemIcon>
                                    <SchemaIcon fontSize="small" />
                                  </ListItemIcon>
                                  <ListItemText 
                                    primary={col} 
                                    secondary={type.type || 'Unknown'}
                                  />
                                </ListItem>
                              ))}
                            </List>
                          </AccordionDetails>
                        </Accordion>

                        <Accordion>
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography>Metadata</Typography>
                          </AccordionSummary>
                          <AccordionDetails>
                            <Typography variant="body2">
                              <strong>Owner:</strong> {datasetDetails.owner}<br/>
                              <strong>Created:</strong> {new Date(datasetDetails.created_at).toLocaleDateString()}<br/>
                              <strong>Updated:</strong> {new Date(datasetDetails.updated_at).toLocaleDateString()}<br/>
                              <strong>Partitions:</strong> {datasetDetails.partition_keys.join(', ')}<br/>
                            </Typography>
                          </AccordionDetails>
                        </Accordion>
                      </Box>
                    )}

                    {sampleData && (
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography>Sample Data ({sampleData.sample_size} rows)</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <TableContainer sx={{ maxHeight: 300 }}>
                            <Table size="small">
                              <TableHead>
                                <TableRow>
                                  {sampleData.columns.slice(0, 4).map((col: string) => (
                                    <TableCell key={col}>{col}</TableCell>
                                  ))}
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {sampleData.data.slice(0, 10).map((row: any, idx: number) => (
                                  <TableRow key={idx}>
                                    {sampleData.columns.slice(0, 4).map((col: string) => (
                                      <TableCell key={col}>
                                        {typeof row[col] === 'string' && row[col].length > 20 
                                          ? row[col].substring(0, 20) + '...'
                                          : String(row[col] || '')
                                        }
                                      </TableCell>
                                    ))}
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </TableContainer>
                        </AccordionDetails>
                      </Accordion>
                    )}

                    {dataProfile && (
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography>Data Profile</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Typography variant="body2">
                            <strong>Records:</strong> {formatNumber(dataProfile.record_count)}<br/>
                            <strong>Columns:</strong> {dataProfile.column_count}<br/>
                            <strong>Quality Score:</strong> {(dataProfile.quality_metrics.overall_score * 100).toFixed(1)}%<br/>
                          </Typography>
                          
                          {dataProfile.recommendations.length > 0 && (
                            <Box mt={2}>
                              <Typography variant="subtitle2">Recommendations:</Typography>
                              <List dense>
                                {dataProfile.recommendations.map((rec, idx) => (
                                  <ListItem key={idx}>
                                    <ListItemIcon>
                                      <InfoIcon fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText primary={rec} />
                                  </ListItem>
                                ))}
                              </List>
                            </Box>
                          )}
                        </AccordionDetails>
                      </Accordion>
                    )}
                  </Box>
                ) : (
                  <Box textAlign="center" py={4}>
                    <Typography variant="body1" color="textSecondary">
                      Select a dataset to view details
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          {/* Query Interface */}
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 2 }}>
                <Box mb={2}>
                  <Typography variant="h6" gutterBottom>
                    SQL Query Editor
                  </Typography>
                  <TextField
                    fullWidth
                    multiline
                    rows={8}
                    value={sqlQuery}
                    onChange={(e) => setSqlQuery(e.target.value)}
                    placeholder="Enter your SQL query here..."
                    variant="outlined"
                  />
                </Box>
                
                <Box display="flex" gap={2} mb={2}>
                  <Button
                    variant="contained"
                    startIcon={<PlayIcon />}
                    onClick={executeQuery}
                    disabled={loading || !sqlQuery.trim()}
                  >
                    Execute Query
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<ClearIcon />}
                    onClick={() => setSqlQuery('')}
                  >
                    Clear
                  </Button>
                </Box>

                {loading && <LinearProgress sx={{ mb: 2 }} />}

                {queryResult && (
                  <Box>
                    <Alert 
                      severity={queryResult.status === 'completed' ? 'success' : 'error'}
                      sx={{ mb: 2 }}
                    >
                      Query {queryResult.status} in {queryResult.execution_time_ms.toFixed(0)}ms
                      {queryResult.status === 'completed' && ` • ${formatNumber(queryResult.record_count)} records`}
                      {queryResult.metadata?.error && ` • ${queryResult.metadata.error}`}
                    </Alert>

                    {queryResult.status === 'completed' && queryResult.data.length > 0 && (
                      <TableContainer sx={{ maxHeight: 400 }}>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              {queryResult.columns.map((col) => (
                                <TableCell key={col}>{col}</TableCell>
                              ))}
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {queryResult.data.slice(0, 50).map((row, idx) => (
                              <TableRow key={idx}>
                                {queryResult.columns.map((col) => (
                                  <TableCell key={col}>
                                    {typeof row[col] === 'string' && row[col].length > 30
                                      ? row[col].substring(0, 30) + '...'
                                      : String(row[col] || '')
                                    }
                                  </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    )}
                  </Box>
                )}
              </Paper>
            </Grid>

            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, height: '600px', overflow: 'auto' }}>
                <Typography variant="h6" gutterBottom>
                  Query History
                </Typography>
                
                {queryHistory.length > 0 ? (
                  <List>
                    {queryHistory.map((query, idx) => (
                      <ListItem key={query.query_id} divider>
                        <ListItemText
                          primary={
                            <Typography variant="body2" noWrap>
                              {query.sql.length > 50 ? query.sql.substring(0, 50) + '...' : query.sql}
                            </Typography>
                          }
                          secondary={
                            <Box>
                              <Typography variant="caption">
                                {query.execution_time_ms.toFixed(0)}ms • {formatNumber(query.record_count)} records
                              </Typography>
                              <br/>
                              <Chip 
                                label={query.status} 
                                size="small" 
                                color={query.status === 'completed' ? 'success' : 'error'}
                              />
                            </Box>
                          }
                        />
                        <IconButton 
                          size="small" 
                          onClick={() => setSqlQuery(query.sql)}
                        >
                          <ViewIcon />
                        </IconButton>
                      </ListItem>
                    ))}
                  </List>
                ) : (
                  <Typography variant="body2" color="textSecondary" textAlign="center">
                    No queries executed yet
                  </Typography>
                )}
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          {/* Analytics Dashboard */}
          <Grid container spacing={3}>
            {dataLakeStats && (
              <>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Box display="flex" alignItems="center">
                        <TableIcon sx={{ mr: 2 }} />
                        <Box>
                          <Typography variant="h4">
                            {formatNumber(dataLakeStats.total_datasets)}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            Total Datasets
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Box display="flex" alignItems="center">
                        <StorageIcon sx={{ mr: 2 }} />
                        <Box>
                          <Typography variant="h4">
                            {formatBytes(dataLakeStats.total_size_bytes)}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            Total Size
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Box display="flex" alignItems="center">
                        <DataIcon sx={{ mr: 2 }} />
                        <Box>
                          <Typography variant="h4">
                            {formatNumber(dataLakeStats.total_records)}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            Total Records
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Box display="flex" alignItems="center">
                        <AssessmentIcon sx={{ mr: 2 }} />
                        <Box>
                          <Typography variant="h4">
                            {(dataLakeStats.quality_metrics?.average_quality * 100 || 0).toFixed(0)}%
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            Avg Quality
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Card>
                    <CardHeader title="Layer Breakdown" />
                    <CardContent>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Layer</TableCell>
                            <TableCell>Datasets</TableCell>
                            <TableCell>Size</TableCell>
                            <TableCell>Records</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(dataLakeStats.layer_breakdown || {}).map(([layer, stats]: [string, any]) => (
                            <TableRow key={layer}>
                              <TableCell>
                                <Chip 
                                  label={layer} 
                                  size="small" 
                                  sx={{ backgroundColor: getLayerColor(layer), color: 'white' }}
                                />
                              </TableCell>
                              <TableCell>{formatNumber(stats.dataset_count)}</TableCell>
                              <TableCell>{formatBytes(stats.total_size)}</TableCell>
                              <TableCell>{formatNumber(stats.total_records)}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Card>
                    <CardHeader title="Storage Costs" />
                    <CardContent>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Storage Type</TableCell>
                            <TableCell>Monthly Cost</TableCell>
                            <TableCell>Efficiency</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(dataLakeStats.storage_costs || {}).map(([type, cost]: [string, any]) => (
                            <TableRow key={type}>
                              <TableCell>{type}</TableCell>
                              <TableCell>${cost.monthly_cost?.toFixed(2) || '0.00'}</TableCell>
                              <TableCell>
                                <Chip 
                                  label={cost.efficiency || 'N/A'} 
                                  size="small" 
                                  color={cost.efficiency === 'High' ? 'success' : 'default'}
                                />
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                </Grid>
              </>
            )}
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          {/* Health Monitoring */}
          <Grid container spacing={3}>
            {healthStatus && (
              <>
                <Grid item xs={12}>
                  <Alert 
                    severity={healthStatus.status === 'healthy' ? 'success' : 'error'}
                    icon={healthStatus.status === 'healthy' ? <SuccessIcon /> : <ErrorIcon />}
                  >
                    Data Lake Status: {healthStatus.status.toUpperCase()}
                    {healthStatus.last_check && (
                      <Typography variant="caption" display="block">
                        Last checked: {new Date(healthStatus.last_check).toLocaleString()}
                      </Typography>
                    )}
                  </Alert>
                </Grid>

                {healthStatus.checks && (
                  <Grid item xs={12}>
                    <Card>
                      <CardHeader title="Health Checks" />
                      <CardContent>
                        <List>
                          {Object.entries(healthStatus.checks).map(([check, result]: [string, any]) => (
                            <ListItem key={check} divider>
                              <ListItemIcon>
                                {result.status === 'pass' ? 
                                  <SuccessIcon color="success" /> : 
                                  result.status === 'warn' ?
                                  <WarningIcon color="warning" /> :
                                  <ErrorIcon color="error" />
                                }
                              </ListItemIcon>
                              <ListItemText
                                primary={check.replace(/_/g, ' ').toUpperCase()}
                                secondary={result.message || result.description}
                              />
                              <Chip 
                                label={result.status} 
                                size="small"
                                color={
                                  result.status === 'pass' ? 'success' :
                                  result.status === 'warn' ? 'warning' : 'error'
                                }
                              />
                            </ListItem>
                          ))}
                        </List>
                      </CardContent>
                    </Card>
                  </Grid>
                )}

                {healthStatus.recommendations && healthStatus.recommendations.length > 0 && (
                  <Grid item xs={12}>
                    <Card>
                      <CardHeader title="Recommendations" />
                      <CardContent>
                        <List>
                          {healthStatus.recommendations.map((rec: string, idx: number) => (
                            <ListItem key={idx}>
                              <ListItemIcon>
                                <InfoIcon color="info" />
                              </ListItemIcon>
                              <ListItemText primary={rec} />
                            </ListItem>
                          ))}
                        </List>
                      </CardContent>
                    </Card>
                  </Grid>
                )}
              </>
            )}
          </Grid>
        </TabPanel>

        {loading && (
          <Box sx={{ width: '100%', mt: 2 }}>
            <LinearProgress />
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default DataLakeExplorer;