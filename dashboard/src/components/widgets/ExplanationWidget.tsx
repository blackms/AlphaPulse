import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Chip,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
  Button,
  Tooltip,
  IconButton,
  Divider,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Psychology as PsychologyIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Visibility as VisibilityIcon
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface FeatureImportance {
  feature: string;
  importance: number;
  direction: 'positive' | 'negative';
  description: string;
}

interface Explanation {
  explanation_id: string;
  symbol: string;
  timestamp: string;
  method: string;
  signal_details: {
    direction: string;
    confidence: number;
    target_price?: number;
    stop_loss?: number;
  };
  feature_importance: Array<[string, number]>;
  explanation_text: string;
  confidence_score: number;
  key_factors: FeatureImportance[];
  risk_factors: string[];
  compliance_notes?: string[];
}

interface CounterfactualScenario {
  scenario: string;
  changed_features: Record<string, any>;
  predicted_outcome: string;
  likelihood: number;
}

const ExplanationWidget: React.FC = () => {
  const theme = useTheme();
  const [explanations, setExplanations] = useState<Explanation[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [selectedExplanation, setSelectedExplanation] = useState<Explanation | null>(null);
  const [counterfactuals, setCounterfactuals] = useState<CounterfactualScenario[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explanationMethod, setExplanationMethod] = useState('shap');

  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']; // In real app, fetch from API

  const fetchExplanationHistory = async (symbol: string) => {
    if (!symbol) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/v1/explainability/explanation-history/${symbol}?limit=10`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setExplanations(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch explanations');
      console.error('Error fetching explanations:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchDetailedExplanation = async (symbol: string, signalData: any) => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/explainability/explain-decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          signal_data: signalData,
          method: explanationMethod,
          include_counterfactuals: true
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const explanation = await response.json();
      setSelectedExplanation(explanation);
      setCounterfactuals(explanation.counterfactuals || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate explanation');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedSymbol) {
      fetchExplanationHistory(selectedSymbol);
    }
  }, [selectedSymbol]);

  const getDirectionColor = (direction: string) => {
    switch (direction.toLowerCase()) {
      case 'buy':
      case 'long':
        return theme.palette.success.main;
      case 'sell':
      case 'short':
        return theme.palette.error.main;
      default:
        return theme.palette.warning.main;
    }
  };

  const getImportanceColor = (importance: number) => {
    const absImportance = Math.abs(importance);
    if (absImportance > 0.3) return theme.palette.error.main;
    if (absImportance > 0.15) return theme.palette.warning.main;
    return theme.palette.success.main;
  };

  const formatImportance = (importance: number) => {
    return `${importance > 0 ? '+' : ''}${(importance * 100).toFixed(1)}%`;
  };

  return (
    <Card>
      <CardHeader
        title="Trading Decision Explanations"
        avatar={<PsychologyIcon />}
        action={
          <Box display="flex" gap={1} alignItems="center">
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Method</InputLabel>
              <Select
                value={explanationMethod}
                label="Method"
                onChange={(e) => setExplanationMethod(e.target.value)}
              >
                <MenuItem value="shap">SHAP</MenuItem>
                <MenuItem value="lime">LIME</MenuItem>
                <MenuItem value="feature_importance">Feature Importance</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Symbol</InputLabel>
              <Select
                value={selectedSymbol}
                label="Symbol"
                onChange={(e) => setSelectedSymbol(e.target.value)}
              >
                {symbols.map(symbol => (
                  <MenuItem key={symbol} value={symbol}>{symbol}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <IconButton onClick={() => selectedSymbol && fetchExplanationHistory(selectedSymbol)}>
              <RefreshIcon />
            </IconButton>
          </Box>
        }
      />
      <CardContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loading && (
          <Box display="flex" justifyContent="center" p={2}>
            <Typography>Loading explanations...</Typography>
          </Box>
        )}

        {!selectedSymbol && !loading && (
          <Alert severity="info">
            Select a symbol to view trading decision explanations
          </Alert>
        )}

        {selectedSymbol && explanations.length === 0 && !loading && (
          <Alert severity="warning">
            No explanations found for {selectedSymbol}
          </Alert>
        )}

        {/* Recent Explanations List */}
        {explanations.length > 0 && (
          <Box mb={3}>
            <Typography variant="h6" gutterBottom>
              Recent Explanations for {selectedSymbol}
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Time</TableCell>
                  <TableCell>Method</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Key Features</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {explanations.slice(0, 5).map((exp, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      {new Date(exp.timestamp).toLocaleTimeString()}
                    </TableCell>
                    <TableCell>
                      <Chip label={exp.method} size="small" />
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          {(exp.confidence * 100).toFixed(0)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={exp.confidence * 100}
                          sx={{ width: 60, height: 4 }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption">
                        {exp.key_features?.slice(0, 2).map(f => f.feature).join(', ') || 'N/A'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <IconButton
                        size="small"
                        onClick={() => setSelectedExplanation(exp)}
                      >
                        <VisibilityIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        )}

        {/* Detailed Explanation View */}
        {selectedExplanation && (
          <Box>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6" gutterBottom>
              Detailed Explanation
            </Typography>

            {/* Signal Overview */}
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Signal Details
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1} mt={1}>
                      <Chip
                        label={selectedExplanation.signal_details.direction}
                        sx={{
                          backgroundColor: getDirectionColor(selectedExplanation.signal_details.direction),
                          color: 'white'
                        }}
                      />
                      <Typography variant="body1">
                        Confidence: {(selectedExplanation.signal_details.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    {selectedExplanation.signal_details.target_price && (
                      <Typography variant="body2" color="textSecondary">
                        Target: ${selectedExplanation.signal_details.target_price.toFixed(2)}
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Explanation Quality
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1} mt={1}>
                      <LinearProgress
                        variant="determinate"
                        value={selectedExplanation.confidence_score * 100}
                        sx={{ flex: 1, height: 8 }}
                        color={selectedExplanation.confidence_score > 0.8 ? 'success' : 
                               selectedExplanation.confidence_score > 0.6 ? 'warning' : 'error'}
                      />
                      <Typography variant="body2">
                        {(selectedExplanation.confidence_score * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="textSecondary">
                      Explanation reliability score
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Key Factors */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Key Decision Factors</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <List>
                  {selectedExplanation.key_factors?.map((factor, index) => (
                    <ListItem key={index} divider>
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center" justifyContent="space-between">
                            <Typography variant="body1">{factor.feature}</Typography>
                            <Box display="flex" alignItems="center" gap={1}>
                              {factor.direction === 'positive' ? 
                                <TrendingUpIcon color="success" fontSize="small" /> :
                                <TrendingDownIcon color="error" fontSize="small" />
                              }
                              <Typography
                                variant="body2"
                                sx={{ color: getImportanceColor(factor.importance) }}
                                fontWeight="bold"
                              >
                                {formatImportance(factor.importance)}
                              </Typography>
                            </Box>
                          </Box>
                        }
                        secondary={factor.description}
                      />
                    </ListItem>
                  ))}
                </List>
              </AccordionDetails>
            </Accordion>

            {/* Feature Importance Chart */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Feature Importance</Typography>
              </AccordionSummary>
              <AccordionDetails>
                {selectedExplanation.feature_importance?.slice(0, 10).map(([feature, importance], index) => (
                  <Box key={index} sx={{ mb: 1 }}>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="body2">{feature}</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {formatImportance(importance)}
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={Math.abs(importance) * 100}
                      sx={{
                        height: 6,
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: importance > 0 ? theme.palette.success.main : theme.palette.error.main
                        }
                      }}
                    />
                  </Box>
                ))}
              </AccordionDetails>
            </Accordion>

            {/* Risk Factors */}
            {selectedExplanation.risk_factors && selectedExplanation.risk_factors.length > 0 && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Risk Factors</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <List>
                    {selectedExplanation.risk_factors.map((risk, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={risk}
                          secondary="Consider this factor in your risk assessment"
                        />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>
            )}

            {/* Counterfactual Scenarios */}
            {counterfactuals.length > 0 && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">What-If Scenarios</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {counterfactuals.map((scenario, index) => (
                    <Card key={index} variant="outlined" sx={{ mb: 1 }}>
                      <CardContent>
                        <Typography variant="subtitle2" gutterBottom>
                          {scenario.scenario}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Predicted Outcome: {scenario.predicted_outcome}
                        </Typography>
                        <Box display="flex" alignItems="center" mt={1}>
                          <Typography variant="caption" sx={{ mr: 1 }}>
                            Likelihood:
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={scenario.likelihood * 100}
                            sx={{ flex: 1, height: 4 }}
                          />
                          <Typography variant="caption" sx={{ ml: 1 }}>
                            {(scenario.likelihood * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  ))}
                </AccordionDetails>
              </Accordion>
            )}

            {/* Explanation Text */}
            <Box mt={2}>
              <Typography variant="subtitle2" gutterBottom>
                Natural Language Explanation
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ 
                p: 2, 
                backgroundColor: theme.palette.grey[100], 
                borderRadius: 1,
                fontStyle: 'italic'
              }}>
                {selectedExplanation.explanation_text}
              </Typography>
            </Box>

            {/* Compliance Notes */}
            {selectedExplanation.compliance_notes && selectedExplanation.compliance_notes.length > 0 && (
              <Box mt={2}>
                <Typography variant="subtitle2" gutterBottom>
                  Compliance Notes
                </Typography>
                <List dense>
                  {selectedExplanation.compliance_notes.map((note, index) => (
                    <ListItem key={index}>
                      <InfoIcon fontSize="small" sx={{ mr: 1, color: 'info.main' }} />
                      <ListItemText primary={note} />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}

            {/* Actions */}
            <Box display="flex" gap={1} mt={2}>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                size="small"
                onClick={() => {
                  // Export explanation
                  const exportData = {
                    explanation: selectedExplanation,
                    counterfactuals,
                    exported_at: new Date().toISOString()
                  };
                  const blob = new Blob([JSON.stringify(exportData, null, 2)], 
                    { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `explanation_${selectedExplanation.symbol}_${selectedExplanation.explanation_id}.json`;
                  a.click();
                }}
              >
                Export
              </Button>
              <Button
                variant="outlined"
                size="small"
                onClick={() => setSelectedExplanation(null)}
              >
                Close
              </Button>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ExplanationWidget;