# AlphaPulse Monitoring System Examples

This directory contains examples demonstrating the AlphaPulse monitoring system.

## Available Examples

### `demo_monitoring.py`

A comprehensive demonstration of the monitoring system, including:

- Initializing the metrics collector
- Collecting and storing metrics
- Querying historical metrics
- Real-time monitoring
- Visualizing metrics

The example generates sample portfolio, trade, and agent data to simulate a real trading environment.

## Running the Examples

### Using the Run Script

The easiest way to run the examples is using the provided script:

```bash
./run_demo.sh
```

This script will:
1. Ensure you're in the project root directory
2. Install required dependencies
3. Run the demo

### Manual Execution

Alternatively, you can run the examples manually:

```bash
# From the project root directory
python examples/monitoring/demo_monitoring.py
```

## Expected Output

The demo will:

1. Generate sample data
2. Process historical data
3. Query and plot metrics
4. Run real-time monitoring for a short period
5. Display the latest metrics

Plots will be saved to the `plots/` directory, including:
- Performance metrics (Sharpe ratio, Sortino ratio, max drawdown)
- Risk metrics (leverage, concentration, portfolio value)

## Customizing the Examples

You can modify the examples to test different scenarios:

- Change the storage backend in the configuration
- Adjust the data generation parameters
- Modify the metrics collection interval
- Add custom metrics calculations

## Additional Resources

For more information on the monitoring system, see:
- [Monitoring System README](../../src/alpha_pulse/monitoring/README.md)
- [API Documentation](../../API_DOCUMENTATION.md)