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

### `demo_alerting.py`

A demonstration of the alerting system, including:

- Creating alert rules
- Processing metrics and generating alerts
- Sending notifications through different channels
- Managing alert history
- Acknowledging alerts

The example simulates various metrics and triggers alerts based on predefined conditions.

## Running the Examples

### Using the Run Scripts

The easiest way to run the examples is using the provided scripts:

```bash
# For monitoring demo
./run_demo.sh

# For alerting demo
./run_alerting_demo.sh
```

These scripts will:
1. Ensure you're in the project root directory
2. Install required dependencies
3. Run the demo

### Manual Execution

Alternatively, you can run the examples manually:

```bash
# From the project root directory
python examples/monitoring/demo_monitoring.py

# Or for the alerting demo
python examples/monitoring/demo_alerting.py
```

## Expected Output

### Monitoring Demo

The monitoring demo will:

1. Generate sample data
2. Process historical data
3. Query and plot metrics
4. Run real-time monitoring for a short period
5. Display the latest metrics

Plots will be saved to the `plots/` directory, including:
- Performance metrics (Sharpe ratio, Sortino ratio, max drawdown)
- Risk metrics (leverage, concentration, portfolio value)

### Alerting Demo

The alerting demo will:

1. Create alert rules for various metrics
2. Simulate metrics with random variations
3. Occasionally trigger alerts by simulating extreme values
4. Process metrics and generate alerts
5. Send notifications through configured channels
6. Display alert history

## Customizing the Examples

You can modify the examples to test different scenarios:

- Change the storage backend in the configuration
- Adjust the data generation parameters
- Modify the metrics collection interval
- Add custom metrics calculations
- Create new alert rules with different conditions
- Implement custom notification channels

## Additional Resources

For more information on the monitoring and alerting systems, see:
- [Monitoring System README](../../src/alpha_pulse/monitoring/README.md)
- [Alerting System README](../../src/alpha_pulse/monitoring/alerting/README.md)
- [API Documentation](../../API_DOCUMENTATION.md)