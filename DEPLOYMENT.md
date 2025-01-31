# AlphaPulse Deployment Guide

This guide explains how to deploy AlphaPulse in a production environment using Docker and Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- Git repository access
- Python 3.11 or higher (for local development)

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Exchange API credentials
EXCHANGE_API_KEY=your_api_key
EXCHANGE_API_SECRET=your_api_secret

# MLflow settings
MLFLOW_TRACKING_URI=http://mlflow:5000

# Monitoring
PROMETHEUS_PORT=8000
GRAFANA_ADMIN_PASSWORD=alphapulse  # Change this in production
```

## Building and Deploying

1. Build and start all services:
```bash
docker-compose up -d --build
```

2. Verify all services are running:
```bash
docker-compose ps
```

Expected output:
```
Name                    Status
alphapulse             Up
mlflow                 Up
prometheus             Up
grafana                Up
```

## Accessing Services

- **AlphaPulse Metrics**: http://localhost:8000/metrics
- **MLflow Dashboard**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (default login: admin/alphapulse)

## Initial Setup

1. Configure Grafana:
   - Log in to Grafana at http://localhost:3000
   - Add Prometheus as a data source (URL: http://prometheus:9090)
   - Import the provided dashboard templates from `monitoring/dashboards/`

2. Test the deployment:
```bash
# Run the paper trading demo
docker-compose exec alphapulse python -m alpha_pulse.examples.demo_paper_trading

# Check the logs
docker-compose logs -f alphapulse
```

## Monitoring and Metrics

The system exposes several key metrics:

1. Trading Metrics:
   - Total trades (`alphapulse_trades_total`)
   - Current PnL (`alphapulse_pnl_current`)
   - Position sizes (`alphapulse_position_size`)
   - Win rates (`alphapulse_win_rate`)

2. Model Metrics:
   - Prediction distributions (`alphapulse_model_predictions`)
   - Inference latency (`alphapulse_model_latency_seconds`)

3. System Metrics:
   - Error counts (`alphapulse_errors_total`)
   - API latency (`alphapulse_api_latency_seconds`)

## MLflow Model Management

Models are tracked using MLflow:

1. View experiments and runs:
   - Open MLflow UI at http://localhost:5000
   - Navigate to the "Experiments" tab

2. Deploy a model:
```bash
# Get the run ID from MLflow UI
export RUN_ID=your_run_id

# Download and use the model
docker-compose exec alphapulse python -c "
import mlflow
model = mlflow.sklearn.load_model('runs:/$RUN_ID/model')
"
```

## Backup and Maintenance

1. Backup volumes:
```bash
docker run --rm -v alphapulse_mlruns:/data -v /backup:/backup \
    ubuntu tar czf /backup/mlruns_backup.tar.gz /data
```

2. Clean up old data:
```bash
# Clean up old Prometheus data (adjust retention as needed)
docker-compose exec prometheus promtool clean-tombstones
```

## Troubleshooting

1. Check service logs:
```bash
docker-compose logs -f [service_name]
```

2. Common issues:
   - If metrics aren't showing up, ensure the Prometheus port (8000) is exposed
   - For MLflow connection issues, verify the tracking URI is correctly set
   - If models aren't loading, check the MLflow artifact store permissions

## Production Considerations

1. Security:
   - Change default passwords
   - Use secrets management for API keys
   - Enable TLS for all services
   - Implement proper authentication

2. Scaling:
   - Consider using Kubernetes for larger deployments
   - Implement proper backup strategies
   - Monitor disk usage for Prometheus and MLflow

3. Monitoring:
   - Set up alerting rules in Prometheus
   - Configure email/Slack notifications
   - Monitor system resources (CPU, memory, disk)

## Support

For issues or questions:
1. Check the logs using `docker-compose logs`
2. Review the metrics in Grafana
3. Consult the project documentation
4. Open an issue in the repository