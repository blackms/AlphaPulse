# AlphaPulse Helm Chart

This Helm chart deploys the AlphaPulse multi-tenant SaaS platform to Kubernetes.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.8+
- PostgreSQL 14+ (external RDS recommended)
- Cert-manager (for TLS certificates)
- Ingress controller (nginx recommended)

## Installation

### 1. Add Dependencies

```bash
# Add Bitnami repository (for Redis)
helm repo add bitnami https://charts.bitnami.com/bitnami

# Add HashiCorp repository (for Vault)
helm repo add hashicorp https://helm.releases.hashicorp.com

# Update repositories
helm repo update
```

### 2. Create Namespace

```bash
kubectl create namespace alphapulse
```

### 3. Create Secrets

Create a `secrets.yaml` file with actual values:

```yaml
database-url: <base64-encoded-postgresql-connection-string>
redis-password: <base64-encoded-redis-password>
postgres-password: <base64-encoded-postgres-password>
jwt-secret: <base64-encoded-jwt-secret>
openai-api-key: <base64-encoded-openai-api-key>
exchange-api-key: <base64-encoded-exchange-api-key>
exchange-api-secret: <base64-encoded-exchange-api-secret>
```

Apply secrets:
```bash
kubectl create secret generic alphapulse-secrets \
  --from-file=secrets.yaml \
  --namespace alphapulse
```

**IMPORTANT**: DO NOT commit `secrets.yaml` to git! Use AWS Secrets Manager, HashiCorp Vault, or similar.

### 4. Install Chart

**Development** (local Minikube/Kind):
```bash
helm install alphapulse ./helm/alphapulse \
  --namespace alphapulse \
  --values ./helm/alphapulse/values-dev.yaml
```

**Staging**:
```bash
helm install alphapulse ./helm/alphapulse \
  --namespace alphapulse-staging \
  --create-namespace \
  --values ./helm/alphapulse/values-staging.yaml
```

**Production**:
```bash
helm install alphapulse ./helm/alphapulse \
  --namespace alphapulse \
  --values ./helm/alphapulse/values-production.yaml
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods -n alphapulse

# Check services
kubectl get svc -n alphapulse

# Check ingress
kubectl get ingress -n alphapulse

# View logs
kubectl logs -n alphapulse -l app.kubernetes.io/component=api
```

## Configuration

### Key Configuration Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api.replicaCount` | Number of API pods | `10` |
| `api.autoscaling.enabled` | Enable HPA for API | `true` |
| `api.autoscaling.maxReplicas` | Max API replicas | `50` |
| `redis.enabled` | Deploy Redis cluster | `true` |
| `redis.replica.replicaCount` | Number of Redis replicas | `2` |
| `vault.enabled` | Deploy Vault | `true` |
| `vault.server.ha.replicas` | Number of Vault replicas | `3` |
| `ingress.enabled` | Enable ingress | `true` |
| `monitoring.enabled` | Enable monitoring stack | `true` |

### Custom Values Files

**`values-dev.yaml`** - Local development:
```yaml
api:
  replicaCount: 1
  autoscaling:
    enabled: false
  resources:
    requests:
      cpu: 500m
      memory: 1Gi

redis:
  replica:
    replicaCount: 0  # Single Redis instance

vault:
  server:
    ha:
      enabled: false  # Single Vault instance

monitoring:
  enabled: false  # Disable monitoring for local dev
```

**`values-staging.yaml`** - Staging environment:
```yaml
api:
  replicaCount: 3
  autoscaling:
    maxReplicas: 10
  image:
    tag: staging

ingress:
  hosts:
    - host: staging.alphapulse.ai
```

**`values-production.yaml`** - Production environment:
```yaml
api:
  replicaCount: 10
  autoscaling:
    maxReplicas: 50
  image:
    tag: v1.0.0

ingress:
  hosts:
    - host: api.alphapulse.ai
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
```

## Upgrading

```bash
# Upgrade to new version
helm upgrade alphapulse ./helm/alphapulse \
  --namespace alphapulse \
  --values ./helm/alphapulse/values-production.yaml

# View history
helm history alphapulse -n alphapulse

# Rollback to previous version
helm rollback alphapulse -n alphapulse
```

## Uninstalling

```bash
helm uninstall alphapulse -n alphapulse
```

## Monitoring

### Prometheus Metrics

API exposes metrics at `/metrics` endpoint:
- `alphapulse_api_requests_total` - Total API requests
- `alphapulse_api_request_duration_seconds` - Request latency
- `alphapulse_api_errors_total` - API errors
- `alphapulse_agent_signals_total` - Trading signals generated

### Grafana Dashboards

Access Grafana:
```bash
kubectl port-forward -n alphapulse svc/grafana 3000:80
```

Open browser: http://localhost:3000

Default dashboards:
- AlphaPulse API Dashboard
- AlphaPulse Trading Agents Dashboard
- AlphaPulse Database Dashboard

## Troubleshooting

### Pods Not Starting

Check pod events:
```bash
kubectl describe pod <pod-name> -n alphapulse
```

Common issues:
- **ImagePullBackOff**: Docker image not found or credentials missing
- **CrashLoopBackOff**: Application crashing on startup (check logs)
- **Pending**: Insufficient resources or PVC not bound

### Database Connection Issues

Verify database connectivity:
```bash
kubectl exec -it <api-pod> -n alphapulse -- sh
psql $DATABASE_URL -c "SELECT 1"
```

### Redis Connection Issues

Verify Redis connectivity:
```bash
kubectl exec -it <api-pod> -n alphapulse -- sh
redis-cli -h redis-master -a $REDIS_PASSWORD ping
```

### Ingress Not Working

Check ingress controller logs:
```bash
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

Verify DNS:
```bash
nslookup api.alphapulse.ai
```

## Development Workflow

### Local Testing with Minikube

```bash
# Start Minikube
minikube start --cpus 4 --memory 8192

# Enable ingress addon
minikube addons enable ingress

# Install chart
helm install alphapulse ./helm/alphapulse \
  --values ./helm/alphapulse/values-dev.yaml

# Get Minikube IP
minikube ip

# Add to /etc/hosts
echo "$(minikube ip) api.alphapulse.local" | sudo tee -a /etc/hosts

# Test API
curl http://api.alphapulse.local/health
```

### Local Testing with Kind

```bash
# Create cluster
kind create cluster --name alphapulse

# Install ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# Install chart
helm install alphapulse ./helm/alphapulse \
  --values ./helm/alphapulse/values-dev.yaml
```

## Security

### Network Policies

Network policies restrict traffic between pods:
- API pods can access Redis and Vault
- Worker pods can access Redis and Vault
- External traffic only via Ingress

### Pod Security

- Pods run as non-root user (UID 1000)
- Read-only root filesystem
- Dropped all capabilities except NET_BIND_SERVICE

### Secrets Management

**IMPORTANT**: DO NOT commit secrets to git!

Use external secrets manager:
- AWS Secrets Manager + External Secrets Operator
- HashiCorp Vault + Vault Secrets Operator
- Azure Key Vault + Secrets Store CSI Driver

## References

- [Helm Documentation](https://helm.sh/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [AlphaPulse Architecture](../../docs/HLD-MULTI-TENANT-SAAS.md)
- [AlphaPulse Deployment Strategy](../../docs/HLD-MULTI-TENANT-SAAS.md#deployment)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
