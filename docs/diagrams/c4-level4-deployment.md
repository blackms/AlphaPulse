# C4 Level 4: Deployment/Runtime Diagram

**Date**: 2025-10-21
**Sprint**: 3 (Design & Alignment)
**Author**: Tech Lead
**Related**: [HLD-MULTI-TENANT-SAAS.md](../HLD-MULTI-TENANT-SAAS.md), Issue #180

---

## Purpose

This Deployment diagram shows how the AlphaPulse SaaS platform is deployed on Kubernetes, including runtime configuration, networking, and infrastructure.

---

## Diagram: Kubernetes Deployment Architecture

```plantuml
@startuml C4_Level4_Deployment
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Deployment.puml

LAYOUT_WITH_LEGEND()

title Deployment Diagram - AlphaPulse SaaS on Kubernetes

Deployment_Node(cloud, "Cloud Provider (AWS/GCP)", "Region: us-east-1") {
    Deployment_Node(k8s_cluster, "Kubernetes Cluster (EKS/GKE)", "3 worker nodes, t3.xlarge") {

        Deployment_Node(ns_prod, "Namespace: alphapulse-prod") {

            Deployment_Node(ingress, "Ingress Controller", "NGINX/Traefik") {
                Container(lb, "Load Balancer", "NGINX", "TLS termination, rate limiting")
            }

            Deployment_Node(api_deployment, "Deployment: API", "10-50 replicas, HPA") {
                Container(api_pod1, "API Pod 1", "FastAPI", "2 CPU, 4GB RAM")
                Container(api_pod2, "API Pod 2", "FastAPI", "2 CPU, 4GB RAM")
                Container(api_pod_n, "API Pod N", "FastAPI", "...")
            }

            Deployment_Node(agent_deployment, "Deployment: Agent Workers", "30-120 replicas") {
                Container(agent_pod1, "Technical Agent Pod", "Celery", "1 CPU, 2GB RAM")
                Container(agent_pod2, "Fundamental Agent Pod", "Celery", "1 CPU, 2GB RAM")
                Container(agent_pod_n, "Agent Pod N", "Celery", "...")
            }

            Deployment_Node(risk_deployment, "Deployment: Risk Workers", "5-10 replicas") {
                Container(risk_pod1, "Risk Worker Pod 1", "Celery", "2 CPU, 4GB RAM")
                Container(risk_pod2, "Risk Worker Pod 2", "Celery", "...")
            }

            Deployment_Node(billing_deployment, "Deployment: Billing Service", "2-3 replicas") {
                Container(billing_pod1, "Billing Pod 1", "FastAPI", "1 CPU, 2GB RAM")
                Container(billing_pod2, "Billing Pod 2", "FastAPI", "1 CPU, 2GB RAM")
            }

            Deployment_Node(redis_statefulset, "StatefulSet: Redis Cluster", "6 pods (3 masters + 3 replicas)") {
                ContainerDb(redis_master1, "Redis Master 1", "cache.t3.medium", "2 CPU, 3GB RAM")
                ContainerDb(redis_master2, "Redis Master 2", "cache.t3.medium", "...")
                ContainerDb(redis_replica1, "Redis Replica 1", "cache.t3.medium", "...")
            }

            Deployment_Node(vault_statefulset, "StatefulSet: Vault HA", "3 pods (Raft consensus)") {
                ContainerDb(vault_pod1, "Vault Pod 1", "t3.medium", "2 CPU, 4GB RAM")
                ContainerDb(vault_pod2, "Vault Pod 2", "t3.medium", "...")
                ContainerDb(vault_pod3, "Vault Pod 3", "t3.medium", "...")
            }
        }
    }

    Deployment_Node(rds, "AWS RDS", "Multi-AZ deployment") {
        ContainerDb(postgres_primary, "PostgreSQL Primary", "db.r5.xlarge", "4 CPU, 32GB RAM, 3000 IOPS")
        ContainerDb(postgres_replica, "PostgreSQL Replica", "db.r5.xlarge", "Read-only replica")
    }

    Deployment_Node(s3, "AWS S3", "Object storage") {
        Container(backup_bucket, "Backup Bucket", "S3", "Database backups, Vault snapshots")
        Container(logs_bucket, "Logs Bucket", "S3", "Application logs, audit logs")
    }

    Deployment_Node(kms, "AWS KMS", "Key Management Service") {
        Container(vault_unseal_key, "Vault Unseal Key", "KMS", "Auto-unseal for Vault")
    }

    Deployment_Node(monitoring, "Monitoring Stack") {
        Container(prometheus, "Prometheus", "Metrics", "Scrapes metrics from pods")
        Container(grafana, "Grafana", "Dashboards", "Visualizes metrics")
        Container(loki, "Loki", "Logs", "Aggregates logs from pods")
    }
}

System_Ext(cloudflare, "Cloudflare CDN", "DNS, DDoS protection")
System_Ext(stripe, "Stripe API", "Payment processing")
System_Ext(exchanges, "Exchange APIs", "Binance, Coinbase")

Rel(cloudflare, lb, "Routes traffic", "HTTPS")
Rel(lb, api_pod1, "Routes requests", "HTTP")
Rel(lb, api_pod2, "Routes requests", "HTTP")

Rel(api_pod1, postgres_primary, "SQL queries", "PostgreSQL protocol")
Rel(api_pod1, redis_master1, "Cache ops", "Redis protocol")
Rel(api_pod1, vault_pod1, "Get credentials", "HTTPS")
Rel(api_pod1, agent_pod1, "Enqueue tasks", "Celery/Redis")

Rel(agent_pod1, postgres_primary, "Write signals", "PostgreSQL")
Rel(agent_pod1, redis_master1, "Cache results", "Redis")

Rel(postgres_primary, postgres_replica, "Replication", "Async")
Rel(postgres_primary, backup_bucket, "Backups", "S3 API")

Rel(vault_pod1, vault_pod2, "Raft consensus", "gRPC")
Rel(vault_pod1, kms, "Auto-unseal", "KMS API")
Rel(vault_pod1, backup_bucket, "Snapshots", "S3 API")

Rel(prometheus, api_pod1, "Scrapes metrics", "/metrics")
Rel(grafana, prometheus, "Queries metrics", "PromQL")

@enduml
```

---

## Infrastructure Components

### 1. Kubernetes Cluster (EKS/GKE)
**Specification**:
- **Region**: us-east-1 (primary), eu-west-1 (future)
- **Worker Nodes**: 3-10 nodes (auto-scaling)
- **Instance Type**: t3.xlarge (4 vCPU, 16GB RAM) or equivalent
- **Kubernetes Version**: 1.28+ (latest stable)

**Node Pool Configuration**:
```yaml
# General workload pool
nodePool:
  name: general
  instanceType: t3.xlarge
  minNodes: 3
  maxNodes: 10
  diskSize: 100GB

# CPU-intensive pool (for Risk Workers)
nodePool:
  name: compute-optimized
  instanceType: c5.2xlarge  # 8 vCPU, 16GB RAM
  minNodes: 2
  maxNodes: 5
```

**Networking**:
- VPC with public/private subnets
- NAT Gateway for outbound traffic
- Security groups (egress to exchanges, Stripe)

---

### 2. Ingress Controller (NGINX/Traefik)
**Specification**:
- **Type**: LoadBalancer service (AWS ELB/GCP LB)
- **Replicas**: 2-3 (high availability)
- **Resources**: 1 CPU, 2GB RAM per replica

**Configuration**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: alphapulse-ingress
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"  # Per IP
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - api.alphapulse.com
      secretName: alphapulse-tls
  rules:
    - host: api.alphapulse.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 80
```

**Features**:
- TLS termination (Let's Encrypt certificates)
- Rate limiting (100 req/min per IP, 1000 req/min global)
- Request routing (path-based, host-based)
- Health checks (liveness, readiness probes)

---

### 3. API Deployment
**Specification**:
- **Replicas**: 10-50 (Horizontal Pod Autoscaler)
- **Resources**: 2 CPU, 4GB RAM per pod
- **Image**: `alphapulse/api:latest`

**Deployment YAML**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 10
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
        - name: api
          image: alphapulse/api:latest
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: database-credentials
                  key: url
            - name: REDIS_URL
              value: redis://redis-cluster:6379
            - name: VAULT_ADDR
              value: http://vault:8200
          resources:
            requests:
              cpu: 2
              memory: 4Gi
            limits:
              cpu: 4
              memory: 8Gi
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
```

**Horizontal Pod Autoscaler (HPA)**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 10
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

---

### 4. Agent Workers Deployment
**Specification**:
- **Replicas**: 30-120 (6 agent types × 5-20 workers)
- **Resources**: 1 CPU, 2GB RAM per pod
- **Image**: `alphapulse/agent-worker:latest`

**Deployment Strategy**:
```yaml
# Separate deployment per agent type for independent scaling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-worker-technical
spec:
  replicas: 20
  selector:
    matchLabels:
      app: agent-worker
      type: technical
  template:
    metadata:
      labels:
        app: agent-worker
        type: technical
    spec:
      containers:
        - name: worker
          image: alphapulse/agent-worker:latest
          env:
            - name: CELERY_QUEUES
              value: technical_agent
            - name: CELERY_CONCURRENCY
              value: "4"  # 4 tasks per worker
          resources:
            requests:
              cpu: 1
              memory: 2Gi
```

**Worker Types**:
- `agent-worker-technical` (20 replicas)
- `agent-worker-fundamental` (20 replicas)
- `agent-worker-sentiment` (20 replicas)
- `agent-worker-value` (20 replicas)
- `agent-worker-activist` (20 replicas)
- `agent-worker-buffett` (20 replicas)

---

### 5. Risk Workers Deployment
**Specification**:
- **Replicas**: 5-10
- **Resources**: 2 CPU, 4GB RAM per pod (CPU-intensive)
- **Node Affinity**: Compute-optimized node pool

**Deployment YAML**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: risk-worker
spec:
  replicas: 5
  template:
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: workload-type
                    operator: In
                    values:
                      - compute-optimized
      containers:
        - name: worker
          image: alphapulse/risk-worker:latest
          resources:
            requests:
              cpu: 2
              memory: 4Gi
            limits:
              cpu: 4
              memory: 8Gi
```

---

### 6. Redis Cluster StatefulSet
**Specification**:
- **Replicas**: 6 pods (3 masters + 3 replicas)
- **Resources**: 2 CPU, 3GB RAM per pod
- **Storage**: 10GB persistent volume per pod

**StatefulSet YAML**:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
            - containerPort: 16379  # Cluster bus
          resources:
            requests:
              cpu: 2
              memory: 3Gi
          volumeMounts:
            - name: data
              mountPath: /data
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: gp3
        resources:
          requests:
            storage: 10Gi
```

**Cluster Configuration**:
```bash
# Initialize Redis Cluster
redis-cli --cluster create \
  redis-0.redis-cluster:6379 \
  redis-1.redis-cluster:6379 \
  redis-2.redis-cluster:6379 \
  redis-3.redis-cluster:6379 \
  redis-4.redis-cluster:6379 \
  redis-5.redis-cluster:6379 \
  --cluster-replicas 1
```

---

### 7. Vault HA StatefulSet
**Specification**:
- **Replicas**: 3 pods (Raft consensus)
- **Resources**: 2 CPU, 4GB RAM per pod
- **Storage**: 10GB persistent volume per pod

**StatefulSet YAML**:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vault
spec:
  serviceName: vault
  replicas: 3
  selector:
    matchLabels:
      app: vault
  template:
    metadata:
      labels:
        app: vault
    spec:
      containers:
        - name: vault
          image: hashicorp/vault:1.15
          ports:
            - containerPort: 8200  # API
            - containerPort: 8201  # Cluster
          env:
            - name: VAULT_ADDR
              value: http://127.0.0.1:8200
            - name: VAULT_API_ADDR
              value: http://$(POD_IP):8200
          resources:
            requests:
              cpu: 2
              memory: 4Gi
          volumeMounts:
            - name: data
              mountPath: /vault/data
            - name: config
              mountPath: /vault/config
      volumes:
        - name: config
          configMap:
            name: vault-config
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
```

**Vault Configuration**:
```hcl
# vault-config.hcl
storage "raft" {
  path = "/vault/data"
  node_id = "${POD_NAME}"
}

listener "tcp" {
  address = "0.0.0.0:8200"
  tls_disable = 1  # TLS terminated at ingress
}

seal "awskms" {
  region = "us-east-1"
  kms_key_id = "arn:aws:kms:..."
}

api_addr = "http://${POD_IP}:8200"
cluster_addr = "http://${POD_IP}:8201"
```

---

### 8. PostgreSQL (AWS RDS)
**Specification**:
- **Instance Type**: db.r5.xlarge (4 vCPU, 32GB RAM)
- **Storage**: GP3 SSD (500GB, 3000 IOPS)
- **Engine**: PostgreSQL 14
- **Multi-AZ**: Yes (automatic failover)
- **Read Replica**: 1 replica (us-east-1b)

**Configuration**:
```sql
-- postgresql.conf settings
max_connections = 200
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 64MB
maintenance_work_mem = 2GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1  # SSD
effective_io_concurrency = 200
```

**Backup**:
- Automated daily backups (7-day retention)
- PITR (Point-In-Time Recovery) enabled
- Manual snapshots before migrations

---

### 9. Monitoring Stack
**Components**:
- **Prometheus**: Metrics collection (scrapes `/metrics` from pods)
- **Grafana**: Dashboards and visualizations
- **Loki**: Log aggregation
- **Alertmanager**: Alert routing (PagerDuty integration)

**Prometheus Configuration**:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

**Key Metrics**:
- `http_request_duration_seconds` (API latency)
- `http_requests_total` (request count)
- `cache_hit_rate` (Redis hit rate)
- `celery_task_duration_seconds` (task execution time)
- `database_connections_active` (PostgreSQL connections)

---

## Networking

### Service Mesh (Optional)
**Technology**: Istio or Linkerd
**Purpose**: mTLS, traffic management, observability

**Not implemented in MVP** (complexity vs benefit), but can be added later.

---

### DNS Configuration
```yaml
# External DNS (Cloudflare)
api.alphapulse.com → Ingress Load Balancer IP

# Internal DNS (Kubernetes)
api-service.alphapulse-prod.svc.cluster.local → API pods
redis-cluster.alphapulse-prod.svc.cluster.local → Redis cluster
vault.alphapulse-prod.svc.cluster.local → Vault pods
```

---

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: ingress
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
    - to:
        - podSelector:
            matchLabels:
              app: vault
      ports:
        - protocol: TCP
          port: 8200
```

---

## Security

### Secrets Management
**Method**: Kubernetes Secrets + External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
spec:
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: database-credentials
  data:
    - secretKey: url
      remoteRef:
        key: /alphapulse/prod/database/url
```

---

### RBAC
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: api-role
rules:
  - apiGroups: [""]
    resources: ["secrets", "configmaps"]
    verbs: ["get", "list"]
```

---

### Pod Security Standards
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  runAsUser:
    rule: MustRunAsNonRoot
```

---

## Disaster Recovery

### Backup Strategy
**Database**:
- Automated daily backups (RDS)
- PITR (7-day recovery window)
- Manual snapshots before migrations

**Vault**:
- Raft snapshots (daily)
- Stored in S3 with encryption
- Tested restore procedure (quarterly)

**Redis**:
- AOF (Append-Only File) persistence
- Snapshots every 6 hours
- Replicas for high availability

### Recovery Time Objective (RTO)
- **Database failure**: <5 minutes (automatic failover to replica)
- **Vault failure**: <10 minutes (promote Raft follower to leader)
- **Redis failure**: <1 minute (automatic failover to replica)
- **API pod failure**: <30 seconds (Kubernetes restarts pod)

### Recovery Point Objective (RPO)
- **Database**: <5 minutes (Multi-AZ replication)
- **Vault**: <1 hour (snapshot frequency)
- **Redis**: <1 second (AOF with fsync=1s)

---

## Cost Estimation

### Monthly Infrastructure Costs (100 tenants)
| Component | Specification | Monthly Cost |
|-----------|--------------|--------------|
| Kubernetes (EKS/GKE) | 3-10 nodes (t3.xlarge) | $600-$2,000 |
| PostgreSQL (RDS) | db.r5.xlarge + replica | $800 |
| Redis (ElastiCache) | 6 nodes (cache.t3.medium) | $400 |
| Vault (self-hosted) | 3 nodes on K8s | Included in K8s |
| Load Balancer | AWS ELB/ALB | $50 |
| S3 Storage | 500GB backups + logs | $15 |
| CloudFront CDN | 1TB transfer | $85 |
| Monitoring (Datadog) | 10 hosts, 200GB logs | $400 |
| **Total** | | **$2,350-$3,750/month** |

**Per-Tenant Cost**: $23-$37/month (at 100 tenants)

**Profitability**:
- Starter ($99/mo) → $62-$76/mo profit per tenant
- Pro ($499/mo) → $462-$476/mo profit per tenant
- Enterprise (custom) → Variable

---

## References

- [C4 Level 3: Component Diagram](c4-level3-component.md)
- [HLD Section 2.3: Deployment View](../HLD-MULTI-TENANT-SAAS.md#23-deployment-view)
- [ADR-002: Tenant Provisioning](../adr/002-tenant-provisioning-architecture.md)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/cluster-administration/)

---

**Diagram Status**: Draft (pending review)
**Review Date**: Sprint 3, Week 1
**Reviewers**: Tech Lead, DevOps Lead, CTO

---

**END OF DOCUMENT**
