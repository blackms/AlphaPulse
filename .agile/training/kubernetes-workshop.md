# Kubernetes Workshop - AlphaPulse Team Training

**Duration**: 2 hours
**Instructor**: DevOps Engineer / Tech Lead
**Audience**: Backend Engineers, Frontend Engineers

---

## Workshop Objectives

By the end of this workshop, you will be able to:
1. Understand core Kubernetes concepts (Pods, Deployments, Services)
2. Deploy AlphaPulse application to local Kubernetes cluster
3. Scale applications using HPA (Horizontal Pod Autoscaler)
4. Debug common Kubernetes issues
5. Use `kubectl` commands confidently

---

## Prerequisites

### Install Required Tools

**macOS**:
```bash
# Install kubectl
brew install kubectl

# Install Minikube
brew install minikube

# Verify installations
kubectl version --client
minikube version
```

**Linux**:
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

**Windows**:
```powershell
# Install kubectl
choco install kubernetes-cli

# Install Minikube
choco install minikube
```

---

## Part 1: Kubernetes Basics (30 minutes)

### 1.1 Core Concepts

**Cluster Architecture**:
```
┌─────────────────────────────────────────┐
│           Kubernetes Cluster            │
├─────────────────────────────────────────┤
│  Control Plane (Master Node)            │
│  ├─ API Server                          │
│  ├─ Scheduler                           │
│  ├─ Controller Manager                  │
│  └─ etcd (cluster data store)           │
├─────────────────────────────────────────┤
│  Worker Nodes                           │
│  ├─ Node 1                              │
│  │  ├─ Kubelet (node agent)             │
│  │  ├─ Kube-proxy (network)             │
│  │  └─ Pods (containers)                │
│  ├─ Node 2                              │
│  └─ Node 3                              │
└─────────────────────────────────────────┘
```

**Key Resources**:

| Resource | Description | Example |
|----------|-------------|---------|
| **Pod** | Smallest deployable unit (1+ containers) | AlphaPulse API container |
| **Deployment** | Manages Pod replicas, rolling updates | AlphaPulse API deployment (10 replicas) |
| **Service** | Exposes Pods to network | AlphaPulse API service (ClusterIP) |
| **Ingress** | HTTP/HTTPS routing to services | Route api.alphapulse.ai → API service |
| **ConfigMap** | Configuration data | Database URLs, feature flags |
| **Secret** | Sensitive data | API keys, passwords |
| **PersistentVolume** | Storage | Redis data, Vault data |
| **StatefulSet** | Stateful applications | Redis, Vault (requires persistent identity) |
| **HPA** | Auto-scaling | Scale API 10-50 replicas based on CPU |

---

### 1.2 Start Minikube

```bash
# Start Minikube with 4 CPUs and 8GB RAM
minikube start --cpus 4 --memory 8192

# Verify cluster is running
kubectl cluster-info
kubectl get nodes

# Enable ingress addon
minikube addons enable ingress
minikube addons enable metrics-server
```

**Expected output**:
```
✅ minikube v1.32.0 on Darwin 14.0
✅ Using the docker driver based on existing profile
✅ Starting control plane node minikube in cluster minikube
✅ Pulling base image ...
✅ Restarting existing docker container for "minikube" ...
✅ Preparing Kubernetes v1.28.3 on Docker 24.0.7 ...
✅ Configuring bridge CNI (Container Networking Interface) ...
✅ Verifying Kubernetes components...
✅ Done! kubectl is now configured to use "minikube" cluster
```

---

## Part 2: Deploying AlphaPulse (45 minutes)

### 2.1 Deploy with Helm

```bash
# Navigate to project root
cd ~/Projects/Personal/AlphaPulse

# Install AlphaPulse with Helm (dev values)
helm install alphapulse ./helm/alphapulse \
  --values ./helm/alphapulse/values-dev.yaml \
  --namespace alphapulse-dev \
  --create-namespace

# Watch deployment progress
kubectl get pods -n alphapulse-dev --watch

# Check deployment status
kubectl get deployments -n alphapulse-dev
kubectl get services -n alphapulse-dev
kubectl get pvc -n alphapulse-dev
```

**Expected Pods**:
- `alphapulse-api-<hash>` (1 replica)
- `alphapulse-worker-agents-<hash>` (1 replica)
- `alphapulse-worker-risk-<hash>` (1 replica)
- `redis-master-0` (StatefulSet)
- `vault-0` (StatefulSet)

---

### 2.2 Access the Application

**Option 1: Port-forward (recommended for dev)**:
```bash
# Port-forward API service
kubectl port-forward -n alphapulse-dev svc/alphapulse-api 8000:80

# Test API
curl http://localhost:8000/health

# Expected response: {"status": "healthy"}
```

**Option 2: Minikube tunnel**:
```bash
# Start tunnel (requires sudo)
minikube tunnel

# Get service URL
minikube service alphapulse-api -n alphapulse-dev --url
```

---

### 2.3 Inspect Resources

**View Pod logs**:
```bash
# Get pod name
kubectl get pods -n alphapulse-dev

# View logs
kubectl logs -n alphapulse-dev alphapulse-api-<hash>

# Follow logs
kubectl logs -n alphapulse-dev -f alphapulse-api-<hash>

# View logs from specific container (if pod has multiple containers)
kubectl logs -n alphapulse-dev alphapulse-api-<hash> -c api
```

**Describe resources**:
```bash
# Describe pod (events, status, resource usage)
kubectl describe pod -n alphapulse-dev alphapulse-api-<hash>

# Describe deployment
kubectl describe deployment -n alphapulse-dev alphapulse-api

# Describe service
kubectl describe service -n alphapulse-dev alphapulse-api
```

**Execute commands in pod**:
```bash
# Open shell in pod
kubectl exec -it -n alphapulse-dev alphapulse-api-<hash> -- /bin/bash

# Run single command
kubectl exec -n alphapulse-dev alphapulse-api-<hash> -- env
kubectl exec -n alphapulse-dev alphapulse-api-<hash> -- ps aux
```

---

## Part 3: Scaling and Updates (30 minutes)

### 3.1 Manual Scaling

```bash
# Scale API deployment to 3 replicas
kubectl scale deployment alphapulse-api -n alphapulse-dev --replicas=3

# Verify scaling
kubectl get pods -n alphapulse-dev

# Check load distribution (make multiple requests)
for i in {1..10}; do
  curl http://localhost:8000/health
  echo ""
done
```

**Expected**: Requests distributed across 3 pods (load balancing).

---

### 3.2 Horizontal Pod Autoscaler (HPA)

**Enable HPA** (requires metrics-server):
```bash
# Check if metrics-server is running
kubectl get pods -n kube-system | grep metrics-server

# Create HPA for API deployment
kubectl autoscale deployment alphapulse-api \
  -n alphapulse-dev \
  --min=2 \
  --max=10 \
  --cpu-percent=70

# View HPA status
kubectl get hpa -n alphapulse-dev

# Expected output:
# NAME              REFERENCE                  TARGETS   MINPODS   MAXPODS   REPLICAS   AGE
# alphapulse-api    Deployment/alphapulse-api  15%/70%   2         10        3          1m
```

**Generate load** (watch HPA scale up):
```bash
# Install stress tool
kubectl run load-generator -n alphapulse-dev --rm -it --image=busybox -- /bin/sh -c "while true; do wget -q -O- http://alphapulse-api/health; done"

# Watch HPA (in another terminal)
kubectl get hpa -n alphapulse-dev --watch

# Expected: REPLICAS increases from 2 → 10 as CPU usage increases
```

Stop load generator (Ctrl+C) and watch replicas scale down.

---

### 3.3 Rolling Updates

**Update deployment image**:
```bash
# Update image tag
kubectl set image deployment/alphapulse-api \
  -n alphapulse-dev \
  api=alphapulse/api:v1.1.0

# Watch rollout
kubectl rollout status deployment/alphapulse-api -n alphapulse-dev

# Expected:
# Waiting for deployment "alphapulse-api" rollout to finish: 1 of 3 updated replicas are available...
# Waiting for deployment "alphapulse-api" rollout to finish: 2 of 3 updated replicas are available...
# deployment "alphapulse-api" successfully rolled out
```

**Rollout history**:
```bash
# View rollout history
kubectl rollout history deployment/alphapulse-api -n alphapulse-dev

# Rollback to previous version
kubectl rollout undo deployment/alphapulse-api -n alphapulse-dev

# Rollback to specific revision
kubectl rollout undo deployment/alphapulse-api -n alphapulse-dev --to-revision=2
```

---

## Part 4: ConfigMaps and Secrets (15 minutes)

### 4.1 ConfigMaps

**Create ConfigMap**:
```bash
# From literal values
kubectl create configmap app-config -n alphapulse-dev \
  --from-literal=LOG_LEVEL=DEBUG \
  --from-literal=ENVIRONMENT=development

# View ConfigMap
kubectl get configmap app-config -n alphapulse-dev -o yaml
```

**Use ConfigMap in Deployment**:
```yaml
# deployment.yaml snippet
env:
  - name: LOG_LEVEL
    valueFrom:
      configMapKeyRef:
        name: app-config
        key: LOG_LEVEL
```

---

### 4.2 Secrets

**Create Secret**:
```bash
# From literal values (base64 encoded automatically)
kubectl create secret generic db-credentials -n alphapulse-dev \
  --from-literal=username=alphapulse \
  --from-literal=password=secret123

# View Secret (values are base64 encoded)
kubectl get secret db-credentials -n alphapulse-dev -o yaml

# Decode secret value
kubectl get secret db-credentials -n alphapulse-dev -o jsonpath='{.data.password}' | base64 --decode
```

**Use Secret in Deployment**:
```yaml
# deployment.yaml snippet
env:
  - name: DATABASE_PASSWORD
    valueFrom:
      secretKeyRef:
        name: db-credentials
        key: password
```

---

## Part 5: Troubleshooting (15 minutes)

### 5.1 Common Issues

#### Issue 1: Pod not starting (ImagePullBackOff)

**Symptoms**:
```bash
kubectl get pods -n alphapulse-dev
# NAME                             READY   STATUS             RESTARTS   AGE
# alphapulse-api-abc123            0/1     ImagePullBackOff   0          2m
```

**Diagnosis**:
```bash
kubectl describe pod alphapulse-api-abc123 -n alphapulse-dev
# Events:
#   Warning  Failed  2m   kubelet  Failed to pull image "alphapulse/api:latest": rpc error: code = Unknown desc = Error response from daemon: pull access denied
```

**Solution**:
- Image doesn't exist or is private (add ImagePullSecrets)
- Wrong image tag (check deployment.yaml)
- Build image locally for Minikube: `eval $(minikube docker-env) && docker build -t alphapulse/api:latest .`

---

#### Issue 2: Pod crashing (CrashLoopBackOff)

**Symptoms**:
```bash
kubectl get pods -n alphapulse-dev
# NAME                             READY   STATUS             RESTARTS   AGE
# alphapulse-api-abc123            0/1     CrashLoopBackOff   5          5m
```

**Diagnosis**:
```bash
# Check logs
kubectl logs -n alphapulse-dev alphapulse-api-abc123

# Check previous container logs (if pod restarted)
kubectl logs -n alphapulse-dev alphapulse-api-abc123 --previous

# Common causes:
# - Application crashes on startup (missing env vars, database connection failed)
# - Health check failures
# - Insufficient resources (OOMKilled)
```

**Solution**:
- Fix application code or configuration
- Check environment variables
- Increase resource limits

---

#### Issue 3: Service not accessible

**Symptoms**:
```bash
curl http://localhost:8000/health
# curl: (7) Failed to connect to localhost port 8000: Connection refused
```

**Diagnosis**:
```bash
# Check if port-forward is running
ps aux | grep port-forward

# Check if service exists
kubectl get svc -n alphapulse-dev

# Check service endpoints (should list pod IPs)
kubectl get endpoints -n alphapulse-dev alphapulse-api
```

**Solution**:
- Restart port-forward: `kubectl port-forward -n alphapulse-dev svc/alphapulse-api 8000:80`
- Check service selector matches pod labels
- Check pod is running and ready

---

### 5.2 Useful Commands Cheat Sheet

```bash
# Get all resources in namespace
kubectl get all -n alphapulse-dev

# View events (troubleshooting)
kubectl get events -n alphapulse-dev --sort-by='.lastTimestamp'

# View resource usage
kubectl top pods -n alphapulse-dev
kubectl top nodes

# Delete resources
kubectl delete pod <pod-name> -n alphapulse-dev
kubectl delete deployment <deployment-name> -n alphapulse-dev

# Edit resource live
kubectl edit deployment alphapulse-api -n alphapulse-dev

# Copy files to/from pod
kubectl cp /local/file.txt alphapulse-dev/alphapulse-api-abc123:/app/file.txt
kubectl cp alphapulse-dev/alphapulse-api-abc123:/app/logs/app.log ./app.log

# View API resources
kubectl api-resources

# View cluster info
kubectl cluster-info
kubectl version
```

---

## Part 6: Hands-On Exercise (15 minutes)

### Exercise: Deploy a New Service

**Task**: Deploy a Redis cache service and connect it to the API.

**Steps**:
1. Create a Redis deployment (1 replica)
2. Create a Redis service (ClusterIP, port 6379)
3. Update API deployment to use Redis service
4. Verify API can connect to Redis

**Solution**:
```bash
# 1. Create Redis deployment
kubectl create deployment redis -n alphapulse-dev --image=redis:7

# 2. Expose Redis as a service
kubectl expose deployment redis -n alphapulse-dev --port=6379 --target-port=6379

# 3. Update API deployment env var
kubectl set env deployment/alphapulse-api -n alphapulse-dev REDIS_URL=redis://redis:6379

# 4. Verify connection
kubectl exec -n alphapulse-dev alphapulse-api-<hash> -- redis-cli -h redis ping
# Expected output: PONG
```

---

## Wrap-Up and Q&A (10 minutes)

### Key Takeaways

1. **Kubernetes abstractions**: Pods, Deployments, Services make it easy to run containers at scale
2. **kubectl is your friend**: Master basic commands (`get`, `describe`, `logs`, `exec`)
3. **Helm simplifies deployment**: Package all resources in a single chart
4. **HPA enables auto-scaling**: Automatically scale based on CPU/memory usage
5. **Troubleshooting workflow**: `get pods` → `describe pod` → `logs` → fix

### Next Steps

- **Practice**: Deploy AlphaPulse to local Minikube daily
- **Read docs**: https://kubernetes.io/docs/
- **Explore**: Try StatefulSets (Redis, Vault), Ingress, PersistentVolumes
- **Vault training**: Tomorrow (Day 3), 2 hours

### Useful Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Helm Documentation](https://helm.sh/docs/)
- [AlphaPulse Helm Chart](../../helm/alphapulse/README.md)

---

## Clean-Up

```bash
# Delete AlphaPulse deployment
helm uninstall alphapulse -n alphapulse-dev

# Delete namespace
kubectl delete namespace alphapulse-dev

# Stop Minikube
minikube stop

# Delete Minikube cluster (optional)
minikube delete
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
