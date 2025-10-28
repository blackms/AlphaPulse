# Staging Environment Setup Guide - AlphaPulse

**Purpose**: Provision staging environment for Sprint 4 load testing (Approval Condition 2)

**Target Date**: Sprint 4, Day 2 (2025-10-29)

**Owner**: DevOps Engineer

---

## Overview

The staging environment is a scaled-down replica of production used for:
- Load testing validation (p99 <500ms, error rate <1%)
- Integration testing
- Pre-production validation
- Team training

**Infrastructure Cost**: ~$300-400/month (can be stopped when not in use)

---

## Prerequisites

### AWS Account Access

**Required Permissions**:
- EC2: Create instances, security groups
- RDS: Create database instances
- EKS: Create Kubernetes clusters
- IAM: Create service accounts, roles
- Route53: Create DNS records (optional)

**Access Request**:
1. Request AWS credentials from infrastructure team
2. Install AWS CLI: `brew install awscli` (macOS) or `apt-get install awscli` (Linux)
3. Configure credentials: `aws configure`
4. Verify access: `aws sts get-caller-identity`

---

## Infrastructure Components

### 1. Kubernetes Cluster (EKS)

**Specifications**:
- **Cluster Name**: `alphapulse-staging`
- **Kubernetes Version**: 1.28+
- **Node Pool**: 2 nodes, t3.xlarge (4 vCPUs, 16GB RAM)
- **Region**: us-east-1 (or closest to your location)
- **Networking**: VPC with 2 subnets (public + private)

**Provisioning** (Option A: eksctl - Recommended):

```bash
# Install eksctl
brew install eksctl  # macOS
# OR
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster
eksctl create cluster \
  --name alphapulse-staging \
  --region us-east-1 \
  --version 1.28 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 2 \
  --nodes-min 2 \
  --nodes-max 4 \
  --managed

# Verify cluster
kubectl get nodes
```

**Provisioning** (Option B: Terraform):

```hcl
# terraform/staging/main.tf
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "alphapulse-staging"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    standard = {
      instance_types = ["t3.xlarge"]
      min_size       = 2
      max_size       = 4
      desired_size   = 2
    }
  }
}
```

**Expected Time**: 15-20 minutes

---

### 2. PostgreSQL Database (RDS)

**Specifications**:
- **Instance**: db.t3.large (2 vCPUs, 8GB RAM)
- **Engine**: PostgreSQL 14.x
- **Storage**: 50GB gp3 SSD
- **Multi-AZ**: No (single-AZ for staging)
- **Backup**: 7-day retention

**Provisioning** (AWS Console):

1. Navigate to RDS → Create database
2. Configuration:
   - Engine: PostgreSQL 14.10
   - Template: Dev/Test
   - Instance: db.t3.large
   - Storage: 50GB gp3
   - Storage autoscaling: Enabled (max 100GB)
3. Connectivity:
   - VPC: Same as EKS cluster
   - Subnet group: Private subnets
   - Public access: No
   - Security group: Allow 5432 from EKS nodes
4. Database authentication:
   - Username: `alphapulse`
   - Password: Generate strong password (save to Vault)
5. Monitoring:
   - Enhanced monitoring: Enabled (60 seconds)
   - Performance Insights: Enabled
6. Backup:
   - Retention: 7 days
   - Backup window: 03:00-04:00 UTC
7. Maintenance:
   - Auto minor version upgrade: Enabled
   - Maintenance window: Sun 04:00-05:00 UTC

**Provisioning** (AWS CLI):

```bash
aws rds create-db-instance \
  --db-instance-identifier alphapulse-staging \
  --db-instance-class db.t3.large \
  --engine postgres \
  --engine-version 14.10 \
  --master-username alphapulse \
  --master-user-password "$(openssl rand -base64 32)" \
  --allocated-storage 50 \
  --storage-type gp3 \
  --vpc-security-group-ids sg-xxxxxxxxx \
  --db-subnet-group-name alphapulse-staging-subnet \
  --backup-retention-period 7 \
  --monitoring-interval 60 \
  --enable-performance-insights \
  --no-publicly-accessible

# Wait for creation
aws rds wait db-instance-available --db-instance-identifier alphapulse-staging

# Get endpoint
aws rds describe-db-instances \
  --db-instance-identifier alphapulse-staging \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text
```

**Expected Time**: 10-15 minutes

**Post-Creation**:
```bash
# Test connection
psql -h <rds-endpoint> -U alphapulse -d postgres

# Create database
CREATE DATABASE alphapulse_staging;
```

---

### 3. Security Groups

**EKS Node Security Group**:
- Inbound:
  - Port 443 (HTTPS): From ALB security group
  - Port 8000 (API): From load testing machine
  - All traffic: From within EKS security group (node-to-node)
- Outbound: All traffic

**RDS Security Group**:
- Inbound:
  - Port 5432: From EKS node security group
- Outbound: All traffic

**Commands**:
```bash
# Create EKS node security group
aws ec2 create-security-group \
  --group-name alphapulse-staging-eks-nodes \
  --description "Security group for AlphaPulse staging EKS nodes" \
  --vpc-id vpc-xxxxxxxxx

# Allow HTTPS from ALB
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 443 \
  --source-group sg-alb-xxxxxxxxx

# Create RDS security group
aws ec2 create-security-group \
  --group-name alphapulse-staging-rds \
  --description "Security group for AlphaPulse staging RDS" \
  --vpc-id vpc-xxxxxxxxx

# Allow PostgreSQL from EKS nodes
aws ec2 authorize-security-group-ingress \
  --group-id sg-rds-xxxxxxxxx \
  --protocol tcp \
  --port 5432 \
  --source-group sg-eks-xxxxxxxxx
```

---

### 4. Secrets Management

**Store secrets in AWS Secrets Manager** (recommended) or **HashiCorp Vault**:

```bash
# Database credentials
aws secretsmanager create-secret \
  --name alphapulse/staging/database \
  --description "AlphaPulse staging database credentials" \
  --secret-string '{
    "username": "alphapulse",
    "password": "GENERATED_PASSWORD",
    "host": "alphapulse-staging.xxxxx.us-east-1.rds.amazonaws.com",
    "port": "5432",
    "database": "alphapulse_staging"
  }'

# Redis password
aws secretsmanager create-secret \
  --name alphapulse/staging/redis \
  --secret-string '{"password": "REDIS_PASSWORD"}'

# API keys
aws secretsmanager create-secret \
  --name alphapulse/staging/api-keys \
  --secret-string '{
    "jwt_secret": "JWT_SECRET",
    "openai_api_key": "OPENAI_KEY",
    "exchange_api_key": "EXCHANGE_KEY",
    "exchange_api_secret": "EXCHANGE_SECRET"
  }'
```

---

## Deployment Steps

### 1. Install Helm and kubectl

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify
kubectl version --client
helm version
```

### 2. Configure kubectl Context

```bash
# Update kubeconfig
aws eks update-kubeconfig --name alphapulse-staging --region us-east-1

# Verify connection
kubectl get nodes
kubectl get namespaces
```

### 3. Create Namespace

```bash
kubectl create namespace alphapulse-staging
kubectl config set-context --current --namespace=alphapulse-staging
```

### 4. Create Kubernetes Secrets

```bash
# Create secret from AWS Secrets Manager
DATABASE_SECRET=$(aws secretsmanager get-secret-value \
  --secret-id alphapulse/staging/database \
  --query SecretString \
  --output text)

DATABASE_URL=$(echo $DATABASE_SECRET | jq -r '"postgresql://\(.username):\(.password)@\(.host):\(.port)/\(.database)"')

kubectl create secret generic alphapulse-secrets \
  --from-literal=database-url="$DATABASE_URL" \
  --from-literal=redis-password="REDIS_PASSWORD" \
  --from-literal=jwt-secret="JWT_SECRET" \
  --from-literal=openai-api-key="OPENAI_KEY" \
  --from-literal=exchange-api-key="EXCHANGE_KEY" \
  --from-literal=exchange-api-secret="EXCHANGE_SECRET" \
  --namespace alphapulse-staging
```

### 5. Deploy with Helm

```bash
# Navigate to project root
cd ~/Projects/Personal/AlphaPulse

# Deploy AlphaPulse
helm install alphapulse ./helm/alphapulse \
  --namespace alphapulse-staging \
  --values ./helm/alphapulse/values-staging.yaml \
  --set postgresql.external.host="<RDS_ENDPOINT>" \
  --wait

# Verify deployment
kubectl get pods -n alphapulse-staging
kubectl get svc -n alphapulse-staging
```

### 6. Initialize Database

```bash
# Run database migrations
kubectl exec -it -n alphapulse-staging \
  $(kubectl get pod -n alphapulse-staging -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}') \
  -- poetry run alembic upgrade head

# Seed load test data
kubectl exec -it -n alphapulse-staging \
  $(kubectl get pod -n alphapulse-staging -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}') \
  -- poetry run python scripts/seed_load_test_users.py --staging
```

### 7. Expose API (for load testing)

```bash
# Port-forward (temporary)
kubectl port-forward -n alphapulse-staging svc/alphapulse-api 8000:80

# OR create LoadBalancer (recommended)
kubectl patch svc alphapulse-api -n alphapulse-staging \
  -p '{"spec": {"type": "LoadBalancer"}}'

# Get external IP
kubectl get svc alphapulse-api -n alphapulse-staging
```

---

## Verification Checklist

- [ ] EKS cluster created and accessible (`kubectl get nodes`)
- [ ] RDS instance created and reachable (test with `psql`)
- [ ] Security groups configured (EKS nodes can reach RDS)
- [ ] Secrets stored in AWS Secrets Manager
- [ ] Kubernetes secrets created
- [ ] AlphaPulse deployed via Helm
- [ ] All pods running (`kubectl get pods`)
- [ ] Database migrations applied
- [ ] Load test data seeded (5 tenants, 10 users)
- [ ] API accessible externally (curl health endpoint)
- [ ] Monitoring enabled (Prometheus, Grafana)

**Verification Commands**:
```bash
# Check all resources
kubectl get all -n alphapulse-staging

# Check pod logs
kubectl logs -n alphapulse-staging -l app.kubernetes.io/component=api

# Test API
curl http://<EXTERNAL_IP>/health
# Expected: {"status": "healthy"}

# Test database connection
kubectl exec -it -n alphapulse-staging \
  $(kubectl get pod -n alphapulse-staging -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}') \
  -- psql $DATABASE_URL -c "SELECT 1"
```

---

## Cost Breakdown

| Resource | Specification | Monthly Cost |
|----------|--------------|--------------|
| EKS Cluster | Control plane | $72 |
| EC2 Instances | 2× t3.xlarge | $120 (2× $60) |
| RDS Instance | db.t3.large | $100 |
| Storage (RDS) | 50GB gp3 | $6 |
| Data Transfer | ~100GB | $10 |
| **Total** | | **~$308/month** |

**Cost Optimization**:
- Stop instances when not in use (nights/weekends): ~$100/month savings
- Use Reserved Instances (1-year): ~30% savings
- Delete after Sprint 4: Save $308/month

---

## Troubleshooting

### Issue: EKS cluster creation fails

**Solution**:
- Check AWS quotas: `aws service-quotas list-service-quotas --service-code eks`
- Verify IAM permissions
- Try different region

### Issue: RDS instance not reachable from EKS

**Solution**:
- Verify security groups: `aws ec2 describe-security-groups --group-ids sg-xxxxx`
- Check VPC routing tables
- Test with bastion host

### Issue: Pods stuck in Pending

**Solution**:
- Check node resources: `kubectl top nodes`
- Describe pod: `kubectl describe pod <pod-name>`
- Scale up nodes if needed

---

## Clean-Up (After Sprint 4)

**Delete resources** to avoid ongoing costs:

```bash
# Delete Helm release
helm uninstall alphapulse -n alphapulse-staging

# Delete namespace
kubectl delete namespace alphapulse-staging

# Delete EKS cluster
eksctl delete cluster --name alphapulse-staging --region us-east-1

# Delete RDS instance
aws rds delete-db-instance \
  --db-instance-identifier alphapulse-staging \
  --skip-final-snapshot

# Delete secrets
aws secretsmanager delete-secret \
  --secret-id alphapulse/staging/database \
  --force-delete-without-recovery
```

**Expected savings**: $308/month

---

## References

- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [AWS RDS Documentation](https://docs.aws.amazon.com/rds/)
- [eksctl Documentation](https://eksctl.io/)
- [Helm Documentation](https://helm.sh/docs/)
- [AlphaPulse Helm Chart](../helm/alphapulse/README.md)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
