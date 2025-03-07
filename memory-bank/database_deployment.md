# Database Deployment Guide

This guide outlines deployment options, security practices, backup procedures, and monitoring for the AI Hedge Fund database infrastructure.

## Deployment Options

### 1. Local Development Environment

For local development:

```bash
# Start PostgreSQL with TimescaleDB and Redis using docker-compose
docker-compose -f docker-compose.db.yml up -d

# Initialize the database
python src/scripts/init_db.py
```

### 2. Docker Swarm or Kubernetes Production Deployment

For production environments:

```yaml
# Example Kubernetes deployment excerpt
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: timescaledb
spec:
  serviceName: timescaledb
  replicas: 1
  selector:
    matchLabels:
      app: timescaledb
  template:
    metadata:
      labels:
        app: timescaledb
    spec:
      containers:
      - name: timescaledb
        image: timescale/timescaledb:latest-pg14
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi
```

### 3. Managed Cloud Services

#### AWS Option:
- Amazon RDS for PostgreSQL with TimescaleDB extension
- Amazon ElastiCache for Redis
- Deploy with Terraform or CloudFormation

#### Azure Option:
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Deploy with Azure Resource Manager templates

#### Google Cloud Option:
- Cloud SQL for PostgreSQL
- Memorystore for Redis
- Deploy with Deployment Manager

## Security Best Practices

### Network Security

1. **Private Subnet**: Deploy databases in private subnets
2. **VPC**: Use VPC with proper network ACLs
3. **Security Groups**: Restrict access to necessary ports

```bash
# Example AWS CLI command to create a security group rule
aws ec2 authorize-security-group-ingress \
  --group-id sg-1234567890abcdef0 \
  --protocol tcp \
  --port 5432 \
  --source-group sg-application-servers
```

### Authentication and Authorization

1. **Strong Passwords**: Use randomly generated passwords
2. **Password Policies**: Implement password rotation
3. **IAM Integration**: Use IAM for authentication when possible
4. **Role-Based Access**: Create specific database roles with least privilege

```sql
-- Example PostgreSQL role setup
CREATE ROLE alphapulse_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA alphapulse TO alphapulse_readonly;

CREATE ROLE alphapulse_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA alphapulse TO alphapulse_app;
```

### Data Protection

1. **Encryption at Rest**: Enable storage encryption
2. **Encryption in Transit**: Force SSL/TLS connections
3. **Column-level Encryption**: Encrypt sensitive data fields
4. **Key Management**: Use a key management service

```yaml
# PostgreSQL configuration for SSL
ssl: on
ssl_cert_file: '/etc/ssl/certs/server.crt'
ssl_key_file: '/etc/ssl/private/server.key'
ssl_ca_file: '/etc/ssl/certs/ca.crt'
```

## Backup and Recovery

### Backup Strategy

1. **Automated Backups**: Schedule regular backups
2. **Point-in-Time Recovery**: Enable WAL archiving for PostgreSQL
3. **Offsite Storage**: Store backups in a different region/location
4. **Backup Encryption**: Encrypt backup files

```bash
# Example backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="alphapulse"
DB_USER="backup_user"

# Create backup
pg_dump -U $DB_USER $DB_NAME | gzip > $BACKUP_DIR/$DB_NAME\_$TIMESTAMP.sql.gz

# Encrypt backup
gpg --encrypt --recipient backup@alphapulse.com $BACKUP_DIR/$DB_NAME\_$TIMESTAMP.sql.gz

# Upload to S3
aws s3 cp $BACKUP_DIR/$DB_NAME\_$TIMESTAMP.sql.gz.gpg s3://alphapulse-backups/

# Remove local file after upload
rm $BACKUP_DIR/$DB_NAME\_$TIMESTAMP.sql.gz.gpg
```

### Recovery Procedures

1. **Recovery Testing**: Regularly test recovery procedures
2. **Documentation**: Document recovery steps in runbooks
3. **Automation**: Create scripts for common recovery scenarios

```bash
# Example restore script
#!/bin/bash
BACKUP_FILE=$1
DB_NAME="alphapulse"
DB_USER="postgres"

# Download from S3 if necessary
if [[ $BACKUP_FILE == s3://* ]]; then
  aws s3 cp $BACKUP_FILE ./backup.sql.gz.gpg
  BACKUP_FILE="./backup.sql.gz.gpg"
fi

# Decrypt backup
gpg --decrypt $BACKUP_FILE > ./backup.sql.gz

# Restore database
gunzip -c ./backup.sql.gz | psql -U $DB_USER $DB_NAME

# Clean up
rm ./backup.sql.gz
```

## Monitoring and Maintenance

### Monitoring Setup

1. **Health Checks**: Implement basic health checks
2. **Performance Metrics**: Monitor key performance indicators
3. **Alerting**: Set up alerts for critical issues
4. **Logging**: Centralize database logs

```yaml
# Prometheus monitoring configuration example
scrape_configs:
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Key Metrics to Monitor

**PostgreSQL:**
- Connection count
- Cache hit ratio
- Transaction rate
- Bloat
- Replication lag
- Query performance
- Disk usage

**Redis:**
- Memory usage
- Connected clients
- Commands per second
- Evictions
- Hit rate
- Keyspace size

### Regular Maintenance Tasks

1. **Vacuuming**: Regular PostgreSQL vacuuming 
2. **Index Maintenance**: Reindex to prevent bloat
3. **Statistics Updates**: Update table statistics
4. **Connection Management**: Monitor and limit connections

```sql
-- PostgreSQL maintenance tasks
-- Update statistics
ANALYZE VERBOSE;

-- Reindex important tables
REINDEX TABLE alphapulse.metrics;

-- Check for bloat
SELECT 
  schemaname, 
  tablename, 
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
  pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size
FROM pg_tables
WHERE schemaname = 'alphapulse'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Scaling Considerations

### Vertical Scaling

- Increase database instance size
- Add more memory and CPU
- Upgrade storage performance

### Horizontal Scaling

**PostgreSQL:**
- Read replicas for read-heavy workloads
- Sharding for large datasets
- Connection pooling with PgBouncer

**Redis:**
- Redis Cluster for distributed data
- Redis Sentinel for high availability
- Redis read replicas

### TimescaleDB-Specific Optimizations

- Optimize chunk intervals
- Configure compression policies
- Use continuous aggregates for faster queries
- Implement data retention policies

```sql
-- TimescaleDB optimization examples
-- Set chunk interval
SELECT set_chunk_time_interval('alphapulse.metrics', INTERVAL '1 day');

-- Create continuous aggregate view
CREATE MATERIALIZED VIEW metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 hour', time) AS hour,
  metric_name,
  avg(value) as avg_value,
  min(value) as min_value,
  max(value) as max_value,
  count(*) as sample_count
FROM alphapulse.metrics
GROUP BY hour, metric_name;

-- Set compression policy
ALTER TABLE alphapulse.metrics SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'metric_name'
);

SELECT add_compression_policy('alphapulse.metrics', INTERVAL '7 days');

-- Set retention policy
SELECT add_retention_policy('alphapulse.metrics', INTERVAL '90 days');
```

## Disaster Recovery Planning

1. **Recovery Point Objective (RPO)**: Define acceptable data loss
2. **Recovery Time Objective (RTO)**: Define acceptable downtime
3. **Disaster Recovery Plan**: Document comprehensive recovery steps
4. **Cross-Region Replication**: Set up replication across regions
5. **Regular DR Testing**: Schedule and perform DR tests

## Implementation Checklist

- [ ] Configure database server instances
- [ ] Set up networking and security
- [ ] Initialize schema and initial data
- [ ] Configure backup system
- [ ] Implement monitoring
- [ ] Test backup and recovery
- [ ] Document operations procedures
- [ ] Train team on maintenance tasks