#!/bin/bash
# Redis Cluster Health Check Script
#
# Performs comprehensive health checks on AlphaPulse Redis Cluster
# Exit codes: 0 (healthy), 1 (unhealthy), 2 (error)

set -euo pipefail

# Configuration
REDIS_PASSWORD="${REDIS_PASSWORD:-alphapulse_redis_secret}"
COMPOSE_FILE="docker-compose.redis-cluster.yml"
HEALTH_CHECK_KEY="health_check:$(date +%s)"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Helper functions
log_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_info() {
    echo "  $1"
}

# Main health check function
main() {
    echo "======================================"
    echo "Redis Cluster Health Check"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================"
    echo ""

    # Check 1: Docker containers running
    echo "[1/8] Docker Containers Status"
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up (healthy)"; then
        RUNNING=$(docker-compose -f "$COMPOSE_FILE" ps | grep -c "Up (healthy)" || echo "0")
        if [ "$RUNNING" -eq 6 ]; then
            log_pass "All 6 Redis nodes running and healthy"
        else
            log_fail "Only $RUNNING/6 nodes healthy"
        fi
    else
        log_fail "No healthy Redis nodes found"
    fi
    echo ""

    # Check 2: Cluster state
    echo "[2/8] Cluster State"
    CLUSTER_STATE=$(docker-compose -f "$COMPOSE_FILE" exec -T redis-master-1 \
        redis-cli -a "$REDIS_PASSWORD" cluster info 2>/dev/null | grep "cluster_state" | cut -d: -f2 | tr -d '\r')

    if [ "$CLUSTER_STATE" = "ok" ]; then
        log_pass "Cluster state: OK"
    else
        log_fail "Cluster state: $CLUSTER_STATE"
    fi
    echo ""

    # Check 3: Hash slot coverage
    echo "[3/8] Hash Slot Coverage"
    SLOTS_ASSIGNED=$(docker-compose -f "$COMPOSE_FILE" exec -T redis-master-1 \
        redis-cli -a "$REDIS_PASSWORD" cluster info 2>/dev/null | grep "cluster_slots_assigned" | cut -d: -f2 | tr -d '\r')
    SLOTS_OK=$(docker-compose -f "$COMPOSE_FILE" exec -T redis-master-1 \
        redis-cli -a "$REDIS_PASSWORD" cluster info 2>/dev/null | grep "cluster_slots_ok" | cut -d: -f2 | tr -d '\r')

    if [ "$SLOTS_ASSIGNED" -eq 16384 ] && [ "$SLOTS_OK" -eq 16384 ]; then
        log_pass "All 16,384 hash slots assigned and OK"
    else
        log_fail "Hash slots: $SLOTS_ASSIGNED assigned, $SLOTS_OK OK (expected: 16384)"
    fi
    echo ""

    # Check 4: Node count
    echo "[4/8] Cluster Node Count"
    NODE_COUNT=$(docker-compose -f "$COMPOSE_FILE" exec -T redis-master-1 \
        redis-cli -a "$REDIS_PASSWORD" cluster nodes 2>/dev/null | wc -l | tr -d ' ')

    if [ "$NODE_COUNT" -eq 6 ]; then
        log_pass "6 nodes in cluster"
    else
        log_fail "Node count: $NODE_COUNT (expected: 6)"
    fi
    echo ""

    # Check 5: Master/Replica topology
    echo "[5/8] Master/Replica Topology"
    MASTER_COUNT=$(docker-compose -f "$COMPOSE_FILE" exec -T redis-master-1 \
        redis-cli -a "$REDIS_PASSWORD" cluster nodes 2>/dev/null | grep -c "master" || echo "0")
    REPLICA_COUNT=$(docker-compose -f "$COMPOSE_FILE" exec -T redis-master-1 \
        redis-cli -a "$REDIS_PASSWORD" cluster nodes 2>/dev/null | grep -c "slave" || echo "0")

    if [ "$MASTER_COUNT" -eq 3 ] && [ "$REPLICA_COUNT" -eq 3 ]; then
        log_pass "Topology correct: 3 masters + 3 replicas"
    else
        log_fail "Topology: $MASTER_COUNT masters, $REPLICA_COUNT replicas (expected: 3+3)"
    fi

    # Check each master has exactly 1 replica
    for i in 1 2 3; do
        REPLICAS=$(docker exec alphapulse-redis-master-$i \
            redis-cli -a "$REDIS_PASSWORD" info replication 2>/dev/null | grep "connected_slaves:" | cut -d: -f2 | tr -d '\r')
        if [ "$REPLICAS" -eq 1 ]; then
            log_info "Master $i: 1 replica connected"
        else
            log_warn "Master $i: $REPLICAS replicas (expected: 1)"
        fi
    done
    echo ""

    # Check 6: Write/Read test
    echo "[6/8] Write/Read Test"
    if docker-compose -f "$COMPOSE_FILE" exec -T redis-master-1 \
        redis-cli -a "$REDIS_PASSWORD" -c set "$HEALTH_CHECK_KEY" "$(date +%s)" >/dev/null 2>&1; then

        VALUE=$(docker-compose -f "$COMPOSE_FILE" exec -T redis-master-1 \
            redis-cli -a "$REDIS_PASSWORD" -c get "$HEALTH_CHECK_KEY" 2>/dev/null | tr -d '\r')

        if [ -n "$VALUE" ]; then
            log_pass "Write/Read successful (value: $VALUE)"
            # Cleanup test key
            docker-compose -f "$COMPOSE_FILE" exec -T redis-master-1 \
                redis-cli -a "$REDIS_PASSWORD" -c del "$HEALTH_CHECK_KEY" >/dev/null 2>&1
        else
            log_fail "Read returned empty value"
        fi
    else
        log_fail "Write operation failed"
    fi
    echo ""

    # Check 7: Memory usage
    echo "[7/8] Memory Usage"
    TOTAL_MEMORY=0
    for i in 1 2 3; do
        MEM_USED=$(docker exec alphapulse-redis-master-$i \
            redis-cli -a "$REDIS_PASSWORD" info memory 2>/dev/null | grep "used_memory:" | cut -d: -f2 | tr -d '\r')
        MEM_MAX=$(docker exec alphapulse-redis-master-$i \
            redis-cli -a "$REDIS_PASSWORD" config get maxmemory 2>/dev/null | tail -1 | tr -d '\r')

        if [ -n "$MEM_USED" ] && [ -n "$MEM_MAX" ]; then
            MEM_PERCENT=$((MEM_USED * 100 / MEM_MAX))
            TOTAL_MEMORY=$((TOTAL_MEMORY + MEM_USED))

            if [ "$MEM_PERCENT" -lt 80 ]; then
                log_info "Master $i: ${MEM_PERCENT}% memory used"
            else
                log_warn "Master $i: ${MEM_PERCENT}% memory used (high)"
            fi
        fi
    done

    # Convert bytes to MB
    TOTAL_MB=$((TOTAL_MEMORY / 1024 / 1024))
    log_pass "Total memory usage: ${TOTAL_MB}MB"
    echo ""

    # Check 8: Redis Exporter health
    echo "[8/8] Monitoring (Redis Exporter)"
    if curl -s http://localhost:9121/health >/dev/null 2>&1; then
        log_pass "Redis Exporter responding on port 9121"

        # Check if metrics are being exposed
        METRIC_COUNT=$(curl -s http://localhost:9121/metrics 2>/dev/null | grep -c "^redis_" || echo "0")
        if [ "$METRIC_COUNT" -gt 0 ]; then
            log_info "Exposing $METRIC_COUNT Redis metrics"
        else
            log_warn "No Redis metrics found"
        fi
    else
        log_fail "Redis Exporter not responding"
    fi
    echo ""

    # Summary
    echo "======================================"
    echo "Health Check Summary"
    echo "======================================"
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"
    if [ "$FAILED_CHECKS" -gt 0 ]; then
        echo -e "${RED}Failed: $FAILED_CHECKS${NC}"
    else
        echo "Failed: 0"
    fi
    echo ""

    # Exit code
    if [ "$FAILED_CHECKS" -eq 0 ]; then
        echo -e "${GREEN}✓ Cluster is HEALTHY${NC}"
        exit 0
    else
        echo -e "${RED}✗ Cluster has ISSUES${NC}"
        exit 1
    fi
}

# Run main function
main
