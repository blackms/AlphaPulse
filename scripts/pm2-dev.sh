#!/bin/bash
# PM2 Development Management Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create logs directory
mkdir -p logs/pm2

function start_services() {
    echo -e "${BLUE}🚀 Starting AlphaPulse services with PM2...${NC}"

    # Start infrastructure first (postgres, redis, vault)
    echo -e "${YELLOW}📦 Starting infrastructure services...${NC}"
    pm2 start ecosystem.config.js --only postgres,redis,vault

    # Wait for services to be ready
    echo -e "${YELLOW}⏳ Waiting for services to start (15s)...${NC}"
    sleep 15

    # Start application services
    echo -e "${YELLOW}🎯 Starting application services...${NC}"
    pm2 start ecosystem.config.js --only alphapulse-api

    echo -e "${GREEN}✅ All services started!${NC}"
    pm2 status
}

function stop_services() {
    echo -e "${YELLOW}🛑 Stopping all AlphaPulse services...${NC}"
    pm2 stop ecosystem.config.js
    echo -e "${GREEN}✅ All services stopped${NC}"
}

function restart_services() {
    echo -e "${YELLOW}🔄 Restarting all AlphaPulse services...${NC}"
    pm2 restart ecosystem.config.js
    echo -e "${GREEN}✅ All services restarted${NC}"
}

function delete_services() {
    echo -e "${YELLOW}🗑️  Deleting all AlphaPulse services from PM2...${NC}"
    pm2 delete ecosystem.config.js || true
    echo -e "${GREEN}✅ Services deleted${NC}"
}

function status_services() {
    echo -e "${BLUE}📊 AlphaPulse Services Status:${NC}"
    pm2 status
}

function logs_service() {
    local service=${1:-alphapulse-api}
    echo -e "${BLUE}📋 Showing logs for: $service${NC}"
    pm2 logs "$service" --lines 100
}

function monitor_services() {
    echo -e "${BLUE}📊 Opening PM2 Monitor...${NC}"
    pm2 monit
}

function show_help() {
    echo -e "${BLUE}AlphaPulse PM2 Development Manager${NC}"
    echo ""
    echo "Usage: $0 {command} [args]"
    echo ""
    echo "Commands:"
    echo "  start       - Start all services"
    echo "  stop        - Stop all services"
    echo "  restart     - Restart all services"
    echo "  delete      - Delete all services from PM2"
    echo "  status      - Show services status"
    echo "  logs [name] - Show logs (default: alphapulse-api)"
    echo "  monitor     - Open PM2 monitor"
    echo "  help        - Show this help"
    echo ""
    echo "Services:"
    echo "  - postgres          (PostgreSQL database)"
    echo "  - redis             (Redis cache)"
    echo "  - vault             (HashiCorp Vault)"
    echo "  - alphapulse-api    (FastAPI backend)"
    echo ""
}

case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    delete)
        delete_services
        ;;
    status)
        status_services
        ;;
    logs)
        logs_service "$2"
        ;;
    monitor)
        monitor_services
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
