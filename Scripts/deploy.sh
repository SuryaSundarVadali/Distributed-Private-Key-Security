#!/bin/bash
# Deployment Script for DeFi Agent System
# Supports: development, staging, production environments

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_ENV=${1:-development}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/Infrastructure/docker-compose.yaml"
BACKUP_DIR="$PROJECT_ROOT/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_info "All requirements met"
}

backup_data() {
    log_info "Creating backup..."
    
    mkdir -p "$BACKUP_DIR"
    BACKUP_FILE="$BACKUP_DIR/backup_${TIMESTAMP}.tar.gz"
    
    cd "$PROJECT_ROOT"
    tar -czf "$BACKUP_FILE" \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        logs/ Infrastructure/
    
    log_info "Backup created: $BACKUP_FILE"
}

validate_environment() {
    log_info "Validating environment: $DEPLOY_ENV"
    
    case $DEPLOY_ENV in
        development|staging|production)
            log_info "Environment validated: $DEPLOY_ENV"
            ;;
        rollback)
            log_info "Rollback requested"
            ;;
        *)
            log_error "Invalid environment: $DEPLOY_ENV"
            log_info "Valid options: development, staging, production, rollback"
            exit 1
            ;;
    esac
}

build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT/Infrastructure"
    
    docker build -f Dockerfile.scheduler -t defi-scheduler:${DEPLOY_ENV} .. || {
        log_error "Failed to build scheduler image"
        exit 1
    }
    
    docker build -f Dockerfile.node -t defi-node:${DEPLOY_ENV} .. || {
        log_error "Failed to build node image"
        exit 1
    }
    
    docker build -f Dockerfile.monitor -t defi-monitor:${DEPLOY_ENV} .. || {
        log_error "Failed to build monitor image"
        exit 1
    }
    
    log_info "All images built successfully"
}

stop_services() {
    log_info "Stopping existing services..."
    
    cd "$PROJECT_ROOT/Infrastructure"
    docker-compose down || log_warn "No running services to stop"
    
    log_info "Services stopped"
}

start_services() {
    log_info "Starting services..."
    
    cd "$PROJECT_ROOT/Infrastructure"
    
    # Pull latest changes if in production
    if [ "$DEPLOY_ENV" == "production" ]; then
        log_info "Pulling latest changes..."
        cd "$PROJECT_ROOT"
        git pull origin main || log_warn "Could not pull latest changes"
        cd "$PROJECT_ROOT/Infrastructure"
    fi
    
    # Start services
    docker-compose up -d
    
    log_info "Services started"
}

health_check() {
    log_info "Running health checks..."
    
    local max_attempts=30
    local attempt=1
    local scheduler_healthy=false
    local monitor_healthy=false
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        # Check scheduler
        if curl -f http://localhost:8000/statistics &> /dev/null; then
            scheduler_healthy=true
            log_info "Scheduler is healthy"
        fi
        
        # Check monitor
        if curl -f http://localhost:8080/health &> /dev/null; then
            monitor_healthy=true
            log_info "Monitor is healthy"
        fi
        
        if $scheduler_healthy && $monitor_healthy; then
            log_info "All services healthy!"
            return 0
        fi
        
        sleep 10
        ((attempt++))
    done
    
    log_error "Health checks failed after $max_attempts attempts"
    return 1
}

view_logs() {
    log_info "Viewing service logs..."
    
    cd "$PROJECT_ROOT/Infrastructure"
    docker-compose logs --tail=50
}

show_status() {
    log_info "Service Status:"
    
    cd "$PROJECT_ROOT/Infrastructure"
    docker-compose ps
    
    echo ""
    log_info "Docker Images:"
    docker images | grep defi-
    
    echo ""
    log_info "Docker Volumes:"
    docker volume ls | grep defi
}

cleanup_old_backups() {
    log_info "Cleaning up old backups (keeping last 10)..."
    
    cd "$BACKUP_DIR"
    ls -t backup_*.tar.gz | tail -n +11 | xargs -r rm --
    
    log_info "Cleanup complete"
}

rollback() {
    log_info "Starting rollback procedure..."
    
    # Find latest backup
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/backup_*.tar.gz 2>/dev/null | head -n 1)
    
    if [ -z "$LATEST_BACKUP" ]; then
        log_error "No backups found for rollback"
        exit 1
    fi
    
    log_info "Rolling back to: $LATEST_BACKUP"
    
    # Stop services
    stop_services
    
    # Restore backup
    cd "$PROJECT_ROOT"
    tar -xzf "$LATEST_BACKUP"
    
    # Rebuild and restart
    build_images
    start_services
    
    if health_check; then
        log_info "Rollback successful"
    else
        log_error "Rollback failed - services not healthy"
        exit 1
    fi
}

run_tests() {
    log_info "Running tests before deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Run Phase 1-4 tests
    for phase in 1 2 3 4; do
        log_info "Running Phase $phase tests..."
        python test_phase${phase}.py || {
            log_error "Phase $phase tests failed"
            return 1
        }
    done
    
    log_info "All tests passed"
    return 0
}

push_to_registry() {
    if [ -z "$DOCKER_REGISTRY" ] || [ -z "$DOCKER_USERNAME" ] || [ -z "$DOCKER_PASSWORD" ]; then
        log_warn "Docker registry credentials not set, skipping push"
        return 0
    fi
    
    log_info "Pushing images to registry..."
    
    echo "$DOCKER_PASSWORD" | docker login "$DOCKER_REGISTRY" -u "$DOCKER_USERNAME" --password-stdin
    
    docker tag defi-scheduler:${DEPLOY_ENV} ${DOCKER_REGISTRY}/defi-scheduler:${DEPLOY_ENV}
    docker tag defi-node:${DEPLOY_ENV} ${DOCKER_REGISTRY}/defi-node:${DEPLOY_ENV}
    docker tag defi-monitor:${DEPLOY_ENV} ${DOCKER_REGISTRY}/defi-monitor:${DEPLOY_ENV}
    
    docker push ${DOCKER_REGISTRY}/defi-scheduler:${DEPLOY_ENV}
    docker push ${DOCKER_REGISTRY}/defi-node:${DEPLOY_ENV}
    docker push ${DOCKER_REGISTRY}/defi-monitor:${DEPLOY_ENV}
    
    log_info "Images pushed to registry"
}

# Main deployment flow
main() {
    log_info "==================================="
    log_info "DeFi Agent System Deployment"
    log_info "Environment: $DEPLOY_ENV"
    log_info "Timestamp: $TIMESTAMP"
    log_info "==================================="
    
    # Validate environment
    validate_environment
    
    # Handle rollback
    if [ "$DEPLOY_ENV" == "rollback" ]; then
        rollback
        exit 0
    fi
    
    # Check requirements
    check_requirements
    
    # Run tests in production
    if [ "$DEPLOY_ENV" == "production" ]; then
        if ! run_tests; then
            log_error "Tests failed, aborting deployment"
            exit 1
        fi
    fi
    
    # Create backup
    backup_data
    
    # Stop existing services
    stop_services
    
    # Build new images
    build_images
    
    # Push to registry (if configured)
    push_to_registry
    
    # Start services
    start_services
    
    # Run health checks
    if health_check; then
        log_info "==================================="
        log_info "Deployment successful!"
        log_info "==================================="
        
        show_status
        
        # Cleanup old backups
        cleanup_old_backups
        
        log_info ""
        log_info "Access points:"
        log_info "  - Scheduler: http://localhost:8000/statistics"
        log_info "  - Monitor: http://localhost:8080/dashboard"
        log_info ""
        log_info "View logs: docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
        
    else
        log_error "Deployment failed - services not healthy"
        log_info "Running automatic rollback..."
        rollback
        exit 1
    fi
}

# Run main function
main
