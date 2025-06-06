# Distributed Private Key Security System

A sophisticated distributed cryptographic system that implements Multi-Factor Key Derivation Function (MFKDF), Shamir's Secret Sharing (SSS), HMAC-based One-Time Passwords (HOTP), Merkle Trees, and Secure Multi-Party Computation (MPC) with intelligent task scheduling using the HEFT (Heterogeneous Earliest Finish Time) algorithm.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [File Structure](#file-structure)
5. [Configuration](#configuration)
6. [Running the System](#running-the-system)
7. [API Endpoints](#api-endpoints)
8. [Monitoring and Debugging](#monitoring-and-debugging)
9. [Security Considerations](#security-considerations)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)

## Architecture Overview

The system implements a distributed cryptographic workflow:

1. **MFKDF Key Generation**: Creates RSA private keys using multi-factor authentication
2. **Secret Sharing**: Distributes private key shares using Shamir's Secret Sharing
3. **HOTP Verification**: Generates and verifies time-based tokens for authentication
4. **Merkle Tree Storage**: Ensures data integrity through cryptographic proofs
5. **MPC Computation**: Performs secure multi-party computations
6. **Key Reconstruction**: Reconstructs private keys from distributed shares
7. **HEFT Scheduling**: Optimally distributes tasks across compute nodes

### Components

- **Scheduler Server**: Central coordinator using HEFT algorithm
- **Distributed Nodes**: Worker nodes executing cryptographic tasks
- **Docker Containers**: Isolated execution environments
- **REST API**: Communication between scheduler and nodes

## Prerequisites

- **Docker**: Version 20.0 or higher
- **Docker Compose**: Version 2.0 or higher
- **Python**: Version 3.9 or higher (for local development)
- **Git**: For cloning the repository

### System Requirements

- **Memory**: Minimum 4GB RAM
- **CPU**: Multi-core processor (4+ cores recommended)
- **Storage**: 2GB free disk space
- **Network**: Open ports 8000-8010

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Private\ Key\ Security
```

### 2. Install Dependencies (Local Development)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Verify Docker Installation

```bash
docker --version
docker-compose --version
```

## File Structure

```
Private Key Security/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── docker-compose.yaml           # Docker orchestration
├── Dockerfile.scheduler          # Scheduler container
├── Dockerfile.node               # Node container
├── 
├── Core Cryptographic Modules:
├── mfkdf.py                      # Multi-Factor Key Derivation
├── secret_sharing.py             # Shamir's Secret Sharing
├── hotp.py                       # HMAC-based OTP
├── merkle_tree.py                # Merkle Tree implementation
├── mpc.py                        # Multi-Party Computation
├── key_generation.py             # MFKDF Key Generator
├── 
├── Distributed System:
├── task_scheduler.py             # HEFT Algorithm (heft_scheduler.py)
├── scheduler_server.py           # Central Scheduler
├── distributed_node.py           # Worker Nodes
├── enhanced_distributed_executor.py  # Main Executor
├── 
└── Scripts:
    ├── run_local.py              # Local execution script
    ├── monitor.py                # System monitoring
    └── test_system.py            # System tests
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# System Configuration
NUM_NODES=5
THRESHOLD=3
SCHEDULER_PORT=8000
HEARTBEAT_INTERVAL=10

# Security Settings
MASTER_SEED_FILE=master.seed
LOG_LEVEL=INFO

# Docker Settings
COMPOSE_PROJECT_NAME=crypto_system
```

### Node Configuration

Each node can be customized with:
- Processing power multiplier
- Available authentication factors
- Maximum concurrent tasks
- Specialized task types

## Running the System

### Option 1: Docker Compose (Recommended)

#### 1. Start the Complete System

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

#### 2. Scale Nodes Dynamically

```bash
# Add more nodes
docker-compose up -d --scale node_0=2 --scale node_1=3

# Remove nodes
docker-compose stop node_4
```

#### 3. Individual Service Management

```bash
# Start only scheduler
docker-compose up -d scheduler

# Start specific nodes
docker-compose up -d node_0 node_1 node_2

# Restart services
docker-compose restart scheduler
```

### Option 2: Local Development

#### 1. Start the Scheduler

```bash
python scheduler_server.py 8000
```

#### 2. Start Worker Nodes

```bash
# Terminal 1
NODE_ID=node_0 SCHEDULER_HOST=localhost python distributed_node.py

# Terminal 2
NODE_ID=node_1 SCHEDULER_HOST=localhost python distributed_node.py

# Terminal 3
NODE_ID=node_2 SCHEDULER_HOST=localhost python distributed_node.py
```

#### 3. Execute Workflow

```bash
python enhanced_distributed_executor.py
```

### Option 3: Mixed Mode

Run scheduler locally, nodes in Docker:

```bash
# Start scheduler locally
python scheduler_server.py 8000

# Start nodes in Docker
docker-compose up -d node_0 node_1 node_2 node_3 node_4
```

## API Endpoints

### Scheduler API

The scheduler exposes the following REST endpoints:

#### Node Management

```http
POST /heartbeat
Content-Type: application/json

{
  "node_id": "node_0",
  "timestamp": 1640995200.0,
  "current_load": 2,
  "max_capacity": 5,
  "status": "active",
  "performance_metrics": {
    "tasks_completed": 10,
    "tasks_failed": 0,
    "average_execution_time": 45.2
  }
}
```

#### Task Management

```http
GET /get_task/{node_id}
# Returns next task assignment for the node

POST /update_task_status
Content-Type: application/json

{
  "task_id": "key_generation_1",
  "node_id": "node_0",
  "status": "completed",
  "timestamp": 1640995200.0,
  "execution_time": 42.5
}
```

#### System Monitoring

```http
GET /statistics
# Returns system-wide statistics

GET /nodes
# Returns information about all nodes

GET /tasks
# Returns information about all tasks

POST /reschedule
# Manually trigger task rescheduling
```

### Example API Usage

```bash
# Check system status
curl http://localhost:8000/statistics

# Get node information
curl http://localhost:8000/nodes

# Get task information
curl http://localhost:8000/tasks

# Trigger manual rescheduling
curl -X POST http://localhost:8000/reschedule
```

## Monitoring and Debugging

### 1. Real-time Monitoring

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f scheduler
docker-compose logs -f node_0

# Monitor system resources
docker stats
```

### 2. System Status Dashboard

Access the monitoring endpoints:

```bash
# System statistics
curl http://localhost:8000/statistics | jq

# Node status
curl http://localhost:8000/nodes | jq

# Task status
curl http://localhost:8000/tasks | jq
```

### 3. Debug Mode

Enable debug logging:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or modify docker-compose.yaml
environment:
  - LOG_LEVEL=DEBUG
```

### 4. Performance Monitoring

```bash
# Monitor container resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Check task execution times
curl http://localhost:8000/statistics | jq '.average_completion_time'
```

## Security Considerations

### 1. Network Security

- All communication uses HTTPS in production
- Firewall rules restrict access to scheduler ports
- VPN or private networks recommended for multi-host deployments

### 2. Cryptographic Security

- Master seeds are generated using cryptographically secure random generators
- Private keys never leave their assigned nodes in plaintext
- All secret sharing uses proven cryptographic schemes

### 3. Authentication Factors

Configure MFKDF factors based on your security requirements:

```python
# Available factors
factors = [
    'biometric',        # Fingerprint, face recognition
    'password',         # User password
    'hardware_token',   # Hardware security key
    'location',         # GPS coordinates
    'time_window'       # Time-based constraints
]
```

### 4. Access Control

- Implement role-based access control (RBAC)
- Use API keys for scheduler access
- Regular security audits and key rotation

## Troubleshooting

### Common Issues

#### 1. Scheduler Not Starting

```bash
# Check port availability
netstat -tlnp | grep 8000

# Check Docker logs
docker-compose logs scheduler

# Verify configuration
docker-compose config
```

#### 2. Nodes Not Connecting

```bash
# Verify network connectivity
docker-compose exec node_0 ping scheduler

# Check environment variables
docker-compose exec node_0 env | grep SCHEDULER

# Restart networking
docker-compose down && docker-compose up -d
```

#### 3. Task Failures

```bash
# Check task status
curl http://localhost:8000/tasks | jq '.[] | select(.status == "failed")'

# Review node logs
docker-compose logs node_0 | grep ERROR

# Check resource utilization
docker stats
```

#### 4. Performance Issues

```bash
# Monitor task execution times
curl http://localhost:8000/statistics | jq '.average_completion_time'

# Check node load distribution
curl http://localhost:8000/nodes | jq '.[] | {id: .node_id, load: .load_factor}'

# Scale up nodes if needed
docker-compose up -d --scale node_0=3
```

### System Recovery

#### 1. Graceful Restart

```bash
# Stop all services
docker-compose down

# Clean up resources
docker system prune -f

# Restart system
docker-compose up -d
```

#### 2. Reset System State

```bash
# Remove all containers and volumes
docker-compose down -v

# Rebuild images
docker-compose build --no-cache

# Start fresh
docker-compose up -d
```

### Log Analysis

#### 1. Structured Logging

All components use structured logging:

```bash
# Filter by severity
docker-compose logs | grep ERROR

# Filter by component
docker-compose logs scheduler | grep "Task assignment"

# Filter by time
docker-compose logs --since="2023-01-01T00:00:00" --until="2023-01-01T23:59:59"
```

#### 2. Performance Metrics

```bash
# Task completion rates
curl http://localhost:8000/statistics | jq '.completed_tasks / .total_tasks'

# Node utilization
curl http://localhost:8000/nodes | jq 'map(.load_factor) | add / length'

# System throughput
curl http://localhost:8000/statistics | jq '.completed_tasks'
```

## Advanced Configuration

### 1. Custom Task Types

Add new cryptographic operations:

```python
# In task_scheduler.py
def register_custom_task_handler(self, task_type: str, handler: Callable):
    self.task_handlers[task_type] = handler

# Example: Digital signature task
def handle_digital_signature(task: CryptoTask, node_id: str) -> Dict[str, Any]:
    # Implementation here
    pass
```

### 2. Load Balancing Strategies

Customize the HEFT algorithm:

```python
# In task_scheduler.py
def custom_node_selection(self, task: CryptoTask) -> str:
    # Custom load balancing logic
    pass
```

### 3. Fault Tolerance

Configure automatic failover:

```yaml
# In docker-compose.yaml
services:
  node_0:
    restart: unless-stopped
    deploy:
      restart_policy:
        condition: on-failure
        max_attempts: 3
```

## Testing

### 1. Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_cryptographic.py
python -m pytest tests/test_scheduling.py
python -m pytest tests/test_integration.py
```

### 2. Integration Tests

```bash
# Test complete workflow
python test_system.py

# Test with different node configurations
NODE_COUNT=3 THRESHOLD=2 python test_system.py
```

### 3. Load Testing

```bash
# Stress test the system
python scripts/load_test.py --tasks=100 --nodes=5 --duration=300
```

## Production Deployment

### 1. Environment Setup

```bash
# Create production environment file
cp .env.example .env.production

# Update configuration for production
# - Set secure random seeds
# - Configure proper hostnames
# - Enable TLS/SSL
# - Set resource limits
```

### 2. Security Hardening

```bash
# Use secrets management
docker swarm init
echo "your-master-seed" | docker secret create master_seed -

# Enable TLS
# Configure reverse proxy (nginx/traefik)
# Set up monitoring (Prometheus/Grafana)
```

### 3. Scaling

```bash
# Docker Swarm deployment
docker stack deploy -c docker-stack.yml crypto-system

# Kubernetes deployment
kubectl apply -f k8s/
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python -m pytest`
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Use type hints for all function signatures

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For technical support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

## Changelog

### v1.0.0 (2024-01-01)
- Initial release
- MFKDF implementation
- HEFT scheduling algorithm
- Docker containerization
- REST API interface

---

**Note**: This is a research prototype. For production use, implement additional security measures, comprehensive testing, and professional security audits.