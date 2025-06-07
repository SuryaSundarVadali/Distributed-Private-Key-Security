# Distributed Private Key Security System

A sophisticated distributed cryptographic system that implements Multi-Factor Key Derivation Function (MFKDF), Shamir's Secret Sharing (SSS), HMAC-based One-Time Passwords (HOTP), Merkle Trees, and Secure Multi-Party Computation (MPC) with intelligent task scheduling using the HEFT (Heterogeneous Earliest Finish Time) algorithm.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [File Structure](#file-structure)
5. [Configuration](#configuration)
6. [Running the System](#running-the-system)
7. [Task Types](#task-types)
8. [API Endpoints](#api-endpoints)
9. [Monitoring](#monitoring)
10. [Security Features](#security-features)
11. [Troubleshooting](#troubleshooting)
12. [Development](#development)

## Architecture Overview

### System Design

The system consists of two main layers:

**1. Cryptographic Infrastructure Layer** (runs on every node):
- **MFKDF Key Generation**: Multi-factor authenticated RSA key generation
- **Secret Sharing**: Shamir's Secret Sharing for key distribution
- **HOTP Authentication**: Time-based token authentication between nodes
- **Merkle Tree Verification**: Data integrity proofs
- **MPC Framework**: Secure multi-party computation capabilities

**2. Distributed Computing Layer**:
- **HEFT Scheduler**: Optimal task distribution using Heterogeneous Earliest Finish Time algorithm
- **Worker Nodes**: Execute computational tasks while maintaining cryptographic security
- **Task Types**: Mathematical, data processing, image processing, ML, scientific computing

### Task Categories

- **Mathematical**: Prime generation, matrix multiplication, Fibonacci calculation
- **Data Processing**: Array sorting, pattern search, data compression
- **Image Processing**: Filters, object detection, image enhancement
- **Network**: Web crawling, API data fetching
- **Machine Learning**: Linear regression, K-means clustering
- **File Operations**: Hashing, encryption, statistical analysis
- **Scientific**: Monte Carlo simulations, statistical analysis

## Prerequisites

### System Requirements
- **Docker**: Version 20.0+ 
- **Docker Compose**: Version 2.0+
- **Memory**: 4GB RAM minimum (8GB recommended)
- **CPU**: Multi-core processor (4+ cores recommended)
- **Network**: Ports 8000-8010 available
- **Storage**: 2GB free space

### For Local Development
- **Python**: 3.9+
- **Git**: Latest version

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd "Private Key Security"

# Make scripts executable (Linux/macOS)
chmod +x Scripts/*.py
```

### 2. Docker Deployment (Recommended)

```bash
# Start the entire system (Infrastructure folder contains Docker files)
docker-compose up --build

# Or run in background
docker-compose up --build -d

# Check logs
docker-compose logs -f

# Stop system
docker-compose down
```

### 3. Local Development Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS) 
source venv/bin/activate

# Install dependencies (Infrastructure folder contains requirements.txt)
pip install -r Infrastructure/requirements.txt

# Run locally
python Scripts/run_local.py --nodes 5 --threshold 3
```

## File Structure

```
Private Key Security/
├── 📁 Core System/
│   ├── scheduler_server.py          # Central HEFT scheduler
│   ├── distributed_node.py          # Worker node implementation
│   ├── task_scheduler.py           # HEFT algorithm & task definitions
│   └── enhanced_distributed_executor.py  # Main system coordinator
│
├── 📁 Cryptographic Modules/
│   ├── mfkdf.py                    # Multi-Factor Key Derivation
│   ├── secret_sharing.py          # Shamir's Secret Sharing
│   ├── hotp.py                     # HMAC-based OTP
│   ├── merkle_tree.py              # Merkle Tree implementation
│   ├── mpc.py                      # Multi-Party Computation
│   └── key_generation.py          # MFKDF key generator
│
├── 📁 Infrastructure/
│   ├── docker-compose.yaml        # Docker orchestration
│   ├── Dockerfile.scheduler       # Scheduler container
│   ├── Dockerfile.node            # Node container
│   ├── requirements.txt           # Python dependencies
│   └── .env                       # Environment configuration
│
├── 📁 Scripts/
│   ├── run_local.py               # Local execution
│   ├── monitor.py                 # System monitoring
│   └── test_system.py             # System tests
│
├── 📁 logs/                        # Runtime logs directory
├── 📁 __pycache__/                 # Python cache files (auto-generated)
├── README.md                       # This file
└── .gitignore                     # Git ignore rules
```

## Configuration

### Environment Variables

Create `.env` file in the `Infrastructure/` folder or set environment variables:

```bash
# System Configuration
NUM_NODES=5                    # Number of worker nodes
THRESHOLD=3                    # Secret sharing threshold
SCHEDULER_PORT=8000           # Scheduler port
HEARTBEAT_INTERVAL=10         # Heartbeat frequency (seconds)

# Security Settings  
LOG_LEVEL=INFO                # Logging level
MASTER_SEED_FILE=master.seed  # Seed file location

# Docker Settings
COMPOSE_PROJECT_NAME=crypto_system
```

### Node Specializations

Nodes automatically specialize in different task types based on their ID:

- **Nodes 0-19**: Mathematical computations
- **Nodes 20-39**: Data processing
- **Nodes 40-59**: Image processing  
- **Nodes 60-79**: Network operations
- **Nodes 80-99**: Machine learning & scientific computing

## Running the System

### Option 1: Docker (Production-like)

```bash
# Navigate to project root
cd "Private Key Security"

# Start all services (Docker files are in Infrastructure/)
docker-compose -f Infrastructure/docker-compose.yaml up --build

# Scale nodes (optional)
docker-compose -f Infrastructure/docker-compose.yaml up --scale node_0=2 --scale node_1=2

# View specific service logs
docker-compose -f Infrastructure/docker-compose.yaml logs scheduler
docker-compose -f Infrastructure/docker-compose.yaml logs node_0

# Stop and cleanup
docker-compose -f Infrastructure/docker-compose.yaml down -v
```

### Option 2: Local Development

```bash
# Terminal 1: Start scheduler (go to Core System folder)
cd "Core System"
python scheduler_server.py 8000

# Terminal 2-6: Start nodes (stay in Core System folder)
python distributed_node.py node_0
python distributed_node.py node_1
python distributed_node.py node_2
python distributed_node.py node_3
python distributed_node.py node_4

# Terminal 7: Monitor system (go to project root)
cd ..
python Scripts/monitor.py --interval 3
```

### Option 3: Automated Local

```bash
# From project root
python Scripts/run_local.py --nodes 5 --threshold 3 --port 8000

# Custom configuration
python Scripts/run_local.py --nodes 10 --threshold 5 --port 8001
```

## Task Types

The system schedules general computational tasks while maintaining cryptographic security on each node. The cryptographic components (MFKDF, SSS, HOTP, Merkle Trees, MPC) run automatically on every node for security purposes.

### Mathematical Computing
```python
# Prime generation
{
    "task_type": "prime_generation",
    "data": {
        "range_start": 1000000,
        "range_end": 1010000,
        "count": 100
    }
}

# Matrix multiplication
{
    "task_type": "matrix_multiplication",
    "data": {
        "matrix_size": 1000,
        "iterations": 5
    }
}

# Fibonacci calculation
{
    "task_type": "fibonacci_calculation",
    "data": {
        "n": 50000,
        "modulo": 1000000007
    }
}
```

### Data Processing
```python
# Large array sorting
{
    "task_type": "large_array_sort",
    "data": {
        "array_size": 1000000,
        "algorithm": "quicksort"
    }
}

# Pattern search
{
    "task_type": "pattern_search",
    "data": {
        "text_size": 10000000,
        "pattern": "cryptography",
        "algorithm": "kmp"
    }
}

# Data compression
{
    "task_type": "data_compression",
    "data": {
        "data_size": 5000000,
        "algorithm": "lz77"
    }
}
```

### Image Processing
```python
# Image filtering
{
    "task_type": "image_processing",
    "data": {
        "operation": "gaussian_blur",
        "image_size": "1920x1080",
        "kernel_size": 15
    }
}

# Object detection
{
    "task_type": "object_detection",
    "data": {
        "algorithm": "edge_detection",
        "image_count": 50
    }
}
```

### Network Operations
```python
# Web crawling
{
    "task_type": "web_crawling",
    "data": {
        "urls": ["https://example.com"],
        "depth": 2
    }
}

# API data fetching
{
    "task_type": "api_data_fetch",
    "data": {
        "endpoint": "https://jsonplaceholder.typicode.com/posts",
        "count": 100
    }
}
```

### Machine Learning
```python
# Linear regression
{
    "task_type": "linear_regression",
    "data": {
        "dataset_size": 10000,
        "features": 20,
        "iterations": 1000
    }
}

# K-means clustering
{
    "task_type": "kmeans_clustering",
    "data": {
        "data_points": 50000,
        "clusters": 10,
        "dimensions": 5
    }
}
```

### File Operations
```python
# File hashing
{
    "task_type": "file_hashing",
    "data": {
        "file_size": 100000000,
        "algorithm": "sha256"
    }
}

# File encryption
{
    "task_type": "file_encryption",
    "data": {
        "file_size": 50000000,
        "algorithm": "aes256"
    }
}
```

### Scientific Computing
```python
# Monte Carlo simulation
{
    "task_type": "monte_carlo_simulation",
    "data": {
        "iterations": 1000000,
        "variables": 3
    }
}

# Statistical analysis
{
    "task_type": "statistical_analysis",
    "data": {
        "dataset_size": 100000,
        "operations": ["mean", "std", "correlation"]
    }
}
```

## API Endpoints

### Scheduler API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/heartbeat` | POST | Node heartbeat registration |
| `/get_task/<node_id>` | GET | Request task assignment |
| `/update_task_status` | POST | Report task completion |
| `/statistics` | GET | System statistics |
| `/nodes` | GET | Node information |
| `/tasks` | GET | Task information |
| `/reschedule` | POST | Trigger manual rescheduling |

### Example API Usage

```bash
# Get system statistics
curl http://localhost:8000/statistics

# Get node information  
curl http://localhost:8000/nodes

# Get task status
curl http://localhost:8000/tasks

# Request task for node
curl http://localhost:8000/get_task/node_0

# Trigger rescheduling
curl -X POST http://localhost:8000/reschedule

# Send heartbeat (from node)
curl -X POST http://localhost:8000/heartbeat \
  -H "Content-Type: application/json" \
  -d '{"node_id": "node_0", "timestamp": 1634567890, "current_load": 2, "max_capacity": 5, "status": "active"}'

# Update task status (from node)
curl -X POST http://localhost:8000/update_task_status \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_123", "status": "completed", "execution_time": 45.2}'
```

## Monitoring

### Real-time Dashboard

```bash
# Start monitoring dashboard (from project root)
python Scripts/monitor.py

# Custom monitoring
python Scripts/monitor.py --host localhost --port 8000 --interval 5
```

### Dashboard Features

- **System Statistics**: Task completion rates, node utilization
- **Node Status**: Active/inactive nodes, current loads, specializations
- **Task Progress**: Pending, running, completed, failed tasks
- **Performance Metrics**: Average execution times, throughput
- **Security Status**: Cryptographic verification, authentication status

### Log Files

```bash
# View logs in real-time (Docker)
docker-compose -f Infrastructure/docker-compose.yaml logs -f

# View specific service logs
docker-compose -f Infrastructure/docker-compose.yaml logs scheduler
docker-compose -f Infrastructure/docker-compose.yaml logs node_0

# For local development (logs stored in logs/ folder)
tail -f logs/scheduler.log
tail -f logs/node_0.log
```

## Security Features

### Cryptographic Infrastructure (Automatic on Every Node)

1. **Multi-Factor Key Derivation Function (MFKDF)**
   - Generates deterministic RSA keys using multiple authentication factors
   - Factors: biometric, password, hardware token, location, time window
   - Threshold-based factor combination (minimum 3 out of 5)

2. **Shamir's Secret Sharing (SSS)**
   - Distributes private keys across multiple nodes
   - Configurable threshold (default 3 out of 5 nodes)
   - Secure reconstruction without revealing individual shares

3. **HMAC-based One-Time Passwords (HOTP)**
   - Time-synchronized authentication between nodes
   - Prevents replay attacks
   - Rolling counter mechanism

4. **Merkle Tree Verification**
   - Data integrity proofs for all shared information
   - Tamper detection and verification
   - Efficient verification without revealing data

5. **Secure Multi-Party Computation (MPC)**
   - Secure computations without revealing private data
   - Threshold-based computation verification
   - Cryptographic proof generation

### Security Best Practices

- **Never commit cryptographic keys**: All keys are generated at runtime
- **Secure communication**: All inter-node communication is authenticated
- **Threshold security**: System remains secure even if minority of nodes are compromised
- **Audit trails**: All cryptographic operations are logged and verifiable
- **Perfect forward secrecy**: Keys are regenerated periodically

## Troubleshooting

### Common Issues

#### 1. Docker Issues

```bash
# Permission denied
sudo usermod -aG docker $USER
newgrp docker

# Port already in use
docker-compose -f Infrastructure/docker-compose.yaml down
netstat -tulpn | grep :8000
kill -9 <process_id>

# Out of memory
docker system prune -a
docker-compose -f Infrastructure/docker-compose.yaml up --build --scale node_0=2  # Reduce nodes
```

#### 2. Node Connection Issues

```bash
# Check network connectivity
docker network ls
docker network inspect infrastructure_crypto_network

# Restart specific node
docker-compose -f Infrastructure/docker-compose.yaml restart node_0

# Check node logs
docker-compose -f Infrastructure/docker-compose.yaml logs node_0
```

#### 3. Task Scheduling Issues

```bash
# Check scheduler status
curl http://localhost:8000/statistics

# Manual reschedule
curl -X POST http://localhost:8000/reschedule

# Check task queue
curl http://localhost:8000/tasks
```

#### 4. Performance Issues

```bash
# Monitor resource usage
docker stats

# Reduce concurrent tasks per node
# Edit Infrastructure/.env: NUM_NODES=3, or reduce task complexity

# Check for deadlocks
python Scripts/monitor.py --interval 1
```

#### 5. Module Import Issues

```bash
# If running locally and getting import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to your shell profile
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)"' >> ~/.bashrc
source ~/.bashrc
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug (if debug compose file exists)
docker-compose -f Infrastructure/docker-compose.yaml -f Infrastructure/docker-compose.debug.yaml up

# Local debug
cd "Core System"
python scheduler_server.py --debug
```

### Health Checks

```bash
# System health
curl http://localhost:8000/statistics

# Node health
curl http://localhost:8000/nodes

# Task health
curl http://localhost:8000/tasks

# Manual health check
python Scripts/test_system.py --health-check
```

## Development

### Setting Up Development Environment

```bash
# Clone and setup
git clone <repository-url>
cd "Private Key Security"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies from Infrastructure folder
pip install -r Infrastructure/requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Pre-commit setup (if .pre-commit-config.yaml exists)
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Unit tests (if tests directory exists)
python -m pytest tests/ -v

# Integration tests
python Scripts/test_system.py

# Performance tests
python Scripts/test_system.py --performance

# Load testing
python Scripts/test_system.py --load-test --nodes 10
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Security scan
bandit -r .
```

### Adding New Task Types

1. **Define task in `Core System/scheduler_server.py`**:
```python
CryptoTask(
    task_id="my_custom_task",
    task_type="custom_computation",
    data={"param1": "value1"},
    priority=TaskPriority.MEDIUM,
    computation_cost=20.0
)
```

2. **Implement handler in `Core System/distributed_node.py`**:
```python
def _execute_custom_computation(self, task: CryptoTask) -> Dict[str, Any]:
    # Your implementation here
    return {"status": "success", "result": "computed_value"}
```

3. **Add to task type mapping**:
```python
def _execute_task_by_type(self, task: CryptoTask) -> Dict[str, Any]:
    if task.task_type == "custom_computation":
        return self._execute_custom_computation(task)
    # ... existing handlers
```

### Performance Tuning

```yaml
# Infrastructure/docker-compose.yaml
services:
  scheduler:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
  
  node_0:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
```

### Monitoring and Metrics

```bash
# Enable detailed metrics
export ENABLE_METRICS=true

# Custom monitoring with export
python Scripts/monitor.py --detailed --export-json metrics.json

# Performance profiling
cd "Core System"
python -m cProfile -o profile.prof scheduler_server.py
# Install snakeviz: pip install snakeviz
snakeviz profile.prof
```

### Working with Folders

```bash
# Navigate to different components
cd "Core System"           # Main system files
cd "Cryptographic Modules" # Crypto implementations
cd "Infrastructure"        # Docker and deployment files
cd "Scripts"              # Utility scripts

# Running components from correct locations
cd "Core System" && python scheduler_server.py 8000
cd "Core System" && python distributed_node.py node_0
python Scripts/monitor.py
```

### Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Project Structure for New Features

```
feature/
├── feature_module.py          # Main implementation
├── tests/
│   ├── test_feature.py       # Unit tests
│   └── test_integration.py   # Integration tests
├── docs/
│   └── feature_doc.md        # Documentation
└── examples/
    └── feature_example.py    # Usage examples
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check this README and inline code documentation
- **Examples**: See `examples/` directory for usage examples

## Changelog

### Version 1.0.0
- Initial release with HEFT scheduling
- Full MFKDF, SSS, HOTP, Merkle Tree, MPC implementation
- Docker containerization
- Real-time monitoring dashboard
- Comprehensive test suite
- Organized folder structure for better maintainability

### Roadmap
- [ ] Kubernetes deployment
- [ ] Advanced fault tolerance
- [ ] Performance optimizations
- [ ] Additional cryptographic algorithms
- [ ] Web-based monitoring UI
- [ ] Distributed storage backend
- [ ] Enhanced testing framework
- [ ] Configuration management improvements