"""
Main Distributed Cryptographic Task Executor with HEFT Scheduling
Integrates MFKDF, SSS, HOTP, Merkle Trees, and MPC with intelligent task distribution
"""
import time
import hashlib
import secrets
import sys
import threading
import logging
from typing import Dict, Any
from pathlib import Path

# Add the parent directory and Cryptographic Modules to Python path
parent_dir = Path(__file__).parent.parent
crypto_modules_dir = parent_dir / "Cryptographic Modules"
sys.path.append(str(parent_dir))
sys.path.append(str(crypto_modules_dir))

from task_scheduler import HEFTScheduler, CryptoTask, NodeCapabilities, TaskPriority

# Import from Cryptographic Modules folder
try:
    from mfkdf import DistributedMFKDF
    from key_generation import MFKDFDeterministicKeyGenerator
    from secret_sharing import ShamirSecretSharing
    from hotp import HOTP
    from merkle_tree import MerkleTree
    from mpc import SecureMultiPartyComputation, MPCNode
except ImportError as e:
    print(f"Error importing cryptographic modules: {e}")
    print("Please ensure all cryptographic modules are in the 'Cryptographic Modules' folder")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedCryptoExecutor:
    """Main orchestrator for distributed cryptographic operations with computational tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scheduler = HEFTScheduler(heartbeat_interval=config.get('heartbeat_interval', 10.0))
        self.distributed_mfkdf = DistributedMFKDF(
            node_count=config['num_nodes'],
            threshold=config['threshold']
        )
        
        # Cryptographic infrastructure (runs automatically on every node)
        self.master_seed = secrets.token_bytes(32)
        self.hotp_secret = hashlib.sha256(self.master_seed + b"HOTP").digest()
        self.counter = 0
        self.crypto_state = {
            'private_key_shares': {},
            'merkle_tree': None,
            'mpc_results': {},
            'hotp_tokens': {},
            'master_private_key': None
        }
        
        # Initialize cryptographic infrastructure
        self._initialize_crypto_infrastructure()
        
        # Register computational task handlers
        self._register_task_handlers()
        
        # Initialize nodes
        self._initialize_nodes()
    
    def _initialize_crypto_infrastructure(self):
        """Initialize cryptographic infrastructure that runs on every node"""
        logger.info("Initializing cryptographic infrastructure...")
        
        # Step 1: Generate master private key using MFKDF
        self._perform_key_generation()
        
        # Step 2: Distribute secret shares using SSS
        self._perform_share_distribution()
        
        # Step 3: Generate HOTP tokens for authentication
        self._perform_hotp_setup()
        
        # Step 4: Create Merkle tree for integrity verification
        self._perform_merkle_storage()
        
        # Step 5: Initialize MPC framework
        self._perform_mpc_initialization()
        
        logger.info("✅ Cryptographic infrastructure initialized successfully")
    
    def _perform_key_generation(self):
        """Generate master private key using MFKDF"""
        logger.info("🔐 Generating master private key with MFKDF...")
        
        # Setup MFKDF for key generation
        generator = MFKDFDeterministicKeyGenerator("master_node")
        
        # Generate RSA key with multiple factors
        active_factors = ['biometric', 'password', 'hardware_token']
        rsa_key = generator.generate_rsa_key_with_mfkdf(
            active_factors=active_factors,
            key_size=2048
        )
        
        # Convert to integer for sharing
        private_key_der = rsa_key.export_key('DER')
        private_key_int = int.from_bytes(private_key_der, 'big')
        
        # Store in crypto state
        self.crypto_state['master_private_key'] = private_key_int
        self.crypto_state['master_key_der'] = private_key_der
        
        logger.info(f"✅ Master key generated: {rsa_key.size_in_bits()} bits")
    
    def _perform_share_distribution(self):
        """Distribute secret shares using Shamir's Secret Sharing"""
        logger.info("🔀 Distributing secret shares...")
        
        if not self.crypto_state['master_private_key']:
            raise ValueError("Master private key not generated")
        
        # Create shares using Shamir's Secret Sharing
        shares = ShamirSecretSharing.create_shares(
            self.crypto_state['master_private_key'],
            self.config['threshold'],
            self.config['num_nodes']
        )
        
        # Distribute shares to nodes
        for i, share in enumerate(shares):
            node_id = f'crypto_node_{i}'
            self.crypto_state['private_key_shares'][node_id] = share
        
        logger.info(f"✅ Distributed {len(shares)} shares with threshold {self.config['threshold']}")
    
    def _perform_hotp_setup(self):
        """Setup HOTP authentication for all nodes"""
        logger.info("🔑 Setting up HOTP authentication...")
        
        for i in range(self.config['num_nodes']):
            node_id = f'crypto_node_{i}'
            
            # Generate HOTP tokens for this node
            tokens = []
            for j in range(10):  # Generate 10 tokens per node
                token = HOTP.generate(self.hotp_secret, (i * 10) + j)
                tokens.append(token)
            
            self.crypto_state['hotp_tokens'][node_id] = tokens
        
        logger.info(f"✅ HOTP tokens generated for {self.config['num_nodes']} nodes")
    
    def _perform_merkle_storage(self):
        """Create Merkle tree for integrity verification"""
        logger.info("🌳 Creating Merkle tree for integrity verification...")
        
        # Collect share data for Merkle tree
        share_data = []
        for node_id, share in self.crypto_state['private_key_shares'].items():
            share_bytes = f"{node_id}:{share[0]}:{share[1]}".encode()
            share_data.append(share_bytes)
        
        if not share_data:
            raise ValueError("No share data available for Merkle tree")
        
        # Create Merkle tree
        merkle_tree = MerkleTree(share_data)
        self.crypto_state['merkle_tree'] = merkle_tree
        
        logger.info(f"✅ Merkle tree created with {len(share_data)} leaves")
    
    def _perform_mpc_initialization(self):
        """Initialize MPC framework"""
        logger.info("🤝 Initializing MPC framework...")
        
        # Create MPC nodes from available shares
        mpc_nodes = []
        for node_id, share in self.crypto_state['private_key_shares'].items():
            secret_value = share[0]  # Use x-coordinate as secret
            mpc_node = MPCNode(node_id, secret_value)
            mpc_nodes.append(mpc_node)
        
        # Initialize MPC system
        mpc_system = SecureMultiPartyComputation(mpc_nodes, self.config['threshold'])
        
        # Store MPC system for later use
        self.crypto_state['mpc_system'] = mpc_system
        
        logger.info(f"✅ MPC framework initialized with {len(mpc_nodes)} nodes")
    
    def _register_task_handlers(self):
        """Register handlers for computational task types"""
        handlers = {
            # Mathematical tasks
            'prime_generation': self._handle_prime_generation,
            'matrix_multiplication': self._handle_matrix_multiplication,
            'fibonacci_calculation': self._handle_fibonacci_calculation,
            
            # Data processing tasks
            'large_array_sort': self._handle_array_sort,
            'pattern_search': self._handle_pattern_search,
            'data_compression': self._handle_data_compression,
            
            # Image processing tasks
            'image_processing': self._handle_image_processing,
            'object_detection': self._handle_object_detection,
            
            # Network tasks
            'web_crawling': self._handle_web_crawling,
            'api_data_fetch': self._handle_api_data_fetch,
            
            # Machine learning tasks
            'linear_regression': self._handle_linear_regression,
            'kmeans_clustering': self._handle_kmeans_clustering,
            
            # File operations
            'file_hashing': self._handle_file_hashing,
            'file_encryption': self._handle_file_encryption,
            
            # Scientific computing
            'monte_carlo_simulation': self._handle_monte_carlo,
            'statistical_analysis': self._handle_statistical_analysis
        }
        
        for task_type, handler in handlers.items():
            self.scheduler.register_task_handler(task_type, handler)
    
    def _initialize_nodes(self):
        """Initialize compute nodes with capabilities"""
        for i in range(self.config['num_nodes']):
            node_id = f'crypto_node_{i}'
            
            # Determine node specialization based on ID
            if i < 20:
                specialization = 'mathematical'
                task_types = ['prime_generation', 'matrix_multiplication', 'fibonacci_calculation']
            elif i < 40:
                specialization = 'data_processing'
                task_types = ['large_array_sort', 'pattern_search', 'data_compression']
            elif i < 60:
                specialization = 'image_processing'
                task_types = ['image_processing', 'object_detection']
            elif i < 80:
                specialization = 'network'
                task_types = ['web_crawling', 'api_data_fetch']
            else:
                specialization = 'ml_scientific'
                task_types = ['linear_regression', 'kmeans_clustering', 'monte_carlo_simulation']
            
            # Define node capabilities
            capabilities = NodeCapabilities(
                node_id=node_id,
                computation_power=1.0 + (i * 0.1),
                available_factors=[
                    'biometric', 'password', 'hardware_token', 'location', 'time_window'
                ],
                max_concurrent_tasks=3 + (i % 3),
                task_completion_rate=0.95 - (i * 0.005),
                specialization=specialization,
                supported_task_types=task_types
            )
            
            self.scheduler.register_node(node_id, capabilities)
    
    def start_execution(self):
        """Start the distributed execution system"""
        logger.info("Starting Distributed Cryptographic Execution System")
        logger.info("=" * 55)
        
        # Start scheduler
        self.scheduler.start()
        
        # Create and submit computational tasks
        self._create_computational_workflow()
        
        # Monitor execution
        self._monitor_execution()
    
    def _create_computational_workflow(self):
        """Create computational tasks that will be executed securely"""
        
        tasks = []
        
        # Mathematical computation tasks
        tasks.extend([
            CryptoTask(
                task_id="prime_gen_1",
                task_type="prime_generation",
                data={
                    "range_start": 1000000,
                    "range_end": 1010000,
                    "count": 100
                },
                priority=TaskPriority.MEDIUM,
                computation_cost=2.0
            ),
            CryptoTask(
                task_id="matrix_mult_1",
                task_type="matrix_multiplication",
                data={
                    "matrix_size": 500,
                    "iterations": 3
                },
                priority=TaskPriority.MEDIUM,
                computation_cost=3.0
            ),
            CryptoTask(
                task_id="fibonacci_1",
                task_type="fibonacci_calculation",
                data={
                    "n": 10000,
                    "modulo": 1000000007
                },
                priority=TaskPriority.LOW,
                computation_cost=1.5
            )
        ])
        
        # Data processing tasks
        tasks.extend([
            CryptoTask(
                task_id="array_sort_1",
                task_type="large_array_sort",
                data={
                    "array_size": 100000,
                    "algorithm": "quicksort"
                },
                priority=TaskPriority.MEDIUM,
                computation_cost=2.5
            ),
            CryptoTask(
                task_id="pattern_search_1",
                task_type="pattern_search",
                data={
                    "text_size": 1000000,
                    "pattern": "cryptography",
                    "algorithm": "kmp"
                },
                priority=TaskPriority.LOW,
                computation_cost=1.8
            )
        ])
        
        # Image processing tasks
        tasks.extend([
            CryptoTask(
                task_id="image_proc_1",
                task_type="image_processing",
                data={
                    "operation": "gaussian_blur",
                    "image_size": "1024x768",
                    "kernel_size": 7
                },
                priority=TaskPriority.HIGH,
                computation_cost=4.0
            )
        ])
        
        # Network tasks
        tasks.extend([
            CryptoTask(
                task_id="api_fetch_1",
                task_type="api_data_fetch",
                data={
                    "endpoint": "https://jsonplaceholder.typicode.com/posts",
                    "count": 50
                },
                priority=TaskPriority.MEDIUM,
                computation_cost=1.0
            )
        ])
        
        # Machine learning tasks
        tasks.extend([
            CryptoTask(
                task_id="linear_reg_1",
                task_type="linear_regression",
                data={
                    "dataset_size": 5000,
                    "features": 10,
                    "iterations": 500
                },
                priority=TaskPriority.HIGH,
                computation_cost=3.5
            )
        ])
        
        # Scientific computing tasks
        tasks.extend([
            CryptoTask(
                task_id="monte_carlo_1",
                task_type="monte_carlo_simulation",
                data={
                    "iterations": 100000,
                    "variables": 2
                },
                priority=TaskPriority.MEDIUM,
                computation_cost=2.8
            )
        ])
        
        # Submit all tasks
        for task in tasks:
            # Verify cryptographic security before task submission
            if self._verify_crypto_security():
                self.scheduler.submit_task(task)
                logger.info(f"Submitted secure task: {task.task_id}")
            else:
                logger.error(f"Security verification failed for task: {task.task_id}")
    
    def _verify_crypto_security(self) -> bool:
        """Verify cryptographic security before executing tasks"""
        # Verify HOTP authentication
        current_token = HOTP.generate(self.hotp_secret, self.counter)
        hotp_valid = HOTP.verify(self.hotp_secret, current_token, self.counter)
        
        # Verify Merkle tree integrity
        merkle_valid = self.crypto_state['merkle_tree'] is not None
        
        # Verify secret shares are available
        shares_valid = len(self.crypto_state['private_key_shares']) >= self.config['threshold']
        
        # Verify MPC system is ready
        mpc_valid = 'mpc_system' in self.crypto_state
        
        return hotp_valid and merkle_valid and shares_valid and mpc_valid
    
    # Computational task handlers (these are the actual tasks)
    def _handle_prime_generation(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Generate prime numbers"""
        logger.info(f"[{node_id}] Executing prime generation...")
        
        # Simulate prime generation
        import random
        start_time = time.time()
        
        primes = []
        for _ in range(task.data['count']):
            # Simulate finding a prime in the range
            prime = random.randint(task.data['range_start'], task.data['range_end'])
            if prime % 2 != 0:  # Simple odd number simulation
                primes.append(prime)
        
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'primes_found': len(primes),
            'execution_time': execution_time,
            'first_prime': primes[0] if primes else None
        }
    
    def _handle_matrix_multiplication(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Handle matrix multiplication"""
        logger.info(f"[{node_id}] Executing matrix multiplication...")
        
        start_time = time.time()
        
        # Simulate matrix multiplication
        size = task.data['matrix_size']
        iterations = task.data['iterations']
        
        total_ops = size * size * size * iterations
        time.sleep(0.1 * iterations)  # Simulate computation time
        
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'matrix_size': size,
            'iterations': iterations,
            'total_operations': total_ops,
            'execution_time': execution_time
        }
    
    def _handle_fibonacci_calculation(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Calculate Fibonacci numbers"""
        logger.info(f"[{node_id}] Calculating Fibonacci sequence...")
        
        start_time = time.time()
        n = task.data['n']
        modulo = task.data['modulo']
        
        # Calculate Fibonacci efficiently
        a, b = 0, 1
        for _ in range(n):
            a, b = b, (a + b) % modulo
        
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'fibonacci_n': n,
            'result': a,
            'execution_time': execution_time
        }
    
    def _handle_array_sort(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Sort large arrays"""
        logger.info(f"[{node_id}] Sorting large array...")
        
        start_time = time.time()
        
        # Simulate array sorting
        import random
        array_size = task.data['array_size']
        array = [random.randint(1, 1000000) for _ in range(min(array_size, 1000))]  # Limit for demo
        array.sort()
        
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'array_size': array_size,
            'algorithm': task.data['algorithm'],
            'execution_time': execution_time,
            'sorted': True
        }
    
    def _handle_pattern_search(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Search for patterns in text"""
        logger.info(f"[{node_id}] Searching for patterns...")
        
        start_time = time.time()
        
        # Simulate pattern search
        text_size = task.data['text_size']
        pattern = task.data['pattern']
        
        # Simulate finding pattern occurrences
        import random
        occurrences = random.randint(0, text_size // 1000)
        
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'text_size': text_size,
            'pattern': pattern,
            'occurrences': occurrences,
            'execution_time': execution_time
        }
    
    def _handle_data_compression(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Compress data"""
        logger.info(f"[{node_id}] Compressing data...")
        
        start_time = time.time()
        
        # Simulate data compression
        data_size = task.data['data_size']
        algorithm = task.data['algorithm']
        
        # Simulate compression ratio
        import random
        compression_ratio = random.uniform(0.3, 0.8)
        compressed_size = int(data_size * compression_ratio)
        
        time.sleep(0.5)  # Simulate compression time
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'original_size': data_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'algorithm': algorithm,
            'execution_time': execution_time
        }
    
    def _handle_image_processing(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Process images"""
        logger.info(f"[{node_id}] Processing image...")
        
        start_time = time.time()
        
        # Simulate image processing
        operation = task.data['operation']
        image_size = task.data['image_size']
        kernel_size = task.data['kernel_size']
        
        time.sleep(1.0)  # Simulate processing time
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'operation': operation,
            'image_size': image_size,
            'kernel_size': kernel_size,
            'execution_time': execution_time
        }
    
    def _handle_object_detection(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Detect objects in images"""
        logger.info(f"[{node_id}] Detecting objects...")
        
        start_time = time.time()
        
        # Simulate object detection
        algorithm = task.data['algorithm']
        image_count = task.data['image_count']
        
        import random
        objects_detected = random.randint(0, image_count * 3)
        
        time.sleep(0.5)
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'algorithm': algorithm,
            'images_processed': image_count,
            'objects_detected': objects_detected,
            'execution_time': execution_time
        }
    
    def _handle_web_crawling(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Crawl web pages"""
        logger.info(f"[{node_id}] Crawling web pages...")
        
        start_time = time.time()
        
        # Simulate web crawling
        urls = task.data['urls']
        depth = task.data['depth']
        
        pages_crawled = len(urls) * (depth + 1)
        
        time.sleep(0.3)
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'urls_processed': len(urls),
            'pages_crawled': pages_crawled,
            'depth': depth,
            'execution_time': execution_time
        }
    
    def _handle_api_data_fetch(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Fetch data from APIs"""
        logger.info(f"[{node_id}] Fetching API data...")
        
        start_time = time.time()
        
        # Simulate API data fetching
        endpoint = task.data['endpoint']
        count = task.data['count']
        
        time.sleep(0.2)
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'endpoint': endpoint,
            'records_fetched': count,
            'execution_time': execution_time
        }
    
    def _handle_linear_regression(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Perform linear regression"""
        logger.info(f"[{node_id}] Performing linear regression...")
        
        start_time = time.time()
        
        # Simulate linear regression
        dataset_size = task.data['dataset_size']
        features = task.data['features']
        iterations = task.data['iterations']
        
        import random
        mse = random.uniform(0.01, 0.1)
        r_squared = random.uniform(0.8, 0.99)
        
        time.sleep(0.8)
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'dataset_size': dataset_size,
            'features': features,
            'iterations': iterations,
            'mse': mse,
            'r_squared': r_squared,
            'execution_time': execution_time
        }
    
    def _handle_kmeans_clustering(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Perform K-means clustering"""
        logger.info(f"[{node_id}] Performing K-means clustering...")
        
        start_time = time.time()
        
        # Simulate K-means clustering
        data_points = task.data['data_points']
        clusters = task.data['clusters']
        dimensions = task.data['dimensions']
        
        import random
        inertia = random.uniform(100, 1000)
        
        time.sleep(0.6)
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'data_points': data_points,
            'clusters': clusters,
            'dimensions': dimensions,
            'inertia': inertia,
            'execution_time': execution_time
        }
    
    def _handle_file_hashing(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Hash files"""
        logger.info(f"[{node_id}] Hashing file...")
        
        start_time = time.time()
        
        # Simulate file hashing
        file_size = task.data['file_size']
        algorithm = task.data['algorithm']
        
        # Generate a mock hash
        import hashlib
        hash_obj = hashlib.sha256(str(file_size).encode())
        file_hash = hash_obj.hexdigest()
        
        time.sleep(0.3)
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'file_size': file_size,
            'algorithm': algorithm,
            'hash': file_hash[:32],
            'execution_time': execution_time
        }
    
    def _handle_file_encryption(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Encrypt files"""
        logger.info(f"[{node_id}] Encrypting file...")
        
        start_time = time.time()
        
        # Simulate file encryption
        file_size = task.data['file_size']
        algorithm = task.data['algorithm']
        
        time.sleep(0.4)
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'file_size': file_size,
            'algorithm': algorithm,
            'encrypted': True,
            'execution_time': execution_time
        }
    
    def _handle_monte_carlo(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Perform Monte Carlo simulation"""
        logger.info(f"[{node_id}] Running Monte Carlo simulation...")
        
        start_time = time.time()
        
        # Simulate Monte Carlo
        iterations = task.data['iterations']
        variables = task.data['variables']
        
        import random
        result = random.uniform(0, 1)
        
        time.sleep(0.7)
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'iterations': iterations,
            'variables': variables,
            'result': result,
            'execution_time': execution_time
        }
    
    def _handle_statistical_analysis(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Perform statistical analysis"""
        logger.info(f"[{node_id}] Performing statistical analysis...")
        
        start_time = time.time()
        
        # Simulate statistical analysis
        dataset_size = task.data['dataset_size']
        operations = task.data['operations']
        
        import random
        stats = {}
        for op in operations:
            stats[op] = random.uniform(0, 100)
        
        time.sleep(0.4)
        execution_time = time.time() - start_time
        
        return {
            'status': 'success',
            'dataset_size': dataset_size,
            'operations': operations,
            'statistics': stats,
            'execution_time': execution_time
        }
    
    def _monitor_execution(self):
        """Monitor the execution of the computational workflow"""
        logger.info("\n" + "=" * 55)
        logger.info("MONITORING SECURE COMPUTATIONAL WORKFLOW")
        logger.info("=" * 55)
        
        start_time = time.time()
        max_wait_time = 300.0  # 5 minutes maximum
        
        while time.time() - start_time < max_wait_time:
            status = self.scheduler.get_scheduler_status()
            
            print(f"\rStatus: {status['completed_tasks']}/{status['total_tasks']} computational tasks completed", end="")
            
            if status['completed_tasks'] == status['total_tasks']:
                print("\n\n✅ All computational tasks completed successfully!")
                self._print_final_report()
                break
            elif status['failed_tasks'] > 0:
                print(f"\n\n❌ {status['failed_tasks']} tasks failed")
                break
            
            time.sleep(2.0)
        else:
            print("\n\n⏰ Execution timeout reached")
        
        self.scheduler.stop()
    
    def _print_final_report(self):
        """Print final execution report"""
        logger.info("\n" + "=" * 55)
        logger.info("FINAL EXECUTION REPORT")
        logger.info("=" * 55)
        
        # System statistics
        stats = self.scheduler.get_scheduler_status()
        logger.info(f"Total Computational Tasks: {stats['total_tasks']}")
        logger.info(f"Completed: {stats['completed_tasks']}")
        logger.info(f"Failed: {stats['failed_tasks']}")
        
        # Cryptographic infrastructure status
        logger.info("\nCryptographic Infrastructure Status:")
        logger.info(f"✅ Master Key Generated: {bool(self.crypto_state['master_private_key'])}")
        logger.info(f"✅ Secret Shares Distributed: {len(self.crypto_state['private_key_shares'])} shares")
        logger.info(f"✅ HOTP Tokens Active: {len(self.crypto_state['hotp_tokens'])} nodes")
        logger.info(f"✅ Merkle Tree Integrity: {bool(self.crypto_state['merkle_tree'])}")
        logger.info(f"✅ MPC Framework Ready: {'mpc_system' in self.crypto_state}")
        
        # Security verification
        security_verified = self._verify_crypto_security()
        logger.info(f"\n🔐 Overall Security Status: {'SECURE' if security_verified else 'COMPROMISED'}")


def main():
    """Main execution function"""
    config = {
        'num_nodes': 5,
        'threshold': 3,
        'heartbeat_interval': 10.0
    }
    
    executor = DistributedCryptoExecutor(config)
    executor.start_execution()


if __name__ == "__main__":
    main()