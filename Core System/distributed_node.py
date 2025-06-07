"""
Distributed Node Implementation
Represents a single compute node in the distributed system
"""
import time
import threading
import logging
import os
import sys
import secrets
import requests
import hashlib
import random
import numpy as np
from typing import Dict, List, Optional, Any

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


class DistributedNode:
    """Individual node in the distributed cryptographic system"""
    
    def __init__(self, node_id: str, scheduler_host: str = "localhost", scheduler_port: int = 8000):
        self.node_id = node_id
        self.scheduler_host = scheduler_host
        self.scheduler_port = scheduler_port
        self.is_running = False

        # Store specialized tasks separately
        self.specialized_tasks = self._determine_specializations()
        
        # Node capabilities and configuration
        self.capability = NodeCapabilities(
            node_id=node_id,
            computation_power=1.0 + (hash(node_id) % 100) / 100,
            max_concurrent_tasks=5,
            available_factors=['biometric', 'password', 'hardware_token', 'location', 'time_window'],
            task_completion_rate=0.95 - (hash(node_id) % 10) / 100,
            communication_latency={}
        )
        
        # Cryptographic infrastructure (runs on every node)
        self.master_seed = None
        self.mfkdf_generator = None
        self.hotp_secret = None
        self.hotp_counter = 0
        self.private_key = None
        self.secret_shares = []
        self.merkle_proofs = {}
        
        # Task execution state
        self.current_tasks = {}
        self.completed_tasks = []
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Communication
        self.heartbeat_interval = 10.0
        self.task_poll_interval = 2.0
        
        logger.info(f"Initialized node {node_id} with capabilities: {self.capability}")

    def _determine_specializations(self) -> List[str]:
        """Determine what task types this node specializes in"""
        # Base specializations on node_id hash for deterministic assignment
        hash_val = hash(self.node_id) % 100
        
        specializations = []
        
        # Mathematical tasks
        if hash_val < 20:
            specializations.extend(['prime_generation', 'fibonacci_calculation', 'matrix_multiplication'])
        
        # Data processing
        if 20 <= hash_val < 40:
            specializations.extend(['large_array_sort', 'pattern_search', 'data_compression'])
        
        # Image processing
        if 40 <= hash_val < 60:
            specializations.extend(['image_processing', 'object_detection'])
        
        # Network tasks
        if 60 <= hash_val < 80:
            specializations.extend(['web_crawling', 'api_data_fetch'])
        
        # Machine learning and scientific
        if 80 <= hash_val < 100:
            specializations.extend(['linear_regression', 'kmeans_clustering', 'monte_carlo_simulation'])
        
        # All nodes can handle basic tasks
        specializations.extend(['file_hashing', 'file_encryption', 'statistical_analysis'])
        
        return specializations
    
    def initialize_cryptographic_components(self, master_seed: bytes):
        """Initialize all cryptographic infrastructure for this node"""
        logger.info(f"Initializing cryptographic infrastructure for {self.node_id}")
        
        self.master_seed = master_seed
        
        # 1. Initialize MFKDF key generation
        self.mfkdf_generator = MFKDFDeterministicKeyGenerator(self.node_id)
        self.mfkdf_generator.setup_authentication_factors(master_seed)
        
        # 2. Generate node's private key using MFKDF
        active_factors = ['biometric', 'password', 'hardware_token']
        self.private_key = self.mfkdf_generator.generate_rsa_key_with_mfkdf(active_factors)
        
        # 3. Initialize HOTP for authentication
        self.hotp_secret = hashlib.sha256(master_seed + self.node_id.encode()).digest()
        self.hotp_counter = 0
        
        # 4. Create secret shares for this node's key
        key_int = int.from_bytes(self.private_key.export_key('DER'), 'big')
        self.secret_shares = ShamirSecretSharing.create_shares(key_int, 3, 5)
        
        # 5. Create Merkle proofs for integrity
        node_data = [
            f"node_id:{self.node_id}".encode(),
            f"key_size:{self.private_key.size_in_bits()}".encode(),
            f"hotp_secret:{self.hotp_secret.hex()}".encode()
        ]
        merkle_tree = MerkleTree(node_data)
        for i, data in enumerate(node_data):
            proof = merkle_tree.get_proof(i)
            self.merkle_proofs[f"data_{i}"] = {
                'proof': proof,
                'root_hash': merkle_tree.get_root_hash()
            }
        
        # 6. Verify cryptographic setup
        hotp_token = HOTP.generate(self.hotp_secret, self.hotp_counter)
        hotp_valid = HOTP.verify(self.hotp_secret, hotp_token, self.hotp_counter)
        
        logger.info(f"Cryptographic setup complete for {self.node_id}:")
        logger.info(f"  - RSA Key: {self.private_key.size_in_bits()} bits")
        logger.info(f"  - Secret Shares: {len(self.secret_shares)}")
        logger.info(f"  - HOTP Authentication: {'PASS' if hotp_valid else 'FAIL'}")
        logger.info(f"  - Merkle Proofs: {len(self.merkle_proofs)}")
    
    def start(self):
        """Start the distributed node"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info(f"Starting distributed node {self.node_id}")
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        
        # Start task execution thread
        task_thread = threading.Thread(target=self._task_execution_loop, daemon=True)
        task_thread.start()
        
        logger.info(f"Node {self.node_id} started successfully")
    
    def stop(self):
        """Stop the distributed node"""
        self.is_running = False
        logger.info(f"Stopping node {self.node_id}")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to scheduler"""
        while self.is_running:
            try:
                self._send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error for {self.node_id}: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _send_heartbeat(self):
        """Send heartbeat to scheduler"""
        try:
            # Generate current HOTP token for authentication
            hotp_token = HOTP.generate(self.hotp_secret, self.hotp_counter)
            self.hotp_counter += 1
            
            heartbeat_data = {
                'node_id': self.node_id,
                'timestamp': time.time(),
                'current_load': len(self.current_tasks),
                'max_capacity': self.capability.max_concurrent_tasks,
                'status': 'active',
                'performance_metrics': self.performance_metrics,
                'hotp_token': hotp_token,  # Include authentication
                'specialized_tasks': self.specialized_tasks
            }
            
            response = requests.post(
                f"http://{self.scheduler_host}:{self.scheduler_port}/heartbeat",
                json=heartbeat_data,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug(f"Heartbeat sent successfully from {self.node_id}")
            else:
                logger.warning(f"Heartbeat failed for {self.node_id}: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Heartbeat request failed for {self.node_id}: {e}")

    def _task_execution_loop(self):
        """Main task execution loop"""
        while self.is_running:
            try:
                if len(self.current_tasks) < self.capability.max_concurrent_tasks:
                    task = self._request_task()
                    if task:
                        # Execute task in separate thread
                        task_thread = threading.Thread(
                            target=self._execute_task,
                            args=(task,),
                            daemon=True
                        )
                        task_thread.start()
                
                time.sleep(self.task_poll_interval)
            except Exception as e:
                logger.error(f"Task execution loop error for {self.node_id}: {e}")
                time.sleep(self.task_poll_interval)

    def _request_task(self) -> Optional[CryptoTask]:
        """Request a new task from the scheduler"""
        try:
            response = requests.get(
                f"http://{self.scheduler_host}:{self.scheduler_port}/get_task/{self.node_id}",
                timeout=5
            )
            
            if response.status_code == 200:
                task_data = response.json()
                if task_data:  # Not empty
                    # Convert back to CryptoTask object
                    task = CryptoTask(
                        task_id=task_data['task_id'],
                        task_type=task_data['task_type'],
                        data=task_data['data'],
                        priority=TaskPriority(task_data['priority']),
                        computation_cost=task_data['computation_cost'],
                        communication_cost=task_data['communication_cost'],
                        dependencies=task_data['dependencies'],
                        timeout=task_data['timeout'],
                        max_retries=task_data['max_retries'],
                        required_factors=task_data['required_factors']
                    )
                    return task
        except Exception as e:
            logger.error(f"Task request failed for {self.node_id}: {e}")
        
        return None

    def _execute_task(self, task: CryptoTask):
        """Execute a computational task"""
        logger.info(f"Executing task {task.task_id} on {self.node_id}")
        
        start_time = time.time()
        
        try:
            # Update task status to in progress
            self._update_task_status(task.task_id, TaskStatus.IN_PROGRESS)
            self.current_tasks[task.task_id] = task
            
            # Execute the specific task type
            result = self._execute_task_by_type(task)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.performance_metrics['tasks_completed'] += 1
            self.performance_metrics['total_execution_time'] += execution_time
            self.performance_metrics['average_execution_time'] = (
                self.performance_metrics['total_execution_time'] / 
                self.performance_metrics['tasks_completed']
            )
            
            # Update task status to completed
            self._update_task_status(task.task_id, TaskStatus.COMPLETED, execution_time)
            
            logger.info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed on {self.node_id}: {e}")
            self.performance_metrics['tasks_failed'] += 1
            self._update_task_status(task.task_id, TaskStatus.FAILED)
        
        finally:
            # Remove task from current tasks
            if task.task_id in self.current_tasks:
                del self.current_tasks[task.task_id]

    def _execute_task_by_type(self, task: CryptoTask) -> Dict[str, Any]:
        """Execute task based on its type"""
        task_type = task.task_type
        
        # Mathematical computation tasks
        if task_type == "prime_generation":
            return self._execute_prime_generation(task)
        elif task_type == "matrix_multiplication":
            return self._execute_matrix_multiplication(task)
        elif task_type == "fibonacci_calculation":
            return self._execute_fibonacci_calculation(task)
        
        # Data processing tasks
        elif task_type == "large_array_sort":
            return self._execute_large_array_sort(task)
        elif task_type == "pattern_search":
            return self._execute_pattern_search(task)
        elif task_type == "data_compression":
            return self._execute_data_compression(task)
        
        # Image processing tasks
        elif task_type == "image_processing":
            return self._execute_image_processing(task)
        elif task_type == "object_detection":
            return self._execute_object_detection(task)
        
        # Network tasks
        elif task_type == "web_crawling":
            return self._execute_web_crawling(task)
        elif task_type == "api_data_fetch":
            return self._execute_api_data_fetch(task)
        
        # Machine learning tasks
        elif task_type == "linear_regression":
            return self._execute_linear_regression(task)
        elif task_type == "kmeans_clustering":
            return self._execute_kmeans_clustering(task)
        
        # File processing tasks
        elif task_type == "file_hashing":
            return self._execute_file_hashing(task)
        elif task_type == "file_encryption":
            return self._execute_file_encryption(task)
        
        # Scientific computation tasks
        elif task_type == "monte_carlo_simulation":
            return self._execute_monte_carlo_simulation(task)
        elif task_type == "statistical_analysis":
            return self._execute_statistical_analysis(task)
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _execute_prime_generation(self, task: CryptoTask) -> Dict[str, Any]:
        """Generate prime numbers in a given range"""
        start = task.data['range_start']
        end = task.data['range_end']
        count = task.data['count']
        
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        primes = []
        num = start
        while len(primes) < count and num <= end:
            if is_prime(num):
                primes.append(num)
            num += 1
        
        return {
            'status': 'success',
            'primes_found': len(primes),
            'largest_prime': max(primes) if primes else None,
            'range_searched': f"{start}-{min(num, end)}"
        }

    def _execute_matrix_multiplication(self, task: CryptoTask) -> Dict[str, Any]:
        """Perform matrix multiplication"""
        size = task.data['matrix_size']
        iterations = task.data['iterations']
        
        results = []
        for i in range(iterations):
            # Generate random matrices
            matrix_a = np.random.randint(0, 100, (size, size))
            matrix_b = np.random.randint(0, 100, (size, size))
            
            # Multiply matrices
            result = np.dot(matrix_a, matrix_b)
            results.append(result.sum())  # Store sum as result metric
        
        return {
            'status': 'success',
            'iterations_completed': len(results),
            'matrix_size': size,
            'average_result_sum': sum(results) / len(results),
            'total_operations': size * size * size * iterations
        }

    def _execute_fibonacci_calculation(self, task: CryptoTask) -> Dict[str, Any]:
        """Calculate Fibonacci numbers"""
        n = task.data['n']
        modulo = task.data.get('modulo', None)
        
        if n <= 1:
            result = n
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
                if modulo:
                    a, b = a % modulo, b % modulo
        
        final_result = b if n > 0 else 0
        
        return {
            'status': 'success',
            'fibonacci_n': n,
            'result': final_result,
            'modulo_used': modulo
        }

    def _execute_large_array_sort(self, task: CryptoTask) -> Dict[str, Any]:
        """Sort a large array"""
        array_size = task.data['array_size']
        algorithm = task.data.get('algorithm', 'quicksort')
        
        # Generate random array
        arr = np.random.randint(0, array_size, array_size)
        
        # Sort using specified algorithm
        if algorithm == 'quicksort':
            sorted_arr = np.sort(arr, kind='quicksort')
        elif algorithm == 'mergesort':
            sorted_arr = np.sort(arr, kind='mergesort')
        else:
            sorted_arr = np.sort(arr)
        
        return {
            'status': 'success',
            'array_size': array_size,
            'algorithm': algorithm,
            'is_sorted': np.all(sorted_arr[:-1] <= sorted_arr[1:]),
            'min_value': int(sorted_arr[0]),
            'max_value': int(sorted_arr[-1])
        }

    def _execute_pattern_search(self, task: CryptoTask) -> Dict[str, Any]:
        """Search for patterns in text"""
        text_size = task.data['text_size']
        pattern = task.data['pattern']
        algorithm = task.data.get('algorithm', 'simple')
        
        # Generate random text
        text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=text_size))
        
        # Insert pattern at random positions
        pattern_positions = []
        for _ in range(random.randint(1, 10)):
            pos = random.randint(0, text_size - len(pattern))
            text = text[:pos] + pattern + text[pos + len(pattern):]
            pattern_positions.append(pos)
        
        # Search for pattern
        found_positions = []
        for i in range(len(text) - len(pattern) + 1):
            if text[i:i + len(pattern)] == pattern:
                found_positions.append(i)
        
        return {
            'status': 'success',
            'text_size': text_size,
            'pattern': pattern,
            'algorithm': algorithm,
            'matches_found': len(found_positions),
            'expected_matches': len(pattern_positions)
        }

    def _execute_data_compression(self, task: CryptoTask) -> Dict[str, Any]:
        """Simulate data compression"""
        data_size = task.data['data_size']
        algorithm = task.data.get('algorithm', 'lz77')
        
        # Generate random data
        data = bytes(random.getrandbits(8) for _ in range(data_size))
        
        # Simulate compression (simplified)
        if algorithm == 'lz77':
            compression_ratio = random.uniform(0.3, 0.7)
        elif algorithm == 'huffman':
            compression_ratio = random.uniform(0.4, 0.8)
        else:
            compression_ratio = random.uniform(0.5, 0.9)
        
        compressed_size = int(data_size * compression_ratio)
        
        return {
            'status': 'success',
            'original_size': data_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'algorithm': algorithm,
            'space_saved': data_size - compressed_size
        }

    def _execute_image_processing(self, task: CryptoTask) -> Dict[str, Any]:
        """Simulate image processing"""
        operation = task.data['operation']
        image_size = task.data['image_size']
        kernel_size = task.data.get('kernel_size', 3)
        
        # Parse image dimensions
        width, height = map(int, image_size.split('x'))
        total_pixels = width * height
        
        # Simulate processing time based on operation
        if operation == 'gaussian_blur':
            operations_per_pixel = kernel_size * kernel_size
        elif operation == 'edge_detection':
            operations_per_pixel = 9  # 3x3 kernel
        else:
            operations_per_pixel = 1
        
        total_operations = total_pixels * operations_per_pixel
        
        # Simulate result
        processed_pixels = total_pixels
        
        return {
            'status': 'success',
            'operation': operation,
            'image_size': image_size,
            'pixels_processed': processed_pixels,
            'total_operations': total_operations,
            'kernel_size': kernel_size
        }

    def _execute_object_detection(self, task: CryptoTask) -> Dict[str, Any]:
        """Simulate object detection"""
        algorithm = task.data['algorithm']
        image_count = task.data['image_count']
        
        # Simulate detection results
        total_objects = 0
        detected_objects = 0
        
        for _ in range(image_count):
            objects_in_image = random.randint(0, 10)
            total_objects += objects_in_image
            
            # Simulate detection accuracy
            detection_rate = random.uniform(0.7, 0.95)
            detected_in_image = int(objects_in_image * detection_rate)
            detected_objects += detected_in_image
        
        accuracy = detected_objects / total_objects if total_objects > 0 else 0
        
        return {
            'status': 'success',
            'algorithm': algorithm,
            'images_processed': image_count,
            'total_objects': total_objects,
            'detected_objects': detected_objects,
            'accuracy': accuracy
        }

    def _execute_web_crawling(self, task: CryptoTask) -> Dict[str, Any]:
        """Simulate web crawling"""
        urls = task.data['urls']
        depth = task.data['depth']
        
        crawled_urls = []
        failed_urls = []
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    crawled_urls.append(url)
                else:
                    failed_urls.append(url)
            except Exception:
                failed_urls.append(url)
        
        # Simulate finding additional URLs at each depth
        total_found = len(crawled_urls)
        for d in range(depth):
            total_found += len(crawled_urls) * random.randint(1, 5)
        
        return {
            'status': 'success',
            'initial_urls': len(urls),
            'successful_crawls': len(crawled_urls),
            'failed_crawls': len(failed_urls),
            'total_urls_found': total_found,
            'depth': depth
        }

    def _execute_api_data_fetch(self, task: CryptoTask) -> Dict[str, Any]:
        """Fetch data from API"""
        endpoint = task.data['endpoint']
        count = task.data['count']
        
        try:
            response = requests.get(endpoint, timeout=30)
            if response.status_code == 200:
                data = response.json()
                fetched_count = min(len(data) if isinstance(data, list) else 1, count)
                
                return {
                    'status': 'success',
                    'endpoint': endpoint,
                    'requested_count': count,
                    'fetched_count': fetched_count,
                    'response_size': len(response.content)
                }
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            return {
                'status': 'failed',
                'endpoint': endpoint,
                'error': str(e),
                'requested_count': count,
                'fetched_count': 0
            }

    def _execute_linear_regression(self, task: CryptoTask) -> Dict[str, Any]:
        """Perform linear regression"""
        dataset_size = task.data['dataset_size']
        features = task.data['features']
        iterations = task.data['iterations']
        
        # Generate synthetic dataset
        X = np.random.randn(dataset_size, features)
        true_weights = np.random.randn(features)
        y = X.dot(true_weights) + np.random.randn(dataset_size) * 0.1
        
        # Simple gradient descent
        weights = np.random.randn(features)
        learning_rate = 0.01
        
        for _ in range(iterations):
            predictions = X.dot(weights)
            error = predictions - y
            gradient = X.T.dot(error) / dataset_size
            weights -= learning_rate * gradient
        
        # Calculate final error
        final_predictions = X.dot(weights)
        mse = np.mean((final_predictions - y) ** 2)
        
        return {
            'status': 'success',
            'dataset_size': dataset_size,
            'features': features,
            'iterations': iterations,
            'final_mse': float(mse),
            'weights_learned': weights.tolist()
        }

    def _execute_kmeans_clustering(self, task: CryptoTask) -> Dict[str, Any]:
        """Perform K-means clustering"""
        data_points = task.data['data_points']
        clusters = task.data['clusters']
        dimensions = task.data['dimensions']
        
        # Generate random data
        data = np.random.randn(data_points, dimensions)
        
        # K-means clustering (simplified)
        centroids = np.random.randn(clusters, dimensions)
        
        for iteration in range(100):  # Max iterations
            # Assign points to clusters
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            assignments = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([data[assignments == k].mean(axis=0) for k in range(clusters)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Calculate within-cluster sum of squares
        wcss = sum(np.sum((data[assignments == k] - centroids[k])**2) for k in range(clusters))
        
        return {
            'status': 'success',
            'data_points': data_points,
            'clusters': clusters,
            'dimensions': dimensions,
            'iterations_to_converge': iteration + 1,
            'wcss': float(wcss)
        }

    def _execute_file_hashing(self, task: CryptoTask) -> Dict[str, Any]:
        """Simulate file hashing"""
        file_size = task.data['file_size']
        algorithm = task.data.get('algorithm', 'sha256')
        
        # Generate random file data
        chunk_size = 1024 * 1024  # 1MB chunks
        hasher = hashlib.sha256() if algorithm == 'sha256' else hashlib.md5()
        
        bytes_processed = 0
        while bytes_processed < file_size:
            chunk_size_current = min(chunk_size, file_size - bytes_processed)
            chunk = bytes(random.getrandbits(8) for _ in range(chunk_size_current))
            hasher.update(chunk)
            bytes_processed += chunk_size_current
        
        file_hash = hasher.hexdigest()
        
        return {
            'status': 'success',
            'file_size': file_size,
            'algorithm': algorithm,
            'hash': file_hash,
            'bytes_processed': bytes_processed
        }

    def _execute_file_encryption(self, task: CryptoTask) -> Dict[str, Any]:
        """Simulate file encryption"""
        file_size = task.data['file_size']
        algorithm = task.data.get('algorithm', 'aes256')
        
        # Simulate encryption time based on file size
        bytes_per_second = 50 * 1024 * 1024  # 50 MB/s
        encryption_time = file_size / bytes_per_second
        
        # Simulate actual encryption delay
        time.sleep(min(encryption_time, 5.0))  # Cap at 5 seconds for demo
        
        # Generate random key and encrypted hash
        key = secrets.token_hex(32 if algorithm == 'aes256' else 16)
        encrypted_hash = hashlib.sha256(key.encode() + str(file_size).encode()).hexdigest()
        
        return {
            'status': 'success',
            'file_size': file_size,
            'algorithm': algorithm,
            'encryption_time': encryption_time,
            'key': key,
            'encrypted_hash': encrypted_hash
        }

    def _execute_monte_carlo_simulation(self, task: CryptoTask) -> Dict[str, Any]:
        """Perform Monte Carlo simulation"""
        iterations = task.data['iterations']
        variables = task.data['variables']
        
        # Simulate estimation of π using Monte Carlo
        inside_circle = 0
        
        for _ in range(iterations):
            # Generate random point in unit square
            point = np.random.uniform(-1, 1, variables)
            
            # Check if point is inside unit circle
            if np.sum(point**2) <= 1:
                inside_circle += 1
        
        pi_estimate = 4 * inside_circle / iterations
        error = abs(pi_estimate - np.pi)
        
        return {
            'status': 'success',
            'iterations': iterations,
            'variables': variables,
            'pi_estimate': pi_estimate,
            'actual_pi': float(np.pi),
            'error': error,
            'points_inside_circle': inside_circle
        }

    def _execute_statistical_analysis(self, task: CryptoTask) -> Dict[str, Any]:
        """Perform statistical analysis"""
        dataset_size = task.data['dataset_size']
        operations = task.data['operations']
        
        # Generate random dataset
        data = np.random.randn(dataset_size)
        
        results = {}
        
        for operation in operations:
            if operation == 'mean':
                results['mean'] = float(np.mean(data))
            elif operation == 'std':
                results['std'] = float(np.std(data))
            elif operation == 'correlation':
                # Generate second dataset for correlation
                data2 = np.random.randn(dataset_size)
                correlation = np.corrcoef(data, data2)[0, 1]
                results['correlation'] = float(correlation)
            elif operation == 'median':
                results['median'] = float(np.median(data))
            elif operation == 'variance':
                results['variance'] = float(np.var(data))
        
        return {
            'status': 'success',
            'dataset_size': dataset_size,
            'operations_performed': operations,
            'results': results
        }

    def _update_task_status(self, task_id: str, status: TaskStatus, execution_time: Optional[float] = None):
        """Update task status with scheduler"""
        try:
            update_data = {
                'task_id': task_id,
                'status': status.value,
                'node_id': self.node_id,
                'timestamp': time.time()
            }
            
            if execution_time:
                update_data['execution_time'] = execution_time
            
            response = requests.post(
                f"http://{self.scheduler_host}:{self.scheduler_port}/update_task_status",
                json=update_data,
                timeout=5
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to update task status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Task status update failed: {e}")
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get comprehensive node information"""
        return {
            'node_id': self.node_id,
            'capability': {
                'computation_power': self.capability.computation_power,
                'max_concurrent_tasks': self.capability.max_concurrent_tasks,
                'current_load': len(self.current_tasks),
                'specialized_tasks': self.specialized_tasks,
                'available_factors': self.capability.available_factors,
                'task_completion_rate': self.capability.task_completion_rate
            },
            'performance_metrics': self.performance_metrics,
            'cryptographic_state': {
                'has_mfkdf_generator': self.mfkdf_generator is not None,
                'has_private_key': self.private_key is not None,
                'secret_shares_count': len(self.secret_shares),
                'hotp_counter': self.hotp_counter,
                'merkle_proofs_count': len(self.merkle_proofs)
            },
            'status': {
                'is_running': self.is_running,
                'current_tasks': len(self.current_tasks),
                'scheduler_connection': f"{self.scheduler_host}:{self.scheduler_port}"
            }
        }


def main():
    """Main function to run a distributed node"""
    
    # Get node configuration from environment or command line
    node_id = os.environ.get('NODE_ID', sys.argv[1] if len(sys.argv) > 1 else 'node_0')
    scheduler_host = os.environ.get('SCHEDULER_HOST', 'localhost')
    scheduler_port = int(os.environ.get('SCHEDULER_PORT', '8000'))
    
    # Create and start node
    node = DistributedNode(node_id, scheduler_host, scheduler_port)
    
    # Initialize cryptographic components
    master_seed = secrets.token_bytes(32)  # In production, this would be shared securely
    node.initialize_cryptographic_components(master_seed)
    
    # Start node
    node.start()
    
    try:
        # Keep node running
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutting down node...")
        node.stop()


if __name__ == "__main__":
    main()