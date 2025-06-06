"""
Distributed Node Implementation
Represents a single compute node in the distributed system
"""
import time
import threading
import requests
import hashlib
import logging
from typing import Dict, List, Optional, Any
import os
import secrets

from secret_sharing import ShamirSecretSharing
from hotp import HOTP
from merkle_tree import MerkleTree
from mpc import SecureMultiPartyComputation, MPCNode
from key_generation import MFKDFDeterministicKeyGenerator
from task_scheduler import CryptoTask, TaskStatus, NodeCapabilities

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
        
        # Node capabilities and configuration
        self.capability = NodeCapabilities(
            node_id=node_id,
            processing_speed=1.0 + (hash(node_id) % 100) / 100,  # Simulate different speeds
            max_capacity=5,
            specialized_tasks=self._determine_specializations(),
            available_factors=['biometric', 'password', 'hardware_token', 'location', 'time_window']
        )
        
        # Cryptographic components
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
        """Determine task specializations based on node characteristics"""
        specializations = []
        
        # Simulate different node specializations
        node_hash = hash(self.node_id)
        
        if node_hash % 3 == 0:
            specializations.extend(['key_generation', 'mfkdf_operations'])
        if node_hash % 3 == 1:
            specializations.extend(['secret_sharing', 'reconstruction'])
        if node_hash % 3 == 2:
            specializations.extend(['mpc_computation', 'verification'])
        
        # All nodes can handle basic operations
        specializations.extend(['hotp_generation', 'merkle_verification'])
        
        return specializations
    
    def initialize_cryptographic_components(self, master_seed: bytes):
        """Initialize cryptographic components for this node"""
        logger.info(f"Initializing cryptographic components for {self.node_id}")
        
        self.master_seed = master_seed
        
        # Initialize MFKDF key generator
        self.mfkdf_generator = MFKDFDeterministicKeyGenerator(self.node_id)
        node_seed = hashlib.sha256(master_seed + self.node_id.encode()).digest()
        self.mfkdf_generator.setup_authentication_factors(node_seed)
        
        # Initialize HOTP
        self.hotp_secret = hashlib.sha256(master_seed + b"HOTP" + self.node_id.encode()).digest()[:20]
        self.hotp_counter = hash(self.node_id) % 1000  # Different starting counters
        
        logger.info(f"Cryptographic initialization complete for {self.node_id}")
    
    def start(self):
        """Start the node's operation threads"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info(f"Starting node {self.node_id}")
        
        # Start background threads
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        threading.Thread(target=self._task_execution_loop, daemon=True).start()
        
        logger.info(f"Node {self.node_id} started successfully")
    
    def stop(self):
        """Stop the node's operation"""
        self.is_running = False
        logger.info(f"Node {self.node_id} stopped")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to scheduler"""
        while self.is_running:
            try:
                self._send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat failed for {self.node_id}: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _send_heartbeat(self):
        """Send heartbeat to scheduler"""
        try:
            heartbeat_data = {
                'node_id': self.node_id,
                'timestamp': time.time(),
                'current_load': self.capability.current_load,
                'max_capacity': self.capability.max_capacity,
                'status': 'active',
                'performance_metrics': self.performance_metrics
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
                # Request task from scheduler
                task = self._request_task()
                
                if task:
                    self._execute_task(task)
                else:
                    time.sleep(self.task_poll_interval)
                    
            except Exception as e:
                logger.error(f"Task execution error in {self.node_id}: {e}")
                time.sleep(self.task_poll_interval)

    def _request_task(self) -> Optional[CryptoTask]:
        """Request a task from the scheduler"""
        try:
            response = requests.get(
                f"http://{self.scheduler_host}:{self.scheduler_port}/get_task/{self.node_id}",
                timeout=5
            )
            
            if response.status_code == 200:
                task_data = response.json()
                if task_data:
                    return CryptoTask(**task_data)

        except requests.exceptions.RequestException as e:
            logger.debug(f"Task request failed for {self.node_id}: {e}")
        
        return None

    def _execute_task(self, task: CryptoTask):
        """Execute a cryptographic task"""
        logger.info(f"Executing task {task.task_id} on {self.node_id}")
        
        start_time = time.time()
        
        try:
            # Update task status to in progress
            self._update_task_status(task.task_id, TaskStatus.IN_PROGRESS)
            self.capability.current_load += 1
            
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
            self.capability.current_load = max(0, self.capability.current_load - 1)

    def _execute_task_by_type(self, task: CryptoTask) -> Dict[str, Any]:
        """Execute task based on its type"""
        task_type = task.task_type
        
        if task_type == 'key_generation':
            return self._execute_key_generation_task(task)
        elif task_type == 'secret_sharing':
            return self._execute_secret_sharing_task(task)
        elif task_type == 'hotp_generation':
            return self._execute_hotp_generation_task(task)
        elif task_type == 'merkle_verification':
            return self._execute_merkle_verification_task(task)
        elif task_type == 'mpc_computation':
            return self._execute_mpc_computation_task(task)
        elif task_type == 'reconstruction':
            return self._execute_reconstruction_task(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _execute_key_generation_task(self, task: CryptoTask) -> Dict[str, Any]:
        """Execute MFKDF key generation task"""
        logger.info(f"Generating RSA key with MFKDF for task {task.task_id}")
        
        if not self.mfkdf_generator:
            raise ValueError("MFKDF generator not initialized")
        
        # Use required factors or default factors
        active_factors = task.required_factors or ['biometric', 'password', 'hardware_token']
        
        # Generate RSA key using MFKDF
        rsa_key = self.mfkdf_generator.generate_rsa_key_with_mfkdf(
            active_factors=active_factors,
            key_size=task.data.get('key_size', 2048)
        )
        
        # Store the private key
        self.private_key = rsa_key
        
        # Generate backup recovery data
        recovery_data = self.mfkdf_generator.generate_backup_recovery_data()
        
        return {
            'key_size': rsa_key.size_in_bits(),
            'public_key_hash': hashlib.sha256(rsa_key.publickey().export_key()).hexdigest(),
            'factors_used': active_factors,
            'recovery_shares_count': len(recovery_data.get('shares', []))
        }
    
    def _execute_secret_sharing_task(self, task: CryptoTask) -> Dict[str, Any]:
        """Execute secret sharing task"""
        logger.info(f"Executing secret sharing for task {task.task_id}")
        
        if not self.private_key:
            raise ValueError("No private key available for sharing")
        
        # Convert private key to shareable format
        private_key_der = self.private_key.export_key('DER')
        secret_int = int.from_bytes(private_key_der[:32], 'big')  # Use first 32 bytes
        
        # Create shares
        threshold = task.data.get('threshold', 3)
        num_shares = task.data.get('num_shares', 5)
        
        shares = ShamirSecretSharing.create_shares(secret_int, threshold, num_shares)
        
        # Store our share
        node_index = int(self.node_id.split('_')[1]) if '_' in self.node_id else 0
        if node_index < len(shares):
            self.secret_shares.append(shares[node_index])
        
        return {
            'shares_created': len(shares),
            'threshold': threshold,
            'share_index': node_index,
            'share_hash': hashlib.sha256(str(shares[node_index]).encode()).hexdigest() if node_index < len(shares) else None
        }
    
    def _execute_hotp_generation_task(self, task: CryptoTask) -> Dict[str, Any]:
        """Execute HOTP token generation task"""
        logger.info(f"Generating HOTP tokens for task {task.task_id}")
        
        if not self.hotp_secret:
            raise ValueError("HOTP secret not initialized")
        
        num_tokens = task.data.get('num_tokens', 5)
        tokens = []
        
        for i in range(num_tokens):
            token = HOTP.generate(self.hotp_secret, self.hotp_counter + i)
            tokens.append(token)
        
        # Verify the first token
        verification_result = HOTP.verify(self.hotp_secret, tokens[0], self.hotp_counter)
        
        self.hotp_counter += num_tokens
        
        return {
            'tokens_generated': len(tokens),
            'first_token': tokens[0] if tokens else None,
            'verification_passed': verification_result,
            'counter_used': self.hotp_counter - num_tokens
        }
    
    def _execute_merkle_verification_task(self, task: CryptoTask) -> Dict[str, Any]:
        """Execute Merkle tree verification task"""
        logger.info(f"Executing Merkle verification for task {task.task_id}")
        
        # Create sample data blocks
        data_blocks = []
        for i in range(task.data.get('num_blocks', 4)):
            block_data = f"{self.node_id}_block_{i}_{task.task_id}".encode()
            data_blocks.append(hashlib.sha256(block_data).digest())
        
        # Create Merkle tree
        merkle_tree = MerkleTree(data_blocks)
        
        # Generate and verify proofs
        proofs_verified = 0
        for i in range(len(data_blocks)):
            proof = merkle_tree.get_proof(i)
            is_valid = MerkleTree.verify_proof(data_blocks[i], proof, merkle_tree.root.hash_value)
            if is_valid:
                proofs_verified += 1
            
            # Store proof for later use
            self.merkle_proofs[f"{task.task_id}_block_{i}"] = proof
        
        return {
            'blocks_processed': len(data_blocks),
            'proofs_verified': proofs_verified,
            'root_hash': merkle_tree.root.hash_value.hex(),
            'verification_success_rate': proofs_verified / len(data_blocks) if data_blocks else 0
        }
    
    def _execute_mpc_computation_task(self, task: CryptoTask) -> Dict[str, Any]:
        """Execute MPC computation task"""
        logger.info(f"Executing MPC computation for task {task.task_id}")
        
        # Use secret share or generate random secret for demonstration
        if self.secret_shares:
            secret_value = self.secret_shares[0][1]  # Use y-coordinate of first share
        else:
            secret_value = hash(self.node_id) % 1000000  # Deterministic secret
        
        # Create MPC node
        mpc_node = MPCNode(self.node_id, secret_value)
        
        # Simulate MPC with other nodes (simplified)
        # In real implementation, this would involve network communication
        other_nodes = []
        for i in range(task.data.get('num_participants', 3)):
            if f"node_{i}" != self.node_id:
                other_secret = hash(f"node_{i}") % 1000000
                other_nodes.append(MPCNode(f"node_{i}", other_secret))
        
        all_nodes = [mpc_node] + other_nodes
        mpc_system = SecureMultiPartyComputation(all_nodes, task.data.get('threshold', 2))
        
        # Execute computation
        results = mpc_system.demonstrate_threshold_computation()
        
        return {
            'computation_type': 'secure_sum',
            'participants': len(all_nodes),
            'my_contribution': secret_value,
            'collective_result': results.get('collective_result', 0),
            'verification_hash': list(results.get('threshold_verification', {}).values())[0] if results.get('threshold_verification') else None
        }
    
    def _execute_reconstruction_task(self, task: CryptoTask) -> Dict[str, Any]:
        """Execute secret reconstruction task"""
        logger.info(f"Executing secret reconstruction for task {task.task_id}")
        
        if not self.secret_shares:
            raise ValueError("No secret shares available for reconstruction")
        
        # Use available shares for reconstruction
        threshold = task.data.get('threshold', 3)
        shares_to_use = self.secret_shares[:threshold]
        
        if len(shares_to_use) < threshold:
            raise ValueError(f"Insufficient shares: need {threshold}, have {len(shares_to_use)}")
        
        # Reconstruct secret
        reconstructed_secret = ShamirSecretSharing.reconstruct_secret(shares_to_use)
        
        # Verify reconstruction by comparing with expected hash if provided
        expected_hash = task.data.get('expected_hash')
        reconstructed_hash = hashlib.sha256(str(reconstructed_secret).encode()).hexdigest()
        
        reconstruction_valid = (expected_hash is None or reconstructed_hash == expected_hash)
        
        return {
            'shares_used': len(shares_to_use),
            'reconstruction_successful': reconstruction_valid,
            'reconstructed_hash': reconstructed_hash[:32],  # First 32 chars for security
            'verification_passed': reconstruction_valid
        }
    
    def _update_task_status(self, task_id: str, status: TaskStatus, execution_time: Optional[float] = None):
        """Update task status with scheduler"""
        try:
            status_data = {
                'task_id': task_id,
                'node_id': self.node_id,
                'status': status.value,
                'timestamp': time.time(),
                'execution_time': execution_time
            }
            
            response = requests.post(
                f"http://{self.scheduler_host}:{self.scheduler_port}/update_task_status",
                json=status_data,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug(f"Task {task_id} status updated to {status.value}")
            else:
                logger.warning(f"Failed to update task status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update task status: {e}")
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get comprehensive node information"""
        return {
            'node_id': self.node_id,
            'capability': {
                'processing_speed': self.capability.processing_speed,
                'max_capacity': self.capability.max_capacity,
                'current_load': self.capability.current_load,
                'specialized_tasks': self.capability.specialized_tasks,
                'available_factors': self.capability.available_factors
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
                'uptime': time.time() - (self.capability.last_heartbeat if hasattr(self.capability, 'start_time') else time.time())
            }
        }


def main():
    """Main function to run a distributed node"""
    import sys
    
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
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down node...")
        node.stop()


if __name__ == "__main__":
    main()