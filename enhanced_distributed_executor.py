"""
Main Distributed Cryptographic Task Executor with HEFT Scheduling
Integrates MFKDF, SSS, HOTP, Merkle Trees, and MPC with intelligent task distribution
"""
import time
import hashlib
import secrets
from typing import Dict, Any

from task_scheduler import HEFTScheduler, CryptoTask, NodeCapabilities, TaskPriority
from mfkdf import DistributedMFKDF
from key_generation import MFKDFDeterministicKeyGenerator
from secret_sharing import ShamirSecretSharing
from hotp import HOTP
from merkle_tree import MerkleTree
from mpc import SecureMultiPartyComputation, MPCNode


class DistributedCryptoExecutor:
    """Main orchestrator for distributed cryptographic operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scheduler = HEFTScheduler(heartbeat_interval=config.get('heartbeat_interval', 10.0))
        self.distributed_mfkdf = DistributedMFKDF(
            node_count=config['num_nodes'],
            threshold=config['threshold']
        )
        
        # Cryptographic state
        self.master_seed = secrets.token_bytes(32)
        self.hotp_secret = hashlib.sha256(self.master_seed + b"HOTP").digest()
        self.counter = 0
        self.global_state = {
            'private_key_shares': {},
            'merkle_tree': None,
            'mpc_results': {},
            'task_results': {}
        }
        
        # Register task handlers
        self._register_task_handlers()
        
        # Initialize nodes
        self._initialize_nodes()
    
    def _register_task_handlers(self):
        """Register handlers for different task types"""
        handlers = {
            'key_generation': self._handle_key_generation,
            'share_distribution': self._handle_share_distribution,
            'hotp_verification': self._handle_hotp_verification,
            'merkle_storage': self._handle_merkle_storage,
            'mpc_computation': self._handle_mpc_computation,
            'key_reconstruction': self._handle_key_reconstruction,
            'factor_verification': self._handle_factor_verification
        }
        
        for task_type, handler in handlers.items():
            self.scheduler.register_task_handler(task_type, handler)
    
    def _initialize_nodes(self):
        """Initialize compute nodes with capabilities"""
        for i in range(self.config['num_nodes']):
            node_id = f'crypto_node_{i}'
            
            # Define node capabilities
            capabilities = NodeCapabilities(
                node_id=node_id,
                computation_power=1.0 + (i * 0.2),  # Varying computation power
                available_factors=[
                    'biometric', 'password', 'hardware_token', 'location', 'time_window'
                ],
                max_concurrent_tasks=3 + i,  # Varying capacity
                task_completion_rate=0.95 - (i * 0.01)  # Slightly varying reliability
            )
            
            self.scheduler.register_node(node_id, capabilities)
    
    def start_execution(self):
        """Start the distributed execution system"""
        print("Starting Distributed Cryptographic Execution System")
        print("=" * 55)
        
        # Start scheduler
        self.scheduler.start()
        
        # Create and submit cryptographic workflow tasks
        self._create_crypto_workflow()
        
        # Monitor execution
        self._monitor_execution()
    
    def _create_crypto_workflow(self):
        """Create the complete cryptographic workflow as tasks"""
        
        # Task 1: Generate master private key
        key_gen_task = CryptoTask(
            task_id="master_key_generation",
            task_type="key_generation",
            data={
                'key_size': 2048,
                'active_factors': ['biometric', 'password', 'hardware_token']
            },
            priority=TaskPriority.CRITICAL,
            timeout=120.0,
            required_factors=['biometric', 'password', 'hardware_token'],
            computation_cost=2.0
        )
        
        # Task 2: Distribute secret shares
        share_dist_task = CryptoTask(
            task_id="share_distribution",
            task_type="share_distribution",
            data={
                'threshold': self.config['threshold'],
                'num_shares': self.config['num_nodes']
            },
            priority=TaskPriority.HIGH,
            dependencies=["master_key_generation"],
            computation_cost=1.0
        )
        
        # Task 3: Generate and distribute HOTP tokens
        hotp_tasks = []
        for i in range(self.config['num_nodes']):
            hotp_task = CryptoTask(
                task_id=f"hotp_verification_{i}",
                task_type="hotp_verification",
                data={
                    'node_id': f'crypto_node_{i}',
                    'counter_start': i * 10
                },
                priority=TaskPriority.MEDIUM,
                dependencies=["share_distribution"],
                computation_cost=0.5
            )
            hotp_tasks.append(hotp_task)
        
        # Task 4: Create Merkle tree storage
        merkle_task = CryptoTask(
            task_id="merkle_storage",
            task_type="merkle_storage",
            data={'storage_type': 'distributed'},
            priority=TaskPriority.HIGH,
            dependencies=["share_distribution"],
            computation_cost=1.5
        )
        
        # Task 5: Execute MPC verification
        mpc_task = CryptoTask(
            task_id="mpc_computation",
            task_type="mpc_computation",
            data={'computation_type': 'secure_sum'},
            priority=TaskPriority.HIGH,
            dependencies=[task.task_id for task in hotp_tasks] + ["merkle_storage"],
            computation_cost=2.5
        )
        
        # Task 6: Reconstruct and verify key
        reconstruction_task = CryptoTask(
            task_id="key_reconstruction",
            task_type="key_reconstruction",
            data={'verification_required': True},
            priority=TaskPriority.CRITICAL,
            dependencies=["mpc_computation"],
            computation_cost=1.0
        )
        
        # Submit all tasks
        all_tasks = [key_gen_task, share_dist_task] + hotp_tasks + [merkle_task, mpc_task, reconstruction_task]
        
        for task in all_tasks:
            self.scheduler.submit_task(task)
            print(f"Submitted task: {task.task_id}")
    
    def _handle_key_generation(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Handle private key generation using MFKDF"""
        print(f"[{node_id}] Generating master private key with MFKDF...")
        
        # Setup MFKDF for key generation
        generator = MFKDFDeterministicKeyGenerator(node_id)
        mfkdf_system = generator.setup_authentication_factors(self.master_seed)
        
        # Generate RSA key
        active_factors = task.data['active_factors']
        rsa_key = generator.generate_rsa_key_with_mfkdf(
            active_factors=active_factors,
            key_size=task.data['key_size']
        )
        
        # Convert to integer for sharing
        private_key_der = rsa_key.export_key('DER')
        private_key_int = int.from_bytes(private_key_der, 'big')
        
        # Store in global state
        self.global_state['master_private_key'] = private_key_int
        self.global_state['master_key_der'] = private_key_der
        
        print(f"[{node_id}] Master key generated: {rsa_key.size_in_bits()} bits")
        
        return {
            'status': 'success',
            'key_size': rsa_key.size_in_bits(),
            'factors_used': active_factors
        }
    
    def _handle_share_distribution(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Handle secret share distribution using SSS"""
        print(f"[{node_id}] Distributing secret shares...")
        
        if 'master_private_key' not in self.global_state:
            raise ValueError("Master private key not generated yet")
        
        # Create shares using Shamir's Secret Sharing
        shares = ShamirSecretSharing.create_shares(
            self.global_state['master_private_key'],
            task.data['threshold'],
            task.data['num_shares']
        )
        
        # Distribute shares to nodes
        for i, share in enumerate(shares):
            share_node_id = f'crypto_node_{i}'
            self.global_state['private_key_shares'][share_node_id] = share
        
        print(f"[{node_id}] Distributed {len(shares)} shares")
        
        return {
            'status': 'success',
            'shares_distributed': len(shares),
            'threshold': task.data['threshold']
        }
    
    def _handle_hotp_verification(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Handle HOTP token generation and verification"""
        target_node = task.data['node_id']
        counter_start = task.data['counter_start']
        
        print(f"[{node_id}] Generating HOTP tokens for {target_node}...")
        
        # Generate HOTP tokens
        tokens = []
        for i in range(5):  # Generate 5 tokens
            token = HOTP.generate(self.hotp_secret, counter_start + i)
            tokens.append(token)
        
        # Verify first token
        is_valid = HOTP.verify(self.hotp_secret, tokens[0], counter_start)
        
        # Store tokens for the target node
        if 'hotp_tokens' not in self.global_state:
            self.global_state['hotp_tokens'] = {}
        self.global_state['hotp_tokens'][target_node] = tokens
        
        print(f"[{node_id}] Generated {len(tokens)} HOTP tokens for {target_node}")
        
        return {
            'status': 'success',
            'tokens_generated': len(tokens),
            'verification_result': is_valid,
            'target_node': target_node
        }
    
    def _handle_merkle_storage(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Handle Merkle tree creation for integrity verification"""
        print(f"[{node_id}] Creating Merkle tree for integrity verification...")
        
        # Collect share data for Merkle tree
        share_data = []
        for share_node_id, share in self.global_state['private_key_shares'].items():
            share_bytes = f"{share_node_id}:{share[0]}:{share[1]}".encode()
            share_data.append(share_bytes)
        
        if not share_data:
            raise ValueError("No share data available for Merkle tree")
        
        # Create Merkle tree
        merkle_tree = MerkleTree(share_data)
        self.global_state['merkle_tree'] = merkle_tree
        
        # Generate proofs for verification
        proofs = {}
        for i, (share_node_id, _) in enumerate(self.global_state['private_key_shares'].items()):
            if i < len(share_data):
                proof = merkle_tree.get_proof(i)
                is_valid = MerkleTree.verify_proof(share_data[i], proof, merkle_tree.root.hash_value)
                proofs[share_node_id] = {'proof': proof, 'valid': is_valid}
        
        print(f"[{node_id}] Merkle tree created with {len(share_data)} leaves")
        
        return {
            'status': 'success',
            'root_hash': merkle_tree.root.hash_value.hex()[:32],
            'proofs_generated': len(proofs),
            'all_proofs_valid': all(p['valid'] for p in proofs.values())
        }
    
    def _handle_mpc_computation(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Handle secure multi-party computation"""
        print(f"[{node_id}] Executing secure multi-party computation...")
        
        # Create MPC nodes from available shares
        mpc_nodes = []
        for share_node_id, share in self.global_state['private_key_shares'].items():
            secret_value = share[0]  # Use x-coordinate as secret
            mpc_node = MPCNode(share_node_id, secret_value)
            mpc_nodes.append(mpc_node)
        
        if not mpc_nodes:
            raise ValueError("No MPC nodes available")
        
        # Execute secure computation
        mpc_system = SecureMultiPartyComputation(mpc_nodes, self.config['threshold'])
        results = mpc_system.execute_secure_sum()
        
        # Verify computation integrity
        is_valid = mpc_system.verify_computation_integrity()
        
        # Store results
        self.global_state['mpc_results'] = results
        
        print(f"[{node_id}] MPC computation completed with {len(results)} nodes")
        
        return {
            'status': 'success',
            'nodes_participated': len(results),
            'computation_valid': is_valid,
            'computation_type': task.data['computation_type']
        }
    
    def _handle_key_reconstruction(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Handle private key reconstruction and verification"""
        print(f"[{node_id}] Reconstructing private key...")
        
        # Collect threshold number of shares
        shares_for_reconstruction = []
        share_count = 0
        
        for share_node_id, share in self.global_state['private_key_shares'].items():
            if share_count < self.config['threshold']:
                shares_for_reconstruction.append(share)
                share_count += 1
        
        if len(shares_for_reconstruction) < self.config['threshold']:
            raise ValueError("Insufficient shares for reconstruction")
        
        # Reconstruct the secret
        reconstructed_secret = ShamirSecretSharing.reconstruct_secret(shares_for_reconstruction)
        
        # Verify reconstruction
        original_secret = self.global_state['master_private_key']
        reconstruction_valid = (reconstructed_secret == original_secret)
        
        # Additional verification with HOTP
        current_token = HOTP.generate(self.hotp_secret, self.counter)
        hotp_valid = HOTP.verify(self.hotp_secret, current_token, self.counter)
        
        print(f"[{node_id}] Key reconstruction: {'SUCCESS' if reconstruction_valid else 'FAILED'}")
        
        return {
            'status': 'success' if reconstruction_valid else 'failed',
            'reconstruction_valid': reconstruction_valid,
            'hotp_verification': hotp_valid,
            'shares_used': len(shares_for_reconstruction)
        }
    
    def _handle_factor_verification(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        """Handle MFKDF factor verification"""
        print(f"[{node_id}] Verifying authentication factors...")
        
        # This would normally verify real factors, but for demo we simulate
        factors_to_verify = task.data.get('factors', ['biometric', 'password', 'hardware_token'])
        verification_results = {}
        
        for factor in factors_to_verify:
            # Simulate factor verification (in real implementation, this would check actual factors)
            verification_results[factor] = True
        
        all_verified = all(verification_results.values())
        
        print(f"[{node_id}] Factor verification: {'PASSED' if all_verified else 'FAILED'}")
        
        return {
            'status': 'success' if all_verified else 'failed',
            'factors_verified': verification_results,
            'overall_result': all_verified
        }
    
    def _monitor_execution(self):
        """Monitor the execution of the cryptographic workflow"""
        print("\n" + "=" * 55)
        print("MONITORING CRYPTOGRAPHIC WORKFLOW EXECUTION")
        print("=" * 55)
        
        start_time = time.time()
        max_wait_time = 300.0  # 5 minutes maximum
        
        while time.time() - start_time < max_wait_time:
            status = self.scheduler.get_scheduler_status()
            
            print(f"\rStatus: {status['completed_tasks']}/{status['total_tasks']} tasks completed", end="")
            
            if status['completed_tasks'] == status['total_tasks']:
                print("\n\n✅ All tasks completed successfully!")
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
        print("\n" + "=" * 55)
        print("FINAL EXECUTION REPORT")
        print("=" * 55)
        
        # System statistics
        # filepath: c:\Surya\Blockchain\Research\Private Key Security\distributed_crypto_executor.py
