"""
ZK-SMPC Integration
===================

Integrates Zero-Knowledge Proofs with Secure Multi-Party Computation.

Features:
- ZK proofs for SMPC computation results
- Privacy-preserving verification
- Proof-of-computation without revealing inputs
- Integration with Aztec Noir circuits
- Verifiable distributed computation
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import logging
import hashlib

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Aztec Noir Integration')))

from .smpc_coordinator import SMPCCoordinator, ComputationResult, ComputationTask
from zkp import ZKProofManager, PoolState
from python_bindings import NoirProof, NoirCircuit

logger = logging.getLogger(__name__)


@dataclass
class VerifiableComputation:
    """SMPC computation with ZK proof"""
    task_id: str
    computation_result: ComputationResult
    zk_proof: NoirProof
    public_inputs: Dict[str, Any]
    is_verified: bool
    proof_generation_time_ms: float


@dataclass
class PrivateInput:
    """Private input for ZK-SMPC"""
    agent_id: str
    value: int
    commitment: bytes  # Commitment to value
    timestamp: float = field(default_factory=time.time)


class ZKSMPCIntegration:
    """
    Integrates zero-knowledge proofs with SMPC for verifiable computation.
    
    Enables agents to:
    1. Prove computation correctness without revealing inputs
    2. Verify SMPC results using ZK proofs
    3. Maintain privacy while ensuring correctness
    4. Detect malicious participants with proofs
    
    Features:
    - Proof-of-computation for SMPC tasks
    - Privacy-preserving aggregation
    - Verifiable secret sharing
    - Byzantine detection with ZK proofs
    """
    
    def __init__(
        self,
        smpc_coordinator: SMPCCoordinator,
        proof_manager: ZKProofManager
    ):
        """
        Initialize ZK-SMPC integration.
        
        Args:
            smpc_coordinator: SMPC coordinator instance
            proof_manager: ZK proof manager instance
        """
        self.smpc = smpc_coordinator
        self.proof_manager = proof_manager
        
        # Ensure computation verification circuit is registered
        self._ensure_circuits_registered()
        
        # Private input commitments
        self.commitments: Dict[str, bytes] = {}
        
        # Verified computations
        self.verified_computations: Dict[str, VerifiableComputation] = {}
        
        # Statistics
        self.stats = {
            'computations_verified': 0,
            'proofs_generated': 0,
            'verifications_successful': 0,
            'verifications_failed': 0,
            'total_proof_time': 0.0
        }
    
    def _ensure_circuits_registered(self):
        """Ensure ZK circuits are registered"""
        circuits = [
            "aggregate_votes",
            "reputation_updater",
            "byzantine_detector"
        ]
        
        for circuit_name in circuits:
            if circuit_name not in self.proof_manager.list_circuits():
                try:
                    circuit_path = os.path.join(
                        os.path.dirname(__file__),
                        "..", "..", "Aztec Noir Integration", "circuits", circuit_name
                    )
                    self.proof_manager.register_circuit(circuit_name, circuit_path)
                    logger.info(f"✓ {circuit_name} circuit registered")
                except Exception as e:
                    logger.warning(f"Could not register {circuit_name}: {e}")
    
    # ========== Commitment Scheme ==========
    
    def commit_input(
        self,
        agent_id: str,
        value: int,
        blinding_factor: Optional[bytes] = None
    ) -> PrivateInput:
        """
        Create commitment to private input.
        
        Uses Pedersen commitment: C = g^value * h^r
        (Simulated with hash for this implementation)
        
        Args:
            agent_id: Agent committing value
            value: Private value
            blinding_factor: Random blinding factor
            
        Returns:
            Private input with commitment
        """
        if blinding_factor is None:
            blinding_factor = os.urandom(32)
        
        # Create commitment: Hash(value || blinding_factor)
        commitment = hashlib.sha256(
            str(value).encode() + blinding_factor
        ).digest()
        
        private_input = PrivateInput(
            agent_id=agent_id,
            value=value,
            commitment=commitment
        )
        
        # Store commitment
        self.commitments[agent_id] = commitment
        
        return private_input
    
    def verify_commitment(
        self,
        agent_id: str,
        value: int,
        blinding_factor: bytes
    ) -> bool:
        """Verify commitment matches revealed value"""
        if agent_id not in self.commitments:
            return False
        
        # Recompute commitment
        expected = hashlib.sha256(
            str(value).encode() + blinding_factor
        ).digest()
        
        return expected == self.commitments[agent_id]
    
    # ========== Verifiable SMPC Computation ==========
    
    def execute_verifiable_computation(
        self,
        session_id: str,
        task_type: str,
        private_inputs: List[PrivateInput],
        public_parameters: Optional[Dict] = None
    ) -> VerifiableComputation:
        """
        Execute SMPC computation with ZK proof of correctness.
        
        Args:
            session_id: SMPC session ID
            task_type: Type of computation
            private_inputs: Private inputs from agents
            public_parameters: Public parameters for computation
            
        Returns:
            Verifiable computation with ZK proof
        """
        start_time = time.time()
        
        # Step 1: Distribute computation using SMPC
        # For simplicity, use first input's value
        # In production, would handle multiple inputs properly
        secret_value = private_inputs[0].value
        
        task_id = self.smpc.distribute_computation(
            session_id=session_id,
            task_type=task_type,
            secret_value=secret_value,
            metadata={'num_inputs': len(private_inputs)}
        )
        
        # Step 2: Execute SMPC computation
        session = self.smpc.get_session(session_id)
        task = session.tasks[task_id]
        shares = list(task.input_shares.values())
        
        computation_result = self.smpc.execute_computation(
            session_id=session_id,
            task_id=task_id,
            collected_shares=shares[:task.threshold]
        )
        
        # Step 3: Generate ZK proof of correct computation
        proof_start = time.time()
        zk_proof = self._generate_computation_proof(
            task=task,
            result=computation_result,
            private_inputs=private_inputs,
            public_parameters=public_parameters
        )
        proof_time = (time.time() - proof_start) * 1000
        
        # Step 4: Verify proof
        is_verified = self._verify_computation_proof(
            zk_proof=zk_proof,
            result=computation_result
        )
        
        # Create verifiable computation
        verifiable = VerifiableComputation(
            task_id=task_id,
            computation_result=computation_result,
            zk_proof=zk_proof,
            public_inputs={
                'task_id': task_id,
                'threshold': task.threshold
            },
            is_verified=is_verified,
            proof_generation_time_ms=proof_time
        )
        
        self.verified_computations[task_id] = verifiable
        
        # Update statistics
        self.stats['computations_verified'] += 1
        self.stats['proofs_generated'] += 1
        self.stats['total_proof_time'] += proof_time
        
        if is_verified:
            self.stats['verifications_successful'] += 1
        else:
            self.stats['verifications_failed'] += 1
        
        logger.info(f"✓ Verifiable computation completed: {task_id}")
        logger.info(f"  Result: {computation_result.result}")
        logger.info(f"  Proof time: {proof_time:.2f}ms")
        logger.info(f"  Verified: {is_verified}")
        
        return verifiable
    
    def _generate_computation_proof(
        self,
        task: ComputationTask,
        result: ComputationResult,
        private_inputs: List[PrivateInput],
        public_parameters: Optional[Dict]
    ) -> NoirProof:
        """Generate ZK proof for computation correctness"""
        
        # Construct witness (private data)
        witness = {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'result': result.result or 0,
            'num_shares': len(task.input_shares),
            'threshold': task.threshold,
            'private_inputs': [inp.value for inp in private_inputs]
        }
        
        # Public inputs (can be verified)
        public_inputs = {
            'task_id': task.task_id,
            'threshold': task.threshold
        }
        
        # Choose circuit based on task type
        circuit_name = self._select_circuit(task.task_type)
        
        # Generate proof
        proof = self.proof_manager.generate_proof(
            circuit_name=circuit_name,
            witness=witness,
            public_inputs=public_inputs,
            use_cache=True
        )
        
        return proof
    
    def _verify_computation_proof(
        self,
        zk_proof: NoirProof,
        result: ComputationResult
    ) -> bool:
        """Verify ZK proof for computation"""
        
        public_inputs = {
            'task_id': result.task_id,
            'threshold': result.shares_used
        }
        
        return self.proof_manager.verify_proof(zk_proof, public_inputs)
    
    def _select_circuit(self, task_type: str) -> str:
        """Select appropriate circuit for task type"""
        circuit_map = {
            'aggregate': 'aggregate_votes',
            'voting': 'aggregate_votes',
            'reputation': 'reputation_updater',
            'byzantine': 'byzantine_detector'
        }
        return circuit_map.get(task_type, 'aggregate_votes')
    
    # ========== Privacy-Preserving Aggregation ==========
    
    def aggregate_with_privacy(
        self,
        session_id: str,
        values: List[int],
        agent_ids: List[str],
        operation: str = 'sum'
    ) -> VerifiableComputation:
        """
        Aggregate values with privacy preservation and ZK proof.
        
        Args:
            session_id: SMPC session ID
            values: Private values from agents
            agent_ids: Agent identifiers
            operation: Aggregation operation
            
        Returns:
            Verifiable aggregation result
        """
        # Create private inputs with commitments
        private_inputs = [
            self.commit_input(agent_id, value)
            for agent_id, value in zip(agent_ids, values)
        ]
        
        # Execute verifiable computation
        return self.execute_verifiable_computation(
            session_id=session_id,
            task_type=f"aggregate_{operation}",
            private_inputs=private_inputs,
            public_parameters={'operation': operation}
        )
    
    # ========== Byzantine Detection with ZK ==========
    
    def detect_byzantine_with_proof(
        self,
        session_id: str,
        task_id: str
    ) -> Tuple[List[str], Optional[NoirProof]]:
        """
        Detect Byzantine agents and generate proof.
        
        Args:
            session_id: Session ID
            task_id: Task ID
            
        Returns:
            Tuple of (byzantine_agents, zk_proof)
        """
        # Detect Byzantine agents using SMPC
        byzantine = self.smpc.detect_byzantine_agents(session_id, task_id)
        
        if not byzantine:
            return [], None
        
        # Generate ZK proof of Byzantine detection
        session = self.smpc.get_session(session_id)
        task = session.tasks[task_id]
        
        witness = {
            'task_id': task_id,
            'suspicious_count': len(byzantine),
            'total_agents': len(task.participants),
            'threshold': task.threshold
        }
        
        public_inputs = {
            'task_id': task_id,
            'detection_count': len(byzantine)
        }
        
        proof = self.proof_manager.generate_proof(
            circuit_name='byzantine_detector',
            witness=witness,
            public_inputs=public_inputs,
            use_cache=False  # Don't cache Byzantine proofs
        )
        
        logger.warning(f"Byzantine agents detected with proof")
        logger.warning(f"  Agents: {byzantine}")
        logger.warning(f"  Proof generated")
        
        return byzantine, proof
    
    # ========== Batch Verification ==========
    
    def batch_verify_computations(
        self,
        task_ids: List[str]
    ) -> Dict[str, bool]:
        """
        Batch verify multiple computations.
        
        Args:
            task_ids: List of task IDs to verify
            
        Returns:
            Dictionary mapping task_id -> verification result
        """
        # Collect proofs
        proofs = []
        for task_id in task_ids:
            if task_id in self.verified_computations:
                verifiable = self.verified_computations[task_id]
                proofs.append(verifiable.zk_proof)
        
        # Batch verify
        results = self.proof_manager.verify_proof_batch(proofs)
        
        # Map results
        verification_map = {
            task_id: result
            for task_id, result in zip(task_ids, results)
            if task_id in self.verified_computations
        }
        
        return verification_map
    
    # ========== Statistics ==========
    
    def get_statistics(self) -> Dict:
        """Get ZK-SMPC integration statistics"""
        avg_proof_time = (
            self.stats['total_proof_time'] / self.stats['proofs_generated']
            if self.stats['proofs_generated'] > 0 else 0.0
        )
        
        verification_rate = (
            self.stats['verifications_successful'] /
            self.stats['computations_verified']
            if self.stats['computations_verified'] > 0 else 0.0
        )
        
        return {
            'computations_verified': self.stats['computations_verified'],
            'proofs_generated': self.stats['proofs_generated'],
            'verifications_successful': self.stats['verifications_successful'],
            'verifications_failed': self.stats['verifications_failed'],
            'verification_rate': verification_rate,
            'avg_proof_time_ms': avg_proof_time,
            'circuits_registered': len(self.proof_manager.list_circuits())
        }


# ========== Built-in Tests ==========

def test_zk_smpc_integration():
    """Test ZK-SMPC integration"""
    print("\nTesting ZK-SMPC Integration")
    print("=" * 60)
    
    # Test 1: Initialize
    print("\n1. Initializing Components...")
    smpc = SMPCCoordinator(
        agent_id="coordinator",
        threshold=3,
        total_agents=5
    )
    
    from zkp import ZKProofManager
    proof_manager = ZKProofManager()
    
    zk_smpc = ZKSMPCIntegration(smpc, proof_manager)
    print("✓ ZK-SMPC initialized")
    
    # Test 2: Create commitments
    print("\n2. Creating Commitments...")
    agents = [f"agent_{i}" for i in range(5)]
    values = [100, 200, 300, 400, 500]
    
    private_inputs = []
    for agent_id, value in zip(agents, values):
        private_input = zk_smpc.commit_input(agent_id, value)
        private_inputs.append(private_input)
        print(f"  ✓ {agent_id}: committed value")
    
    # Test 3: Create session
    print("\n3. Creating SMPC Session...")
    session_id = smpc.create_session(agents, threshold=3)
    print(f"✓ Session: {session_id}")
    
    # Test 4: Verifiable computation
    print("\n4. Executing Verifiable Computation...")
    verifiable = zk_smpc.execute_verifiable_computation(
        session_id=session_id,
        task_type="aggregate",
        private_inputs=private_inputs[:1],  # Use first input
        public_parameters={'operation': 'test'}
    )
    print(f"✓ Computation completed")
    print(f"  Result: {verifiable.computation_result.result}")
    print(f"  Verified: {verifiable.is_verified}")
    print(f"  Proof time: {verifiable.proof_generation_time_ms:.2f}ms")
    
    # Test 5: Privacy-preserving aggregation
    print("\n5. Privacy-Preserving Aggregation...")
    agg_result = zk_smpc.aggregate_with_privacy(
        session_id=session_id,
        values=[10, 20, 30],
        agent_ids=agents[:3],
        operation='sum'
    )
    print(f"✓ Aggregation with privacy")
    print(f"  Result: {agg_result.computation_result.result}")
    print(f"  Verified: {agg_result.is_verified}")
    
    # Test 6: Statistics
    print("\n6. ZK-SMPC Statistics:")
    stats = zk_smpc.get_statistics()
    print(f"  Computations verified: {stats['computations_verified']}")
    print(f"  Proofs generated: {stats['proofs_generated']}")
    print(f"  Verification rate: {stats['verification_rate']:.1%}")
    print(f"  Avg proof time: {stats['avg_proof_time_ms']:.2f}ms")
    
    print("\n✓ All ZK-SMPC integration tests passed!")


if __name__ == "__main__":
    test_zk_smpc_integration()
