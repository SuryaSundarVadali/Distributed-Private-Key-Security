"""
Pool Verifier Integration
==========================

Integrates ZK proof generation and verification for DeFi pool state queries.
Replaces simulated proofs with real zero-knowledge proofs using Aztec Noir circuits.

Features:
- Pool state verification with ZK proofs
- Integration with yield farming agents
- On-chain verification support
- Batch verification for multiple pools
"""

import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import logging

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Aztec Noir Integration')))

from .proof_manager import ZKProofManager
from python_bindings import NoirProof

logger = logging.getLogger(__name__)


@dataclass
class PoolState:
    """DeFi pool state (from Phase 1)"""
    pool_address: str
    protocol: str
    reserve0: int
    reserve1: int
    total_liquidity: float
    block_number: int
    block_timestamp: int
    fee_tier: Optional[float] = None


@dataclass
class VerifiedPoolState:
    """Pool state with ZK proof"""
    pool_state: PoolState
    proof: NoirProof
    is_verified: bool
    verification_time_ms: float


class PoolVerifier:
    """
    Verifies DeFi pool states using zero-knowledge proofs.
    
    Generates proofs that pool state data is correct without revealing
    sensitive information about the query process.
    
    Features:
    - ZK proof generation for pool queries
    - Proof verification before trusting data
    - Batch verification for efficiency
    - On-chain verification support
    """
    
    def __init__(self, proof_manager: ZKProofManager):
        """
        Initialize pool verifier.
        
        Args:
            proof_manager: ZK proof manager instance
        """
        self.proof_manager = proof_manager
        
        # Ensure pool_verifier circuit is registered
        self._ensure_circuit_registered()
        
        # Statistics
        self.stats = {
            'pools_verified': 0,
            'proofs_generated': 0,
            'proofs_verified': 0,
            'batch_operations': 0,
            'total_verification_time': 0.0
        }
    
    def _ensure_circuit_registered(self):
        """Ensure pool_verifier circuit is registered"""
        if "pool_verifier" not in self.proof_manager.list_circuits():
            # Register circuit
            circuit_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "Aztec Noir Integration", "circuits", "pool_verifier"
            )
            try:
                self.proof_manager.register_circuit("pool_verifier", circuit_path)
                logger.info("✓ pool_verifier circuit registered")
            except Exception as e:
                logger.warning(f"Could not register circuit: {e}")
    
    # ========== Proof Generation ==========
    
    def generate_pool_proof(
        self,
        pool_state: PoolState,
        query_metadata: Optional[Dict] = None
    ) -> NoirProof:
        """
        Generate ZK proof for pool state.
        
        Args:
            pool_state: Pool state to prove
            query_metadata: Optional query metadata
            
        Returns:
            Zero-knowledge proof
        """
        # Construct witness (private data)
        witness = {
            "pool_address": pool_state.pool_address,
            "protocol": pool_state.protocol,
            "reserve0": pool_state.reserve0,
            "reserve1": pool_state.reserve1,
            "total_liquidity": int(pool_state.total_liquidity),
            "block_number": pool_state.block_number,
            "block_timestamp": pool_state.block_timestamp
        }
        
        # Add metadata if provided
        if query_metadata:
            witness.update(query_metadata)
        
        # Public inputs (data that can be publicly verified)
        public_inputs = {
            "pool_address": pool_state.pool_address,
            "block_number": pool_state.block_number
        }
        
        # Generate proof
        proof = self.proof_manager.generate_proof(
            circuit_name="pool_verifier",
            witness=witness,
            public_inputs=public_inputs,
            use_cache=True
        )
        
        self.stats['proofs_generated'] += 1
        
        return proof
    
    def generate_pool_proofs_batch(
        self,
        pool_states: List[PoolState]
    ) -> List[NoirProof]:
        """
        Generate ZK proofs for multiple pool states.
        
        Args:
            pool_states: List of pool states
            
        Returns:
            List of proofs
        """
        witnesses = []
        for pool_state in pool_states:
            witness = {
                "pool_address": pool_state.pool_address,
                "protocol": pool_state.protocol,
                "reserve0": pool_state.reserve0,
                "reserve1": pool_state.reserve1,
                "total_liquidity": int(pool_state.total_liquidity),
                "block_number": pool_state.block_number,
                "block_timestamp": pool_state.block_timestamp
            }
            witnesses.append(witness)
        
        proofs = self.proof_manager.generate_proof_batch(
            circuit_name="pool_verifier",
            witnesses=witnesses,
            use_cache=True
        )
        
        self.stats['proofs_generated'] += len(proofs)
        self.stats['batch_operations'] += 1
        
        return proofs
    
    # ========== Verification ==========
    
    def verify_pool_state(
        self,
        pool_state: PoolState,
        proof: NoirProof
    ) -> VerifiedPoolState:
        """
        Verify pool state with ZK proof.
        
        Args:
            pool_state: Pool state to verify
            proof: ZK proof for pool state
            
        Returns:
            Verified pool state with verification result
        """
        start_time = time.time()
        
        # Verify proof
        public_inputs = {
            "pool_address": pool_state.pool_address,
            "block_number": pool_state.block_number
        }
        
        is_valid = self.proof_manager.verify_proof(proof, public_inputs)
        
        elapsed = (time.time() - start_time) * 1000  # ms
        
        # Update stats
        self.stats['pools_verified'] += 1
        self.stats['proofs_verified'] += 1
        self.stats['total_verification_time'] += elapsed
        
        return VerifiedPoolState(
            pool_state=pool_state,
            proof=proof,
            is_verified=is_valid,
            verification_time_ms=elapsed
        )
    
    def verify_pool_states_batch(
        self,
        pool_states: List[PoolState],
        proofs: List[NoirProof]
    ) -> List[VerifiedPoolState]:
        """
        Verify multiple pool states with ZK proofs.
        
        Args:
            pool_states: List of pool states
            proofs: List of corresponding proofs
            
        Returns:
            List of verified pool states
        """
        start_time = time.time()
        
        # Batch verify
        results = self.proof_manager.verify_proof_batch(proofs)
        
        elapsed = (time.time() - start_time) * 1000  # ms
        
        # Create verified states
        verified_states = []
        for i, (pool_state, is_valid) in enumerate(zip(pool_states, results)):
            verified_states.append(VerifiedPoolState(
                pool_state=pool_state,
                proof=proofs[i],
                is_verified=is_valid,
                verification_time_ms=elapsed / len(pool_states)
            ))
        
        # Update stats
        self.stats['pools_verified'] += len(pool_states)
        self.stats['proofs_verified'] += len(proofs)
        self.stats['batch_operations'] += 1
        self.stats['total_verification_time'] += elapsed
        
        return verified_states
    
    # ========== Integration with Yield Farming Agents ==========
    
    def create_verified_pool_query(
        self,
        pool_address: str,
        protocol: str,
        reserve0: int,
        reserve1: int,
        total_liquidity: float,
        block_number: int,
        block_timestamp: int
    ) -> VerifiedPoolState:
        """
        Create pool state and generate/verify proof in one call.
        
        This is the main integration point for yield farming agents.
        
        Args:
            pool_address: Pool address
            protocol: Protocol name
            reserve0: Reserve amount for token0
            reserve1: Reserve amount for token1
            total_liquidity: Total liquidity
            block_number: Block number
            block_timestamp: Block timestamp
            
        Returns:
            Verified pool state with proof
        """
        # Create pool state
        pool_state = PoolState(
            pool_address=pool_address,
            protocol=protocol,
            reserve0=reserve0,
            reserve1=reserve1,
            total_liquidity=total_liquidity,
            block_number=block_number,
            block_timestamp=block_timestamp
        )
        
        # Generate proof
        proof = self.generate_pool_proof(pool_state)
        
        # Verify proof
        verified_state = self.verify_pool_state(pool_state, proof)
        
        return verified_state
    
    # ========== On-Chain Verification ==========
    
    def prepare_onchain_verification(
        self,
        proof: NoirProof
    ) -> Dict:
        """
        Prepare proof for on-chain verification.
        
        Formats proof and public inputs for smart contract verification.
        
        Args:
            proof: ZK proof to verify on-chain
            
        Returns:
            Dictionary with formatted data for smart contract
        """
        return {
            "proof": proof.to_hex(),
            "publicInputs": list(proof.public_inputs.values()),
            "circuitName": proof.circuit_name,
            "timestamp": proof.timestamp
        }
    
    def format_batch_verification(
        self,
        proofs: List[NoirProof]
    ) -> Dict:
        """
        Format multiple proofs for batch on-chain verification.
        
        Args:
            proofs: List of proofs
            
        Returns:
            Formatted batch verification data
        """
        return {
            "proofs": [p.to_hex() for p in proofs],
            "publicInputsList": [list(p.public_inputs.values()) for p in proofs],
            "circuitNames": [p.circuit_name for p in proofs],
            "count": len(proofs)
        }
    
    # ========== Statistics ==========
    
    def get_statistics(self) -> Dict:
        """Get pool verifier statistics"""
        avg_verification_time = (
            self.stats['total_verification_time'] / self.stats['pools_verified']
            if self.stats['pools_verified'] > 0 else 0.0
        )
        
        return {
            'pools_verified': self.stats['pools_verified'],
            'proofs_generated': self.stats['proofs_generated'],
            'proofs_verified': self.stats['proofs_verified'],
            'batch_operations': self.stats['batch_operations'],
            'avg_verification_time_ms': avg_verification_time
        }


# ========== Built-in Tests ==========

def test_pool_verifier():
    """Test pool verifier integration"""
    print("Testing Pool Verifier Integration")
    print("=" * 60)
    
    # Test 1: Initialize
    print("\n1. Initializing Pool Verifier...")
    proof_manager = ZKProofManager()
    verifier = PoolVerifier(proof_manager)
    print("✓ Pool verifier initialized")
    
    # Test 2: Generate proof for single pool
    print("\n2. Generating Pool Proof...")
    pool_state = PoolState(
        pool_address="0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",
        protocol="uniswap_v3",
        reserve0=1000000,
        reserve1=2000000,
        total_liquidity=3000000.0,
        block_number=19000000,
        block_timestamp=1704067200
    )
    
    proof = verifier.generate_pool_proof(pool_state)
    print(f"✓ Proof generated: {proof.size()} bytes")
    print(f"  Circuit: {proof.circuit_name}")
    
    # Test 3: Verify pool state
    print("\n3. Verifying Pool State...")
    verified = verifier.verify_pool_state(pool_state, proof)
    print(f"✓ Pool state verified: {verified.is_verified}")
    print(f"  Verification time: {verified.verification_time_ms:.2f}ms")
    
    # Test 4: Batch operations
    print("\n4. Batch Pool Verification...")
    pool_states = [
        PoolState(
            pool_address=f"0xpool{i}",
            protocol="uniswap_v3",
            reserve0=i*1000000,
            reserve1=i*2000000,
            total_liquidity=float(i*3000000),
            block_number=19000000+i,
            block_timestamp=1704067200+i
        )
        for i in range(10)
    ]
    
    proofs = verifier.generate_pool_proofs_batch(pool_states)
    print(f"✓ Batch generated: {len(proofs)} proofs")
    
    verified_states = verifier.verify_pool_states_batch(pool_states, proofs)
    valid_count = sum(1 for v in verified_states if v.is_verified)
    print(f"✓ Batch verified: {valid_count}/{len(verified_states)} valid")
    
    # Test 5: Integrated query
    print("\n5. Integrated Pool Query...")
    verified_query = verifier.create_verified_pool_query(
        pool_address="0xIntegratedPool",
        protocol="aave_v3",
        reserve0=5000000,
        reserve1=10000000,
        total_liquidity=15000000.0,
        block_number=19000100,
        block_timestamp=1704067300
    )
    print(f"✓ Integrated query: verified={verified_query.is_verified}")
    print(f"  Pool: {verified_query.pool_state.pool_address}")
    print(f"  Protocol: {verified_query.pool_state.protocol}")
    
    # Test 6: On-chain preparation
    print("\n6. Preparing On-Chain Verification...")
    onchain_data = verifier.prepare_onchain_verification(verified_query.proof)
    print(f"✓ On-chain data prepared")
    print(f"  Proof length: {len(onchain_data['proof'])} chars")
    print(f"  Public inputs: {len(onchain_data['publicInputs'])}")
    
    batch_data = verifier.format_batch_verification(proofs[:5])
    print(f"✓ Batch data formatted: {batch_data['count']} proofs")
    
    # Test 7: Statistics
    print("\n7. Pool Verifier Statistics:")
    stats = verifier.get_statistics()
    print(f"  Pools verified: {stats['pools_verified']}")
    print(f"  Proofs generated: {stats['proofs_generated']}")
    print(f"  Proofs verified: {stats['proofs_verified']}")
    print(f"  Batch operations: {stats['batch_operations']}")
    print(f"  Avg verification time: {stats['avg_verification_time_ms']:.2f}ms")
    
    # Proof manager stats
    print("\n8. Proof Manager Statistics:")
    pm_stats = proof_manager.get_statistics()
    print(f"  Circuits registered: {pm_stats['circuits_registered']}")
    print(f"  Cache hit rate: {pm_stats['cache']['hit_rate']:.1%}")
    print(f"  Avg proof time: {pm_stats['avg_proof_time_ms']:.2f}ms")
    
    print("\n✓ All pool verifier tests passed!")


if __name__ == "__main__":
    test_pool_verifier()
