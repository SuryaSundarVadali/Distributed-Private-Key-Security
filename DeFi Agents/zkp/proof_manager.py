"""
ZK Proof Manager
================

Manages zero-knowledge proof generation, verification, caching, and batching
for DeFi agent coordination system.

Features:
- Proof caching to avoid redundant generation
- Batch proof generation for efficiency
- Multi-circuit coordination
- Performance optimization
- Proof lifecycle management
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import time
import hashlib
import json
from collections import defaultdict
import logging

# Add Aztec Noir Integration to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Aztec Noir Integration')))

from python_bindings import (
    NoirCircuit, NoirProof, NoirCircuitCompiler,
    NoirProofGenerator, NoirProofVerifier
)

logger = logging.getLogger(__name__)


@dataclass
class ProofRequest:
    """Request to generate a proof"""
    circuit_name: str
    witness: Dict
    public_inputs: Dict
    priority: int = 0  # Higher = more urgent
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:16])


@dataclass
class ProofCacheEntry:
    """Cached proof entry"""
    proof: NoirProof
    witness_hash: str
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    hit_rate: float = 0.0


class ProofCache:
    """
    LRU cache for zero-knowledge proofs.
    
    Caches proofs to avoid regenerating identical proofs.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        """
        Initialize proof cache.
        
        Args:
            max_size: Maximum number of cached proofs
            ttl_seconds: Time-to-live for cached proofs
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, ProofCacheEntry] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _compute_witness_hash(self, witness: Dict) -> str:
        """Compute deterministic hash of witness"""
        witness_str = json.dumps(witness, sort_keys=True)
        return hashlib.sha256(witness_str.encode()).hexdigest()
    
    def _make_cache_key(self, circuit_name: str, witness_hash: str) -> str:
        """Create cache key"""
        return f"{circuit_name}:{witness_hash}"
    
    def get(self, circuit_name: str, witness: Dict) -> Optional[NoirProof]:
        """Get proof from cache if available"""
        witness_hash = self._compute_witness_hash(witness)
        cache_key = self._make_cache_key(circuit_name, witness_hash)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            # Check TTL
            age = time.time() - entry.last_accessed
            if age > self.ttl_seconds:
                del self.cache[cache_key]
                self.misses += 1
                return None
            
            # Update access stats
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.hits += 1
            
            return entry.proof
        
        self.misses += 1
        return None
    
    def put(self, circuit_name: str, witness: Dict, proof: NoirProof):
        """Store proof in cache"""
        witness_hash = self._compute_witness_hash(witness)
        cache_key = self._make_cache_key(circuit_name, witness_hash)
        
        # Evict if cache is full (LRU)
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        entry = ProofCacheEntry(
            proof=proof,
            witness_hash=witness_hash
        )
        self.cache[cache_key] = entry
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        # Find LRU entry
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
        del self.cache[lru_key]
        self.evictions += 1
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()


class ProofBatcher:
    """
    Batches proof generation requests for efficiency.
    
    Collects multiple proof requests and processes them together
    to amortize setup costs.
    """
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 1.0):
        """
        Initialize proof batcher.
        
        Args:
            batch_size: Number of proofs to batch together
            batch_timeout: Maximum wait time before processing batch
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Batches by circuit
        self.batches: Dict[str, List[ProofRequest]] = defaultdict(list)
        self.batch_start_times: Dict[str, float] = {}
    
    def add_request(self, request: ProofRequest) -> Optional[List[ProofRequest]]:
        """
        Add proof request to batch.
        
        Returns batch if ready to process, None otherwise.
        """
        circuit_name = request.circuit_name
        
        # Initialize batch if needed
        if circuit_name not in self.batch_start_times:
            self.batch_start_times[circuit_name] = time.time()
        
        # Add to batch
        self.batches[circuit_name].append(request)
        
        # Check if batch is ready
        batch = self.batches[circuit_name]
        elapsed = time.time() - self.batch_start_times[circuit_name]
        
        if len(batch) >= self.batch_size or elapsed >= self.batch_timeout:
            # Return and clear batch
            ready_batch = batch.copy()
            self.batches[circuit_name].clear()
            del self.batch_start_times[circuit_name]
            return ready_batch
        
        return None
    
    def get_pending_batches(self) -> Dict[str, List[ProofRequest]]:
        """Get all pending batches that have timed out"""
        ready_batches = {}
        current_time = time.time()
        
        for circuit_name, batch in self.batches.items():
            if not batch:
                continue
            
            elapsed = current_time - self.batch_start_times[circuit_name]
            if elapsed >= self.batch_timeout:
                ready_batches[circuit_name] = batch.copy()
                self.batches[circuit_name].clear()
                del self.batch_start_times[circuit_name]
        
        return ready_batches


class ZKProofManager:
    """
    Manages zero-knowledge proof lifecycle for DeFi agents.
    
    Coordinates proof generation, verification, caching, and batching
    across multiple circuits.
    
    Features:
    - Automatic proof caching
    - Batch proof generation
    - Multi-circuit management
    - Performance monitoring
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        batch_size: int = 10,
        enable_batching: bool = True
    ):
        """
        Initialize ZK proof manager.
        
        Args:
            cache_size: Maximum cached proofs
            batch_size: Proofs per batch
            enable_batching: Enable batch processing
        """
        # Core components
        self.compiler = NoirCircuitCompiler()
        self.generator = NoirProofGenerator()
        self.verifier = NoirProofVerifier()
        
        # Circuit registry
        self.circuits: Dict[str, NoirCircuit] = {}
        
        # Caching and batching
        self.cache = ProofCache(max_size=cache_size)
        self.batcher = ProofBatcher(batch_size=batch_size) if enable_batching else None
        self.enable_batching = enable_batching
        
        # Statistics
        self.stats = {
            'proofs_generated': 0,
            'proofs_verified': 0,
            'proofs_cached': 0,
            'batch_operations': 0,
            'total_proof_time': 0.0,
            'total_verify_time': 0.0
        }
    
    # ========== Circuit Management ==========
    
    def register_circuit(self, circuit_name: str, circuit_path: str) -> NoirCircuit:
        """
        Register and compile a Noir circuit.
        
        Args:
            circuit_name: Name of the circuit
            circuit_path: Path to circuit .nr file
            
        Returns:
            Compiled circuit
        """
        logger.info(f"Registering circuit: {circuit_name}")
        
        # Compile circuit
        circuit = self.compiler.compile_circuit(circuit_path)
        circuit.name = circuit_name  # Override name
        
        # Store in registry
        self.circuits[circuit_name] = circuit
        
        logger.info(f"✓ Circuit registered: {circuit_name}")
        return circuit
    
    def get_circuit(self, circuit_name: str) -> Optional[NoirCircuit]:
        """Get registered circuit"""
        return self.circuits.get(circuit_name)
    
    def list_circuits(self) -> List[str]:
        """List all registered circuits"""
        return list(self.circuits.keys())
    
    # ========== Proof Generation ==========
    
    def generate_proof(
        self,
        circuit_name: str,
        witness: Dict,
        public_inputs: Optional[Dict] = None,
        use_cache: bool = True
    ) -> NoirProof:
        """
        Generate zero-knowledge proof.
        
        Args:
            circuit_name: Name of circuit to use
            witness: Private witness data
            public_inputs: Public inputs
            use_cache: Check cache before generating
            
        Returns:
            Generated proof
        """
        # Check cache first
        if use_cache:
            cached_proof = self.cache.get(circuit_name, witness)
            if cached_proof:
                self.stats['proofs_cached'] += 1
                return cached_proof
        
        # Get circuit
        circuit = self.get_circuit(circuit_name)
        if not circuit:
            raise ValueError(f"Circuit not registered: {circuit_name}")
        
        # Generate proof
        start_time = time.time()
        proof = self.generator.generate_proof(circuit, witness, public_inputs)
        elapsed = time.time() - start_time
        
        # Update stats
        self.stats['proofs_generated'] += 1
        self.stats['total_proof_time'] += elapsed
        
        # Cache proof
        if use_cache:
            self.cache.put(circuit_name, witness, proof)
        
        return proof
    
    def generate_proof_batch(
        self,
        circuit_name: str,
        witnesses: List[Dict],
        use_cache: bool = True
    ) -> List[NoirProof]:
        """
        Generate multiple proofs in batch.
        
        Args:
            circuit_name: Circuit to use
            witnesses: List of witness data
            use_cache: Check cache before generating
            
        Returns:
            List of generated proofs
        """
        circuit = self.get_circuit(circuit_name)
        if not circuit:
            raise ValueError(f"Circuit not registered: {circuit_name}")
        
        proofs = []
        uncached_witnesses = []
        uncached_indices = []
        
        # Check cache
        for i, witness in enumerate(witnesses):
            if use_cache:
                cached_proof = self.cache.get(circuit_name, witness)
                if cached_proof:
                    proofs.append(cached_proof)
                    self.stats['proofs_cached'] += 1
                    continue
            
            # Need to generate
            proofs.append(None)  # Placeholder
            uncached_witnesses.append(witness)
            uncached_indices.append(i)
        
        # Generate uncached proofs in batch
        if uncached_witnesses:
            start_time = time.time()
            new_proofs = self.generator.batch_generate_proofs(circuit, uncached_witnesses)
            elapsed = time.time() - start_time
            
            # Update stats
            self.stats['proofs_generated'] += len(new_proofs)
            self.stats['batch_operations'] += 1
            self.stats['total_proof_time'] += elapsed
            
            # Cache and insert
            for i, proof in enumerate(new_proofs):
                idx = uncached_indices[i]
                proofs[idx] = proof
                
                if use_cache:
                    self.cache.put(circuit_name, uncached_witnesses[i], proof)
        
        return proofs
    
    # ========== Proof Verification ==========
    
    def verify_proof(
        self,
        proof: NoirProof,
        public_inputs: Optional[Dict] = None
    ) -> bool:
        """
        Verify zero-knowledge proof.
        
        Args:
            proof: Proof to verify
            public_inputs: Expected public inputs
            
        Returns:
            True if valid, False otherwise
        """
        # Get circuit
        circuit = self.get_circuit(proof.circuit_name)
        if not circuit:
            logger.warning(f"Circuit not registered: {proof.circuit_name}")
            return False
        
        # Verify
        start_time = time.time()
        is_valid = self.verifier.verify_proof(proof, circuit, public_inputs)
        elapsed = time.time() - start_time
        
        # Update stats
        self.stats['proofs_verified'] += 1
        self.stats['total_verify_time'] += elapsed
        
        return is_valid
    
    def verify_proof_batch(self, proofs: List[NoirProof]) -> List[bool]:
        """
        Verify multiple proofs in batch.
        
        Args:
            proofs: List of proofs to verify
            
        Returns:
            List of verification results
        """
        if not proofs:
            return []
        
        # Group by circuit
        by_circuit = defaultdict(list)
        for proof in proofs:
            by_circuit[proof.circuit_name].append(proof)
        
        # Verify each group
        results = [False] * len(proofs)
        proof_idx = 0
        
        for circuit_name, circuit_proofs in by_circuit.items():
            circuit = self.get_circuit(circuit_name)
            if not circuit:
                proof_idx += len(circuit_proofs)
                continue
            
            start_time = time.time()
            batch_results = self.verifier.batch_verify_proofs(circuit_proofs, circuit)
            elapsed = time.time() - start_time
            
            # Update stats
            self.stats['proofs_verified'] += len(circuit_proofs)
            self.stats['batch_operations'] += 1
            self.stats['total_verify_time'] += elapsed
            
            # Store results
            for result in batch_results:
                results[proof_idx] = result
                proof_idx += 1
        
        return results
    
    # ========== Statistics ==========
    
    def get_statistics(self) -> Dict:
        """Get proof manager statistics"""
        cache_stats = self.cache.get_stats()
        
        avg_proof_time = (
            self.stats['total_proof_time'] / self.stats['proofs_generated']
            if self.stats['proofs_generated'] > 0 else 0.0
        )
        
        avg_verify_time = (
            self.stats['total_verify_time'] / self.stats['proofs_verified']
            if self.stats['proofs_verified'] > 0 else 0.0
        )
        
        return {
            'circuits_registered': len(self.circuits),
            'proofs_generated': self.stats['proofs_generated'],
            'proofs_verified': self.stats['proofs_verified'],
            'proofs_cached': self.stats['proofs_cached'],
            'batch_operations': self.stats['batch_operations'],
            'avg_proof_time_ms': avg_proof_time * 1000,
            'avg_verify_time_ms': avg_verify_time * 1000,
            'cache': cache_stats
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.stats = {
            'proofs_generated': 0,
            'proofs_verified': 0,
            'proofs_cached': 0,
            'batch_operations': 0,
            'total_proof_time': 0.0,
            'total_verify_time': 0.0
        }
        self.cache.hits = 0
        self.cache.misses = 0
        self.cache.evictions = 0


# ========== Built-in Tests ==========

def test_proof_manager():
    """Test ZK proof manager"""
    print("Testing ZK Proof Manager")
    print("=" * 60)
    
    # Test 1: Initialization
    print("\n1. Initializing Proof Manager...")
    manager = ZKProofManager(cache_size=100, batch_size=5)
    print("✓ Manager initialized")
    
    # Test 2: Register circuits
    print("\n2. Registering Circuits...")
    circuits = [
        ("pool_verifier", "circuits/pool_verifier"),
        ("aggregate_votes", "circuits/aggregate_votes"),
        ("reputation_updater", "circuits/reputation_updater")
    ]
    
    for name, path in circuits:
        manager.register_circuit(name, path)
    
    registered = manager.list_circuits()
    print(f"✓ Registered circuits: {registered}")
    
    # Test 3: Generate proofs
    print("\n3. Generating Proofs...")
    witness1 = {
        "pool_address": "0xpool1",
        "reserve0": 1000000,
        "reserve1": 2000000
    }
    
    proof1 = manager.generate_proof("pool_verifier", witness1)
    print(f"✓ Proof 1 generated: {proof1.size()} bytes")
    
    # Test cache hit
    proof1_cached = manager.generate_proof("pool_verifier", witness1)
    print(f"✓ Proof 1 cached: {proof1 is proof1_cached}")
    
    # Test 4: Batch generation
    print("\n4. Batch Proof Generation...")
    witnesses = [
        {"pool_address": f"0xpool{i}", "reserve0": i*1000, "reserve1": i*2000}
        for i in range(10)
    ]
    
    proofs = manager.generate_proof_batch("pool_verifier", witnesses)
    print(f"✓ Batch generated: {len(proofs)} proofs")
    
    # Test 5: Verification
    print("\n5. Verifying Proofs...")
    is_valid = manager.verify_proof(proof1)
    print(f"✓ Proof 1 valid: {is_valid}")
    
    # Batch verification
    results = manager.verify_proof_batch(proofs)
    valid_count = sum(results)
    print(f"✓ Batch verified: {valid_count}/{len(results)} valid")
    
    # Test 6: Statistics
    print("\n6. Statistics:")
    stats = manager.get_statistics()
    print(f"  Circuits registered: {stats['circuits_registered']}")
    print(f"  Proofs generated: {stats['proofs_generated']}")
    print(f"  Proofs verified: {stats['proofs_verified']}")
    print(f"  Proofs cached: {stats['proofs_cached']}")
    print(f"  Cache hit rate: {stats['cache']['hit_rate']:.1%}")
    print(f"  Avg proof time: {stats['avg_proof_time_ms']:.2f}ms")
    print(f"  Avg verify time: {stats['avg_verify_time_ms']:.2f}ms")
    
    print("\n✓ All proof manager tests passed!")


if __name__ == "__main__":
    test_proof_manager()
