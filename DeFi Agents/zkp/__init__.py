"""
Zero-Knowledge Proof (ZKP) Module
===================================

Phase 3 implementation - Aztec Noir integration for DeFi agents.

Components:
- proof_manager: ZK proof lifecycle management with caching and batching
- pool_verifier: Pool state verification with ZK proofs
"""

from .proof_manager import (
    ProofCache,
    ProofBatcher,
    ProofRequest,
    ZKProofManager
)

from .pool_verifier import (
    PoolState,
    VerifiedPoolState,
    PoolVerifier
)

__all__ = [
    # Proof Manager
    'ProofCache',
    'ProofBatcher',
    'ProofRequest',
    'ZKProofManager',
    
    # Pool Verifier
    'PoolState',
    'VerifiedPoolState',
    'PoolVerifier'
]
