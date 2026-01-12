"""
Secure Multi-Party Computation (SMPC) Module
=============================================

Phase 4 implementation - Distributed computation with ZK proofs.

Components:
- smpc_coordinator: Orchestrates multi-party computation
- zk_smpc_integration: Integrates ZK proofs with SMPC
- distributed_key_generation: Secure distributed key generation
"""

from .smpc_coordinator import (
    SMPCCoordinator,
    ComputationTask,
    ComputationResult,
    SMPCSession
)

from .zk_smpc_integration import (
    ZKSMPCIntegration,
    VerifiableComputation,
    PrivateInput
)

from .distributed_key_generation import (
    DistributedKeyGenerator,
    KeyShare,
    DistributedKey,
    KeyRefreshRequest
)

__all__ = [
    # SMPC Coordinator
    'SMPCCoordinator',
    'ComputationTask',
    'ComputationResult',
    'SMPCSession',
    
    # ZK-SMPC Integration
    'ZKSMPCIntegration',
    'VerifiableComputation',
    'PrivateInput',
    
    # Distributed Key Generation
    'DistributedKeyGenerator',
    'KeyShare',
    'DistributedKey',
    'KeyRefreshRequest'
]
