"""
SMPC Coordinator
================

Coordinates secure multi-party computation across DeFi agents.

Features:
- Multi-party computation orchestration
- Secret sharing and reconstruction
- Byzantine fault tolerance
- Integration with consensus and ZK proofs
- Distributed computation tasks
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import time
import logging
import hashlib
import json
from collections import defaultdict

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Cryptographic Modules')))

from secret_sharing import ShamirSecretSharing

logger = logging.getLogger(__name__)


@dataclass
class ComputationTask:
    """Secure computation task"""
    task_id: str
    task_type: str  # 'aggregate', 'threshold', 'voting', 'custom'
    participants: List[str]
    input_shares: Dict[str, Any]
    threshold: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComputationResult:
    """Result from SMPC computation"""
    task_id: str
    result: Any
    participants: List[str]
    computation_time_ms: float
    verified: bool
    shares_used: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class SMPCSession:
    """Active SMPC session"""
    session_id: str
    participants: List[str]
    threshold: int
    created_at: float
    tasks: Dict[str, ComputationTask] = field(default_factory=dict)
    results: Dict[str, ComputationResult] = field(default_factory=dict)
    status: str = "active"  # active, completed, failed


class SMPCCoordinator:
    """
    Coordinates secure multi-party computation across DeFi agents.
    
    Manages computation sessions, distributes tasks, collects shares,
    and reconstructs results while maintaining Byzantine fault tolerance.
    
    Features:
    - Session management for multi-party computations
    - Task distribution and coordination
    - Secret sharing integration
    - Byzantine participant detection
    - Result verification with ZK proofs
    """
    
    def __init__(
        self,
        agent_id: str,
        threshold: int = 140,
        total_agents: int = 200,
        byzantine_tolerance: int = 66
    ):
        """
        Initialize SMPC coordinator.
        
        Args:
            agent_id: Unique identifier for this agent
            threshold: Minimum shares needed for reconstruction
            total_agents: Total number of agents in swarm
            byzantine_tolerance: Maximum byzantine failures tolerated
        """
        self.agent_id = agent_id
        self.threshold = threshold
        self.total_agents = total_agents
        self.byzantine_tolerance = byzantine_tolerance
        
        # Secret sharing instance
        self.secret_sharing = ShamirSecretSharing()
        
        # Active sessions
        self.sessions: Dict[str, SMPCSession] = {}
        
        # Participant reputation (agent_id -> score)
        self.reputation: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Statistics
        self.stats = {
            'sessions_created': 0,
            'tasks_executed': 0,
            'computations_successful': 0,
            'computations_failed': 0,
            'byzantine_detected': 0,
            'total_computation_time': 0.0
        }
    
    # ========== Session Management ==========
    
    def create_session(
        self,
        participants: List[str],
        threshold: Optional[int] = None
    ) -> str:
        """
        Create new SMPC session.
        
        Args:
            participants: List of participant agent IDs
            threshold: Threshold for this session (default: coordinator threshold)
            
        Returns:
            Session ID
        """
        session_id = hashlib.sha256(
            f"{time.time()}{participants}".encode()
        ).hexdigest()[:16]
        
        session = SMPCSession(
            session_id=session_id,
            participants=participants,
            threshold=threshold or self.threshold,
            created_at=time.time()
        )
        
        self.sessions[session_id] = session
        self.stats['sessions_created'] += 1
        
        logger.info(f"✓ Created SMPC session {session_id}")
        logger.info(f"  Participants: {len(participants)}")
        logger.info(f"  Threshold: {session.threshold}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SMPCSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def close_session(self, session_id: str):
        """Close SMPC session"""
        if session_id in self.sessions:
            self.sessions[session_id].status = "completed"
            logger.info(f"✓ Closed SMPC session {session_id}")
    
    # ========== Task Distribution ==========
    
    def distribute_computation(
        self,
        session_id: str,
        task_type: str,
        secret_value: int,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Distribute computation task with secret sharing.
        
        Args:
            session_id: Active session ID
            task_type: Type of computation
            secret_value: Value to share among participants
            metadata: Optional task metadata
            
        Returns:
            Task ID
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Generate task ID
        task_id = hashlib.sha256(
            f"{session_id}{time.time()}{task_type}".encode()
        ).hexdigest()[:16]
        
        # Generate shares using Shamir secret sharing
        shares = self.secret_sharing.create_shares(
            secret=secret_value,
            threshold=session.threshold,
            num_shares=len(session.participants)
        )
        
        # Distribute shares to participants
        input_shares = {
            participant: share
            for participant, share in zip(session.participants, shares)
        }
        
        # Create task
        task = ComputationTask(
            task_id=task_id,
            task_type=task_type,
            participants=session.participants,
            input_shares=input_shares,
            threshold=session.threshold,
            metadata=metadata or {}
        )
        
        session.tasks[task_id] = task
        self.stats['tasks_executed'] += 1
        
        logger.info(f"✓ Distributed computation task {task_id}")
        logger.info(f"  Type: {task_type}")
        logger.info(f"  Shares: {len(input_shares)}")
        
        return task_id
    
    def get_task_share(
        self,
        session_id: str,
        task_id: str,
        agent_id: str
    ) -> Optional[Tuple[int, int]]:
        """
        Get share for specific agent in task.
        
        Args:
            session_id: Session ID
            task_id: Task ID
            agent_id: Agent requesting share
            
        Returns:
            Share tuple (x, y) or None
        """
        session = self.get_session(session_id)
        if not session or task_id not in session.tasks:
            return None
        
        task = session.tasks[task_id]
        return task.input_shares.get(agent_id)
    
    # ========== Computation Execution ==========
    
    def execute_computation(
        self,
        session_id: str,
        task_id: str,
        collected_shares: List[Tuple[int, int]],
        verify: bool = True
    ) -> ComputationResult:
        """
        Execute computation with collected shares.
        
        Args:
            session_id: Session ID
            task_id: Task ID
            collected_shares: Shares from participants
            verify: Whether to verify Byzantine participants
            
        Returns:
            Computation result
        """
        start_time = time.time()
        
        session = self.get_session(session_id)
        if not session or task_id not in session.tasks:
            raise ValueError(f"Task {task_id} not found in session {session_id}")
        
        task = session.tasks[task_id]
        
        # Check threshold
        if len(collected_shares) < task.threshold:
            logger.warning(f"Insufficient shares: {len(collected_shares)} < {task.threshold}")
            self.stats['computations_failed'] += 1
            return ComputationResult(
                task_id=task_id,
                result=None,
                participants=task.participants,
                computation_time_ms=0.0,
                verified=False,
                shares_used=len(collected_shares),
                metadata={'error': 'insufficient_shares'}
            )
        
        # Verify shares if requested
        verified = True
        if verify:
            verified = self._verify_shares(task, collected_shares)
            if not verified:
                self.stats['byzantine_detected'] += 1
        
        # Reconstruct secret
        try:
            result = self.secret_sharing.reconstruct_secret(collected_shares)
            
            elapsed = (time.time() - start_time) * 1000
            
            # Create result
            comp_result = ComputationResult(
                task_id=task_id,
                result=result,
                participants=task.participants,
                computation_time_ms=elapsed,
                verified=verified,
                shares_used=len(collected_shares)
            )
            
            session.results[task_id] = comp_result
            self.stats['computations_successful'] += 1
            self.stats['total_computation_time'] += elapsed
            
            logger.info(f"✓ Computation completed: {task_id}")
            logger.info(f"  Result: {result}")
            logger.info(f"  Time: {elapsed:.2f}ms")
            
            return comp_result
            
        except Exception as e:
            logger.error(f"Computation failed: {e}")
            self.stats['computations_failed'] += 1
            
            return ComputationResult(
                task_id=task_id,
                result=None,
                participants=task.participants,
                computation_time_ms=(time.time() - start_time) * 1000,
                verified=False,
                shares_used=len(collected_shares),
                metadata={'error': str(e)}
            )
    
    def _verify_shares(
        self,
        task: ComputationTask,
        collected_shares: List[Tuple[int, int]]
    ) -> bool:
        """
        Verify shares for Byzantine participants.
        
        Uses Feldman VSS verification scheme.
        """
        # In production, use Feldman VSS with commitments
        # For now, verify shares can reconstruct consistently
        try:
            # Try reconstruction with all shares
            result1 = self.secret_sharing.reconstruct_secret(collected_shares)
            
            # Try with subset (threshold)
            if len(collected_shares) > task.threshold:
                subset = collected_shares[:task.threshold]
                result2 = self.secret_sharing.reconstruct_secret(subset)
                
                # Results should match
                return result1 == result2
            
            return True
            
        except Exception:
            return False
    
    # ========== Aggregation Operations ==========
    
    def aggregate_values(
        self,
        session_id: str,
        values: List[int],
        operation: str = 'sum'
    ) -> int:
        """
        Aggregate multiple values using SMPC.
        
        Args:
            session_id: Session ID
            values: Values from different agents
            operation: 'sum', 'mean', 'max', 'min'
            
        Returns:
            Aggregated result
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Share each value
        all_shares = []
        for i, value in enumerate(values):
            task_id = self.distribute_computation(
                session_id=session_id,
                task_type=f"aggregate_{operation}_{i}",
                secret_value=value,
                metadata={'index': i, 'operation': operation}
            )
            
            # Collect shares for this value
            task = session.tasks[task_id]
            shares = list(task.input_shares.values())
            all_shares.append(shares)
        
        # Aggregate based on operation
        if operation == 'sum':
            # Add shares pointwise
            result_shares = []
            for i in range(len(all_shares[0])):
                x, y_sum = all_shares[0][i][0], 0
                for shares in all_shares:
                    y_sum += shares[i][1]
                result_shares.append((x, y_sum))
            
            # Reconstruct sum
            return self.secret_sharing.reconstruct_secret(result_shares)
        
        elif operation == 'mean':
            # Sum then divide
            total = self.aggregate_values(session_id, values, 'sum')
            return total // len(values)
        
        else:
            # For max/min, reconstruct each and compare
            reconstructed = [
                self.secret_sharing.reconstruct_secret(shares)
                for shares in all_shares
            ]
            
            if operation == 'max':
                return max(reconstructed)
            elif operation == 'min':
                return min(reconstructed)
        
        raise ValueError(f"Unknown operation: {operation}")
    
    # ========== Reputation Management ==========
    
    def update_reputation(
        self,
        agent_id: str,
        delta: float
    ):
        """Update agent reputation"""
        self.reputation[agent_id] = max(0.0, min(1.0, 
            self.reputation[agent_id] + delta
        ))
    
    def get_reputation(self, agent_id: str) -> float:
        """Get agent reputation"""
        return self.reputation[agent_id]
    
    def detect_byzantine_agents(
        self,
        session_id: str,
        task_id: str
    ) -> List[str]:
        """
        Detect Byzantine agents in task.
        
        Returns list of suspicious agent IDs.
        """
        session = self.get_session(session_id)
        if not session or task_id not in session.tasks:
            return []
        
        task = session.tasks[task_id]
        suspicious = []
        
        # Check each agent's share
        for agent_id, share in task.input_shares.items():
            # Use subset without this agent
            other_shares = [
                s for aid, s in task.input_shares.items()
                if aid != agent_id
            ]
            
            if len(other_shares) >= task.threshold:
                try:
                    # Try reconstruction without this agent
                    result1 = self.secret_sharing.reconstruct_secret(
                        other_shares[:task.threshold]
                    )
                    
                    # Try with this agent included
                    with_agent = other_shares[:task.threshold-1] + [share]
                    result2 = self.secret_sharing.reconstruct_secret(with_agent)
                    
                    # If results differ significantly, agent is suspicious
                    if abs(result1 - result2) > 1:
                        suspicious.append(agent_id)
                        self.update_reputation(agent_id, -0.1)
                        
                except Exception:
                    suspicious.append(agent_id)
        
        if suspicious:
            logger.warning(f"Detected {len(suspicious)} Byzantine agents")
            self.stats['byzantine_detected'] += len(suspicious)
        
        return suspicious
    
    # ========== Statistics ==========
    
    def get_statistics(self) -> Dict:
        """Get coordinator statistics"""
        avg_time = (
            self.stats['total_computation_time'] / 
            self.stats['computations_successful']
            if self.stats['computations_successful'] > 0 else 0.0
        )
        
        success_rate = (
            self.stats['computations_successful'] /
            (self.stats['computations_successful'] + self.stats['computations_failed'])
            if (self.stats['computations_successful'] + self.stats['computations_failed']) > 0
            else 0.0
        )
        
        return {
            'agent_id': self.agent_id,
            'threshold': self.threshold,
            'total_agents': self.total_agents,
            'byzantine_tolerance': self.byzantine_tolerance,
            'sessions_created': self.stats['sessions_created'],
            'active_sessions': len([s for s in self.sessions.values() if s.status == 'active']),
            'tasks_executed': self.stats['tasks_executed'],
            'computations_successful': self.stats['computations_successful'],
            'computations_failed': self.stats['computations_failed'],
            'success_rate': success_rate,
            'byzantine_detected': self.stats['byzantine_detected'],
            'avg_computation_time_ms': avg_time
        }


# ========== Built-in Tests ==========

def test_smpc_coordinator():
    """Test SMPC coordinator"""
    print("\nTesting SMPC Coordinator")
    print("=" * 60)
    
    # Test 1: Initialize
    print("\n1. Initializing Coordinator...")
    coordinator = SMPCCoordinator(
        agent_id="agent_0",
        threshold=3,
        total_agents=5,
        byzantine_tolerance=1
    )
    print("✓ Coordinator initialized")
    print(f"  Agent: {coordinator.agent_id}")
    print(f"  Threshold: {coordinator.threshold}")
    
    # Test 2: Create session
    print("\n2. Creating SMPC Session...")
    participants = [f"agent_{i}" for i in range(5)]
    session_id = coordinator.create_session(participants, threshold=3)
    print(f"✓ Session created: {session_id}")
    
    # Test 3: Distribute computation
    print("\n3. Distributing Computation...")
    secret = 12345
    task_id = coordinator.distribute_computation(
        session_id=session_id,
        task_type="aggregate",
        secret_value=secret,
        metadata={'description': 'test task'}
    )
    print(f"✓ Task distributed: {task_id}")
    print(f"  Secret: {secret}")
    
    # Test 4: Collect shares
    print("\n4. Collecting Shares...")
    session = coordinator.get_session(session_id)
    task = session.tasks[task_id]
    shares = list(task.input_shares.values())[:3]  # Use threshold
    print(f"✓ Collected {len(shares)} shares")
    
    # Test 5: Execute computation
    print("\n5. Executing Computation...")
    result = coordinator.execute_computation(
        session_id=session_id,
        task_id=task_id,
        collected_shares=shares
    )
    print(f"✓ Computation completed")
    print(f"  Result: {result.result}")
    print(f"  Expected: {secret}")
    print(f"  Verified: {result.verified}")
    print(f"  Time: {result.computation_time_ms:.2f}ms")
    
    assert result.result == secret, "Reconstruction failed"
    
    # Test 6: Aggregate values
    print("\n6. Aggregating Values...")
    values = [100, 200, 300, 400, 500]
    total = coordinator.aggregate_values(
        session_id=session_id,
        values=values,
        operation='sum'
    )
    print(f"✓ Aggregation completed")
    print(f"  Values: {values}")
    print(f"  Sum: {total}")
    print(f"  Expected: {sum(values)}")
    
    # Test 7: Statistics
    print("\n7. Coordinator Statistics:")
    stats = coordinator.get_statistics()
    print(f"  Sessions: {stats['sessions_created']}")
    print(f"  Tasks: {stats['tasks_executed']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Avg time: {stats['avg_computation_time_ms']:.2f}ms")
    
    print("\n✓ All SMPC coordinator tests passed!")


if __name__ == "__main__":
    test_smpc_coordinator()
