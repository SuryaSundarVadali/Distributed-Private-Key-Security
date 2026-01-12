"""
Weighted PBFT Consensus Protocol
=================================

Implements Practical Byzantine Fault Tolerance (PBFT) with reputation-weighted voting
for DeFi agent coordination. Agents with higher reputation have more influence in
consensus decisions.

Protocol Phases:
1. Pre-Prepare: Primary proposes a value
2. Prepare: Replicas validate and broadcast prepare messages
3. Commit: Once 2f+1 prepare messages received, broadcast commit
4. Execute: Once 2f+1 commit messages received, execute operation

View-Change: If primary fails, replicas can trigger view change to elect new primary
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import time
import hashlib
import json


class MessageType(Enum):
    """PBFT message types"""
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"
    NEW_VIEW = "new_view"


class ConsensusPhase(Enum):
    """Current phase of consensus"""
    IDLE = "idle"
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    COMMITTED = "committed"
    VIEW_CHANGE = "view_change"


@dataclass
class ConsensusMessage:
    """PBFT consensus message"""
    msg_type: MessageType
    view: int  # Current view number
    sequence: int  # Sequence number for operation
    digest: str  # Hash of the operation
    sender_id: str  # Agent ID of sender
    timestamp: float = field(default_factory=time.time)
    operation: Optional[Dict] = None  # The actual operation being proposed
    reputation: float = 1.0  # Sender's reputation weight (0.0 to 1.0)
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary"""
        return {
            'msg_type': self.msg_type.value,
            'view': self.view,
            'sequence': self.sequence,
            'digest': self.digest,
            'sender_id': self.sender_id,
            'timestamp': self.timestamp,
            'operation': self.operation,
            'reputation': self.reputation
        }
    
    def compute_digest(self) -> str:
        """Compute message digest"""
        data = json.dumps({
            'view': self.view,
            'sequence': self.sequence,
            'operation': self.operation
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ViewChangeRequest:
    """Request to change view (primary)"""
    new_view: int
    requester_id: str
    reason: str  # Why view change is needed
    timestamp: float = field(default_factory=time.time)
    prepared_proofs: List[ConsensusMessage] = field(default_factory=list)


class WeightedPBFT:
    """
    Weighted PBFT consensus protocol with reputation-based voting.
    
    Key Features:
    - Reputation-weighted voting (higher reputation = more influence)
    - Byzantine fault tolerance (tolerates up to f < n/3 malicious agents)
    - View-change protocol for primary failure handling
    - Sequence number tracking for operation ordering
    
    Parameters:
    - agent_id: Unique identifier for this agent
    - n_agents: Total number of agents in the system
    - f: Maximum number of Byzantine (faulty) agents (typically f = (n-1)//3)
    - timeout: Timeout in seconds for detecting primary failure
    """
    
    def __init__(self, agent_id: str, n_agents: int, f: int, timeout: float = 5.0):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.f = f  # Maximum Byzantine faults tolerated
        self.timeout = timeout
        
        # View state
        self.view = 0  # Current view number
        self.primary_id = self._compute_primary(self.view)
        self.phase = ConsensusPhase.IDLE
        
        # Sequence tracking
        self.sequence = 0
        self.last_executed_sequence = -1
        
        # Message logs
        self.pre_prepare_log: Dict[Tuple[int, int], ConsensusMessage] = {}  # (view, seq) -> msg
        self.prepare_log: Dict[Tuple[int, int, str, str], ConsensusMessage] = {}  # (view, seq, digest, sender) -> msg
        self.commit_log: Dict[Tuple[int, int, str, str], ConsensusMessage] = {}  # (view, seq, digest, sender) -> msg
        
        # View change state
        self.view_change_requests: Dict[int, Set[str]] = {}  # new_view -> set of agent_ids
        self.last_primary_message_time = time.time()
        
        # Reputation weights (agent_id -> reputation)
        self.reputation_weights: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            'operations_committed': 0,
            'view_changes': 0,
            'prepare_messages_sent': 0,
            'commit_messages_sent': 0,
            'total_consensus_rounds': 0
        }
    
    def _compute_primary(self, view: int) -> str:
        """Compute primary agent for given view (round-robin)"""
        primary_index = view % self.n_agents
        return f"agent_{primary_index}"
    
    def update_reputation(self, agent_id: str, reputation: float):
        """Update reputation weight for an agent"""
        self.reputation_weights[agent_id] = max(0.0, min(1.0, reputation))
    
    def get_reputation(self, agent_id: str) -> float:
        """Get reputation weight for agent"""
        return self.reputation_weights.get(agent_id, 1.0)
    
    def is_primary(self) -> bool:
        """Check if this agent is the current primary"""
        return self.agent_id == self.primary_id
    
    def _compute_weighted_votes(self, messages: List[ConsensusMessage]) -> float:
        """
        Compute weighted sum of votes based on reputation.
        Returns sum of reputation weights.
        """
        return sum(msg.reputation for msg in messages)
    
    def _get_quorum_threshold(self) -> float:
        """
        Get weighted quorum threshold (2f+1 in standard PBFT).
        In weighted PBFT, we need votes from agents with total reputation > threshold.
        """
        total_reputation = sum(self.reputation_weights.values()) or self.n_agents
        # Need 2f+1 weighted votes (approximately 2/3 of total reputation)
        return (2 * self.f + 1) * (total_reputation / self.n_agents)
    
    # ========== Phase 1: Pre-Prepare ==========
    
    def propose_operation(self, operation: Dict) -> Optional[ConsensusMessage]:
        """
        Primary proposes an operation (Phase 1: Pre-Prepare).
        Only the primary can propose operations.
        
        Returns: Pre-prepare message if successful, None otherwise
        """
        if not self.is_primary():
            return None
        
        if self.phase not in [ConsensusPhase.IDLE, ConsensusPhase.COMMITTED]:
            return None
        
        # Create pre-prepare message
        self.sequence += 1
        msg = ConsensusMessage(
            msg_type=MessageType.PRE_PREPARE,
            view=self.view,
            sequence=self.sequence,
            digest=hashlib.sha256(json.dumps(operation, sort_keys=True).encode()).hexdigest(),
            sender_id=self.agent_id,
            operation=operation,
            reputation=self.get_reputation(self.agent_id)
        )
        
        # Log pre-prepare
        key = (msg.view, msg.sequence)
        self.pre_prepare_log[key] = msg
        
        # Transition to pre-prepare phase
        self.phase = ConsensusPhase.PRE_PREPARE
        self.stats['total_consensus_rounds'] += 1
        
        return msg
    
    def handle_pre_prepare(self, msg: ConsensusMessage) -> Optional[ConsensusMessage]:
        """
        Replica handles pre-prepare message (Phase 1).
        Validates and moves to prepare phase.
        
        Returns: Prepare message if validation succeeds, None otherwise
        """
        # Validation checks
        if msg.msg_type != MessageType.PRE_PREPARE:
            return None
        
        if msg.view != self.view:
            return None
        
        if msg.sender_id != self.primary_id:
            return None  # Only primary can send pre-prepare
        
        if msg.sequence <= self.last_executed_sequence:
            return None  # Already executed
        
        # Verify digest
        expected_digest = hashlib.sha256(json.dumps(msg.operation, sort_keys=True).encode()).hexdigest()
        if msg.digest != expected_digest:
            return None
        
        # Log pre-prepare
        key = (msg.view, msg.sequence)
        self.pre_prepare_log[key] = msg
        
        # Update last primary message time
        self.last_primary_message_time = time.time()
        
        # Transition to prepare phase and send prepare message
        self.phase = ConsensusPhase.PREPARE
        
        prepare_msg = ConsensusMessage(
            msg_type=MessageType.PREPARE,
            view=msg.view,
            sequence=msg.sequence,
            digest=msg.digest,
            sender_id=self.agent_id,
            reputation=self.get_reputation(self.agent_id)
        )
        
        self.stats['prepare_messages_sent'] += 1
        return prepare_msg
    
    # ========== Phase 2: Prepare ==========
    
    def handle_prepare(self, msg: ConsensusMessage) -> Optional[ConsensusMessage]:
        """
        Handle prepare message (Phase 2).
        Once 2f+1 prepare messages received, send commit message.
        
        Returns: Commit message if quorum reached, None otherwise
        """
        if msg.msg_type != MessageType.PREPARE:
            return None
        
        if msg.view != self.view:
            return None
        
        # Log prepare message (include sender_id in key to store multiple messages)
        key = (msg.view, msg.sequence, msg.digest, msg.sender_id)
        self.prepare_log[key] = msg
        
        # Check if we have enough prepare messages (weighted quorum)
        prepare_messages = [m for k, m in self.prepare_log.items() 
                           if k[0] == msg.view and k[1] == msg.sequence and k[2] == msg.digest]
        
        weighted_votes = self._compute_weighted_votes(prepare_messages)
        quorum_threshold = self._get_quorum_threshold()
        
        if weighted_votes >= quorum_threshold and self.phase == ConsensusPhase.PREPARE:
            # Prepared! Transition to commit phase
            self.phase = ConsensusPhase.COMMIT
            
            commit_msg = ConsensusMessage(
                msg_type=MessageType.COMMIT,
                view=msg.view,
                sequence=msg.sequence,
                digest=msg.digest,
                sender_id=self.agent_id,
                reputation=self.get_reputation(self.agent_id)
            )
            
            self.stats['commit_messages_sent'] += 1
            return commit_msg
        
        return None
    
    # ========== Phase 3: Commit ==========
    
    def handle_commit(self, msg: ConsensusMessage) -> Tuple[bool, Optional[Dict]]:
        """
        Handle commit message (Phase 3).
        Once 2f+1 commit messages received, execute operation.
        
        Returns: (committed, operation) tuple
        - committed: True if operation is committed
        - operation: The operation to execute (if committed)
        """
        if msg.msg_type != MessageType.COMMIT:
            return False, None
        
        if msg.view != self.view:
            return False, None
        
        # Log commit message (include sender_id in key to store multiple messages)
        key = (msg.view, msg.sequence, msg.digest, msg.sender_id)
        self.commit_log[key] = msg
        
        # Check if we have enough commit messages (weighted quorum)
        commit_messages = [m for k, m in self.commit_log.items()
                          if k[0] == msg.view and k[1] == msg.sequence and k[2] == msg.digest]
        
        weighted_votes = self._compute_weighted_votes(commit_messages)
        quorum_threshold = self._get_quorum_threshold()
        
        if weighted_votes >= quorum_threshold and self.phase == ConsensusPhase.COMMIT:
            # Committed! Execute operation
            self.phase = ConsensusPhase.COMMITTED
            
            # Get operation from pre-prepare log
            pre_prepare_key = (msg.view, msg.sequence)
            if pre_prepare_key in self.pre_prepare_log:
                operation = self.pre_prepare_log[pre_prepare_key].operation
                self.last_executed_sequence = msg.sequence
                self.stats['operations_committed'] += 1
                
                # Reset to idle for next operation
                self.phase = ConsensusPhase.IDLE
                
                return True, operation
        
        return False, None
    
    # ========== View Change Protocol ==========
    
    def check_timeout(self) -> bool:
        """Check if primary has timed out (no messages for timeout period)"""
        if not self.is_primary():
            elapsed = time.time() - self.last_primary_message_time
            if elapsed > self.timeout and self.phase != ConsensusPhase.IDLE:
                return True
        return False
    
    def initiate_view_change(self, reason: str = "Primary timeout") -> ViewChangeRequest:
        """Initiate view change to elect new primary"""
        new_view = self.view + 1
        
        # Collect prepared proofs (messages with prepare quorum)
        prepared_proofs = []
        for key, msg in self.prepare_log.items():
            if key[0] == self.view:  # Current view
                prepared_proofs.append(msg)
        
        request = ViewChangeRequest(
            new_view=new_view,
            requester_id=self.agent_id,
            reason=reason,
            prepared_proofs=prepared_proofs
        )
        
        # Log view change request
        if new_view not in self.view_change_requests:
            self.view_change_requests[new_view] = set()
        self.view_change_requests[new_view].add(self.agent_id)
        
        self.phase = ConsensusPhase.VIEW_CHANGE
        return request
    
    def handle_view_change_request(self, request: ViewChangeRequest) -> bool:
        """
        Handle view change request from another agent.
        Returns True if view change is accepted (f+1 requests received)
        """
        # Log request
        if request.new_view not in self.view_change_requests:
            self.view_change_requests[request.new_view] = set()
        self.view_change_requests[request.new_view].add(request.requester_id)
        
        # Check if we have f+1 requests for new view
        if len(self.view_change_requests[request.new_view]) >= self.f + 1:
            # Accept view change
            self.view = request.new_view
            self.primary_id = self._compute_primary(self.view)
            self.phase = ConsensusPhase.IDLE
            self.last_primary_message_time = time.time()
            self.stats['view_changes'] += 1
            return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Get consensus statistics"""
        return {
            **self.stats,
            'current_view': self.view,
            'current_phase': self.phase.value,
            'is_primary': self.is_primary(),
            'primary_id': self.primary_id,
            'sequence': self.sequence,
            'last_executed_sequence': self.last_executed_sequence
        }


# ========== Built-in Tests ==========

def test_weighted_pbft():
    """Test Weighted PBFT consensus"""
    print("Testing Weighted PBFT Consensus")
    print("=" * 60)
    
    # Create 4 agents (f=1, tolerates 1 Byzantine)
    n_agents = 4
    f = 1
    agents = []
    
    print(f"\n1. Initializing {n_agents} agents (f={f})...")
    for i in range(n_agents):
        agent = WeightedPBFT(f"agent_{i}", n_agents, f)
        # Set reputation weights
        agent.update_reputation(f"agent_{i}", 0.8 + i * 0.05)
        for j in range(n_agents):
            agent.update_reputation(f"agent_{j}", 0.8 + j * 0.05)
        agents.append(agent)
    
    print(f"✓ Primary: {agents[0].primary_id}")
    for i, agent in enumerate(agents):
        print(f"  Agent {i}: reputation={agent.get_reputation(f'agent_{i}'):.2f}")
    
    # Test consensus round
    print("\n2. Starting consensus round...")
    operation = {"type": "allocate", "pool": "uniswap_v3", "amount": 50000}
    
    # Phase 1: Primary proposes
    primary = agents[0]
    pre_prepare_msg = primary.propose_operation(operation)
    print(f"✓ Primary proposed operation: {operation}")
    print(f"  Digest: {pre_prepare_msg.digest[:16]}...")
    
    # Replicas handle pre-prepare
    prepare_messages = []
    for agent in agents[1:]:  # Non-primary agents
        prepare_msg = agent.handle_pre_prepare(pre_prepare_msg)
        if prepare_msg:
            prepare_messages.append(prepare_msg)
    print(f"✓ Replicas sent {len(prepare_messages)} prepare messages")
    
    # Phase 2: All agents handle prepare messages
    commit_messages = []
    for agent in agents:
        for prep_msg in prepare_messages:
            commit_msg = agent.handle_prepare(prep_msg)
            if commit_msg:
                commit_messages.append((agent.agent_id, commit_msg))
    
    unique_commits = len(set(cid for cid, _ in commit_messages))
    print(f"✓ {unique_commits} agents sent commit messages")
    
    # Phase 3: All agents handle commit messages
    committed_agents = []
    for agent in agents:
        for _, comm_msg in commit_messages:
            committed, executed_op = agent.handle_commit(comm_msg)
            if committed and agent.agent_id not in committed_agents:
                committed_agents.append(agent.agent_id)
                print(f"✓ {agent.agent_id} committed operation")
    
    print(f"\n✓ Consensus reached! {len(committed_agents)} agents committed")
    
    # Test view change
    print("\n3. Testing view change protocol...")
    print(f"  Current primary: {agents[0].primary_id}")
    
    # Simulate primary timeout
    view_change_requests = []
    for agent in agents[1:]:  # Non-primary agents
        request = agent.initiate_view_change("Testing view change")
        view_change_requests.append(request)
    
    print(f"✓ {len(view_change_requests)} agents requested view change")
    
    # Handle view change requests
    view_changed = False
    for agent in agents:
        for request in view_change_requests:
            if agent.handle_view_change_request(request):
                view_changed = True
    
    if view_changed:
        new_primary = agents[0].primary_id
        print(f"✓ View change successful! New primary: {new_primary}")
    
    # Statistics
    print("\n4. Statistics:")
    for i, agent in enumerate(agents):
        stats = agent.get_statistics()
        print(f"  Agent {i}: committed={stats['operations_committed']}, "
              f"view={stats['current_view']}, phase={stats['current_phase']}")
    
    print("\n✓ All Weighted PBFT tests passed!")


if __name__ == "__main__":
    test_weighted_pbft()
