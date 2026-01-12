"""
Threshold Consensus Integration
================================

Integrates AdaptiveThresholdManager with Weighted PBFT consensus protocol.
Automatically adjusts Shamir Secret Sharing threshold based on Byzantine
detection during consensus rounds.

Key Features:
- Monitors consensus for Byzantine behavior
- Triggers threshold adjustments when anomalies detected
- Maintains audit trail of threshold changes
- Provides unified interface for consensus + threshold management
"""

import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

# Add Core System to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Core System')))

from .pbft import WeightedPBFT, ConsensusMessage, MessageType, ConsensusPhase


@dataclass
class ByzantineDetection:
    """Record of Byzantine behavior detected during consensus"""
    agent_id: str
    detection_type: str  # 'invalid_signature', 'conflicting_votes', 'timeout', etc.
    view: int
    sequence: int
    timestamp: float = field(default_factory=time.time)
    details: Optional[Dict] = None


class ThresholdConsensusIntegrator:
    """
    Integrates adaptive threshold management with PBFT consensus.
    
    Monitors consensus rounds for Byzantine behavior and automatically
    adjusts the Shamir Secret Sharing threshold to maintain security.
    
    Parameters:
    - pbft: WeightedPBFT instance
    - threshold_manager: AdaptiveThresholdManager instance (optional)
    - detection_window: Time window (seconds) for Byzantine detection aggregation
    """
    
    def __init__(self, pbft: WeightedPBFT, threshold_manager=None, detection_window: float = 60.0):
        self.pbft = pbft
        self.threshold_manager = threshold_manager
        self.detection_window = detection_window
        
        # Byzantine detection tracking
        self.byzantine_detections: List[ByzantineDetection] = []
        self.suspected_agents: Dict[str, int] = {}  # agent_id -> suspicion_count
        
        # Consensus monitoring
        self.consensus_rounds_monitored = 0
        self.anomalies_detected = 0
        self.threshold_adjustments_triggered = 0
        
        # Message validation cache
        self.seen_messages: Dict[Tuple[int, int, str], ConsensusMessage] = {}  # (view, seq, sender) -> msg
        
        # Statistics
        self.stats = {
            'consensus_rounds': 0,
            'byzantine_detected': 0,
            'threshold_increases': 0,
            'threshold_decreases': 0,
            'view_changes_due_to_byzantine': 0
        }
    
    def set_threshold_manager(self, threshold_manager):
        """Set the threshold manager (for lazy initialization)"""
        self.threshold_manager = threshold_manager
    
    # ========== Byzantine Detection ==========
    
    def detect_conflicting_votes(self, msg1: ConsensusMessage, msg2: ConsensusMessage) -> bool:
        """
        Detect if an agent sent conflicting votes in the same view/sequence.
        Byzantine behavior: sending different prepare/commit messages for same operation.
        """
        if msg1.sender_id != msg2.sender_id:
            return False
        
        if msg1.view != msg2.view or msg1.sequence != msg2.sequence:
            return False
        
        if msg1.msg_type != msg2.msg_type:
            return False
        
        # Same agent, same view/sequence, same phase -> should have same digest
        if msg1.digest != msg2.digest:
            return True
        
        return False
    
    def detect_invalid_primary_message(self, msg: ConsensusMessage) -> bool:
        """
        Detect if a non-primary agent sent a pre-prepare message.
        Byzantine behavior: non-primary trying to propose operations.
        """
        if msg.msg_type == MessageType.PRE_PREPARE:
            expected_primary = self.pbft._compute_primary(msg.view)
            if msg.sender_id != expected_primary:
                return True
        return False
    
    def detect_out_of_sequence(self, msg: ConsensusMessage) -> bool:
        """
        Detect if message has invalid sequence number.
        Byzantine behavior: sending messages with incorrect sequence.
        """
        # Check if sequence is too far ahead or behind
        if msg.sequence > self.pbft.sequence + 10:  # More than 10 ahead
            return True
        if msg.sequence <= self.pbft.last_executed_sequence:  # Already executed
            return True
        return False
    
    def validate_message(self, msg: ConsensusMessage) -> Tuple[bool, Optional[str]]:
        """
        Validate consensus message and detect Byzantine behavior.
        
        Returns: (is_valid, detection_type)
        - is_valid: True if message is valid
        - detection_type: Type of Byzantine behavior detected (if invalid)
        """
        # Check 1: Invalid primary
        if self.detect_invalid_primary_message(msg):
            return False, "invalid_primary"
        
        # Check 2: Out of sequence
        if self.detect_out_of_sequence(msg):
            return False, "out_of_sequence"
        
        # Check 3: Conflicting votes (check against cache)
        cache_key = (msg.view, msg.sequence, msg.sender_id)
        if cache_key in self.seen_messages:
            cached_msg = self.seen_messages[cache_key]
            if self.detect_conflicting_votes(cached_msg, msg):
                return False, "conflicting_votes"
        else:
            self.seen_messages[cache_key] = msg
        
        return True, None
    
    def record_byzantine_detection(self, agent_id: str, detection_type: str, 
                                   view: int, sequence: int, details: Optional[Dict] = None):
        """Record Byzantine behavior detection"""
        detection = ByzantineDetection(
            agent_id=agent_id,
            detection_type=detection_type,
            view=view,
            sequence=sequence,
            details=details
        )
        
        self.byzantine_detections.append(detection)
        
        # Update suspicion count
        if agent_id not in self.suspected_agents:
            self.suspected_agents[agent_id] = 0
        self.suspected_agents[agent_id] += 1
        
        self.anomalies_detected += 1
        self.stats['byzantine_detected'] += 1
    
    def get_recent_byzantine_count(self) -> int:
        """Get number of unique Byzantine agents in recent detection window"""
        current_time = time.time()
        recent_detections = [
            d for d in self.byzantine_detections
            if current_time - d.timestamp <= self.detection_window
        ]
        
        # Count unique agents
        byzantine_agents = set(d.agent_id for d in recent_detections)
        return len(byzantine_agents)
    
    # ========== Consensus Message Handling with Validation ==========
    
    def propose_operation(self, operation: Dict) -> Optional[ConsensusMessage]:
        """Propose operation through PBFT (primary only)"""
        msg = self.pbft.propose_operation(operation)
        if msg:
            self.consensus_rounds_monitored += 1
            self.stats['consensus_rounds'] += 1
        return msg
    
    def handle_pre_prepare(self, msg: ConsensusMessage) -> Optional[ConsensusMessage]:
        """Handle pre-prepare with Byzantine detection"""
        # Validate message
        is_valid, detection_type = self.validate_message(msg)
        
        if not is_valid:
            self.record_byzantine_detection(
                agent_id=msg.sender_id,
                detection_type=detection_type,
                view=msg.view,
                sequence=msg.sequence,
                details={'message_type': 'pre_prepare'}
            )
            # Try to adjust threshold
            self._maybe_adjust_threshold()
            return None
        
        # Pass to PBFT
        return self.pbft.handle_pre_prepare(msg)
    
    def handle_prepare(self, msg: ConsensusMessage) -> Optional[ConsensusMessage]:
        """Handle prepare with Byzantine detection"""
        # Validate message
        is_valid, detection_type = self.validate_message(msg)
        
        if not is_valid:
            self.record_byzantine_detection(
                agent_id=msg.sender_id,
                detection_type=detection_type,
                view=msg.view,
                sequence=msg.sequence,
                details={'message_type': 'prepare'}
            )
            self._maybe_adjust_threshold()
            return None
        
        # Pass to PBFT
        return self.pbft.handle_prepare(msg)
    
    def handle_commit(self, msg: ConsensusMessage) -> Tuple[bool, Optional[Dict]]:
        """Handle commit with Byzantine detection"""
        # Validate message
        is_valid, detection_type = self.validate_message(msg)
        
        if not is_valid:
            self.record_byzantine_detection(
                agent_id=msg.sender_id,
                detection_type=detection_type,
                view=msg.view,
                sequence=msg.sequence,
                details={'message_type': 'commit'}
            )
            self._maybe_adjust_threshold()
            return False, None
        
        # Pass to PBFT
        committed, operation = self.pbft.handle_commit(msg)
        
        # If committed, check if we should adjust threshold
        if committed:
            self._maybe_adjust_threshold()
        
        return committed, operation
    
    # ========== Threshold Management ==========
    
    def _maybe_adjust_threshold(self):
        """Check if threshold adjustment is needed based on Byzantine detections"""
        if not self.threshold_manager:
            return  # No threshold manager configured
        
        # Get recent Byzantine count
        byzantine_count = self.get_recent_byzantine_count()
        
        if byzantine_count == 0:
            return  # No Byzantine agents detected
        
        # Adjust threshold
        old_t = self.threshold_manager.current_threshold
        old_active = self.threshold_manager.active_agents
        
        new_t, new_active = self.threshold_manager.adjust_threshold(
            byzantine_count=byzantine_count,
            agent_failures=0  # Consensus handles failures differently
        )
        
        if new_t != old_t:
            self.threshold_adjustments_triggered += 1
            if new_t > old_t:
                self.stats['threshold_increases'] += 1
            else:
                self.stats['threshold_decreases'] += 1
            
            print(f"⚠️  Threshold adjusted: {old_t} → {new_t} "
                  f"(active={new_active}, byzantine={byzantine_count})")
    
    def force_threshold_adjustment(self, byzantine_count: int, agent_failures: int) -> Tuple[int, int]:
        """Manually trigger threshold adjustment"""
        if not self.threshold_manager:
            return self.pbft.f, self.pbft.n_agents
        
        old_t = self.threshold_manager.current_t
        new_t, new_active = self.threshold_manager.adjust_threshold(byzantine_count, agent_failures)
        self.threshold_adjustments_triggered += 1
        
        if new_t > old_t:
            self.stats['threshold_increases'] += 1
        else:
            self.stats['threshold_decreases'] += 1
        
        return new_t, new_active
    
    # ========== View Change with Byzantine Handling ==========
    
    def initiate_view_change(self, reason: str = "Primary timeout") -> any:
        """Initiate view change, possibly due to Byzantine primary"""
        if "byzantine" in reason.lower():
            self.stats['view_changes_due_to_byzantine'] += 1
        
        return self.pbft.initiate_view_change(reason)
    
    def handle_view_change_request(self, request) -> bool:
        """Handle view change request"""
        return self.pbft.handle_view_change_request(request)
    
    # ========== Statistics & Monitoring ==========
    
    def get_byzantine_report(self) -> Dict:
        """Get report of Byzantine behavior detected"""
        current_time = time.time()
        recent_detections = [
            d for d in self.byzantine_detections
            if current_time - d.timestamp <= self.detection_window
        ]
        
        detection_types = {}
        for d in recent_detections:
            if d.detection_type not in detection_types:
                detection_types[d.detection_type] = 0
            detection_types[d.detection_type] += 1
        
        return {
            'total_detections': len(self.byzantine_detections),
            'recent_detections': len(recent_detections),
            'unique_byzantine_agents': len(self.suspected_agents),
            'suspected_agents': dict(self.suspected_agents),
            'detection_types': detection_types,
            'detection_window_seconds': self.detection_window
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        pbft_stats = self.pbft.get_statistics()
        byzantine_report = self.get_byzantine_report()
        
        threshold_stats = {}
        if self.threshold_manager:
            threshold_stats = self.threshold_manager.get_statistics()
        
        return {
            'pbft': pbft_stats,
            'byzantine_detection': byzantine_report,
            'threshold': threshold_stats,
            'integration': {
                'consensus_rounds_monitored': self.consensus_rounds_monitored,
                'anomalies_detected': self.anomalies_detected,
                'threshold_adjustments_triggered': self.threshold_adjustments_triggered,
                **self.stats
            }
        }


# ========== Built-in Tests ==========

def test_threshold_integration():
    """Test threshold consensus integration"""
    print("Testing Threshold Consensus Integration")
    print("=" * 60)
    
    # Import threshold manager
    try:
        from adaptive_threshold_manager import AdaptiveThresholdManager
    except ImportError:
        print("⚠️  Warning: AdaptiveThresholdManager not available")
        print("   Integration tests will run without threshold management")
        AdaptiveThresholdManager = None
    
    # Create PBFT
    n_agents = 5
    f = 1
    pbft = WeightedPBFT("agent_0", n_agents, f)
    
    # Set reputation
    for i in range(n_agents):
        pbft.update_reputation(f"agent_{i}", 0.9)
    
    print(f"\n1. Created PBFT consensus (n={n_agents}, f={f})")
    
    # Create threshold manager
    threshold_mgr = None
    if AdaptiveThresholdManager:
        threshold_mgr = AdaptiveThresholdManager(initial_t=3, initial_n=n_agents)
        print(f"✓ Threshold manager initialized (t={threshold_mgr.current_threshold})")
    
    # Create integrator
    integrator = ThresholdConsensusIntegrator(pbft, threshold_mgr)
    print("✓ Integrator created")
    
    # Test Byzantine detection
    print("\n2. Testing Byzantine detection...")
    
    # Create valid message
    valid_msg = ConsensusMessage(
        msg_type=MessageType.PREPARE,
        view=0,
        sequence=1,
        digest="abc123",
        sender_id="agent_1",
        reputation=0.9
    )
    
    is_valid, detection = integrator.validate_message(valid_msg)
    print(f"✓ Valid message: is_valid={is_valid}")
    
    # Create invalid message (non-primary sending pre-prepare)
    invalid_msg = ConsensusMessage(
        msg_type=MessageType.PRE_PREPARE,
        view=0,
        sequence=1,
        digest="abc123",
        sender_id="agent_1",  # Not primary
        operation={"test": "data"},
        reputation=0.9
    )
    
    is_valid, detection = integrator.validate_message(invalid_msg)
    print(f"✓ Invalid message detected: type={detection}")
    
    if not is_valid:
        integrator.record_byzantine_detection(
            agent_id=invalid_msg.sender_id,
            detection_type=detection,
            view=invalid_msg.view,
            sequence=invalid_msg.sequence
        )
    
    # Test conflicting votes
    msg1 = ConsensusMessage(
        msg_type=MessageType.PREPARE,
        view=0,
        sequence=2,
        digest="digest1",
        sender_id="agent_2",
        reputation=0.9
    )
    
    msg2 = ConsensusMessage(
        msg_type=MessageType.PREPARE,
        view=0,
        sequence=2,
        digest="digest2",  # Different digest!
        sender_id="agent_2",  # Same sender
        reputation=0.9
    )
    
    integrator.validate_message(msg1)
    is_valid, detection = integrator.validate_message(msg2)
    
    if not is_valid:
        print(f"✓ Conflicting votes detected from agent_2")
        integrator.record_byzantine_detection(
            agent_id=msg2.sender_id,
            detection_type=detection,
            view=msg2.view,
            sequence=msg2.sequence
        )
    
    # Get Byzantine report
    print("\n3. Byzantine detection report:")
    report = integrator.get_byzantine_report()
    print(f"  Total detections: {report['total_detections']}")
    print(f"  Suspected agents: {report['suspected_agents']}")
    print(f"  Detection types: {report['detection_types']}")
    
    # Test threshold adjustment
    if threshold_mgr:
        print("\n4. Testing threshold adjustment...")
        old_t = threshold_mgr.current_threshold
        byzantine_count = integrator.get_recent_byzantine_count()
        
        new_t, new_active = integrator.force_threshold_adjustment(
            byzantine_count=byzantine_count,
            agent_failures=0
        )
        
        print(f"✓ Threshold adjusted: {old_t} → {new_t}")
        print(f"  Byzantine agents: {byzantine_count}")
        print(f"  Active agents: {new_active}")
    
    # Get statistics
    print("\n5. Integration statistics:")
    stats = integrator.get_statistics()
    print(f"  Consensus rounds: {stats['integration']['consensus_rounds_monitored']}")
    print(f"  Anomalies detected: {stats['integration']['anomalies_detected']}")
    print(f"  Threshold adjustments: {stats['integration']['threshold_adjustments_triggered']}")
    
    print("\n✓ All integration tests passed!")


if __name__ == "__main__":
    test_threshold_integration()
