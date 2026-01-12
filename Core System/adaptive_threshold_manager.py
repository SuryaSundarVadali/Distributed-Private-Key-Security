"""
Adaptive Threshold Manager
Dynamically adjusts Shamir Secret Sharing threshold based on Byzantine detection
and agent failures to maintain security while ensuring reconstruction is possible.

Author: Venkata Surya Sundar Vadali
Date: January 11, 2026
"""

import time
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdAdjustmentEvent:
    """Record of threshold adjustment"""
    timestamp: float
    old_threshold: int
    new_threshold: int
    active_agents: int
    byzantine_count: int
    reason: str


class AdaptiveThresholdManager:
    """
    Manages dynamic threshold adjustment for Shamir Secret Sharing.
    
    Key Features:
    - Increases threshold during Byzantine attacks for extra security
    - Decreases threshold when agents go offline to maintain liveness
    - Maintains adjustment history for audit trails
    - Configurable min/max threshold bounds
    
    Performance: <100ms adjustment computation
    """
    
    def __init__(
        self, 
        initial_t: int, 
        initial_n: int,
        min_threshold_offset: int = 2,
        max_threshold_offset: int = 3
    ):
        """
        Initialize adaptive threshold manager.
        
        Args:
            initial_t: Initial threshold value
            initial_n: Initial total number of agents
            min_threshold_offset: Minimum reduction from initial_t (default: 2)
            max_threshold_offset: Maximum increase from initial_t (default: 3)
        """
        self.initial_t = initial_t
        self.initial_n = initial_n
        self.current_t = initial_t
        self.current_n = initial_n
        
        # Define safe threshold bounds
        self.min_t = max(2, initial_t - min_threshold_offset)
        self.max_t = min(initial_n - 1, initial_t + max_threshold_offset)
        
        # Track adjustment history for audit trail
        self.adjustment_history: List[ThresholdAdjustmentEvent] = []
        
        # Track Byzantine agents over time
        self.byzantine_detection_count = 0
        self.total_adjustments = 0
        
        logger.info(
            f"Initialized AdaptiveThresholdManager: "
            f"t={initial_t}, n={initial_n}, "
            f"bounds=[{self.min_t}, {self.max_t}]"
        )
    
    def adjust_threshold(
        self, 
        byzantine_count: int, 
        agent_failures: int,
        reason: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Adjust threshold based on detected Byzantine agents and failures.
        
        Strategy:
        1. Increase t if Byzantine agents detected (security priority)
        2. Decrease t if too many agents offline (liveness priority)
        3. Ensure t is within safe bounds
        4. Maintain n > 3f Byzantine fault tolerance guarantee
        
        Args:
            byzantine_count: Number of detected Byzantine agents
            agent_failures: Number of agents currently offline/failed
            reason: Optional explanation for adjustment
            
        Returns:
            Tuple of (new_threshold, active_agents)
        """
        start_time = time.time()
        old_t = self.current_t
        active_agents = self.current_n - agent_failures
        
        # Strategy 1: Increase threshold during attacks
        if byzantine_count > 0:
            self.byzantine_detection_count += byzantine_count
            # Increase by number of Byzantine agents detected
            new_t = min(self.max_t, self.current_t + byzantine_count)
            adjustment_reason = reason or f"Byzantine attack detected ({byzantine_count} agents)"
            
        # Strategy 2: Decrease threshold if too many agents offline
        elif self.current_t > active_agents - 1:
            # Ensure reconstruction is still possible
            # Need at least t shares from active agents
            new_t = max(self.min_t, active_agents // 2 + 1)
            adjustment_reason = reason or f"Agent failures ({agent_failures} offline)"
            
        # Strategy 3: Gradually restore to baseline during normal operation
        elif byzantine_count == 0 and agent_failures == 0 and self.current_t > self.initial_t:
            # Slowly restore to baseline (reduce by 1)
            new_t = max(self.initial_t, self.current_t - 1)
            adjustment_reason = reason or "Gradual restoration to baseline"
            
        else:
            # No adjustment needed
            new_t = self.current_t
            adjustment_reason = reason or "No adjustment required"
        
        # Ensure Byzantine fault tolerance: n > 3f → t > n/3
        # If too many Byzantine agents, increase threshold
        if byzantine_count > 0:
            min_required_t = (active_agents // 3) + 1
            if new_t < min_required_t:
                new_t = min(self.max_t, min_required_t)
                adjustment_reason += f" (BFT requirement: t > n/3)"
        
        # Apply adjustment
        self.current_t = new_t
        self.total_adjustments += 1
        
        # Record adjustment event
        event = ThresholdAdjustmentEvent(
            timestamp=time.time(),
            old_threshold=old_t,
            new_threshold=new_t,
            active_agents=active_agents,
            byzantine_count=byzantine_count,
            reason=adjustment_reason
        )
        self.adjustment_history.append(event)
        
        # Log adjustment
        elapsed_ms = (time.time() - start_time) * 1000
        if new_t != old_t:
            logger.warning(
                f"Threshold adjusted: {old_t} → {new_t} "
                f"(active={active_agents}, byzantine={byzantine_count}) "
                f"Reason: {adjustment_reason} "
                f"[{elapsed_ms:.2f}ms]"
            )
        else:
            logger.info(
                f"Threshold unchanged: {new_t} "
                f"(active={active_agents}, byzantine={byzantine_count}) "
                f"[{elapsed_ms:.2f}ms]"
            )
        
        return self.current_t, active_agents
    
    def get_current_threshold(self) -> int:
        """Get current threshold value"""
        return self.current_t
    
    def get_adjustment_history(self) -> List[ThresholdAdjustmentEvent]:
        """Get complete adjustment history for audit trail"""
        return self.adjustment_history
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about threshold adjustments.
        
        Returns:
            Dictionary with adjustment statistics
        """
        if not self.adjustment_history:
            return {
                "total_adjustments": 0,
                "current_threshold": self.current_t,
                "byzantine_detections": self.byzantine_detection_count,
                "threshold_range": [self.min_t, self.max_t]
            }
        
        threshold_values = [event.new_threshold for event in self.adjustment_history]
        
        return {
            "total_adjustments": self.total_adjustments,
            "current_threshold": self.current_t,
            "initial_threshold": self.initial_t,
            "min_threshold": self.min_t,
            "max_threshold": self.max_t,
            "byzantine_detections": self.byzantine_detection_count,
            "threshold_range_observed": [min(threshold_values), max(threshold_values)],
            "average_threshold": sum(threshold_values) / len(threshold_values),
            "total_events": len(self.adjustment_history)
        }
    
    def reset_to_baseline(self):
        """Reset threshold to initial baseline value"""
        old_t = self.current_t
        self.current_t = self.initial_t
        
        event = ThresholdAdjustmentEvent(
            timestamp=time.time(),
            old_threshold=old_t,
            new_threshold=self.current_t,
            active_agents=self.current_n,
            byzantine_count=0,
            reason="Manual reset to baseline"
        )
        self.adjustment_history.append(event)
        
        logger.info(f"Threshold reset to baseline: {old_t} → {self.current_t}")


def test_adaptive_threshold_manager():
    """Test adaptive threshold manager functionality"""
    print("=" * 80)
    print("Testing Adaptive Threshold Manager")
    print("=" * 80)
    
    # Initialize manager with t=3, n=5
    manager = AdaptiveThresholdManager(initial_t=3, initial_n=5)
    
    # Test 1: Normal operation (no adjustment)
    print("\n1. Normal operation (no Byzantine, no failures):")
    t, active = manager.adjust_threshold(byzantine_count=0, agent_failures=0)
    print(f"   Result: t={t}, active_agents={active}")
    assert t == 3, "Threshold should remain unchanged"
    
    # Test 2: Byzantine attack detected (increase threshold)
    print("\n2. Byzantine attack detected (2 agents):")
    t, active = manager.adjust_threshold(byzantine_count=2, agent_failures=0)
    print(f"   Result: t={t}, active_agents={active}")
    assert t > 3, "Threshold should increase during attack"
    
    # Test 3: Agent failures (decrease threshold to maintain liveness)
    print("\n3. Agent failures (2 agents offline):")
    manager.reset_to_baseline()  # Reset first
    t, active = manager.adjust_threshold(byzantine_count=0, agent_failures=2)
    print(f"   Result: t={t}, active_agents={active}")
    assert t >= 2, "Threshold should be at least 2"
    assert t <= active - 1, "Threshold should allow reconstruction"
    
    # Test 4: Gradual restoration to baseline
    print("\n4. Gradual restoration (after attack resolved):")
    manager.current_t = 5  # Simulate increased threshold
    t, active = manager.adjust_threshold(byzantine_count=0, agent_failures=0)
    print(f"   Result: t={t}, active_agents={active}")
    
    # Test 5: Statistics
    print("\n5. Adjustment statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_adaptive_threshold_manager()
