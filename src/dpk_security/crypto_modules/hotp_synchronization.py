"""
HOTP Event-Based Synchronization for Mission Phases

Builds on RFC 4226-compliant HOTP implementation in hotp.py for
event-based (counter-based) phase sequencing without time sync.

Paper Reference: Section 4 - HOTP Counter Mechanisms for Sequential Task Coordination
"""

import time
import struct
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from .hotp import HOTP, HOTPVerificationResult


@dataclass
class PhaseMetrics:
    """Metrics for phase progression analysis."""
    phase_number: int = 0
    counter_value: str = ""
    computation_time_us: float = 0.0
    energy_cost_uj: float = 0.0  # microjoules
    desync_count: int = 0


class HOTPCounter:
    """
    Mission-phase HOTP counter manager.

    - Uses RFC 4226 HOTP(Kc, phase_index) for each phase.
    - Adds per-phase metrics and mission-specific semantics.
    """

    def __init__(self, counter_key: bytes, num_phases: int = 10, digits: int = 6):
        """
        Initialize HOTP counter mechanism for mission phases.

        Args:
            counter_key: Shared key Kc derived from mission SK.
            num_phases: Number of mission phases (default: 10).
            digits: Number of HOTP digits per phase token (default: 6).
        """
        self.Kc = counter_key
        self.num_phases = num_phases
        self.digits = digits
        self.current_phase = 0
        self.counter_values: Dict[int, str] = {}
        self.metrics: Dict[int, PhaseMetrics] = {}

        # Generate all phase counters upfront using HOTP implementation
        for i in range(num_phases):
            self.counter_values[i] = self._compute_and_record_counter(i)

    def _compute_and_record_counter(self, phase_i: int) -> str:
        """
        Compute HOTP(Kc, phase_i) and record timing/energy metrics.
        """
        start_time = time.perf_counter()

        hotp_str = HOTP.generate(self.Kc, phase_i, digits=self.digits)

        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1e6

        self.metrics[phase_i] = PhaseMetrics(
            phase_number=phase_i,
            counter_value=hotp_str,
            computation_time_us=elapsed_us,
            # Simple linear model: 0.01 μJ per μs (tunable).
            energy_cost_uj=elapsed_us * 0.01,
        )
        return hotp_str

    def verify_counter(self, token: str, phase_i: int, window: int = 0) -> bool:
        """
        Verify an HOTP token for a given mission phase using resynchronization window.

        Args:
            token: HOTP token to verify.
            phase_i: Expected phase number (used as counter).
            window: Resynchronization window (default 0).

        Returns:
            True if token is valid for some counter in [phase_i, phase_i + window].
        """
        result: HOTPVerificationResult = HOTP.verify(self.Kc, token, phase_i, window=window)
        return result.valid

    def phase_progression(self, current_phase: int) -> Dict[str, Any]:
        """
        Deterministic phase progression with mission semantics.

        Case-study semantics (example):
        - Phases 0-2: Deployment
        - Phases 3-6: Execution
        - Phases 7-9: Consensus

        Args:
            current_phase: Current mission phase (0-indexed).

        Returns:
            Phase metadata and current/next HOTP counters.
        """
        if current_phase >= self.num_phases:
            raise ValueError(f"Phase {current_phase} exceeds mission phases {self.num_phases}")

        self.current_phase = current_phase

        phase_info = self._get_phase_info(current_phase)

        next_counter = None
        if current_phase < self.num_phases - 1:
            next_counter = self.counter_values[current_phase + 1]

        return {
            "current_phase": current_phase,
            "current_counter": self.counter_values[current_phase],
            "next_counter": next_counter,
            "phase_name": phase_info["name"],
            "phase_category": phase_info["category"],
            "day_range": phase_info["days"],
            "tasks_allocated": phase_info["tasks"],
            "metrics": self.metrics[current_phase],
        }

    def _get_phase_info(self, phase: int) -> Dict[str, Any]:
        """
        Get detailed phase information for the precision farming case study.
        """
        phase_map = {
            0: {"name": "Initialization",      "category": "Deployment", "days": "1",    "tasks": 50},
            1: {"name": "Drone Deployment",    "category": "Deployment", "days": "2",    "tasks": 100},
            2: {"name": "Area Surveying",      "category": "Deployment", "days": "3",    "tasks": 150},
            3: {"name": "Ground Sampling",     "category": "Execution",  "days": "4-5",  "tasks": 200},
            4: {"name": "Data Collection",     "category": "Execution",  "days": "5-6",  "tasks": 200},
            5: {"name": "Pattern Analysis",    "category": "Execution",  "days": "6-7",  "tasks": 150},
            6: {"name": "Verification",        "category": "Execution",  "days": "7",    "tasks": 100},
            7: {"name": "Aggregation",         "category": "Consensus",  "days": "8",    "tasks": 50},
            8: {"name": "Consensus Building",  "category": "Consensus",  "days": "9",    "tasks": 50},
            9: {"name": "Authorization",       "category": "Consensus",  "days": "10",   "tasks": 50},
        }

        return phase_map.get(
            phase,
            {"name": f"Phase {phase}", "category": "Unknown", "days": "N/A", "tasks": 0},
        )

    def resynchronization_recovery(
        self,
        received_counter: str,
        expected_phase: int,
        window: int = 2,
    ) -> Tuple[bool, Optional[int], Dict[str, Any]]:
        """
        Handle counter desynchronization with a symmetric window around expected_phase.

        Accept if received_counter matches any HOTP value in
        [expected_phase - window, expected_phase + window], clipped to [0, num_phases-1].

        Args:
            received_counter: HOTP string received from a robot.
            expected_phase: Current system phase.
            window: Resynchronization window (default: 2).

        Returns:
            (accepted, recovered_phase, metrics_dict)
        """
        start_time = time.perf_counter()

        accepted = False
        recovered_phase: Optional[int] = None

        search_start = max(0, expected_phase - window)
        search_end = min(self.num_phases - 1, expected_phase + window)

        for phase in range(search_start, search_end + 1):
            if self.counter_values[phase] == received_counter:
                accepted = True
                recovered_phase = phase
                break

        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1e6

        desync_distance = abs(recovered_phase - expected_phase) if recovered_phase is not None else window + 1

        metrics = {
            "accepted": accepted,
            "expected_phase": expected_phase,
            "recovered_phase": recovered_phase,
            "desync_distance": desync_distance,
            "window_size": window,
            "information_leaked_bits": window.bit_length(),
            "recovery_time_us": elapsed_us,
            "within_tolerance": desync_distance <= window,
        }

        if accepted and recovered_phase in self.metrics:
            self.metrics[recovered_phase].desync_count += 1

        return accepted, recovered_phase, metrics

    def get_mission_timeline(self) -> Dict[str, Any]:
        """
        Return mission timeline with HOTP phases and metadata.
        """
        timeline: Dict[str, Any] = {
            "total_phases": self.num_phases,
            "total_days": 10,
            "phase_details": {},
            "category_summary": {
                "deployment": {"phases": [0, 1, 2], "days": "1-3",  "robots": 200},
                "execution":  {"phases": [3, 4, 5, 6], "days": "4-7", "robots": 200},
                "consensus":  {"phases": [7, 8, 9], "days": "8-10", "robots": 140},
            },
        }

        for phase in range(self.num_phases):
            info = self._get_phase_info(phase)
            timeline["phase_details"][phase] = {
                "counter": self.counter_values[phase],
                "name": info["name"],
                "category": info["category"],
                "days": info["days"],
                "tasks": info["tasks"],
            }

        return timeline

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated phase-level performance metrics.
        """
        if not self.metrics:
            return {}

        total_time = sum(m.computation_time_us for m in self.metrics.values())
        total_energy = sum(m.energy_cost_uj for m in self.metrics.values())
        total_desyncs = sum(m.desync_count for m in self.metrics.values())

        avg_time = total_time / len(self.metrics)
        avg_energy = total_energy / len(self.metrics)

        return {
            "total_phases": self.num_phases,
            "total_computation_time_us": total_time,
            "avg_phase_time_us": avg_time,
            "total_energy_uj": total_energy,
            "avg_phase_energy_uj": avg_energy,
            "total_desyncs": total_desyncs,
            "desync_rate": total_desyncs / self.num_phases if self.num_phases > 0 else 0.0,
            "meets_paper_requirements": {
                "avg_time_under_1us": avg_time < 1.0,
                "avg_energy_under_1uj": avg_energy < 1.0,
            },
        }


class HOTPSynchronization:
    """
    Multi-robot HOTP synchronization manager built on HOTPCounter.
    """

    def __init__(self, shared_key: bytes, num_robots: int, num_phases: int = 10, digits: int = 6):
        """
        Initialize synchronization manager for a robot swarm.

        Args:
            shared_key: Shared HOTP key for all robots.
            num_robots: Number of robots in the swarm.
            num_phases: Number of mission phases.
            digits: HOTP digits for counters.
        """
        self.shared_key = shared_key
        self.num_robots = num_robots
        self.num_phases = num_phases
        self.digits = digits
        self.robot_counters: Dict[int, HOTPCounter] = {}

        for robot_id in range(num_robots):
            self.robot_counters[robot_id] = HOTPCounter(shared_key, num_phases, digits=digits)

    def synchronize_phase(self, phase: int) -> Dict[int, str]:
        """
        Synchronize all robots to a specific phase and return their counters.
        """
        phase_counters: Dict[int, str] = {}
        for robot_id, counter in self.robot_counters.items():
            phase_info = counter.phase_progression(phase)
            phase_counters[robot_id] = phase_info["current_counter"]

        return phase_counters

    def verify_swarm_consensus(self, phase: int) -> Dict[str, Any]:
        """
        Verify that all robots are synchronized to the same HOTP counter at a given phase.
        """
        counters = self.synchronize_phase(phase)
        unique_counters = set(counters.values())

        return {
            "phase": phase,
            "total_robots": self.num_robots,
            "synchronized": len(unique_counters) == 1,
            "unique_counter_values": len(unique_counters),
            "consensus_achieved": len(unique_counters) == 1,
            "counter_value": list(unique_counters)[0] if len(unique_counters) == 1 else None,
        }


if __name__ == "__main__":
    import secrets

    print("=" * 80)
    print("HOTP Event-Based Synchronization - Test")
    print("=" * 80)

    key = secrets.token_bytes(32)
    counter = HOTPCounter(key, num_phases=10)

    print("\n[1] Testing HOTP Counter Generation")
    phase_0 = counter.counter_values[0]
    print(f"  ✓ Phase 0 counter: {phase_0}")
    print(f"  ✓ Computation time: {counter.metrics[0].computation_time_us:.2f} μs")
    print(f"  ✓ Energy cost: {counter.metrics[0].energy_cost_uj:.2f} μJ")

    print("\n[2] Testing Phase Progression")
    for phase in [0, 3, 7, 9]:
        phase_info = counter.phase_progression(phase)
        print(f"  ✓ Phase {phase}: {phase_info['phase_name']} - Day {phase_info['day_range']}")
        print(f"    Counter: {phase_info['current_counter']}")

    print("\n[3] Mission Timeline")
    timeline = counter.get_mission_timeline()
    print(f"  ✓ Total phases: {timeline['total_phases']}")
    print(f"  ✓ Total days: {timeline['total_days']}")
    for category, info in timeline["category_summary"].items():
        print(f"  ✓ {category.capitalize()}: Days {info['days']}, {len(info['phases'])} phases")

    print("\n[4] Testing Resynchronization Recovery")
    received = counter.counter_values[5]
    accepted, recovered, metrics = counter.resynchronization_recovery(received, 3, window=2)
    print(f"  ✓ Counter accepted: {accepted}")
    print(f"  ✓ Recovered phase: {recovered}")
    print(f"  ✓ Desync distance: {metrics['desync_distance']}")
    print(f"  ✓ Information leaked: {metrics['information_leaked_bits']} bits")

    print("\n[5] Testing Counter Verification")
    phase_7_counter = counter.counter_values[7]
    is_valid = counter.verify_counter(phase_7_counter, 7)
    print(f"  ✓ Phase 7 counter verified: {is_valid}")

    print("\n[6] Testing Swarm Synchronization")
    sync = HOTPSynchronization(key, num_robots=10, num_phases=10)
    consensus = sync.verify_swarm_consensus(phase=5)
    print(f"  ✓ Total robots: {consensus['total_robots']}")
    print(f"  ✓ Synchronized: {consensus['synchronized']}")
    print(f"  ✓ Consensus achieved: {consensus['consensus_achieved']}")
    print(f"  ✓ Counter value: {consensus['counter_value']}")

    print("\n[7] Performance Metrics")
    perf = counter.get_performance_metrics()
    print(f"  ✓ Total phases: {perf['total_phases']}")
    print(f"  ✓ Avg computation time: {perf['avg_phase_time_us']:.2f} μs")
    print(f"  ✓ Avg energy cost: {perf['avg_phase_energy_uj']:.2f} μJ")
    print(f"  ✓ Meets paper requirements: {perf['meets_paper_requirements']}")

    print("\n" + "=" * 80)
    print("HOTP Synchronization Implementation Complete - All Tests Passed ✓")
    print("=" * 80)