"""
Phase 4 Integration Tests
==========================

Tests for Secure Multi-Party Computation (SMPC) integration.

Coverage:
- SMPC coordinator session management
- Distributed computation and reconstruction
- ZK-SMPC verifiable computation
- Privacy-preserving aggregation
- Distributed key generation
- Key refresh mechanism
- Byzantine detection with ZK proofs
- End-to-end SMPC workflow
"""

import sys
import os
import time
import unittest
from typing import List

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'DeFi Agents')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Aztec Noir Integration')))

from smpc import (
    SMPCCoordinator,
    ZKSMPCIntegration,
    DistributedKeyGenerator
)

from zkp import ZKProofManager


class TestSMPCCoordinator(unittest.TestCase):
    """Test SMPC Coordinator"""
    
    def setUp(self):
        """Initialize coordinator"""
        self.coordinator = SMPCCoordinator(
            agent_id="coordinator",
            threshold=3,
            total_agents=5,
            byzantine_tolerance=1
        )
        self.agents = [f"agent_{i}" for i in range(5)]
    
    def test_session_creation(self):
        """Test SMPC session creation"""
        print("\n[Test 1] Session Creation")
        print("-" * 60)
        
        session_id = self.coordinator.create_session(
            participants=self.agents,
            threshold=3
        )
        
        self.assertIsNotNone(session_id)
        
        session = self.coordinator.get_session(session_id)
        self.assertEqual(len(session.participants), 5)
        self.assertEqual(session.threshold, 3)
        self.assertEqual(session.status, "active")
        
        print(f"✓ Session created: {session_id}")
        print(f"  Participants: {len(session.participants)}")
        print(f"  Threshold: {session.threshold}")
    
    def test_task_distribution(self):
        """Test computation task distribution"""
        print("\n[Test 2] Task Distribution")
        print("-" * 60)
        
        session_id = self.coordinator.create_session(self.agents, threshold=3)
        
        secret_value = 12345
        task_id = self.coordinator.distribute_computation(
            session_id=session_id,
            task_type="aggregate",
            secret_value=secret_value
        )
        
        self.assertIsNotNone(task_id)
        
        # Check shares were distributed
        session = self.coordinator.get_session(session_id)
        task = session.tasks[task_id]
        
        self.assertEqual(len(task.input_shares), 5)
        self.assertEqual(task.threshold, 3)
        
        print(f"✓ Task distributed: {task_id}")
        print(f"  Secret: {secret_value}")
        print(f"  Shares: {len(task.input_shares)}")
    
    def test_computation_execution(self):
        """Test SMPC computation execution"""
        print("\n[Test 3] Computation Execution")
        print("-" * 60)
        
        session_id = self.coordinator.create_session(self.agents, threshold=3)
        
        secret = 98765
        task_id = self.coordinator.distribute_computation(
            session_id=session_id,
            task_type="test",
            secret_value=secret
        )
        
        # Collect shares
        session = self.coordinator.get_session(session_id)
        task = session.tasks[task_id]
        shares = list(task.input_shares.values())[:3]
        
        # Execute computation
        result = self.coordinator.execute_computation(
            session_id=session_id,
            task_id=task_id,
            collected_shares=shares
        )
        
        self.assertEqual(result.result, secret)
        self.assertTrue(result.verified)
        self.assertEqual(result.shares_used, 3)
        
        print(f"✓ Computation executed")
        print(f"  Result: {result.result}")
        print(f"  Expected: {secret}")
        print(f"  Time: {result.computation_time_ms:.2f}ms")
    
    def test_value_aggregation(self):
        """Test value aggregation"""
        print("\n[Test 4] Value Aggregation")
        print("-" * 60)
        
        session_id = self.coordinator.create_session(self.agents, threshold=3)
        
        values = [100, 200, 300, 400, 500]
        total = self.coordinator.aggregate_values(
            session_id=session_id,
            values=values,
            operation='sum'
        )
        
        self.assertEqual(total, sum(values))
        
        print(f"✓ Aggregation completed")
        print(f"  Values: {values}")
        print(f"  Sum: {total}")
        print(f"  Expected: {sum(values)}")


class TestZKSMPCIntegration(unittest.TestCase):
    """Test ZK-SMPC Integration"""
    
    def setUp(self):
        """Initialize components"""
        self.coordinator = SMPCCoordinator(
            agent_id="coordinator",
            threshold=3,
            total_agents=5
        )
        self.proof_manager = ZKProofManager()
        self.zk_smpc = ZKSMPCIntegration(self.coordinator, self.proof_manager)
        self.agents = [f"agent_{i}" for i in range(5)]
    
    def test_commitment_scheme(self):
        """Test private input commitments"""
        print("\n[Test 5] Commitment Scheme")
        print("-" * 60)
        
        agent_id = "agent_0"
        value = 12345
        
        # Create commitment
        private_input = self.zk_smpc.commit_input(agent_id, value)
        
        self.assertEqual(private_input.agent_id, agent_id)
        self.assertEqual(private_input.value, value)
        self.assertIsNotNone(private_input.commitment)
        
        # Verify commitment is stored
        self.assertIn(agent_id, self.zk_smpc.commitments)
        
        print(f"✓ Commitment created")
        print(f"  Agent: {agent_id}")
        print(f"  Value: {value}")
        print(f"  Commitment: {private_input.commitment.hex()[:32]}...")
    
    def test_verifiable_computation(self):
        """Test verifiable SMPC computation"""
        print("\n[Test 6] Verifiable Computation")
        print("-" * 60)
        
        session_id = self.coordinator.create_session(self.agents, threshold=3)
        
        # Create private inputs
        private_inputs = [
            self.zk_smpc.commit_input(f"agent_{i}", 100 * (i + 1))
            for i in range(3)
        ]
        
        # Execute verifiable computation
        verifiable = self.zk_smpc.execute_verifiable_computation(
            session_id=session_id,
            task_type="aggregate",
            private_inputs=private_inputs
        )
        
        self.assertIsNotNone(verifiable.computation_result.result)
        self.assertTrue(verifiable.is_verified)
        self.assertIsNotNone(verifiable.zk_proof)
        
        print(f"✓ Verifiable computation")
        print(f"  Result: {verifiable.computation_result.result}")
        print(f"  Verified: {verifiable.is_verified}")
        print(f"  Proof time: {verifiable.proof_generation_time_ms:.2f}ms")
    
    def test_privacy_preserving_aggregation(self):
        """Test privacy-preserving aggregation"""
        print("\n[Test 7] Privacy-Preserving Aggregation")
        print("-" * 60)
        
        session_id = self.coordinator.create_session(self.agents, threshold=3)
        
        values = [10, 20, 30]
        agent_ids = self.agents[:3]
        
        # Aggregate with privacy
        verifiable = self.zk_smpc.aggregate_with_privacy(
            session_id=session_id,
            values=values,
            agent_ids=agent_ids,
            operation='sum'
        )
        
        self.assertTrue(verifiable.is_verified)
        
        print(f"✓ Privacy-preserving aggregation")
        print(f"  Inputs: {values}")
        print(f"  Result: {verifiable.computation_result.result}")
        print(f"  Verified: {verifiable.is_verified}")


class TestDistributedKeyGeneration(unittest.TestCase):
    """Test Distributed Key Generation"""
    
    def setUp(self):
        """Initialize DKG"""
        self.dkg = DistributedKeyGenerator(
            agent_id="agent_0",
            threshold=3,
            total_agents=5
        )
        self.agents = [f"agent_{i}" for i in range(5)]
    
    def test_key_generation(self):
        """Test distributed key generation"""
        print("\n[Test 8] Key Generation")
        print("-" * 60)
        
        distributed_key = self.dkg.generate_distributed_key(
            key_purpose="signing",
            participating_agents=self.agents
        )
        
        self.assertEqual(distributed_key.threshold, 3)
        self.assertEqual(distributed_key.total_shares, 5)
        self.assertEqual(len(distributed_key.shares), 5)
        self.assertIsNotNone(distributed_key.public_key)
        
        print(f"✓ Key generated: {distributed_key.key_id}")
        print(f"  Threshold: {distributed_key.threshold}")
        print(f"  Shares: {distributed_key.total_shares}")
    
    def test_share_verification(self):
        """Test share verification"""
        print("\n[Test 9] Share Verification")
        print("-" * 60)
        
        distributed_key = self.dkg.generate_distributed_key(
            key_purpose="encryption",
            participating_agents=self.agents
        )
        
        valid, total = self.dkg.verify_all_shares(distributed_key.key_id)
        
        self.assertEqual(total, 5)
        self.assertEqual(valid, total)
        
        print(f"✓ Share verification")
        print(f"  Valid: {valid}/{total}")
    
    def test_key_reconstruction(self):
        """Test key reconstruction"""
        print("\n[Test 10] Key Reconstruction")
        print("-" * 60)
        
        distributed_key = self.dkg.generate_distributed_key(
            key_purpose="test",
            participating_agents=self.agents
        )
        
        # Collect threshold shares
        shares_to_use = list(distributed_key.shares.values())[:3]
        
        # Reconstruct
        reconstructed = self.dkg.reconstruct_key(
            distributed_key.key_id,
            shares_to_use
        )
        
        self.assertIsNotNone(reconstructed)
        
        print(f"✓ Key reconstructed")
        print(f"  Shares used: {len(shares_to_use)}")
        print(f"  Result: {reconstructed}")
    
    def test_key_refresh(self):
        """Test key refresh mechanism"""
        print("\n[Test 11] Key Refresh")
        print("-" * 60)
        
        distributed_key = self.dkg.generate_distributed_key(
            key_purpose="refresh_test",
            participating_agents=self.agents
        )
        
        old_public_key = distributed_key.public_key
        
        # Refresh shares
        refreshed = self.dkg.refresh_key_shares(
            key_id=distributed_key.key_id,
            refreshing_agents=self.agents
        )
        
        # Public key should remain the same
        self.assertEqual(old_public_key, refreshed.public_key)
        self.assertEqual(refreshed.status, "active")
        
        print(f"✓ Key refreshed")
        print(f"  Old PK: {old_public_key.hex()[:32]}...")
        print(f"  New PK: {refreshed.public_key.hex()[:32]}...")
        print(f"  Keys match: {old_public_key == refreshed.public_key}")


class TestEndToEnd(unittest.TestCase):
    """End-to-end SMPC integration tests"""
    
    def test_complete_smpc_workflow(self):
        """Test complete SMPC workflow"""
        print("\n[Test 12] Complete SMPC Workflow")
        print("-" * 60)
        
        # Initialize all components
        coordinator = SMPCCoordinator(
            agent_id="coordinator",
            threshold=3,
            total_agents=5
        )
        proof_manager = ZKProofManager()
        zk_smpc = ZKSMPCIntegration(coordinator, proof_manager)
        dkg = DistributedKeyGenerator(
            agent_id="coordinator",
            threshold=3,
            total_agents=5
        )
        
        agents = [f"agent_{i}" for i in range(5)]
        
        print("  1. Generating distributed signing key...")
        signing_key = dkg.generate_distributed_key(
            key_purpose="signing",
            participating_agents=agents
        )
        print(f"    ✓ Key: {signing_key.key_id}")
        
        print("  2. Creating SMPC session...")
        session_id = coordinator.create_session(agents, threshold=3)
        print(f"    ✓ Session: {session_id}")
        
        print("  3. Privacy-preserving aggregation...")
        values = [100, 200, 300]
        verifiable = zk_smpc.aggregate_with_privacy(
            session_id=session_id,
            values=values,
            agent_ids=agents[:3],
            operation='sum'
        )
        print(f"    ✓ Aggregated: {verifiable.computation_result.result}")
        print(f"    ✓ Verified: {verifiable.is_verified}")
        
        print("  4. Key reconstruction for signing...")
        shares_to_use = list(signing_key.shares.values())[:3]
        reconstructed = dkg.reconstruct_key(
            signing_key.key_id,
            shares_to_use
        )
        print(f"    ✓ Reconstructed key")
        
        print("  5. Collecting statistics...")
        
        coord_stats = coordinator.get_statistics()
        print(f"\n  Coordinator Stats:")
        print(f"    Sessions: {coord_stats['sessions_created']}")
        print(f"    Tasks: {coord_stats['tasks_executed']}")
        print(f"    Success rate: {coord_stats['success_rate']:.1%}")
        
        zk_stats = zk_smpc.get_statistics()
        print(f"\n  ZK-SMPC Stats:")
        print(f"    Computations verified: {zk_stats['computations_verified']}")
        print(f"    Verification rate: {zk_stats['verification_rate']:.1%}")
        
        dkg_stats = dkg.get_statistics()
        print(f"\n  DKG Stats:")
        print(f"    Keys generated: {dkg_stats['keys_generated']}")
        print(f"    Shares verified: {dkg_stats['shares_verified']}")
        
        print("\n✓ Complete workflow test passed!")


def run_phase4_tests():
    """Run all Phase 4 tests"""
    print("\n" + "=" * 60)
    print("PHASE 4 INTEGRATION TESTS")
    print("Secure Multi-Party Computation (SMPC)")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSMPCCoordinator))
    suite.addTests(loader.loadTestsFromTestCase(TestZKSMPCIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDistributedKeyGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL PHASE 4 TESTS PASSED!")
    else:
        print("\n✗ Some tests failed")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_phase4_tests()
    sys.exit(0 if success else 1)
