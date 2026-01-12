#!/usr/bin/env python3
"""
Performance Testing Suite for DeFi Agent System
Tests load handling, latency, throughput, and resource usage across all phases
"""

import os
import sys
import time
import json
import asyncio
import threading
import statistics
import requests
from typing import List, Dict, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Cryptographic_Modules.key_generation import generate_master_keys
from Cryptographic_Modules.shamir_secret_sharing import ShamirSecretSharing
from Cryptographic_Modules.mfkdf import MFKDFGenerator
from DeFi_Agents.adaptive_threshold import AdaptiveThreshold
from DeFi_Agents.defi_yield_agent import DeFiYieldAgent
from DeFi_Agents.consensus.pbft_consensus import PBFTConsensus
from DeFi_Agents.smpc.smpc_coordinator import SMPCCoordinator

class PerformanceMetrics:
    """Collects and analyzes performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'latencies': [],
            'throughput': [],
            'errors': [],
            'resource_usage': []
        }
        
    def add_latency(self, operation: str, duration: float):
        """Add latency measurement"""
        self.metrics['latencies'].append({
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
    def add_throughput(self, operation: str, count: int, duration: float):
        """Add throughput measurement"""
        ops_per_second = count / duration if duration > 0 else 0
        self.metrics['throughput'].append({
            'operation': operation,
            'count': count,
            'duration': duration,
            'ops_per_second': ops_per_second,
            'timestamp': datetime.now().isoformat()
        })
        
    def add_error(self, operation: str, error: str):
        """Record an error"""
        self.metrics['errors'].append({
            'operation': operation,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        
    def get_summary(self) -> Dict:
        """Get performance summary"""
        latencies = [m['duration'] for m in self.metrics['latencies']]
        throughputs = [m['ops_per_second'] for m in self.metrics['throughput']]
        
        return {
            'latency': {
                'min': min(latencies) if latencies else 0,
                'max': max(latencies) if latencies else 0,
                'mean': statistics.mean(latencies) if latencies else 0,
                'median': statistics.median(latencies) if latencies else 0,
                'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
                'p99': statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
            },
            'throughput': {
                'min': min(throughputs) if throughputs else 0,
                'max': max(throughputs) if throughputs else 0,
                'mean': statistics.mean(throughputs) if throughputs else 0,
            },
            'errors': {
                'count': len(self.metrics['errors']),
                'rate': len(self.metrics['errors']) / len(self.metrics['latencies']) if self.metrics['latencies'] else 0
            },
            'total_operations': len(self.metrics['latencies'])
        }

class Phase1PerformanceTests:
    """Performance tests for Phase 1 - Adaptive Thresholds"""
    
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        
    def test_key_generation_throughput(self, iterations: int = 1000):
        """Test key generation throughput"""
        print(f"Testing key generation ({iterations} iterations)...")
        
        start_time = time.time()
        for i in range(iterations):
            try:
                op_start = time.time()
                generate_master_keys(password="test_pass", salt=os.urandom(16))
                self.metrics.add_latency('key_generation', time.time() - op_start)
            except Exception as e:
                self.metrics.add_error('key_generation', str(e))
                
        duration = time.time() - start_time
        self.metrics.add_throughput('key_generation', iterations, duration)
        print(f"  Completed in {duration:.2f}s ({iterations/duration:.2f} ops/sec)")
        
    def test_shamir_sharing_performance(self, iterations: int = 500):
        """Test Shamir secret sharing performance"""
        print(f"Testing Shamir secret sharing ({iterations} iterations)...")
        
        sss = ShamirSecretSharing(threshold=3, num_shares=5)
        secret = os.urandom(32)
        
        start_time = time.time()
        for i in range(iterations):
            try:
                op_start = time.time()
                shares = sss.create_shares(secret)
                recovered = sss.combine_shares(shares[:3])
                assert recovered == secret
                self.metrics.add_latency('shamir_sharing', time.time() - op_start)
            except Exception as e:
                self.metrics.add_error('shamir_sharing', str(e))
                
        duration = time.time() - start_time
        self.metrics.add_throughput('shamir_sharing', iterations, duration)
        print(f"  Completed in {duration:.2f}s ({iterations/duration:.2f} ops/sec)")
        
    def test_adaptive_threshold_performance(self, iterations: int = 300):
        """Test adaptive threshold adjustment performance"""
        print(f"Testing adaptive threshold ({iterations} iterations)...")
        
        threshold_manager = AdaptiveThreshold(
            initial_threshold=100,
            total_agents=200,
            adjustment_factor=0.1
        )
        
        start_time = time.time()
        for i in range(iterations):
            try:
                op_start = time.time()
                # Simulate various conditions
                threshold_manager.adjust(
                    success_rate=0.95,
                    active_agents=180,
                    avg_response_time=0.5
                )
                self.metrics.add_latency('adaptive_threshold', time.time() - op_start)
            except Exception as e:
                self.metrics.add_error('adaptive_threshold', str(e))
                
        duration = time.time() - start_time
        self.metrics.add_throughput('adaptive_threshold', iterations, duration)
        print(f"  Completed in {duration:.2f}s ({iterations/duration:.2f} ops/sec)")

class Phase2PerformanceTests:
    """Performance tests for Phase 2 - PBFT Consensus"""
    
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        
    def test_consensus_latency(self, num_nodes: int = 7, rounds: int = 50):
        """Test consensus latency with varying node counts"""
        print(f"Testing PBFT consensus ({num_nodes} nodes, {rounds} rounds)...")
        
        # Create consensus instances
        nodes = []
        for i in range(num_nodes):
            node = PBFTConsensus(
                node_id=f"node_{i}",
                total_nodes=num_nodes,
                byzantine_tolerance=1
            )
            nodes.append(node)
            
        # Connect nodes (simplified)
        for node in nodes:
            node.peers = [n for n in nodes if n.node_id != node.node_id]
            
        # Run consensus rounds
        for round_num in range(rounds):
            try:
                op_start = time.time()
                
                # Propose a value
                transaction = {
                    'round': round_num,
                    'data': f'test_transaction_{round_num}',
                    'timestamp': time.time()
                }
                
                # Simulate consensus
                primary = nodes[round_num % num_nodes]
                result = primary.propose(transaction)
                
                self.metrics.add_latency('consensus_round', time.time() - op_start)
            except Exception as e:
                self.metrics.add_error('consensus_round', str(e))
                
        print(f"  Completed {rounds} consensus rounds")
        
    def test_byzantine_tolerance(self, iterations: int = 100):
        """Test system performance under Byzantine conditions"""
        print(f"Testing Byzantine tolerance ({iterations} iterations)...")
        
        num_nodes = 10
        byzantine_nodes = 3  # 30% Byzantine
        
        for i in range(iterations):
            try:
                op_start = time.time()
                
                # Create nodes
                nodes = [
                    PBFTConsensus(f"node_{j}", num_nodes, 3)
                    for j in range(num_nodes)
                ]
                
                # Simulate Byzantine behavior in some nodes
                for j in range(byzantine_nodes):
                    nodes[j].is_byzantine = True
                    
                # Run consensus
                transaction = {'iteration': i, 'data': f'test_{i}'}
                primary = nodes[i % num_nodes]
                
                if not getattr(primary, 'is_byzantine', False):
                    result = primary.propose(transaction)
                    
                self.metrics.add_latency('byzantine_consensus', time.time() - op_start)
            except Exception as e:
                self.metrics.add_error('byzantine_consensus', str(e))
                
        print(f"  Completed Byzantine tolerance test")

class Phase3PerformanceTests:
    """Performance tests for Phase 3 - Zero-Knowledge Proofs"""
    
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        
    def test_proof_generation_latency(self, iterations: int = 50):
        """Test ZK proof generation latency"""
        print(f"Testing ZK proof generation ({iterations} iterations)...")
        
        # Note: Actual Noir proof generation would be used in production
        # This simulates the performance characteristics
        
        for i in range(iterations):
            try:
                op_start = time.time()
                
                # Simulate proof generation time (Noir compilation + proving)
                time.sleep(0.01)  # Simulated proving time
                
                # Create proof data structure
                proof_data = {
                    'inputs': {'value': 1000 + i, 'commitment': 'hash'},
                    'proof': 'simulated_proof_bytes',
                    'public_inputs': [1000 + i]
                }
                
                self.metrics.add_latency('zk_proof_generation', time.time() - op_start)
            except Exception as e:
                self.metrics.add_error('zk_proof_generation', str(e))
                
        print(f"  Completed proof generation test")
        
    def test_proof_verification_throughput(self, iterations: int = 500):
        """Test ZK proof verification throughput"""
        print(f"Testing ZK proof verification ({iterations} iterations)...")
        
        start_time = time.time()
        for i in range(iterations):
            try:
                op_start = time.time()
                
                # Simulate verification (much faster than generation)
                time.sleep(0.001)  # Simulated verification time
                
                # Verification logic
                verified = True  # Simulated result
                
                self.metrics.add_latency('zk_proof_verification', time.time() - op_start)
            except Exception as e:
                self.metrics.add_error('zk_proof_verification', str(e))
                
        duration = time.time() - start_time
        self.metrics.add_throughput('zk_proof_verification', iterations, duration)
        print(f"  Completed in {duration:.2f}s ({iterations/duration:.2f} ops/sec)")

class Phase4PerformanceTests:
    """Performance tests for Phase 4 - SMPC"""
    
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        
    def test_smpc_session_throughput(self, num_parties: int = 5, iterations: int = 20):
        """Test SMPC session throughput"""
        print(f"Testing SMPC sessions ({num_parties} parties, {iterations} iterations)...")
        
        coordinator = SMPCCoordinator(num_parties=num_parties, threshold=3)
        
        for i in range(iterations):
            try:
                op_start = time.time()
                
                # Create session
                session_id = coordinator.create_session(
                    computation_type='sum',
                    participants=[f"party_{j}" for j in range(num_parties)]
                )
                
                # Simulate computation
                for j in range(num_parties):
                    coordinator.submit_share(session_id, f"party_{j}", j * 10)
                    
                # Compute result
                result = coordinator.compute_result(session_id)
                
                self.metrics.add_latency('smpc_session', time.time() - op_start)
            except Exception as e:
                self.metrics.add_error('smpc_session', str(e))
                
        print(f"  Completed SMPC session test")
        
    def test_distributed_key_generation(self, iterations: int = 10):
        """Test distributed key generation performance"""
        print(f"Testing DKG ({iterations} iterations)...")
        
        for i in range(iterations):
            try:
                op_start = time.time()
                
                # Simulate DKG protocol
                num_parties = 5
                threshold = 3
                
                # Each party generates shares
                shares = []
                for j in range(num_parties):
                    sss = ShamirSecretSharing(threshold=threshold, num_shares=num_parties)
                    party_secret = os.urandom(32)
                    party_shares = sss.create_shares(party_secret)
                    shares.append(party_shares)
                    
                self.metrics.add_latency('dkg', time.time() - op_start)
            except Exception as e:
                self.metrics.add_error('dkg', str(e))
                
        print(f"  Completed DKG test")

class LoadTests:
    """Load testing for the entire system"""
    
    def __init__(self, metrics: PerformanceMetrics, base_url: str = "http://localhost:8000"):
        self.metrics = metrics
        self.base_url = base_url
        
    def test_concurrent_requests(self, num_requests: int = 100, concurrency: int = 10):
        """Test system under concurrent load"""
        print(f"Testing concurrent requests ({num_requests} requests, {concurrency} concurrent)...")
        
        def make_request(request_id: int):
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}/statistics", timeout=10)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    self.metrics.add_latency('concurrent_request', duration)
                    return True
                else:
                    self.metrics.add_error('concurrent_request', f"HTTP {response.status_code}")
                    return False
            except Exception as e:
                self.metrics.add_error('concurrent_request', str(e))
                return False
                
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]
            
        duration = time.time() - start_time
        successful = sum(results)
        
        print(f"  Completed: {successful}/{num_requests} successful in {duration:.2f}s")
        print(f"  Throughput: {successful/duration:.2f} req/sec")
        
    def test_stress_sustained_load(self, duration_seconds: int = 60, requests_per_second: int = 10):
        """Test system under sustained load"""
        print(f"Testing sustained load ({duration_seconds}s, {requests_per_second} req/s)...")
        
        end_time = time.time() + duration_seconds
        request_count = 0
        
        while time.time() < end_time:
            batch_start = time.time()
            
            for i in range(requests_per_second):
                try:
                    op_start = time.time()
                    response = requests.get(f"{self.base_url}/statistics", timeout=5)
                    self.metrics.add_latency('sustained_load', time.time() - op_start)
                    request_count += 1
                except Exception as e:
                    self.metrics.add_error('sustained_load', str(e))
                    
            # Wait for next batch
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
                
        print(f"  Completed {request_count} requests over {duration_seconds}s")

def run_performance_suite():
    """Run complete performance test suite"""
    print("\n" + "="*60)
    print("DeFi Agent System - Performance Test Suite")
    print("="*60 + "\n")
    
    metrics = PerformanceMetrics()
    
    # Phase 1 Tests
    print("\n--- Phase 1: Adaptive Thresholds ---")
    phase1 = Phase1PerformanceTests(metrics)
    phase1.test_key_generation_throughput(iterations=100)
    phase1.test_shamir_sharing_performance(iterations=50)
    phase1.test_adaptive_threshold_performance(iterations=30)
    
    # Phase 2 Tests
    print("\n--- Phase 2: PBFT Consensus ---")
    phase2 = Phase2PerformanceTests(metrics)
    phase2.test_consensus_latency(num_nodes=7, rounds=20)
    phase2.test_byzantine_tolerance(iterations=20)
    
    # Phase 3 Tests
    print("\n--- Phase 3: Zero-Knowledge Proofs ---")
    phase3 = Phase3PerformanceTests(metrics)
    phase3.test_proof_generation_latency(iterations=10)
    phase3.test_proof_verification_throughput(iterations=50)
    
    # Phase 4 Tests
    print("\n--- Phase 4: Secure Multi-Party Computation ---")
    phase4 = Phase4PerformanceTests(metrics)
    phase4.test_smpc_session_throughput(num_parties=5, iterations=10)
    phase4.test_distributed_key_generation(iterations=5)
    
    # Load Tests (if system is running)
    try:
        print("\n--- Load Testing ---")
        load_tests = LoadTests(metrics)
        load_tests.test_concurrent_requests(num_requests=50, concurrency=10)
        # load_tests.test_stress_sustained_load(duration_seconds=30, requests_per_second=5)
    except Exception as e:
        print(f"  Skipping load tests (system not running): {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    
    summary = metrics.get_summary()
    
    print(f"\nLatency Statistics:")
    print(f"  Min: {summary['latency']['min']*1000:.2f}ms")
    print(f"  Max: {summary['latency']['max']*1000:.2f}ms")
    print(f"  Mean: {summary['latency']['mean']*1000:.2f}ms")
    print(f"  Median: {summary['latency']['median']*1000:.2f}ms")
    print(f"  P95: {summary['latency']['p95']*1000:.2f}ms")
    print(f"  P99: {summary['latency']['p99']*1000:.2f}ms")
    
    print(f"\nThroughput Statistics:")
    print(f"  Min: {summary['throughput']['min']:.2f} ops/sec")
    print(f"  Max: {summary['throughput']['max']:.2f} ops/sec")
    print(f"  Mean: {summary['throughput']['mean']:.2f} ops/sec")
    
    print(f"\nError Statistics:")
    print(f"  Total Errors: {summary['errors']['count']}")
    print(f"  Error Rate: {summary['errors']['rate']*100:.2f}%")
    
    print(f"\nTotal Operations: {summary['total_operations']}")
    
    # Save results to file
    results_dir = "performance-results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"performance_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'summary': summary,
            'detailed_metrics': metrics.metrics
        }, f, indent=2)
        
    print(f"\nResults saved to: {results_file}")
    print("\n" + "="*60)
    
    return summary['errors']['count'] == 0

if __name__ == '__main__':
    success = run_performance_suite()
    sys.exit(0 if success else 1)
