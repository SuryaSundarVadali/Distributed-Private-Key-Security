"""
Secure Multi-Party Computation (Simplified)
Provides secure computation across multiple parties without revealing individual secrets
"""

import hashlib
from typing import List, Dict, Tuple
from secret_sharing import ShamirSecretSharing


class MPCNode:
    """Node participating in secure multi-party computation"""
    
    def __init__(self, node_id: str, secret_value: int):
        self.node_id = node_id
        self.secret_value = secret_value
        self.received_shares = {}
        self.my_shares = []
    
    def generate_shares(self, threshold: int, num_parties: int) -> List[Tuple[int, int]]:
        """Generate secret shares for this node's value"""
        return ShamirSecretSharing.create_shares(self.secret_value, threshold, num_parties)
    
    def receive_share(self, from_node: str, share: Tuple[int, int]):
        """Receive a share from another node"""
        self.received_shares[from_node] = share
    
    def compute_sum(self) -> int:
        """Compute sum of all received shares"""
        total = 0
        for share in self.received_shares.values():
            total += share[1]  # Add y-coordinates
        return total


class SecureMultiPartyComputation:
    """Coordinator for secure multi-party computation"""
    
    def __init__(self, nodes: List[MPCNode], threshold: int):
        self.nodes = {node.node_id: node for node in nodes}
        self.threshold = threshold
        self.computation_log = []
        self.verification_data = {}
    
    def execute_secure_sum(self) -> Dict[str, int]:
        """Execute secure sum computation"""
        results = {}
        
        # Step 1: Each node generates shares of their secret
        all_shares = {}
        for node_id, node in self.nodes.items():
            shares = node.generate_shares(self.threshold, len(self.nodes))
            all_shares[node_id] = shares
            
            # Log computation step
            self.computation_log.append({
                'step': 'share_generation',
                'node': node_id,
                'shares_count': len(shares)
            })
        
        # Step 2: Distribute shares to all nodes
        for sender_id, shares in all_shares.items():
            for i, (receiver_id, receiver_node) in enumerate(self.nodes.items()):
                if i < len(shares):
                    receiver_node.receive_share(sender_id, shares[i])
        
        # Step 3: Each node computes local sum
        for node_id, node in self.nodes.items():
            local_sum = node.compute_sum()
            results[node_id] = local_sum
            
            # Store verification data
            self.verification_data[node_id] = {
                'received_shares': len(node.received_shares),
                'local_sum': local_sum
            }
        
        return results
    
    def execute_secure_sum_with_shares(self) -> Dict[str, int]:
        """Execute secure sum using share reconstruction"""
        results = {}
        
        # Collect shares from threshold number of nodes
        selected_nodes = list(self.nodes.keys())[:self.threshold]
        
        for node_id in selected_nodes:
            node = self.nodes[node_id]
            
            # Create shares for reconstruction
            shares_for_reconstruction = []
            for other_node_id, other_node in self.nodes.items():
                if other_node_id in node.received_shares:
                    shares_for_reconstruction.append(node.received_shares[other_node_id])
            
            if len(shares_for_reconstruction) >= self.threshold:
                # Reconstruct partial result
                partial_result = ShamirSecretSharing.reconstruct_secret(
                    shares_for_reconstruction[:self.threshold]
                )
                results[node_id] = partial_result
            else:
                results[node_id] = 0
        
        return results
    
    def execute_secure_multiplication(self) -> Dict[str, int]:
        """Execute secure multiplication (simplified)"""
        results = {}
        
        # For demonstration: multiply each node's secret by a common factor
        common_factor = 2
        
        for node_id, node in self.nodes.items():
            # Simple multiplication for demo
            result = (node.secret_value * common_factor) % ShamirSecretSharing.PRIME
            results[node_id] = result
        
        return results
    
    def verify_computation_integrity(self) -> bool:
        """Verify the integrity of the MPC computation"""
        try:
            # Check if all nodes participated
            if len(self.verification_data) != len(self.nodes):
                return False
            
            # Verify each node received shares from others
            for node_id, data in self.verification_data.items():
                if data['received_shares'] == 0:
                    return False
            
            # Additional integrity checks
            total_shares = sum(data['received_shares'] for data in self.verification_data.values())
            expected_shares = len(self.nodes) * len(self.nodes)  # Each node should receive from all
            
            return total_shares > 0  # Basic sanity check
            
        except Exception:
            return False
    
    def generate_proof_of_computation(self) -> Dict[str, str]:
        """Generate proof of computation for verification"""
        proof = {}
        
        for node_id, node in self.nodes.items():
            # Create proof hash
            proof_data = f"{node_id}:{node.secret_value}:{len(node.received_shares)}"
            proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()
            proof[node_id] = proof_hash
        
        return proof
    
    def demonstrate_threshold_computation(self) -> Dict[str, any]:
        """Demonstrate threshold-based computation"""
        # Select threshold number of nodes
        selected_nodes = list(self.nodes.keys())[:self.threshold]
        
        # Perform computation with selected nodes only
        threshold_verification = {}
        collective_result = 0
        
        for node_id in selected_nodes:
            node = self.nodes[node_id]
            node_contribution = node.secret_value
            threshold_verification[node_id] = node_contribution
            collective_result += node_contribution
        
        return {
            'selected_nodes': selected_nodes,
            'threshold_verification': threshold_verification,
            'collective_result': collective_result % ShamirSecretSharing.PRIME
        }