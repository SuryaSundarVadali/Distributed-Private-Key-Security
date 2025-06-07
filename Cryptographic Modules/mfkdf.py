"""
Multi-Factor Key Derivation Function (MFKDF) Implementation
Provides secure key derivation using multiple authentication factors
"""
import hashlib
import hmac
import secrets
from typing import List, Dict
from dataclasses import dataclass
from secret_sharing import ShamirSecretSharing

@dataclass
class MFKDFPolicy:
    """Policy configuration for MFKDF"""
    threshold: int
    factors: List[str]
    factor_weights: Dict[str, int]


class MFKDFFactor:
    """Individual factor in MFKDF system"""
    
    def __init__(self, factor_type: str, factor_id: str, secret: bytes, params: Dict = None):
        self.factor_type = factor_type
        self.factor_id = factor_id
        self.secret = secret
        self.params = params or {}
        self.salt = hashlib.sha256(f"{factor_type}:{factor_id}".encode()).digest()
    
    def derive_partial_key(self, context: bytes, length: int) -> bytes:
        """Derive partial key material from this factor"""
        # Use PBKDF2 with factor-specific salt
        from hashlib import pbkdf2_hmac
        iterations = self.params.get('iterations', 10000)
        
        # Combine secret with context for key derivation
        combined_input = self.secret + context + self.factor_id.encode()
        
        return pbkdf2_hmac('sha256', combined_input, self.salt, iterations, length)
    
    def get_commitment(self) -> bytes:
        """Generate cryptographic commitment for this factor"""
        return hashlib.sha256(self.secret + self.salt).digest()


class MFKDF:
    """Multi-Factor Key Derivation Function implementation"""
    
    def __init__(self, policy: MFKDFPolicy):
        self.policy = policy
        self.factors = {}
        self.master_salt = secrets.token_bytes(32)
    
    def add_factor(self, factor: MFKDFFactor):
        """Add a factor to the MFKDF system"""
        if factor.factor_id in self.factors:
            raise ValueError(f"Factor {factor.factor_id} already exists")
        self.factors[factor.factor_id] = factor
    
    def derive_key(self, active_factors: List[str], context: bytes, length: int) -> bytes:
        """Derive key using specified active factors"""
        if len(active_factors) < self.policy.threshold:
            raise ValueError(f"Insufficient factors: need {self.policy.threshold}, got {len(active_factors)}")
        
        # Verify all active factors are available
        for factor_id in active_factors:
            if factor_id not in self.factors:
                raise ValueError(f"Factor {factor_id} not available")
        
        # Derive partial keys from each active factor
        partial_keys = []
        total_weight = 0
        
        for factor_id in active_factors:
            factor = self.factors[factor_id]
            weight = self.policy.factor_weights.get(factor_id, 1)
            
            # Derive partial key with weight consideration
            partial_key = factor.derive_partial_key(context, length)
            
            # Apply weight by repeating the partial key
            weighted_key = partial_key
            for _ in range(weight - 1):
                weighted_key = hashlib.sha256(weighted_key + partial_key).digest()[:length]
            
            partial_keys.append(weighted_key)
            total_weight += weight
        
        # Combine partial keys using secure mixing
        return self._combine_partial_keys(partial_keys, context, length)
    
    def _combine_partial_keys(self, partial_keys: List[bytes], context: bytes, length: int) -> bytes:
        """Securely combine partial keys into final derived key"""
        if not partial_keys:
            raise ValueError("No partial keys to combine")
        
        # Use iterative hashing to combine keys
        combined = partial_keys[0]
        
        for i, partial_key in enumerate(partial_keys[1:], 1):
            # Mix with master salt and context
            mix_input = combined + partial_key + self.master_salt + context + i.to_bytes(4, 'big')
            combined = hashlib.sha256(mix_input).digest()
        
        # Final extraction with HKDF-like expand
        return self._hkdf_expand(combined, context, length)
    
    def _hkdf_expand(self, prk: bytes, info: bytes, length: int) -> bytes:
        """HKDF expand phase for final key derivation"""
        hash_len = hashlib.sha256().digest_size
        n = (length + hash_len - 1) // hash_len
        
        okm = b''
        t = b''
        
        for i in range(1, n + 1):
            t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
            okm += t
        
        return okm[:length]
    
    def generate_recovery_shares(self, num_shares: int, threshold: int) -> List[bytes]:
        """Generate recovery shares for the MFKDF system"""
        # Create recovery secret from all factor commitments
        recovery_input = b''
        for factor in self.factors.values():
            recovery_input += factor.get_commitment()
        
        recovery_secret = hashlib.sha256(recovery_input + self.master_salt).digest()
        
        # Convert to integer for Shamir's Secret Sharing
        secret_int = int.from_bytes(recovery_secret, 'big')
        shares = ShamirSecretSharing.create_shares(secret_int, threshold, num_shares)
        
        # Convert shares back to bytes
        recovery_shares = []
        for x, y in shares:
            share_data = f"{x}:{y}".encode()
            recovery_shares.append(hashlib.sha256(share_data).digest())
        
        return recovery_shares
    
    def verify_factor_integrity(self, factor_id: str, provided_commitment: bytes) -> bool:
        """Verify the integrity of a factor using its commitment"""
        if factor_id not in self.factors:
            return False
        
        expected_commitment = self.factors[factor_id].get_commitment()
        return hmac.compare_digest(expected_commitment, provided_commitment)


class DistributedMFKDF:
    """Distributed Multi-Factor Key Derivation for secure computing"""
    
    def __init__(self, node_count: int, threshold: int):
        self.node_count = node_count
        self.threshold = threshold
        self.node_policies = {}
        self.global_context = secrets.token_bytes(32)
    
    def setup_node_factors(self, node_id: str, factor_configs: List[Dict]) -> MFKDF:
        """Setup MFKDF factors for a specific node"""
        # Create policy for this node
        factor_ids = [config['id'] for config in factor_configs]
        factor_weights = {config['id']: config.get('weight', 1) for config in factor_configs}
        
        policy = MFKDFPolicy(
            threshold=min(len(factor_configs), self.threshold),
            factors=factor_ids,
            factor_weights=factor_weights
        )
        
        # Create MFKDF instance
        mfkdf = MFKDF(policy)
        
        # Add factors
        for config in factor_configs:
            factor = MFKDFFactor(
                factor_type=config['type'],
                factor_id=config['id'],
                secret=config['secret'],
                params=config.get('params', {})
            )
            mfkdf.add_factor(factor)
        
        self.node_policies[node_id] = mfkdf
        return mfkdf
    
    def derive_distributed_key(self, node_id: str, active_factors: List[str], length: int = 32) -> bytes:
        """Derive key for specific node using MFKDF"""
        if node_id not in self.node_policies:
            raise ValueError(f"Node {node_id} not configured")
        
        mfkdf = self.node_policies[node_id]
        node_context = self.global_context + node_id.encode()
        
        return mfkdf.derive_key(active_factors, node_context, length)