"""
Enhanced Deterministic Key Generation with MFKDF
Provides secure RSA key generation using multi-factor authentication
"""
import hashlib
from typing import List, Dict
from Crypto.PublicKey import RSA
from mfkdf import MFKDF, MFKDFPolicy, MFKDFFactor


class MFKDFDeterministicKeyGenerator:
    """Generate deterministic RSA keys using MFKDF with multiple authentication factors"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.mfkdf_system = None
        self.factor_secrets = {}
        
    def setup_authentication_factors(self, master_seed: bytes) -> MFKDF:
        """Setup multi-factor authentication system"""
        # Define factor weights (higher = more important)
        factor_weights = {
            'biometric': 3,
            'password': 2, 
            'hardware_token': 2,
            'location': 1,
            'time_window': 1
        }
        
        # Create policy requiring at least 3 factors
        policy = MFKDFPolicy(
            threshold=3,
            factors=list(factor_weights.keys()),
            factor_weights=factor_weights
        )
        
        # Initialize MFKDF system
        self.mfkdf_system = MFKDF(policy)
        
        # Generate deterministic secrets for each factor
        for factor_type in factor_weights.keys():
            # Create factor-specific seed
            factor_seed = hashlib.sha256(
                master_seed + self.node_id.encode() + factor_type.encode()
            ).digest()
            
            # Generate factor secret
            factor_secret = hashlib.pbkdf2_hmac(
                'sha256', 
                factor_seed, 
                b'MFKDF_FACTOR_' + factor_type.encode(), 
                10000, 
                32
            )
            
            # Create and add factor
            factor = MFKDFFactor(
                factor_type=factor_type,
                factor_id=f"{self.node_id}_{factor_type}",
                secret=factor_secret,
                params={'iterations': 10000}
            )
            
            self.mfkdf_system.add_factor(factor)
            self.factor_secrets[factor_type] = factor_secret
        
        return self.mfkdf_system
    
    def generate_rsa_key_with_mfkdf(self, active_factors: List[str], key_size: int = 2048) -> RSA.RsaKey:
        """Generate RSA key using MFKDF-derived entropy"""
        if not self.mfkdf_system:
            raise ValueError("MFKDF system not initialized")
        
        # Derive key material using MFKDF
        context = f"RSA_KEY_GEN:{self.node_id}".encode()
        key_material = self.mfkdf_system.derive_key(active_factors, context, 64)
        
        # Use derived material as entropy for RSA key generation
        rng = MFKDFSecureRNG(key_material, self.node_id)
        
        # Generate RSA key with MFKDF-derived randomness
        return RSA.generate(key_size, randfunc=rng.get_random_bytes)
    
    def derive_factor_verification_key(self, factor_id: str, context: bytes) -> bytes:
        """Derive verification key for a specific factor"""
        if not self.mfkdf_system:
            raise ValueError("MFKDF system not initialized")
        
        # Extract factor type from factor_id
        factor_type = factor_id.split('_')[-1] if '_' in factor_id else factor_id
        
        if factor_type not in self.factor_secrets:
            raise ValueError(f"Factor {factor_type} not found")
        
        # Derive verification key
        factor_secret = self.factor_secrets[factor_type]
        verification_key = hashlib.pbkdf2_hmac(
            'sha256',
            factor_secret,
            context + b'_VERIFICATION',
            5000,
            32
        )
        
        return verification_key
    
    def generate_backup_recovery_data(self) -> Dict[str, any]:
        """Generate backup and recovery data"""
        if not self.mfkdf_system:
            raise ValueError("MFKDF system not initialized")
        
        # Generate recovery shares
        recovery_shares = self.mfkdf_system.generate_recovery_shares(
            num_shares=5, 
            threshold=3
        )
        
        # Generate factor commitments for verification
        commitments = {}
        for factor_id, factor in self.mfkdf_system.factors.items():
            commitments[factor_id] = factor.get_commitment()
        
        # Generate master recovery key
        master_recovery_context = f"RECOVERY:{self.node_id}".encode()
        master_recovery_key = hashlib.pbkdf2_hmac(
            'sha256',
            self.mfkdf_system.master_salt,
            master_recovery_context,
            10000,
            32
        )
        
        return {
            'shares': recovery_shares,
            'commitments': commitments,
            'master_recovery_key': master_recovery_key,
            'node_id': self.node_id
        }
    
    def verify_factor_authenticity(self, factor_type: str, provided_secret: bytes) -> bool:
        """Verify the authenticity of a provided factor secret"""
        if factor_type not in self.factor_secrets:
            return False
        
        expected_secret = self.factor_secrets[factor_type]
        return hashlib.compare_digest(expected_secret, provided_secret)


class MFKDFSecureRNG:
    """Cryptographically secure RNG using MFKDF-derived entropy"""
    
    def __init__(self, master_entropy: bytes, node_id: str):
        self.master_entropy = master_entropy
        self.node_id = node_id
        self.counter = 0
        self.buffer = b''
        
    def get_random_bytes(self, n: int) -> bytes:
        """Generate n random bytes using MFKDF entropy"""
        while len(self.buffer) < n:
            # Generate more entropy
            counter_bytes = self.counter.to_bytes(8, 'big')
            entropy_input = self.master_entropy + self.node_id.encode() + counter_bytes
            new_bytes = hashlib.sha256(entropy_input).digest()
            self.buffer += new_bytes
            self.counter += 1
        
        # Extract requested bytes
        result = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return result
    
    def reseed(self, additional_entropy: bytes):
        """Reseed the RNG with additional entropy"""
        self.master_entropy = hashlib.sha256(
            self.master_entropy + additional_entropy
        ).digest()
        self.counter = 0
        self.buffer = b''