"""
Shamir's Secret Sharing Implementation
Provides secure secret distribution and reconstruction
"""
import secrets
from typing import List, Tuple


class ShamirSecretSharing:
    """Implementation of Shamir's Secret Sharing scheme"""
    
    # Large prime for finite field arithmetic
    PRIME = 2**256 - 189
    
    @staticmethod
    def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean algorithm"""
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = ShamirSecretSharing._extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    @staticmethod
    def _mod_inverse(a: int, m: int) -> int:
        """Modular multiplicative inverse"""
        gcd, x, _ = ShamirSecretSharing._extended_gcd(a % m, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m
    
    @staticmethod
    def _polynomial_evaluate(coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x"""
        result = 0
        for i, coeff in enumerate(coefficients):
            result = (result + coeff * pow(x, i, ShamirSecretSharing.PRIME)) % ShamirSecretSharing.PRIME
        return result
    
    @staticmethod
    def create_shares(secret: int, threshold: int, num_shares: int) -> List[Tuple[int, int]]:
        """Create shares for secret sharing"""
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")
        
        # Generate random coefficients for polynomial
        coefficients = [secret % ShamirSecretSharing.PRIME]
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(ShamirSecretSharing.PRIME))
        
        # Generate shares
        shares = []
        for i in range(1, num_shares + 1):
            x = i
            y = ShamirSecretSharing._polynomial_evaluate(coefficients, x)
            shares.append((x, y))
        
        return shares
    
    @staticmethod
    def reconstruct_secret(shares: List[Tuple[int, int]]) -> int:
        """Reconstruct secret from shares using Lagrange interpolation"""
        if len(shares) < 2:
            raise ValueError("Need at least 2 shares to reconstruct")
        
        secret = 0
        prime = ShamirSecretSharing.PRIME
        
        for i, (xi, yi) in enumerate(shares):
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % prime
                    denominator = (denominator * (xi - xj)) % prime
            
            # Calculate Lagrange coefficient
            lagrange_coeff = (numerator * ShamirSecretSharing._mod_inverse(denominator, prime)) % prime
            secret = (secret + yi * lagrange_coeff) % prime
        
        return secret
    
    @staticmethod
    def verify_shares(shares: List[Tuple[int, int]], threshold: int) -> bool:
        """Verify that shares are consistent"""
        if len(shares) < threshold:
            return False
        
        try:
            # Try to reconstruct with different subsets
            for i in range(len(shares) - threshold + 1):
                subset1 = shares[i:i+threshold]
                subset2 = shares[i+1:i+threshold+1] if i+threshold+1 <= len(shares) else shares[:threshold]
                
                if len(subset2) == threshold:
                    secret1 = ShamirSecretSharing.reconstruct_secret(subset1)
                    secret2 = ShamirSecretSharing.reconstruct_secret(subset2)
                    
                    if secret1 != secret2:
                        return False
            return True
        except:
            return False