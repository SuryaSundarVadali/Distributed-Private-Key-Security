"""
HOTP (HMAC-based One-Time Password) Implementation
Provides time-based token generation and verification
"""

import hashlib
import hmac
import struct


class HOTP:
    """HMAC-based One-Time Password implementation"""
    
    @staticmethod
    def generate(secret: bytes, counter: int, digits: int = 6) -> str:
        """Generate HOTP value"""
        # Convert counter to 8-byte big-endian
        counter_bytes = struct.pack('>Q', counter)
        
        # Generate HMAC-SHA1
        hmac_hash = hmac.new(secret, counter_bytes, hashlib.sha1).digest()
        
        # Dynamic truncation
        offset = hmac_hash[-1] & 0x0f
        truncated = struct.unpack('>I', hmac_hash[offset:offset+4])[0]
        truncated &= 0x7fffffff  # Remove sign bit
        
        # Generate final HOTP value
        hotp_value = truncated % (10 ** digits)
        return str(hotp_value).zfill(digits)
    
    @staticmethod
    def verify(secret: bytes, token: str, counter: int, window: int = 0) -> bool:
        """Verify HOTP token within synchronization window"""
        for i in range(counter, counter + window + 1):
            if HOTP.generate(secret, i, len(token)) == token:
                return True
        return False