"""
HOTP (HMAC-based One-Time Password) Implementation
RFC 4226 compliant token generation and verification.

Reference: RFC 4226 - HOTP: An HMAC-Based One-Time Password Algorithm.
"""

import hashlib
import hmac
import struct
from dataclasses import dataclass
from typing import Optional


# RFC 4226 defines HOTP length as 6–8 digits; 6 is the de facto default.
# We allow up to 9 digits because the 31-bit truncated value is < 10^10,
# so beyond 9 digits the extra digits are just zero-padding.
MIN_DIGITS = 6
MAX_DIGITS = 9

# Limit the resynchronization window to bound brute-force surface.
# RFC 4226 examples use small windows (single digits); we cap at 50.
MAX_WINDOW = 50


@dataclass(frozen=True)
class HOTPVerificationResult:
    """
    Result of an HOTP verification attempt.

    matched_counter is the counter value that produced the match, or None
    if verification failed. Callers MUST persist next_counter = matched_counter + 1
    (not the old stored counter) after a successful verification. Without
    this, a token that matched inside the resync window can be replayed,
    since the server counter would never advance past skipped values.
    """
    valid: bool
    matched_counter: Optional[int] = None


class HOTP:
    """HMAC-based One-Time Password implementation (RFC 4226)."""

    @staticmethod
    def generate(secret: bytes, counter: int, digits: int = 6) -> str:
        """
        Generate an HOTP value per RFC 4226 §5.3.

        Steps:
          1. HMAC-SHA1(secret, counter)
          2. Dynamic truncation to a 31-bit unsigned integer
          3. Reduce mod 10^digits
          4. Zero-pad to 'digits' characters

        Args:
            secret: Shared secret key as raw bytes (not base32/hex encoded).
            counter: Non-negative 64-bit counter value.
            digits: Number of output decimal digits, between MIN_DIGITS and
                    MAX_DIGITS inclusive.

        Raises:
            ValueError: on invalid secret, counter, or digits.
        """
        if not isinstance(secret, (bytes, bytearray)) or len(secret) == 0:
            raise ValueError("secret must be non-empty bytes")
        if not isinstance(counter, int) or counter < 0:
            raise ValueError("counter must be a non-negative integer")
        if counter > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("counter must fit in an unsigned 64-bit integer")
        if not (MIN_DIGITS <= digits <= MAX_DIGITS):
            raise ValueError(f"digits must be between {MIN_DIGITS} and {MAX_DIGITS}")

        # 8-byte big-endian counter
        counter_bytes = struct.pack('>Q', counter)

        # HMAC-SHA1 per RFC 4226
        hmac_hash = hmac.new(bytes(secret), counter_bytes, hashlib.sha1).digest()

        # Dynamic truncation (RFC 4226 §5.3)
        offset = hmac_hash[-1] & 0x0F
        truncated = struct.unpack('>I', hmac_hash[offset:offset + 4])[0]
        truncated &= 0x7FFFFFFF  # 31-bit unsigned value

        hotp_value = truncated % (10 ** digits)
        return str(hotp_value).zfill(digits)

    @staticmethod
    def verify(
        secret: bytes,
        token: str,
        counter: int,
        window: int = 0,
    ) -> HOTPVerificationResult:
        """
        Verify an HOTP token against a resynchronization window
        [counter, counter + window].

        Returns HOTPVerificationResult with:
          - valid: True/False
          - matched_counter: the counter that produced the token (if any)

        Callers MUST advance server-side state to matched_counter + 1
        after a successful verification.

        Args:
            secret: Shared secret key as raw bytes.
            token: Token string to verify, e.g. "123456".
            counter: Server's current expected counter.
            window: Number of counter values ahead of 'counter' to also
                    accept for drift tolerance (capped at MAX_WINDOW).

        Raises:
            ValueError: on invalid counter or window.
        """
        if not isinstance(counter, int) or counter < 0:
            raise ValueError("counter must be a non-negative integer")
        if not isinstance(window, int) or window < 0:
            raise ValueError("window must be a non-negative integer")
        if window > MAX_WINDOW:
            raise ValueError(f"window must not exceed {MAX_WINDOW}")
        if not isinstance(token, str) or not token.isdigit():
            # Reject malformed input without doing HMAC work.
            return HOTPVerificationResult(valid=False)

        token_len = len(token)

        for candidate_counter in range(counter, counter + window + 1):
            candidate = HOTP.generate(secret, candidate_counter, digits=token_len)
            # Constant-time comparison to avoid timing side-channels.
            if hmac.compare_digest(candidate, token):
                return HOTPVerificationResult(valid=True, matched_counter=candidate_counter)

        return HOTPVerificationResult(valid=False)


if __name__ == "__main__":
    import os

    secret = os.urandom(20)  # RFC 4226 recommends ≥160-bit (20-byte) secrets.
    counter = 42

    token = HOTP.generate(secret, counter)
    print(f"Generated token at counter {counter}: {token}")

    # Simulate client being 2 steps ahead
    client_ahead_token = HOTP.generate(secret, counter + 2)
    result = HOTP.verify(secret, client_ahead_token, counter, window=5)
    print(f"Verify (drifted by 2, window=5): valid={result.valid}, matched_counter={result.matched_counter}")

    bad_result = HOTP.verify(secret, "000000", counter, window=5)
    print(f"Verify (wrong token): valid={bad_result.valid}, matched_counter={bad_result.matched_counter}")