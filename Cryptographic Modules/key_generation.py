"""
Deterministic RSA Key Generation from Master Seed

Provides deterministic RSA key generation for a node, using a master
seed and node_id to derive a per-node entropy pool. This avoids the
complexity and external dependencies of MFKDF while still giving a
well-scoped, reproducible source of key material.

Intended usage in this project:
- For testing and simulation, where you want reproducible RSA keys for
  a given (master_seed, node_id) pair across runs.
- For any place you need a node-scoped asymmetric key that is tied to a
  long-term secret seed.

Security notes:
- This is NOT a replacement for proper human multi-factor
  authentication. It is a deterministic KDF + DRBG on top of a master
  seed. If the master seed is compromised, all derived keys are
  compromised.
- For production systems, ensure master_seed has at least 256 bits of
  entropy and is stored in a secure hardware-bound keystore.
"""

import hashlib
from typing import Dict

from Crypto.PublicKey import RSA


def hkdf_extract_and_expand(
    ikm: bytes,
    salt: bytes,
    info: bytes,
    length: int,
    hash_name: str = "sha256",
) -> bytes:
    """
    Simple HKDF-style extract-and-expand using HMAC based on RFC 5869.

    This is not a full general-purpose HKDF implementation; it is scoped
    for this module's use: deriving a few dozen bytes of key material
    from a master seed, node_id, and context.

    Args:
        ikm: Input keying material (e.g., master seed).
        salt: Optional salt (can be empty but should be random).
        info: Context string for domain separation (e.g., b"RSA_KEY_GEN:node_1").
        length: Number of bytes of output key material.
        hash_name: Underlying hash function name (default: sha256).

    Returns:
        Pseudorandom key material of 'length' bytes.
    """
    if not isinstance(ikm, (bytes, bytearray)) or len(ikm) == 0:
        raise ValueError("ikm must be non-empty bytes")

    hash_mod = getattr(hashlib, hash_name)
    hash_len = hash_mod().digest_size

    # Extract
    if salt is None:
        salt = b"\x00" * hash_len
    prk = hashlib.pbkdf2_hmac(hash_name, ikm, salt, 1, dklen=hash_len)

    # Expand
    okm = b""
    prev = b""
    counter = 1
    while len(okm) < length:
        prev = hashlib.pbkdf2_hmac(
            hash_name,
            prev + info + bytes([counter]),
            prk,
            1,
            dklen=hash_len,
        )
        okm += prev
        counter += 1

    return okm[:length]


class DeterministicRSAKeyGenerator:
    """
    Deterministic RSA key generator for a specific node_id.

    Given a master_seed and node_id, this class derives a per-node
    entropy pool using hkdf_extract_and_expand() and feeds it into a
    DRBG (DeterministicRNG) that PyCryptodome's RSA.generate uses as
    its entropy source.

    The resulting RSA key is deterministic for a fixed (master_seed,
    node_id, context, key_size) tuple, which is useful for reproducible
    experiments and simulations.
    """

    def __init__(self, node_id: str, master_seed: bytes):
        """
        Initialize the deterministic key generator.

        Args:
            node_id: Unique identifier for this node (e.g., "node_0").
            master_seed: High-entropy master seed shared (or stored) by the
                         system. Must be the same across runs to reproduce
                         keys.
        """
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a non-empty string")
        if not isinstance(master_seed, (bytes, bytearray)) or len(master_seed) < 32:
            raise ValueError("master_seed must be >= 32 bytes for 256-bit security")

        self.node_id = node_id
        self.master_seed = bytes(master_seed)

    def _derive_node_entropy(self, context: str, length: int = 64) -> bytes:
        """
        Derive node-specific entropy from the master seed and a context string.

        Args:
            context: ASCII context label (e.g., "RSA_KEY_GEN").
            length: Number of bytes to derive (default: 64).

        Returns:
            Derived entropy bytes.
        """
        info = f"{context}:{self.node_id}".encode("utf-8")
        salt = hashlib.sha256(b"NODE_DERIVATION_SALT").digest()
        return hkdf_extract_and_expand(
            ikm=self.master_seed,
            salt=salt,
            info=info,
            length=length,
            hash_name="sha256",
        )

    def generate_rsa_key(self, key_size: int = 2048) -> RSA.RsaKey:
        """
        Generate a deterministic RSA key pair for this node.

        Args:
            key_size: RSA modulus size in bits (default: 2048).

        Returns:
            RSA.RsaKey object (PyCryptodome).
        """
        # Derive 64 bytes of entropy for this node and this purpose.
        node_entropy = self._derive_node_entropy("RSA_KEY_GEN", length=64)

        rng = DeterministicRNG(node_entropy, self.node_id)
        return RSA.generate(key_size, randfunc=rng.get_random_bytes)

    def derive_subkey(self, purpose: str, length: int = 32) -> bytes:
        """
        Derive a symmetric subkey for this node and a given purpose.

        This is useful if you want node-scoped symmetric keys (e.g.,
        for local storage encryption or MAC keys) without adding another
        KDF elsewhere.

        Args:
            purpose: Application-specific purpose label, e.g. "storage",
                     "local_mac", "logging_hmac".
            length: Number of bytes to derive (default: 32).

        Returns:
            Derived key bytes.
        """
        info = f"SUBKEY:{purpose}:{self.node_id}".encode("utf-8")
        salt = hashlib.sha256(b"SUBKEY_DERIVATION_SALT").digest()
        return hkdf_extract_and_expand(
            ikm=self.master_seed,
            salt=salt,
            info=info,
            length=length,
            hash_name="sha256",
        )


class DeterministicRNG:
    """
    Simple deterministic RNG backed by a hash-chain DRBG.

    Given a seed (master_entropy) and node_id, this produces a
    deterministic stream of pseudorandom bytes suitable for feeding into
    RSA.generate's randfunc parameter.

    This is NOT a system-wide CSPRNG and should NEVER be used as a
    drop-in replacement for os.urandom. It is scoped to generating test
    keys from a fixed seed.
    """

    def __init__(self, master_entropy: bytes, node_id: str):
        if not isinstance(master_entropy, (bytes, bytearray)) or len(master_entropy) == 0:
            raise ValueError("master_entropy must be non-empty bytes")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a non-empty string")

        self.master_entropy = bytes(master_entropy)
        self.node_id = node_id
        self.counter = 0
        self.buffer = b""

    def get_random_bytes(self, n: int) -> bytes:
        """
        Generate n pseudorandom bytes.

        Uses a simple hash-chain construction:
            block_i = SHA-256(master_entropy || node_id || counter_i)
        concatenated until at least n bytes are available.
        """
        while len(self.buffer) < n:
            counter_bytes = self.counter.to_bytes(8, "big")
            entropy_input = self.master_entropy + self.node_id.encode("utf-8") + counter_bytes
            new_bytes = hashlib.sha256(entropy_input).digest()
            self.buffer += new_bytes
            self.counter += 1

        result = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return result

    def reseed(self, additional_entropy: bytes) -> None:
        """
        Reseed the DRBG with additional entropy.

        This mixes the existing master_entropy with additional_entropy
        via SHA-256 and resets the internal counter and buffer.
        """
        if not isinstance(additional_entropy, (bytes, bytearray)):
            raise ValueError("additional_entropy must be bytes")

        self.master_entropy = hashlib.sha256(
            self.master_entropy + additional_entropy
        ).digest()
        self.counter = 0
        self.buffer = b""


if __name__ == "__main__":
    import os

    print("=" * 80)
    print("Deterministic RSA Key Generation - Test")
    print("=" * 80)

    master_seed = os.urandom(32)
    node_id = "node_0"

    gen1 = DeterministicRSAKeyGenerator(node_id, master_seed)
    key1 = gen1.generate_rsa_key(2048)

    # Re-create with same seed and node_id: keys should match
    gen2 = DeterministicRSAKeyGenerator(node_id, master_seed)
    key2 = gen2.generate_rsa_key(2048)

    same_n = key1.n == key2.n
    same_e = key1.e == key2.e
    same_d = key1.d == key2.d

    print(f"  ✓ Modulus equal: {same_n}")
    print(f"  ✓ Public exponent equal: {same_e}")
    print(f"  ✓ Private exponent equal: {same_d}")

    # Derive a subkey for storage
    storage_key = gen1.derive_subkey("storage", length=32)
    print(f"  ✓ Derived storage key (hex): {storage_key.hex()}")

    print("\n" + "=" * 80)
    print("Deterministic RSA Key Generation - Complete")
    print("=" * 80)