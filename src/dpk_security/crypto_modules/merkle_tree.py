"""
Merkle Tree Implementation
Provides data integrity verification through cryptographic hashing.

Construction follows RFC 6962 ("Certificate Transparency") Section 2.1's
Merkle Tree Hash (MTH) definition: the tree is built by splitting the
input at the largest power of two strictly less than the remaining leaf
count, producing an unbalanced binary tree, rather than pairing adjacent
hashes level-by-level and duplicating a lone leftover leaf when the count
is odd.

This matters because "duplicate the last leaf if the count is odd" is
the exact construction behind CVE-2012-2459: an attacker who controls
part of the input can append a duplicate of the last data block and
produce an IDENTICAL root hash for a data set that differs from the
original by one repeated entry. A verifier checking only the root hash
(as Bitcoin block headers do) cannot tell the two apart, and a leaf
proof can end up validating against data the tree owner never actually
committed to. The power-of-two-split construction never pads or
duplicates, so tree shape is a pure function of leaf count and this
class of attack does not apply.

Hashing also uses domain separation: leaf hashes are HASH(0x00 || data)
and internal node hashes are HASH(0x01 || left || right). Without this,
an internal node's hash HASH(leaf_L || leaf_R) is structurally
indistinguishable from a leaf hash HASH(data) for some data that happens
to equal leaf_L || leaf_R -- letting an attacker pass an internal node
off as a leaf (or vice versa) in a forged proof. The prefix removes
that ambiguity entirely.
"""

import hashlib
import hmac
import json
from dataclasses import dataclass, is_dataclass, asdict
from typing import List, Tuple, Optional, Any


# Single source of truth for the hash function used throughout this
# module. Every hash in this file (leaf, internal node, and proof
# verification) goes through this constant. Do not call hashlib.sha256
# (or any other hash) directly elsewhere -- mixing hash functions
# between leaves, internal nodes, and verification would make proofs
# non-portable and could reopen the exact ambiguity domain separation
# is meant to close.
HASH_FN = hashlib.sha256

LEAF_PREFIX = b'\x00'
NODE_PREFIX = b'\x01'


def canonical_serialize(obj: Any) -> bytes:
    """
    Canonical serialization for Merkle tree leaf inputs.

    Hashes are only meaningful as integrity checks if every party
    (potentially in different languages/implementations) derives the
    exact same bytes for the exact same logical data. Passing arbitrary
    Python objects to hashlib -- or relying on dict/JSON key ordering
    that Python happens to preserve -- does not guarantee that.

    Rules:
        - bytes/bytearray are used as-is (already canonical).
        - dataclass instances are converted via dataclasses.asdict()
          and then serialized as below.
        - dict/list/tuple/str/int/float/bool/None are serialized as
          JSON with sorted keys and no incidental whitespace, so two
          logically-equal objects built in different key/construction
          order always produce identical bytes.

    Raises:
        TypeError: for inputs with no well-defined canonical form
            (e.g. sets, arbitrary class instances, NaN/inf floats).
    """
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if obj is None or isinstance(obj, (dict, list, tuple, str, int, bool)):
        return json.dumps(
            obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True
        ).encode('utf-8')
    if isinstance(obj, float):
        import math
        if not math.isfinite(obj):
            raise TypeError("Cannot canonically serialize non-finite float (NaN/inf)")
        return json.dumps(
            obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True
        ).encode('utf-8')
    raise TypeError(
        f"Cannot canonically serialize object of type {type(obj)!r}; "
        f"pass raw bytes, or a dict/list/dataclass/JSON-serializable value"
    )


@dataclass
class MerkleNode:
    """Node in a Merkle tree"""
    hash_value: bytes
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    is_leaf: bool = False


class MerkleTree:
    """
    Merkle tree for integrity verification.

    Construction: RFC 6962-style Merkle Tree Hash over an unbalanced
    binary tree (power-of-two split), with domain-separated leaf/internal
    hashing. See module docstring for why this matters.
    """

    def __init__(self, data_blocks: List[bytes]):
        if not data_blocks:
            raise ValueError("Cannot create Merkle tree with empty data")

        # Copy defensively. The original implementation passed
        # self.leaves directly into a recursive builder that mutated it
        # in place (appending a duplicate hash when the count was odd),
        # silently corrupting self.leaves so its length no longer
        # matched self.data_blocks. Never mutate a list we intend to
        # keep as authoritative state, and never mutate a caller's list.
        self.data_blocks: List[bytes] = list(data_blocks)
        self.leaves: List[bytes] = [self._hash_leaf(block) for block in self.data_blocks]
        self.root: MerkleNode = self._build_tree(self.leaves)

    @staticmethod
    def _hash_leaf(data: bytes) -> bytes:
        """Hash a single data block with the leaf domain-separation prefix."""
        return HASH_FN(LEAF_PREFIX + data).digest()

    @staticmethod
    def _hash_internal(left: bytes, right: bytes) -> bytes:
        """Hash a pair of child hashes with the internal-node prefix."""
        return HASH_FN(NODE_PREFIX + left + right).digest()

    @staticmethod
    def _split_point(n: int) -> int:
        """
        Largest power of two strictly less than n (RFC 6962 MTH
        definition). This determines how a range of n leaves is split
        into a left subtree of size k and a right subtree of size n-k.
        Used identically during both tree construction and proof
        generation -- they must never drift out of sync, or proofs
        generated here will fail to verify against roots built here.
        """
        k = 1
        while k * 2 < n:
            k *= 2
        return k

    def _build_tree(self, leaf_hashes: List[bytes]) -> MerkleNode:
        """
        Recursively build the tree over leaf_hashes following the
        RFC 6962 Merkle Tree Hash definition:
            MTH({d0})   = leaf_hash(d0)
            MTH(D[0:n]) = hash_internal(MTH(D[0:k]), MTH(D[k:n]))
                          where k = largest power of two < n
        No padding or duplication is ever introduced, which is what
        prevents the CVE-2012-2459-style duplicate-leaf attack (see
        module docstring).
        """
        n = len(leaf_hashes)
        if n == 1:
            return MerkleNode(hash_value=leaf_hashes[0], is_leaf=True)

        k = self._split_point(n)
        left_node = self._build_tree(leaf_hashes[:k])
        right_node = self._build_tree(leaf_hashes[k:])
        parent_hash = self._hash_internal(left_node.hash_value, right_node.hash_value)
        return MerkleNode(hash_value=parent_hash, left=left_node, right=right_node)

    def get_proof(self, index: int) -> List[Tuple[bytes, bool]]:
        """
        Get the Merkle audit proof for the data block at `index`.

        Returns a list of (sibling_hash, sibling_is_left) pairs ordered
        from the leaf level up to the root, so verify_proof can apply
        them in sequence starting from the leaf's own hash. Walks the
        actual node tree built in __init__ (O(log n)) using the same
        _split_point logic used to build it, rather than recomputing
        subtree hashes from scratch.
        """
        if not (0 <= index < len(self.data_blocks)):
            raise ValueError("Index out of range")

        # Collected root-to-leaf (top-down) while descending; reversed
        # at the end since verify_proof needs to apply the nearest
        # (leaf-level) sibling first and work up to the root.
        proof_top_down: List[Tuple[bytes, bool]] = []
        node = self.root
        n = len(self.leaves)
        idx = index

        while n > 1:
            k = self._split_point(n)
            if idx < k:
                proof_top_down.append((node.right.hash_value, False))  # sibling is on the right
                node = node.left
                n = k
            else:
                proof_top_down.append((node.left.hash_value, True))  # sibling is on the left
                node = node.right
                idx -= k
                n -= k

        proof_top_down.reverse()
        return proof_top_down

    @staticmethod
    def verify_proof(data: bytes, proof: List[Tuple[bytes, bool]], root_hash: bytes) -> bool:
        """
        Verify a Merkle audit proof for `data` against `root_hash`.
        `proof` must be ordered leaf-to-root, as returned by get_proof.
        """
        current_hash = MerkleTree._hash_leaf(data)

        for sibling_hash, sibling_is_left in proof:
            if sibling_is_left:
                current_hash = MerkleTree._hash_internal(sibling_hash, current_hash)
            else:
                current_hash = MerkleTree._hash_internal(current_hash, sibling_hash)

        # Constant-time comparison: a plain == on the final hash can
        # leak timing information about how many leading bytes matched,
        # which is unnecessary risk for a security-relevant comparison.
        return hmac.compare_digest(current_hash, root_hash)

    def get_root_hash(self) -> bytes:
        """Get the root hash of the tree"""
        return self.root.hash_value

    def verify_tree_integrity(self) -> bool:
        """
        Verify that every leaf's proof correctly reconstructs the root.
        """
        try:
            for i in range(len(self.data_blocks)):
                proof = self.get_proof(i)
                if not self.verify_proof(self.data_blocks[i], proof, self.root.hash_value):
                    return False
            return True
        except (ValueError, IndexError):
            # Only catch errors we understand the cause of. A bare
            # `except:` would also swallow KeyboardInterrupt/SystemExit
            # and silently mask real bugs as "tree is invalid".
            return False


if __name__ == "__main__":
    # Basic sanity checks, including the duplicate-leaf scenario that
    # broke the naive pairwise-duplicate construction (CVE-2012-2459).

    def make_tree(blocks):
        return MerkleTree([canonical_serialize(b) if not isinstance(b, bytes) else b for b in blocks])

    # 1) Odd count, proofs verify correctly for every index.
    t = make_tree([b"a", b"b", b"c"])
    assert t.verify_tree_integrity()
    print("Odd-count tree: integrity OK, root =", t.get_root_hash().hex())

    # 2) Single element.
    t1 = make_tree([b"only"])
    assert t1.verify_tree_integrity()
    print("Single-leaf tree: integrity OK, root =", t1.get_root_hash().hex())

    # 3) Even count.
    t2 = make_tree([b"a", b"b", b"c", b"d"])
    assert t2.verify_tree_integrity()
    print("Even-count tree: integrity OK, root =", t2.get_root_hash().hex())

    # 4) Tampered data must fail verification.
    proof = t.get_proof(0)
    assert not MerkleTree.verify_proof(b"tampered", proof, t.get_root_hash())
    print("Tampered-data proof correctly rejected")

    # 5) Duplicate-leaf malleability check: [a, b, c] vs [a, b, c, c]
    #    must NOT produce the same root (this was the CVE-2012-2459 bug).
    t3 = make_tree([b"a", b"b", b"c"])
    t4 = make_tree([b"a", b"b", b"c", b"c"])
    assert t3.get_root_hash() != t4.get_root_hash()
    print("Duplicate-leaf malleability check passed: roots differ as expected")

    # 6) Canonical serialization: key order shouldn't affect the hash.
    obj1 = {"b": 2, "a": 1}
    obj2 = {"a": 1, "b": 2}
    assert canonical_serialize(obj1) == canonical_serialize(obj2)
    print("Canonical serialization is order-independent, as expected")

    print("\nAll checks passed.")