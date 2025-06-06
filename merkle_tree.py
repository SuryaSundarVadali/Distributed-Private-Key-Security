"""
Merkle Tree Implementation
Provides data integrity verification through cryptographic hashing
"""
import hashlib
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MerkleNode:
    """Node in a Merkle tree"""
    hash_value: bytes
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    is_leaf: bool = False


class MerkleTree:
    """Merkle tree for integrity verification"""
    
    def __init__(self, data_blocks: List[bytes]):
        if not data_blocks:
            raise ValueError("Cannot create Merkle tree with empty data")
        
        self.data_blocks = data_blocks
        self.leaves = [self._hash_data(block) for block in data_blocks]
        self.root = self._build_tree(self.leaves)
    
    @staticmethod
    def _hash_data(data: bytes) -> bytes:
        """Hash a single data block"""
        return hashlib.sha256(data).digest()
    
    @staticmethod
    def _hash_pair(left: bytes, right: bytes) -> bytes:
        """Hash a pair of hashes"""
        return hashlib.sha256(left + right).digest()
    
    def _build_tree(self, hashes: List[bytes]) -> MerkleNode:
        """Build Merkle tree from leaf hashes"""
        if len(hashes) == 1:
            return MerkleNode(hash_value=hashes[0], is_leaf=True)
        
        # Ensure even number of hashes (duplicate last if odd)
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])
        
        # Build next level
        next_level = []
        nodes = []
        
        for i in range(0, len(hashes), 2):
            left_hash = hashes[i]
            right_hash = hashes[i + 1]
            parent_hash = self._hash_pair(left_hash, right_hash)
            next_level.append(parent_hash)
            
            # Create nodes for current level if this is the leaf level
            if not hasattr(self, '_nodes_created'):
                left_node = MerkleNode(hash_value=left_hash, is_leaf=True)
                right_node = MerkleNode(hash_value=right_hash, is_leaf=True)
                parent_node = MerkleNode(hash_value=parent_hash, left=left_node, right=right_node)
                nodes.append(parent_node)
        
        if len(next_level) == 1:
            return MerkleNode(hash_value=next_level[0])
        
        return self._build_tree(next_level)
    
    def get_proof(self, index: int) -> List[Tuple[bytes, bool]]:
        """Get Merkle proof for data at given index"""
        if index >= len(self.data_blocks):
            raise ValueError("Index out of range")
        
        proof = []
        current_hashes = self.leaves[:]
        current_index = index
        
        while len(current_hashes) > 1:
            # Ensure even number of hashes
            if len(current_hashes) % 2 == 1:
                current_hashes.append(current_hashes[-1])
            
            # Find sibling
            if current_index % 2 == 0:
                # Current is left child, sibling is right
                sibling_index = current_index + 1
                is_left = False
            else:
                # Current is right child, sibling is left
                sibling_index = current_index - 1
                is_left = True
            
            if sibling_index < len(current_hashes):
                proof.append((current_hashes[sibling_index], is_left))
            
            # Move to next level
            next_level = []
            for i in range(0, len(current_hashes), 2):
                left_hash = current_hashes[i]
                right_hash = current_hashes[i + 1]
                parent_hash = self._hash_pair(left_hash, right_hash)
                next_level.append(parent_hash)
            
            current_hashes = next_level
            current_index = current_index // 2
        
        return proof
    
    @staticmethod
    def verify_proof(data: bytes, proof: List[Tuple[bytes, bool]], root_hash: bytes) -> bool:
        """Verify Merkle proof"""
        current_hash = MerkleTree._hash_data(data)
        
        for sibling_hash, is_left in proof:
            if is_left:
                current_hash = MerkleTree._hash_pair(sibling_hash, current_hash)
            else:
                current_hash = MerkleTree._hash_pair(current_hash, sibling_hash)
        
        return current_hash == root_hash
    
    def get_root_hash(self) -> bytes:
        """Get the root hash of the tree"""
        return self.root.hash_value
    
    def verify_tree_integrity(self) -> bool:
        """Verify the integrity of the entire tree"""
        try:
            # Verify each leaf can be proven
            for i in range(len(self.data_blocks)):
                proof = self.get_proof(i)
                if not self.verify_proof(self.data_blocks[i], proof, self.root.hash_value):
                    return False
            return True
        except:
            return False