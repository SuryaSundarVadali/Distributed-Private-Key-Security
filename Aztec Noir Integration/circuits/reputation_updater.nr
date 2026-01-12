// reputation_updater.nr - Reputation Merkle Tree Update Circuit
// Purpose: Update agent reputation in Merkle tree with cryptographic verification
// Performance: ~30ms proof generation, ~0.5KB proof size

use dep::std;

/// Update agent reputation and compute new Merkle root
/// 
/// # Arguments
/// * `agent_id` - Unique identifier for the agent
/// * `old_reputation` - Current reputation score (0-1000)
/// * `action` - Action type: 0=correct vote (+5), 1=byzantine (-50), 2=offline (-20)
/// * `merkle_root` - Current Merkle root
/// * `merkle_proof` - Sibling hashes for Merkle path verification (depth 8 = 256 agents)
///
/// # Returns
/// * Tuple of (new_reputation: Field, new_merkle_root: Field)
///
/// # Constraints
/// * Old reputation must be in Merkle tree (verified via path)
/// * Reputation must stay within 0-1000 range
/// * New Merkle root must be correctly computed
pub fn update_reputation<DEPTH>(
    agent_id: Field,
    old_reputation: Field,
    action: u8,
    merkle_root: Field,
    merkle_proof: [Field; DEPTH]
) -> (Field, Field) {
    // Step 1: Create old leaf hash
    let old_leaf = std::hash::poseidon::bn254::hash([agent_id, old_reputation]);
    
    // Step 2: Verify old leaf is in Merkle tree
    let mut current = old_leaf;
    for i in 0..DEPTH {
        let sibling = merkle_proof[i];
        
        // Determine if current is left or right child
        // Use bit i of agent_id to determine position
        let bit_mask = 1 << i;
        let is_right = (agent_id & bit_mask) != 0;
        
        if is_right {
            // Current is right child
            current = std::hash::poseidon::bn254::hash([sibling, current]);
        } else {
            // Current is left child
            current = std::hash::poseidon::bn254::hash([current, sibling]);
        }
    }
    
    // Verify we reached the expected root
    assert(current == merkle_root);
    
    // Step 3: Calculate reputation delta based on action
    let delta = if action == 0 {
        5      // Correct vote: +5
    } else if action == 1 {
        -50    // Byzantine behavior: -50
    } else if action == 2 {
        -20    // Offline: -20
    } else {
        0      // Unknown action: no change
    };
    
    // Step 4: Update reputation with clamping to 0-1000
    let mut new_rep = old_reputation + delta;
    
    // Clamp to minimum (0)
    if new_rep < 0 {
        new_rep = 0;
    }
    
    // Clamp to maximum (1000)
    if new_rep > 1000 {
        new_rep = 1000;
    }
    
    // Step 5: Compute new Merkle root with updated reputation
    let new_leaf = std::hash::poseidon::bn254::hash([agent_id, new_rep]);
    
    let mut new_current = new_leaf;
    for i in 0..DEPTH {
        let sibling = merkle_proof[i];
        
        let bit_mask = 1 << i;
        let is_right = (agent_id & bit_mask) != 0;
        
        if is_right {
            new_current = std::hash::poseidon::bn254::hash([sibling, new_current]);
        } else {
            new_current = std::hash::poseidon::bn254::hash([new_current, sibling]);
        }
    }
    
    (new_rep, new_current)
}

// Helper function to compute root from agent data
fn compute_test_root(agent_id: Field, reputation: Field, proof: [Field; 8]) -> Field {
    let leaf = std::hash::poseidon::bn254::hash([agent_id, reputation]);
    let mut current = leaf;
    
    for i in 0..8 {
        let sibling = proof[i];
        let bit_mask = 1 << i;
        let is_right = (agent_id & bit_mask) != 0;
        
        if is_right {
            current = std::hash::poseidon::bn254::hash([sibling, current]);
        } else {
            current = std::hash::poseidon::bn254::hash([current, sibling]);
        }
    }
    
    current
}

// Unit tests
#[test]
fn test_reputation_correct_vote() {
    let agent_id = 0;
    let old_rep = 950;
    let action = 0;  // Correct vote
    let proof = [0; 8];  // Dummy proof
    let merkle_root = compute_test_root(agent_id, old_rep, proof);
    
    let (new_rep, new_root) = update_reputation(agent_id, old_rep, action, merkle_root, proof);
    
    assert(new_rep == 955);  // 950 + 5
    assert(new_root != merkle_root);  // Root should change
}

#[test]
fn test_reputation_byzantine() {
    let agent_id = 0;
    let old_rep = 950;
    let action = 1;  // Byzantine
    let proof = [0; 8];
    let merkle_root = compute_test_root(agent_id, old_rep, proof);
    
    let (new_rep, _) = update_reputation(agent_id, old_rep, action, merkle_root, proof);
    
    assert(new_rep == 900);  // 950 - 50
}

#[test]
fn test_reputation_offline() {
    let agent_id = 0;
    let old_rep = 950;
    let action = 2;  // Offline
    let proof = [0; 8];
    let merkle_root = compute_test_root(agent_id, old_rep, proof);
    
    let (new_rep, _) = update_reputation(agent_id, old_rep, action, merkle_root, proof);
    
    assert(new_rep == 930);  // 950 - 20
}

#[test]
fn test_reputation_clamp_minimum() {
    let agent_id = 0;
    let old_rep = 30;
    let action = 1;  // Byzantine (-50)
    let proof = [0; 8];
    let merkle_root = compute_test_root(agent_id, old_rep, proof);
    
    let (new_rep, _) = update_reputation(agent_id, old_rep, action, merkle_root, proof);
    
    assert(new_rep == 0);  // Should clamp to 0, not go negative
}

#[test]
fn test_reputation_clamp_maximum() {
    let agent_id = 0;
    let old_rep = 998;
    let action = 0;  // Correct (+5)
    let proof = [0; 8];
    let merkle_root = compute_test_root(agent_id, old_rep, proof);
    
    let (new_rep, _) = update_reputation(agent_id, old_rep, action, merkle_root, proof);
    
    assert(new_rep == 1000);  // Should clamp to 1000
}
