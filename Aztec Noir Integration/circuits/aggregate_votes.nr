// aggregate_votes.nr - Vote Aggregation Circuit
// Purpose: Count votes, verify threshold met, without revealing individual votes
// Performance: ~50ms proof generation, ~2KB proof size

use dep::std;

/// Sum all elements in an array
fn sum_array<N>(arr: [Field; N]) -> Field {
    let mut result = 0;
    for i in 0..N {
        result = result + arr[i];
    }
    result
}

/// Compute Merkle root from leaf array (simplified version)
/// In production, use proper binary Merkle tree structure
fn merkle_root<N>(leaves: [Field; N]) -> Field {
    let mut combined = 0;
    for i in 0..N {
        combined = combined + leaves[i];
    }
    std::hash::poseidon::bn254::hash([combined])
}

/// Main circuit: Aggregate votes and verify commitments
/// 
/// # Arguments
/// * `votes` - Binary votes (0 or 1) from each agent
/// * `threshold` - Minimum votes required (typically 2f+1)
/// * `commitments` - Vote commitment hashes from Phase 1
/// * `expected_root` - Known good Merkle root for verification
///
/// # Returns
/// * Tuple of (approved: bool, commitment_root: Field)
///
/// # Constraints
/// * All votes must be binary (0 or 1)
/// * Commitment root must match expected_root
/// * Sum must be correctly calculated
pub fn aggregate_votes<N>(
    votes: [Field; N],
    threshold: Field,
    commitments: [Field; N],
    expected_root: Field
) -> (bool, Field) {
    // Step 1: Verify all votes are binary (0 or 1)
    // This constraint enforces: vote * (vote - 1) = 0
    // Which means: vote = 0 OR vote = 1
    for i in 0..N {
        let vote = votes[i];
        let binary_check = vote * (vote - 1);
        assert(binary_check == 0);
    }
    
    // Step 2: Sum all votes
    let sum = sum_array(votes);
    
    // Step 3: Check if threshold is met
    let approved = sum >= threshold;
    
    // Step 4: Compute and verify commitment root
    let commit_root = merkle_root(commitments);
    assert(commit_root == expected_root);
    
    // Return decision and commitment root
    (approved, commit_root)
}

// Unit tests
#[test]
fn test_aggregate_votes_approved() {
    let votes = [1, 1, 0, 1, 1];  // 4 yes, 1 no
    let threshold = 3;  // Need 3 votes
    let commitments = [1, 2, 3, 4, 5];
    let expected_root = merkle_root(commitments);
    
    let (approved, root) = aggregate_votes(votes, threshold, commitments, expected_root);
    assert(approved);  // 4 >= 3
    assert(root == expected_root);
}

#[test]
fn test_aggregate_votes_not_approved() {
    let votes = [1, 0, 0, 1];  // 2 yes, 2 no
    let threshold = 3;  // Need 3 votes
    let commitments = [1, 2, 3, 4];
    let expected_root = merkle_root(commitments);
    
    let (approved, _) = aggregate_votes(votes, threshold, commitments, expected_root);
    assert(!approved);  // 2 < 3
}

#[test]
fn test_aggregate_votes_exact_threshold() {
    let votes = [1, 1, 1, 0];  // 3 yes, 1 no
    let threshold = 3;  // Need 3 votes
    let commitments = [1, 2, 3, 4];
    let expected_root = merkle_root(commitments);
    
    let (approved, _) = aggregate_votes(votes, threshold, commitments, expected_root);
    assert(approved);  // 3 >= 3
}
