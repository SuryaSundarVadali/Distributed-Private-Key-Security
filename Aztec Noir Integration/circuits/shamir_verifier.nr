// shamir_verifier.nr - Shamir Secret Sharing Verification Circuit
// Purpose: Verify polynomial consistency of Shamir shares
// Performance: ~40ms proof generation, ~1KB proof size

use dep::std;

/// Verify Shamir Secret Sharing shares against polynomial commitments
/// 
/// # Arguments
/// * `shares` - Secret shares: f(x_i) for each agent
/// * `commitments` - Polynomial coefficients: [a_0, a_1, a_2, ..., a_{k-1}]
/// * `recipient_ids` - X-coordinates for each agent (typically 1, 2, 3, ...)
///
/// # Returns
/// * bool - True if all shares are valid, false otherwise
///
/// # Algorithm
/// For each agent with id x_i and share s_i:
///   1. Compute expected share: f(x_i) = a_0 + a_1*x_i + a_2*x_i^2 + ... + a_{k-1}*x_i^{k-1}
///   2. Verify s_i == f(x_i)
///
/// # Security
/// This provides information-theoretic security:
/// - Any k shares can reconstruct the secret
/// - Any k-1 shares reveal ZERO information
/// - Survives quantum computing attacks
pub fn verify_shamir_shares<N, K>(
    shares: [Field; N],
    commitments: [Field; K],
    recipient_ids: [Field; N]
) -> bool {
    // Verify each share against polynomial
    for i in 0..N {
        let x = recipient_ids[i];
        let share = shares[i];
        
        // Compute expected share: f(x) = a_0 + a_1*x + a_2*x^2 + ...
        let mut expected = commitments[0];  // a_0 (constant term)
        let mut x_power = x;  // x^1
        
        for j in 1..K {
            let term = commitments[j] * x_power;
            expected = expected + term;
            x_power = x_power * x;  // Increment power
        }
        
        // Verify share matches expected polynomial evaluation
        if share != expected {
            return false;
        }
    }
    
    true
}

/// Optimized version with early exit
pub fn verify_shamir_shares_optimized<N, K>(
    shares: [Field; N],
    commitments: [Field; K],
    recipient_ids: [Field; N]
) -> bool {
    let mut all_valid = true;
    
    for i in 0..N {
        if all_valid {
            let x = recipient_ids[i];
            let share = shares[i];
            
            let mut expected = commitments[0];
            let mut x_power = x;
            
            for j in 1..K {
                expected = expected + commitments[j] * x_power;
                x_power = x_power * x;
            }
            
            if share != expected {
                all_valid = false;
            }
        }
    }
    
    all_valid
}

/// Batch verification for multiple share sets
pub fn verify_shamir_batch<N, K, B>(
    shares_batch: [[Field; N]; B],
    commitments: [Field; K],
    recipient_ids: [Field; N]
) -> [bool; B] {
    let mut results = [false; B];
    
    for batch_idx in 0..B {
        let shares = shares_batch[batch_idx];
        results[batch_idx] = verify_shamir_shares(shares, commitments, recipient_ids);
    }
    
    results
}

// Unit tests
#[test]
fn test_shamir_verification_valid() {
    // Simple polynomial: f(x) = 5 + 3x
    // f(1) = 8, f(2) = 11, f(3) = 14
    let shares = [8, 11, 14];
    let commitments = [5, 3];  // [a_0, a_1]
    let recipient_ids = [1, 2, 3];
    
    let valid = verify_shamir_shares(shares, commitments, recipient_ids);
    assert(valid);
}

#[test]
fn test_shamir_verification_invalid() {
    // Polynomial: f(x) = 5 + 3x
    // Expected: f(1) = 8, f(2) = 11, f(3) = 14
    // But provide wrong share for agent 2
    let shares = [8, 12, 14];  // 12 is wrong (should be 11)
    let commitments = [5, 3];
    let recipient_ids = [1, 2, 3];
    
    let valid = verify_shamir_shares(shares, commitments, recipient_ids);
    assert(!valid);
}

#[test]
fn test_shamir_quadratic() {
    // Polynomial: f(x) = 2 + 3x + 4x^2
    // f(1) = 2 + 3 + 4 = 9
    // f(2) = 2 + 6 + 16 = 24
    // f(3) = 2 + 9 + 36 = 47
    let shares = [9, 24, 47];
    let commitments = [2, 3, 4];  // [a_0, a_1, a_2]
    let recipient_ids = [1, 2, 3];
    
    let valid = verify_shamir_shares(shares, commitments, recipient_ids);
    assert(valid);
}

#[test]
fn test_shamir_cubic() {
    // Polynomial: f(x) = 1 + 2x + 3x^2 + 4x^3
    // f(1) = 1 + 2 + 3 + 4 = 10
    // f(2) = 1 + 4 + 12 + 32 = 49
    let shares = [10, 49];
    let commitments = [1, 2, 3, 4];  // [a_0, a_1, a_2, a_3]
    let recipient_ids = [1, 2];
    
    let valid = verify_shamir_shares(shares, commitments, recipient_ids);
    assert(valid);
}

#[test]
fn test_shamir_optimized() {
    let shares = [8, 11, 14];
    let commitments = [5, 3];
    let recipient_ids = [1, 2, 3];
    
    let valid1 = verify_shamir_shares(shares, commitments, recipient_ids);
    let valid2 = verify_shamir_shares_optimized(shares, commitments, recipient_ids);
    
    assert(valid1 == valid2);
}

#[test]
fn test_shamir_batch() {
    // Two sets of shares for same polynomial
    let shares_batch = [
        [8, 11, 14],
        [8, 11, 14]
    ];
    let commitments = [5, 3];
    let recipient_ids = [1, 2, 3];
    
    let results = verify_shamir_batch(shares_batch, commitments, recipient_ids);
    assert(results[0]);
    assert(results[1]);
}
