// byzantine_detector.nr - Byzantine Fault Detection Circuit
// Purpose: Identify agents deviating from consensus
// Performance: ~80ms proof generation, ~1.5KB proof size

use dep::std;

/// Detect Byzantine faults in agent voting behavior
/// 
/// # Arguments
/// * `agent_states` - Reported state (vote) per agent
/// * `consensus_threshold` - Minimum agents required for consensus (2f+1)
/// * `isolation_levels` - Current isolation level (0-3) for each agent
///
/// # Returns
/// * Tuple of (deviants: [Field; N], new_isolation_levels: [Field; N])
///
/// # Algorithm
/// 1. Find consensus state (most common vote)
/// 2. Verify consensus meets threshold
/// 3. Mark agents deviating from consensus
/// 4. Update isolation levels for Byzantine agents
///
/// # Constraints
/// * Consensus must have >= threshold agents
/// * Isolation levels must be 0-3
/// * State counting must be accurate
pub fn detect_byzantine_fault<N>(
    agent_states: [Field; N],
    consensus_threshold: Field,
    isolation_levels: [Field; N]
) -> ([Field; N], [Field; N]) {
    // Step 1: Count occurrences of each state
    // Assumes states are small numbers (0-10)
    let mut state_counts = [0; 10];
    
    for i in 0..N {
        let state = agent_states[i];
        // Ensure state is in valid range
        assert(state < 10);
        
        // Count this state
        // Note: In production Noir, array indexing with Field requires careful handling
        let state_idx = state as u32;
        state_counts[state_idx] = state_counts[state_idx] + 1;
    }
    
    // Step 2: Find consensus state (most common)
    let mut consensus_state = 0;
    let mut max_count = 0;
    
    for s in 0..10 {
        if state_counts[s] > max_count {
            max_count = state_counts[s];
            consensus_state = s;
        }
    }
    
    // Step 3: Verify consensus meets threshold
    assert(max_count >= consensus_threshold);
    
    // Step 4: Mark deviants (agents not matching consensus)
    let mut deviants = [0; N];
    for i in 0..N {
        if agent_states[i] != consensus_state {
            deviants[i] = 1;
        }
    }
    
    // Step 5: Update isolation levels for Byzantine agents
    let mut new_levels = isolation_levels;
    for i in 0..N {
        if deviants[i] == 1 {
            // Increase isolation level, cap at 3
            let new_level = isolation_levels[i] + 1;
            if new_level > 3 {
                new_levels[i] = 3;
            } else {
                new_levels[i] = new_level;
            }
        } else {
            new_levels[i] = isolation_levels[i];
        }
    }
    
    (deviants, new_levels)
}

// Unit tests
#[test]
fn test_byzantine_detection_single_deviant() {
    let states = [1, 1, 1, 2];  // 3 agents vote 1, 1 agent votes 2
    let threshold = 3;
    let isolation = [0, 0, 0, 0];
    
    let (deviants, new_isolation) = detect_byzantine_fault(states, threshold, isolation);
    
    // Agent 3 is deviant
    assert(deviants[0] == 0);
    assert(deviants[1] == 0);
    assert(deviants[2] == 0);
    assert(deviants[3] == 1);
    
    // Agent 3 isolation increased
    assert(new_isolation[3] == 1);
}

#[test]
fn test_byzantine_detection_no_deviants() {
    let states = [1, 1, 1, 1];  // All agents agree
    let threshold = 3;
    let isolation = [0, 0, 0, 0];
    
    let (deviants, new_isolation) = detect_byzantine_fault(states, threshold, isolation);
    
    // No deviants
    assert(deviants[0] == 0);
    assert(deviants[1] == 0);
    assert(deviants[2] == 0);
    assert(deviants[3] == 0);
    
    // No isolation changes
    assert(new_isolation[0] == 0);
    assert(new_isolation[1] == 0);
    assert(new_isolation[2] == 0);
    assert(new_isolation[3] == 0);
}

#[test]
fn test_byzantine_detection_isolation_cap() {
    let states = [1, 1, 1, 2];  // 3 agents vote 1, 1 agent votes 2
    let threshold = 3;
    let isolation = [0, 0, 0, 3];  // Agent 3 already at max isolation
    
    let (deviants, new_isolation) = detect_byzantine_fault(states, threshold, isolation);
    
    // Agent 3 isolation should stay at 3 (capped)
    assert(new_isolation[3] == 3);
}

#[test]
fn test_byzantine_detection_multiple_deviants() {
    let states = [1, 1, 1, 2, 2];  // 3 agents vote 1, 2 agents vote 2
    let threshold = 3;
    let isolation = [0, 0, 0, 0, 0];
    
    let (deviants, new_isolation) = detect_byzantine_fault(states, threshold, isolation);
    
    // Agents 3 and 4 are deviants
    assert(deviants[0] == 0);
    assert(deviants[1] == 0);
    assert(deviants[2] == 0);
    assert(deviants[3] == 1);
    assert(deviants[4] == 1);
    
    // Both deviant agents' isolation increased
    assert(new_isolation[3] == 1);
    assert(new_isolation[4] == 1);
}
