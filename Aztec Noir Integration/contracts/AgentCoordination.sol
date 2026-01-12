// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title AgentCoordination
 * @dev Byzantine Fault Tolerant Agent Coordination with Aztec ZK Proofs
 * @notice This contract manages agent voting, reputation, and Byzantine detection
 *         using Noir circuits and Barretenberg proof verification
 */
contract AgentCoordination {
    // ============================================================================
    // State Variables
    // ============================================================================
    
    /// @notice Current Merkle root of reputation tree
    bytes32 public currentMerkleRoot;
    
    /// @notice Total number of registered agents
    uint256 public totalAgents;
    
    /// @notice Current voting round number
    uint256 public currentRound;
    
    /// @notice Contract owner/admin
    address public owner;
    
    /// @notice Mapping of agent ID to Agent struct
    mapping(uint256 => Agent) public agents;
    
    /// @notice Mapping of round number to RoundData
    mapping(uint256 => RoundData) public rounds;
    
    /// @notice Mapping to prevent proof reuse (proof hash => used)
    mapping(bytes32 => bool) public usedProofs;
    
    // ============================================================================
    // Structs
    // ============================================================================
    
    /// @notice Agent information
    struct Agent {
        uint256 id;
        uint256 reputation;      // 0-1000 scale
        uint8 isolationLevel;     // 0-3 (0=none, 3=max isolation)
        bool active;
        address agentAddress;
        uint256 registeredAt;
    }
    
    /// @notice Voting round data
    struct RoundData {
        uint256 roundNumber;
        bytes32 commitmentRoot;
        bool decision;
        bytes32 reputationRoot;
        uint256 timestamp;
        uint256 participantCount;
        uint256 byzantineCount;
        bool finalized;
    }
    
    // ============================================================================
    // Events
    // ============================================================================
    
    event AgentRegistered(uint256 indexed agentId, address indexed agentAddress);
    event VoteAggregated(uint256 indexed round, bool decision, bytes32 proofHash);
    event ByzantineDetected(uint256 indexed round, uint256[] agentIds, uint8[] newLevels);
    event ReputationUpdated(uint256 indexed agentId, uint256 oldReputation, uint256 newReputation);
    event RoundStored(uint256 indexed roundNumber, bytes32 roundHash);
    event RoundFinalized(uint256 indexed roundNumber, bool decision);
    
    // ============================================================================
    // Modifiers
    // ============================================================================
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyActiveAgent(uint256 agentId) {
        require(agents[agentId].active, "Agent is not active");
        _;
    }
    
    modifier proofNotUsed(bytes memory proof) {
        bytes32 proofHash = keccak256(proof);
        require(!usedProofs[proofHash], "Proof already used");
        _;
    }
    
    // ============================================================================
    // Constructor
    // ============================================================================
    
    constructor() {
        owner = msg.sender;
        currentRound = 1;
        currentMerkleRoot = bytes32(0);
    }
    
    // ============================================================================
    // Agent Management
    // ============================================================================
    
    /**
     * @notice Register a new agent in the system
     * @param agentId Unique identifier for the agent
     * @param agentAddress Ethereum address of the agent
     */
    function registerAgent(uint256 agentId, address agentAddress) external onlyOwner {
        require(!agents[agentId].active, "Agent already registered");
        require(agentAddress != address(0), "Invalid agent address");
        
        agents[agentId] = Agent({
            id: agentId,
            reputation: 500,  // Start with neutral reputation
            isolationLevel: 0,
            active: true,
            agentAddress: agentAddress,
            registeredAt: block.timestamp
        });
        
        totalAgents++;
        emit AgentRegistered(agentId, agentAddress);
    }
    
    /**
     * @notice Deactivate an agent
     * @param agentId Agent to deactivate
     */
    function deactivateAgent(uint256 agentId) external onlyOwner {
        require(agents[agentId].active, "Agent already inactive");
        agents[agentId].active = false;
        totalAgents--;
    }
    
    // ============================================================================
    // Voting & Proof Verification
    // ============================================================================
    
    /**
     * @notice Verify aggregated votes and update decision
     * @param proof Noir circuit proof from aggregate_votes circuit
     * @param decision Final voting decision (approved/rejected)
     * @param commitmentRoot Merkle root of vote commitments
     * @param roundNumber Round identifier
     */
    function verifyAggregateVotes(
        bytes calldata proof,
        bool decision,
        bytes32 commitmentRoot,
        uint256 roundNumber
    ) external proofNotUsed(proof) {
        require(roundNumber == currentRound, "Invalid round number");
        require(!rounds[roundNumber].finalized, "Round already finalized");
        
        // Verify Noir circuit proof
        bool proofValid = _verifyProof(proof, "aggregate_votes");
        require(proofValid, "Invalid proof");
        
        // Mark proof as used
        bytes32 proofHash = keccak256(proof);
        usedProofs[proofHash] = true;
        
        // Store round data
        rounds[roundNumber].decision = decision;
        rounds[roundNumber].commitmentRoot = commitmentRoot;
        
        emit VoteAggregated(roundNumber, decision, proofHash);
    }
    
    /**
     * @notice Verify Byzantine detection and update isolation levels
     * @param proof Noir circuit proof from byzantine_detector circuit
     * @param deviatingAgentIds Array of agent IDs that deviated
     * @param newIsolationLevels Updated isolation levels
     * @param roundNumber Round identifier
     */
    function verifyByzantineDetection(
        bytes calldata proof,
        uint256[] calldata deviatingAgentIds,
        uint8[] calldata newIsolationLevels,
        uint256 roundNumber
    ) external proofNotUsed(proof) {
        require(deviatingAgentIds.length == newIsolationLevels.length, "Length mismatch");
        require(roundNumber == currentRound, "Invalid round number");
        
        // Verify proof
        bool proofValid = _verifyProof(proof, "byzantine_detector");
        require(proofValid, "Invalid proof");
        
        // Update isolation levels
        for (uint256 i = 0; i < deviatingAgentIds.length; i++) {
            uint256 agentId = deviatingAgentIds[i];
            require(agents[agentId].active, "Agent not active");
            require(newIsolationLevels[i] <= 3, "Invalid isolation level");
            
            agents[agentId].isolationLevel = newIsolationLevels[i];
        }
        
        // Update round Byzantine count
        rounds[roundNumber].byzantineCount = deviatingAgentIds.length;
        
        // Mark proof as used
        usedProofs[keccak256(proof)] = true;
        
        emit ByzantineDetected(roundNumber, deviatingAgentIds, newIsolationLevels);
    }
    
    /**
     * @notice Update agent reputation with Merkle proof
     * @param proof Noir circuit proof from reputation_updater circuit
     * @param agentId Agent whose reputation is being updated
     * @param newReputation Updated reputation score (0-1000)
     * @param newMerkleRoot New Merkle root after update
     * @param roundNumber Round identifier
     */
    function updateReputation(
        bytes calldata proof,
        uint256 agentId,
        uint256 newReputation,
        bytes32 newMerkleRoot,
        uint256 roundNumber
    ) external proofNotUsed(proof) onlyActiveAgent(agentId) {
        require(newReputation <= 1000, "Reputation must be 0-1000");
        require(roundNumber == currentRound, "Invalid round number");
        
        // Verify proof
        bool proofValid = _verifyProof(proof, "reputation_updater");
        require(proofValid, "Invalid proof");
        
        // Get old reputation for event
        uint256 oldReputation = agents[agentId].reputation;
        
        // Update agent reputation
        agents[agentId].reputation = newReputation;
        
        // Update global Merkle root
        currentMerkleRoot = newMerkleRoot;
        rounds[roundNumber].reputationRoot = newMerkleRoot;
        
        // Mark proof as used
        usedProofs[keccak256(proof)] = true;
        
        emit ReputationUpdated(agentId, oldReputation, newReputation);
    }
    
    /**
     * @notice Verify Shamir secret shares
     * @param proof Noir circuit proof from shamir_verifier circuit
     */
    function verifyShamirShares(bytes calldata proof) external proofNotUsed(proof) {
        bool proofValid = _verifyProof(proof, "shamir_verifier");
        require(proofValid, "Invalid Shamir proof");
        
        usedProofs[keccak256(proof)] = true;
    }
    
    /**
     * @notice Store round data in encrypted PST
     * @param proof Noir circuit proof from round_storage circuit
     * @param roundNumber Round identifier
     * @param roundHash Hash of round data
     * @param timestamp Round completion timestamp
     */
    function storeRound(
        bytes calldata proof,
        uint256 roundNumber,
        bytes32 roundHash,
        uint256 timestamp
    ) external proofNotUsed(proof) {
        require(roundNumber == currentRound, "Invalid round number");
        
        bool proofValid = _verifyProof(proof, "round_storage");
        require(proofValid, "Invalid proof");
        
        // Store round metadata
        rounds[roundNumber].roundNumber = roundNumber;
        rounds[roundNumber].timestamp = timestamp;
        rounds[roundNumber].finalized = true;
        
        // Mark proof as used
        usedProofs[keccak256(proof)] = true;
        
        emit RoundStored(roundNumber, roundHash);
        emit RoundFinalized(roundNumber, rounds[roundNumber].decision);
        
        // Increment round counter
        currentRound++;
    }
    
    // ============================================================================
    // Internal Functions
    // ============================================================================
    
    /**
     * @dev Verify Noir circuit proof using Barretenberg verifier
     * @param proof The proof bytes
     * @param circuitName Name of the circuit for verification key lookup
     * @return bool True if proof is valid
     * 
     * NOTE: In production, this would call Aztec's IVerifier interface
     * For now, this is a placeholder that performs basic validation
     */
    function _verifyProof(bytes calldata proof, string memory circuitName) 
        internal 
        view 
        returns (bool) 
    {
        // Basic validation
        require(proof.length > 0, "Empty proof");
        
        // In production, delegate to Aztec verifier contract:
        // return IAztecVerifier(verifierAddress).verify(proof, circuitName);
        
        // For development/testing: always return true
        // IMPORTANT: Replace with actual verifier in production
        return true;
    }
    
    // ============================================================================
    // View Functions
    // ============================================================================
    
    /**
     * @notice Get agent information
     * @param agentId Agent identifier
     * @return Agent struct
     */
    function getAgent(uint256 agentId) external view returns (Agent memory) {
        return agents[agentId];
    }
    
    /**
     * @notice Get agent reputation
     * @param agentId Agent identifier
     * @return uint256 Reputation score (0-1000)
     */
    function getReputation(uint256 agentId) external view returns (uint256) {
        return agents[agentId].reputation;
    }
    
    /**
     * @notice Get agent isolation level
     * @param agentId Agent identifier
     * @return uint8 Isolation level (0-3)
     */
    function getIsolationLevel(uint256 agentId) external view returns (uint8) {
        return agents[agentId].isolationLevel;
    }
    
    /**
     * @notice Get current Merkle root
     * @return bytes32 Current reputation Merkle root
     */
    function getMerkleRoot() external view returns (bytes32) {
        return currentMerkleRoot;
    }
    
    /**
     * @notice Get round data
     * @param roundNumber Round identifier
     * @return RoundData struct
     */
    function getRound(uint256 roundNumber) external view returns (RoundData memory) {
        return rounds[roundNumber];
    }
    
    /**
     * @notice Check if agent is active
     * @param agentId Agent identifier
     * @return bool True if agent is active
     */
    function isAgentActive(uint256 agentId) external view returns (bool) {
        return agents[agentId].active;
    }
    
    /**
     * @notice Check if proof has been used
     * @param proof Proof bytes
     * @return bool True if proof has been used
     */
    function isProofUsed(bytes calldata proof) external view returns (bool) {
        return usedProofs[keccak256(proof)];
    }
}
