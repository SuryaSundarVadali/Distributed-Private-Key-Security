// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/IIdentityRegistry.sol";
import "./interfaces/IReputationRegistry.sol";
import "./interfaces/IValidationRegistry.sol";

/**
 * @title SwarmMissionController
 * @notice On-chain controller for autonomous robot swarm missions.
 *         Bridges off-chain cryptographic protocols (HKDF, Shamir SSS, Merkle,
 *         HOTP, SMPC) with ERC-8004 Trustless Agents registries.
 * @dev Implements ERC-1271 so this contract can serve as the swarm's agentWallet
 *      and validate EIP-712 typed data signatures on behalf of the collective.
 */
contract SwarmMissionController {
    // ---------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------

    /// @dev ERC-1271 magic value returned on valid signature
    bytes4 public constant ERC1271_MAGIC = 0x1626ba7e;

    /// @dev EIP-712 domain separator type hash
    bytes32 public constant DOMAIN_TYPEHASH =
        keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)");

    /// @dev EIP-712 type hash for mission start authorization
    bytes32 public constant START_MISSION_TYPEHASH =
        keccak256("StartMission(uint256 agentId,bytes32 missionRoot,string missionURI,uint256 nonce)");

    // ---------------------------------------------------------------
    // State
    // ---------------------------------------------------------------

    enum MissionStatus { None, Active, Completed, Validated }

    struct Mission {
        uint256 agentId;
        bytes32 missionRoot;       // Merkle root of task tree
        string  missionURI;        // IPFS/HTTPS pointer to mission metadata
        MissionStatus status;
        uint256 startedAt;
        uint256 completedAt;
        bytes32 validationRequestHash;
    }

    /// @notice Contract owner (deployer)
    address public owner;

    /// @notice Authorized signers whose ECDSA signatures are accepted by isValidSignature
    mapping(address => bool) public authorizedSigners;

    /// @notice ERC-8004 registry addresses
    IIdentityRegistry   public identityRegistry;
    IReputationRegistry public reputationRegistry;
    IValidationRegistry public validationRegistry;

    /// @notice Default validator address for validation requests
    address public defaultValidator;

    /// @notice Mission storage keyed by a sequential mission ID
    mapping(uint256 => Mission) public missions;
    uint256 public missionCount;

    /// @notice Nonce for EIP-712 replay protection
    uint256 public nonce;

    /// @notice Cached EIP-712 domain separator
    bytes32 public immutable DOMAIN_SEPARATOR;

    // ---------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------

    event MissionStarted(uint256 indexed missionId, uint256 indexed agentId, bytes32 missionRoot, string missionURI);
    event MissionCompleted(uint256 indexed missionId, bytes32 requestHash);
    event SignerAuthorized(address indexed signer, bool authorized);

    // ---------------------------------------------------------------
    // Modifiers
    // ---------------------------------------------------------------

    modifier onlyOwner() {
        require(msg.sender == owner, "SwarmMissionController: caller is not the owner");
        _;
    }

    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------

    /**
     * @param _identityRegistry   ERC-8004 Identity Registry address
     * @param _reputationRegistry ERC-8004 Reputation Registry address
     * @param _validationRegistry ERC-8004 Validation Registry address
     * @param _defaultValidator   Default validator for validation requests
     */
    constructor(
        address _identityRegistry,
        address _reputationRegistry,
        address _validationRegistry,
        address _defaultValidator
    ) {
        owner = msg.sender;
        authorizedSigners[msg.sender] = true;

        identityRegistry   = IIdentityRegistry(_identityRegistry);
        reputationRegistry = IReputationRegistry(_reputationRegistry);
        validationRegistry = IValidationRegistry(_validationRegistry);
        defaultValidator   = _defaultValidator;

        DOMAIN_SEPARATOR = keccak256(abi.encode(
            DOMAIN_TYPEHASH,
            keccak256("SwarmMissionController"),
            keccak256("1"),
            block.chainid,
            address(this)
        ));
    }

    // ---------------------------------------------------------------
    // Signer Management
    // ---------------------------------------------------------------

    /// @notice Authorize or revoke an ECDSA signer
    function setAuthorizedSigner(address signer, bool authorized) external onlyOwner {
        authorizedSigners[signer] = authorized;
        emit SignerAuthorized(signer, authorized);
    }

    // ---------------------------------------------------------------
    // Mission Lifecycle
    // ---------------------------------------------------------------

    /**
     * @notice Start a new mission by publishing the Merkle root on-chain
     * @param agentId     ERC-8004 agent identity token ID
     * @param missionRoot Merkle root of the off-chain task tree
     * @param missionURI  URI pointing to mission metadata (IPFS/HTTPS)
     * @return missionId  The sequential mission identifier
     */
    function startMission(
        uint256 agentId,
        bytes32 missionRoot,
        string calldata missionURI
    ) external onlyOwner returns (uint256 missionId) {
        missionId = missionCount++;

        missions[missionId] = Mission({
            agentId: agentId,
            missionRoot: missionRoot,
            missionURI: missionURI,
            status: MissionStatus.Active,
            startedAt: block.timestamp,
            completedAt: 0,
            validationRequestHash: bytes32(0)
        });

        emit MissionStarted(missionId, agentId, missionRoot, missionURI);
    }

    /**
     * @notice Mark a mission as completed (after SMPC success) and open
     *         a validation request in the ERC-8004 Validation Registry.
     * @param missionId   The mission to complete
     * @param requestURI  URI pointing to the off-chain validation report
     * @param requestHash keccak256 commitment to the validation report
     */
    function markMissionCompleted(
        uint256 missionId,
        string calldata requestURI,
        bytes32 requestHash
    ) external onlyOwner {
        Mission storage m = missions[missionId];
        require(m.status == MissionStatus.Active, "SwarmMissionController: mission not active");

        m.status = MissionStatus.Completed;
        m.completedAt = block.timestamp;
        m.validationRequestHash = requestHash;

        // Open a validation request in the ERC-8004 Validation Registry
        validationRegistry.validationRequest(
            defaultValidator,
            m.agentId,
            requestURI,
            requestHash
        );

        emit MissionCompleted(missionId, requestHash);
    }

    // ---------------------------------------------------------------
    // Query Helpers
    // ---------------------------------------------------------------

    /// @notice Get full mission data
    function getMission(uint256 missionId) external view returns (
        uint256 agentId,
        bytes32 missionRoot,
        string memory missionURI,
        MissionStatus status,
        uint256 startedAt,
        uint256 completedAt,
        bytes32 validationRequestHash
    ) {
        Mission storage m = missions[missionId];
        return (m.agentId, m.missionRoot, m.missionURI, m.status, m.startedAt, m.completedAt, m.validationRequestHash);
    }

    // ---------------------------------------------------------------
    // ERC-1271: Smart Contract Signature Validation
    // ---------------------------------------------------------------

    /**
     * @notice Validates a signature on behalf of the swarm.
     *         Recovers the ECDSA signer and checks authorizedSigners.
     * @param hash      The EIP-191 / EIP-712 digest that was signed
     * @param signature 65-byte ECDSA signature (r, s, v)
     * @return magicValue ERC1271_MAGIC if valid, 0xffffffff otherwise
     */
    function isValidSignature(bytes32 hash, bytes memory signature) public view returns (bytes4) {
        if (signature.length != 65) {
            return 0xffffffff;
        }

        bytes32 r;
        bytes32 s;
        uint8 v;

        assembly {
            r := mload(add(signature, 32))
            s := mload(add(signature, 64))
            v := byte(0, mload(add(signature, 96)))
        }

        // Normalize v
        if (v < 27) {
            v += 27;
        }

        if (v != 27 && v != 28) {
            return 0xffffffff;
        }

        address recovered = ecrecover(hash, v, r, s);
        if (recovered == address(0)) {
            return 0xffffffff;
        }

        if (authorizedSigners[recovered]) {
            return ERC1271_MAGIC;
        }

        return 0xffffffff;
    }
}
