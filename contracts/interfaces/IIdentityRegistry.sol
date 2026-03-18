// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IIdentityRegistry
 * @notice Minimal ERC-8004 Identity Registry interface (ERC-721 based)
 * @dev Each registered agent receives a unique NFT tokenId as its agentId.
 */
interface IIdentityRegistry {
    /// @notice Emitted when a new agent is registered
    event AgentRegistered(uint256 indexed agentId, address indexed owner, string agentURI);

    /// @notice Emitted when an agent's wallet is updated
    event AgentWalletSet(uint256 indexed agentId, address indexed wallet);

    /// @notice Register a new agent and mint an identity NFT
    /// @param agentURI JSON metadata URI describing the agent
    /// @return agentId The newly minted token ID
    function register(string calldata agentURI) external returns (uint256 agentId);

    /// @notice Set or update the wallet address associated with an agent
    /// @param agentId The agent's token ID
    /// @param wallet The wallet address to associate
    /// @param deadline Timestamp after which the signature expires
    /// @param sig EIP-712 signature authorizing the wallet change
    function setAgentWallet(
        uint256 agentId,
        address wallet,
        uint256 deadline,
        bytes calldata sig
    ) external;

    /// @notice Get the owner of an agent NFT
    function ownerOf(uint256 agentId) external view returns (address);

    /// @notice Get the metadata URI for an agent
    function agentURI(uint256 agentId) external view returns (string memory);

    /// @notice Get the wallet address associated with an agent
    function agentWallet(uint256 agentId) external view returns (address);
}
