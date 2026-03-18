// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../interfaces/IIdentityRegistry.sol";

/**
 * @title MockIdentityRegistry
 * @notice Minimal mock of ERC-8004 Identity Registry for testing
 */
contract MockIdentityRegistry is IIdentityRegistry {
    uint256 private _nextTokenId = 1;

    struct AgentData {
        address owner;
        string  uri;
        address wallet;
    }

    mapping(uint256 => AgentData) private _agents;

    function register(string calldata agentURI) external returns (uint256 agentId) {
        agentId = _nextTokenId++;
        _agents[agentId] = AgentData({
            owner: msg.sender,
            uri: agentURI,
            wallet: address(0)
        });
        emit AgentRegistered(agentId, msg.sender, agentURI);
    }

    function setAgentWallet(
        uint256 agentId,
        address wallet,
        uint256 /* deadline */,
        bytes calldata /* sig */
    ) external {
        require(_agents[agentId].owner == msg.sender, "not owner");
        _agents[agentId].wallet = wallet;
        emit AgentWalletSet(agentId, wallet);
    }

    function ownerOf(uint256 agentId) external view returns (address) {
        return _agents[agentId].owner;
    }

    function agentURI(uint256 agentId) external view returns (string memory) {
        return _agents[agentId].uri;
    }

    function agentWallet(uint256 agentId) external view returns (address) {
        return _agents[agentId].wallet;
    }
}
