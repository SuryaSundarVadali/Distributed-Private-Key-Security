// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IReputationRegistry
 * @notice Minimal ERC-8004 Reputation Registry interface
 * @dev Allows any address to leave tagged, quantified feedback for an agent.
 */
interface IReputationRegistry {
    /// @notice Emitted when feedback is recorded
    event FeedbackGiven(
        uint256 indexed agentId,
        address indexed from,
        uint256 value,
        string tag1,
        string tag2
    );

    /// @notice Record feedback for an agent
    /// @param agentId       Target agent's identity token ID
    /// @param value         Numeric reputation score
    /// @param valueDecimals Decimal precision of `value`
    /// @param tag1          Primary category tag (e.g. "mission_success")
    /// @param tag2          Secondary category tag (e.g. "precision_farming")
    /// @param feedbackURI   URI pointing to detailed feedback data
    /// @param feedbackHash  keccak256 of the feedback payload for integrity
    function giveFeedback(
        uint256 agentId,
        uint256 value,
        uint8 valueDecimals,
        string calldata tag1,
        string calldata tag2,
        string calldata feedbackURI,
        bytes32 feedbackHash
    ) external;
}
