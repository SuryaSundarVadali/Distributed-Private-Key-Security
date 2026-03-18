// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../interfaces/IReputationRegistry.sol";

/**
 * @title MockReputationRegistry
 * @notice Minimal mock of ERC-8004 Reputation Registry for testing
 */
contract MockReputationRegistry is IReputationRegistry {
    struct Feedback {
        address from;
        uint256 value;
        uint8   valueDecimals;
        string  tag1;
        string  tag2;
        string  feedbackURI;
        bytes32 feedbackHash;
    }

    mapping(uint256 => Feedback[]) public feedbackHistory;

    function giveFeedback(
        uint256 agentId,
        uint256 value,
        uint8 valueDecimals,
        string calldata tag1,
        string calldata tag2,
        string calldata feedbackURI,
        bytes32 feedbackHash
    ) external {
        feedbackHistory[agentId].push(Feedback({
            from: msg.sender,
            value: value,
            valueDecimals: valueDecimals,
            tag1: tag1,
            tag2: tag2,
            feedbackURI: feedbackURI,
            feedbackHash: feedbackHash
        }));

        emit FeedbackGiven(agentId, msg.sender, value, tag1, tag2);
    }

    function getFeedbackCount(uint256 agentId) external view returns (uint256) {
        return feedbackHistory[agentId].length;
    }
}
