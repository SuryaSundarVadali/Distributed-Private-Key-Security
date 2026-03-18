// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IValidationRegistry
 * @notice Minimal ERC-8004 Validation Registry interface
 * @dev Manages validation request/response lifecycle for agent actions.
 */
interface IValidationRegistry {
    /// @notice Emitted when a validation request is created
    event ValidationRequested(
        bytes32 indexed requestHash,
        uint256 indexed agentId,
        address indexed validator,
        string requestURI
    );

    /// @notice Emitted when a validation response is submitted
    event ValidationResponded(
        bytes32 indexed requestHash,
        uint256 value,
        string responseURI
    );

    /// @notice Create a validation request
    /// @param validator    Address of the designated validator
    /// @param agentId      Agent requesting validation
    /// @param requestURI   URI pointing to the validation request payload
    /// @param requestHash  keccak256 commitment to the request payload
    function validationRequest(
        address validator,
        uint256 agentId,
        string calldata requestURI,
        bytes32 requestHash
    ) external;

    /// @notice Submit a validation response
    /// @param requestHash  The original request hash being responded to
    /// @param value        Numeric validation result
    /// @param responseURI  URI pointing to the validation response payload
    /// @param responseHash keccak256 commitment to the response payload
    /// @param tag1         Category tag for the response
    function validationResponse(
        bytes32 requestHash,
        uint256 value,
        string calldata responseURI,
        bytes32 responseHash,
        string calldata tag1
    ) external;
}
