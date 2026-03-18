// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../interfaces/IValidationRegistry.sol";

/**
 * @title MockValidationRegistry
 * @notice Minimal mock of ERC-8004 Validation Registry for testing
 */
contract MockValidationRegistry is IValidationRegistry {
    enum RequestStatus { None, Pending, Responded }

    struct Request {
        address validator;
        uint256 agentId;
        string  requestURI;
        RequestStatus status;
    }

    struct Response {
        uint256 value;
        string  responseURI;
        bytes32 responseHash;
        string  tag1;
    }

    mapping(bytes32 => Request)  public requests;
    mapping(bytes32 => Response) public responses;

    function validationRequest(
        address validator,
        uint256 agentId,
        string calldata requestURI,
        bytes32 requestHash
    ) external {
        require(requests[requestHash].status == RequestStatus.None, "request exists");

        requests[requestHash] = Request({
            validator: validator,
            agentId: agentId,
            requestURI: requestURI,
            status: RequestStatus.Pending
        });

        emit ValidationRequested(requestHash, agentId, validator, requestURI);
    }

    function validationResponse(
        bytes32 requestHash,
        uint256 value,
        string calldata responseURI,
        bytes32 responseHash,
        string calldata tag1
    ) external {
        Request storage req = requests[requestHash];
        require(req.status == RequestStatus.Pending, "not pending");
        require(msg.sender == req.validator, "not validator");

        req.status = RequestStatus.Responded;
        responses[requestHash] = Response({
            value: value,
            responseURI: responseURI,
            responseHash: responseHash,
            tag1: tag1
        });

        emit ValidationResponded(requestHash, value, responseURI);
    }
}
