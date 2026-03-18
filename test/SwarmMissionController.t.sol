// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "@contracts/SwarmMissionController.sol";
import "@contracts/mocks/MockIdentityRegistry.sol";
import "@contracts/mocks/MockReputationRegistry.sol";
import "@contracts/mocks/MockValidationRegistry.sol";

contract SwarmMissionControllerTest is Test {
    SwarmMissionController public controller;
    MockIdentityRegistry public identityReg;
    MockReputationRegistry public reputationReg;
    MockValidationRegistry public validationReg;

    address public owner;
    uint256 public ownerKey;
    address public validator;
    address public unauthorized;

    function setUp() public {
        ownerKey = 0xA11CE;
        owner = vm.addr(ownerKey);
        validator = makeAddr("validator");
        unauthorized = makeAddr("unauthorized");

        vm.startPrank(owner);

        identityReg = new MockIdentityRegistry();
        reputationReg = new MockReputationRegistry();
        validationReg = new MockValidationRegistry();

        controller = new SwarmMissionController(
            address(identityReg),
            address(reputationReg),
            address(validationReg),
            validator
        );

        vm.stopPrank();
    }

    // ---- Deployment ----

    function test_DeploymentSetsOwner() public view {
        assertEq(controller.owner(), owner);
    }

    function test_DeploymentSetsRegistries() public view {
        assertEq(address(controller.identityRegistry()), address(identityReg));
        assertEq(address(controller.reputationRegistry()), address(reputationReg));
        assertEq(address(controller.validationRegistry()), address(validationReg));
    }

    function test_OwnerIsAuthorizedSigner() public view {
        assertTrue(controller.authorizedSigners(owner));
    }

    // ---- Start Mission ----

    function test_StartMission() public {
        bytes32 root = keccak256("merkle-root-1");
        string memory uri = "ipfs://QmMissionData";

        vm.prank(owner);
        uint256 missionId = controller.startMission(1, root, uri);

        assertEq(missionId, 0);
        assertEq(controller.missionCount(), 1);

        (
            uint256 agentId,
            bytes32 missionRoot,
            string memory missionURI,
            SwarmMissionController.MissionStatus status,
            uint256 startedAt,
            ,
        ) = controller.getMission(0);

        assertEq(agentId, 1);
        assertEq(missionRoot, root);
        assertEq(keccak256(bytes(missionURI)), keccak256(bytes(uri)));
        assertEq(uint8(status), uint8(SwarmMissionController.MissionStatus.Active));
        assertGt(startedAt, 0);
    }

    function test_StartMission_RevertsForNonOwner() public {
        vm.prank(unauthorized);
        vm.expectRevert("SwarmMissionController: caller is not the owner");
        controller.startMission(1, keccak256("root"), "uri");
    }

    function test_MultipleMissions() public {
        vm.startPrank(owner);
        uint256 id1 = controller.startMission(1, keccak256("root1"), "uri1");
        uint256 id2 = controller.startMission(2, keccak256("root2"), "uri2");
        vm.stopPrank();

        assertEq(id1, 0);
        assertEq(id2, 1);
        assertEq(controller.missionCount(), 2);
    }

    // ---- Mark Mission Completed ----

    function test_MarkMissionCompleted() public {
        bytes32 root = keccak256("merkle-root");
        bytes32 reqHash = keccak256("validation-report");

        vm.startPrank(owner);
        uint256 missionId = controller.startMission(1, root, "ipfs://mission");
        controller.markMissionCompleted(missionId, "ipfs://report", reqHash);
        vm.stopPrank();

        (, , , SwarmMissionController.MissionStatus status, , uint256 completedAt, bytes32 valHash) =
            controller.getMission(missionId);

        assertEq(uint8(status), uint8(SwarmMissionController.MissionStatus.Completed));
        assertGt(completedAt, 0);
        assertEq(valHash, reqHash);
    }

    function test_MarkMissionCompleted_RevertsIfNotActive() public {
        bytes32 root = keccak256("merkle-root");
        bytes32 reqHash = keccak256("report");

        vm.startPrank(owner);
        uint256 missionId = controller.startMission(1, root, "uri");
        controller.markMissionCompleted(missionId, "uri", reqHash);

        // Try completing again
        vm.expectRevert("SwarmMissionController: mission not active");
        controller.markMissionCompleted(missionId, "uri2", keccak256("report2"));
        vm.stopPrank();
    }

    function test_MarkMissionCompleted_OpensValidationRequest() public {
        bytes32 reqHash = keccak256("report-hash");

        vm.startPrank(owner);
        controller.startMission(1, keccak256("root"), "ipfs://mission");
        controller.markMissionCompleted(0, "ipfs://report", reqHash);
        vm.stopPrank();

        // Verify the mock validation registry received the request
        (address reqValidator, uint256 reqAgentId, , MockValidationRegistry.RequestStatus reqStatus) =
            validationReg.requests(reqHash);

        assertEq(reqValidator, validator);
        assertEq(reqAgentId, 1);
        assertEq(uint8(reqStatus), uint8(MockValidationRegistry.RequestStatus.Pending));
    }

    // ---- ERC-1271 ----

    function test_IsValidSignature_ValidSigner() public view {
        bytes32 digest = keccak256("test message");
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(ownerKey, digest);
        bytes memory sig = abi.encodePacked(r, s, v);

        bytes4 result = controller.isValidSignature(digest, sig);
        assertEq(result, controller.ERC1271_MAGIC());
    }

    function test_IsValidSignature_UnauthorizedSigner() public view {
        uint256 randomKey = 0xBEEF;
        bytes32 digest = keccak256("test message");
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(randomKey, digest);
        bytes memory sig = abi.encodePacked(r, s, v);

        bytes4 result = controller.isValidSignature(digest, sig);
        assertEq(result, bytes4(0xffffffff));
    }

    function test_IsValidSignature_InvalidLength() public view {
        bytes32 digest = keccak256("test");
        bytes memory badSig = new bytes(32); // wrong length

        bytes4 result = controller.isValidSignature(digest, badSig);
        assertEq(result, bytes4(0xffffffff));
    }

    // ---- Signer Management ----

    function test_SetAuthorizedSigner() public {
        address newSigner = makeAddr("newSigner");
        uint256 newSignerKey = 0xCAFE;
        newSigner = vm.addr(newSignerKey);

        vm.prank(owner);
        controller.setAuthorizedSigner(newSigner, true);

        assertTrue(controller.authorizedSigners(newSigner));

        // Verify the new signer's signatures are now accepted
        bytes32 digest = keccak256("test");
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(newSignerKey, digest);
        bytes memory sig = abi.encodePacked(r, s, v);

        assertEq(controller.isValidSignature(digest, sig), controller.ERC1271_MAGIC());
    }

    function test_RevokeAuthorizedSigner() public {
        address newSigner = vm.addr(0xCAFE);

        vm.startPrank(owner);
        controller.setAuthorizedSigner(newSigner, true);
        controller.setAuthorizedSigner(newSigner, false);
        vm.stopPrank();

        assertFalse(controller.authorizedSigners(newSigner));
    }

    function test_SetAuthorizedSigner_RevertsForNonOwner() public {
        vm.prank(unauthorized);
        vm.expectRevert("SwarmMissionController: caller is not the owner");
        controller.setAuthorizedSigner(makeAddr("x"), true);
    }

    // ---- End-to-End Flow ----

    function test_FullMissionLifecycle() public {
        // 1. Register agent
        vm.prank(owner);
        uint256 agentId = identityReg.register('{"name":"SwarmAlpha","type":"precision_farming"}');

        // 2. Start mission
        bytes32 missionRoot = keccak256("task1task2task3");
        vm.prank(owner);
        uint256 missionId = controller.startMission(agentId, missionRoot, "ipfs://QmMission");

        // 3. Complete mission → opens validation request
        bytes32 reportHash = keccak256("smpc-validated-report");
        vm.prank(owner);
        controller.markMissionCompleted(missionId, "ipfs://QmReport", reportHash);

        // 4. Validator responds
        vm.prank(validator);
        validationReg.validationResponse(reportHash, 95, "ipfs://QmResponse", keccak256("response"), "approved");

        // 5. Give reputation feedback
        vm.prank(owner);
        reputationReg.giveFeedback(
            agentId, 95, 0, "mission_success", "precision_farming", "ipfs://QmFeedback", keccak256("feedback")
        );

        // Verify final state
        (, , , SwarmMissionController.MissionStatus status, , , ) = controller.getMission(missionId);
        assertEq(uint8(status), uint8(SwarmMissionController.MissionStatus.Completed));
        assertEq(reputationReg.getFeedbackCount(agentId), 1);
    }
}
