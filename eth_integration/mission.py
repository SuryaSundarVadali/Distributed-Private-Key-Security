"""
Mission lifecycle management bridging off-chain crypto with on-chain ERC-8004.

Provides the MissionManager class that orchestrates:
    1. Publishing Merkle roots on-chain via SwarmMissionController
    2. Generating validation reports from SMPC results
    3. Completing missions and opening validation requests
"""

import json
import hashlib
from typing import Any, Dict, Optional, Tuple

from web3 import Web3
from eth_account import Account

from eth_integration.config import ChainConfig
from eth_integration.secure_key_vault import SecureKeyVault
from eth_integration.abis import SWARM_MISSION_CONTROLLER_ABI


class MissionManager:
    """
    Manages the on-chain mission lifecycle via SwarmMissionController.

    Bridges the off-chain cryptographic outputs (Merkle roots from merkle_tree.py,
    SMPC verification results from smpc_verification.py) with on-chain state.
    """

    def __init__(self, config: ChainConfig):
        self.config = config
        self.w3 = Web3(Web3.HTTPProvider(config.rpc_url))

        self.account = None
        if config.deployer_key_name:
            if config.vault_path == "env":
                vault = SecureKeyVault.from_environment()
            else:
                try:
                    vault = SecureKeyVault.from_encrypted_file(config.vault_path)
                except Exception:
                    vault = None

            if vault and vault.has_key(config.deployer_key_name):
                self.account = Account.from_key(vault.get_key(config.deployer_key_name))

        self._controller = None
        if config.swarm_controller:
            self._controller = self.w3.eth.contract(
                address=Web3.to_checksum_address(config.swarm_controller),
                abi=SWARM_MISSION_CONTROLLER_ABI,
            )

    # ------------------------------------------------------------------
    # Mission Lifecycle
    # ------------------------------------------------------------------

    def start_mission(
        self,
        agent_id: int,
        mission_root: bytes,
        mission_uri: str,
    ) -> Tuple[int, str]:
        """
        Start a new mission by publishing the Merkle root on-chain.

        This should be called after:
            1. HKDF key derivation (hkdf_entropy.py)
            2. Shamir share distribution (shamir_secret_sharing.py)
            3. Merkle tree construction (merkle_tree.py)

        Args:
            agent_id: ERC-8004 agent token ID
            mission_root: 32-byte Merkle root of the task tree
            mission_uri: IPFS/HTTPS URI pointing to mission metadata

        Returns:
            Tuple of (missionId, tx_hash_hex)
        """
        if not self._controller:
            raise RuntimeError("SwarmMissionController address not configured")

        # Ensure mission_root is exactly 32 bytes
        if len(mission_root) != 32:
            mission_root = Web3.keccak(mission_root)

        tx = self._controller.functions.startMission(
            agent_id, mission_root, mission_uri
        ).build_transaction(self._tx_params())

        receipt = self._send_tx(tx)

        # Parse MissionStarted event
        logs = self._controller.events.MissionStarted().process_receipt(receipt)
        mission_id = logs[0]["args"]["missionId"] if logs else 0

        return mission_id, receipt.transactionHash.hex()

    def complete_mission(
        self,
        mission_id: int,
        request_uri: str,
        request_hash: bytes,
    ) -> str:
        """
        Mark a mission as completed and open a validation request.

        This should be called after successful SMPC verification.
        The SwarmMissionController will automatically open a validation
        request in the ERC-8004 Validation Registry.

        Args:
            mission_id: The on-chain mission ID from startMission
            request_uri: URI pointing to the validation report
            request_hash: 32-byte keccak256 of the validation report

        Returns:
            Transaction hash hex string
        """
        if not self._controller:
            raise RuntimeError("SwarmMissionController address not configured")

        if len(request_hash) != 32:
            request_hash = Web3.keccak(request_hash)

        tx = self._controller.functions.markMissionCompleted(
            mission_id, request_uri, request_hash
        ).build_transaction(self._tx_params())

        receipt = self._send_tx(tx)
        return receipt.transactionHash.hex()

    def get_mission(self, mission_id: int) -> Dict[str, Any]:
        """
        Query on-chain mission data.

        Returns:
            Dict with keys: agentId, missionRoot, missionURI, status,
                           startedAt, completedAt, validationRequestHash
        """
        if not self._controller:
            raise RuntimeError("SwarmMissionController address not configured")

        result = self._controller.functions.getMission(mission_id).call()
        status_names = {0: "None", 1: "Active", 2: "Completed", 3: "Validated"}
        return {
            "agentId": result[0],
            "missionRoot": result[1].hex(),
            "missionURI": result[2],
            "status": status_names.get(result[3], "Unknown"),
            "startedAt": result[4],
            "completedAt": result[5],
            "validationRequestHash": result[6].hex(),
        }

    def get_mission_count(self) -> int:
        """Get total number of missions."""
        if not self._controller:
            raise RuntimeError("SwarmMissionController address not configured")
        return self._controller.functions.missionCount().call()

    # ------------------------------------------------------------------
    # Validation Report Generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_validation_report(
        mission_root: bytes,
        smpc_results: Dict[str, Any],
        agent_id: int = 0,
        mission_id: int = 0,
    ) -> Tuple[str, bytes]:
        """
        Generate a validation report JSON from SMPC verification results.

        This bridges the off-chain SMPC output (smpc_verification.py)
        with the on-chain validation request.

        Args:
            mission_root: Merkle root of the mission task tree
            smpc_results: Dict from SMPC verification containing at minimum
                         'verified' (bool) and optionally 'aggregated_result',
                         'participants', 'privacy_metrics'
            agent_id: Agent token ID (for metadata)
            mission_id: On-chain mission ID (for metadata)

        Returns:
            Tuple of (report_json_string, report_keccak256_hash)
        """
        report = {
            "version": "1.0",
            "agentId": agent_id,
            "missionId": mission_id,
            "missionRoot": mission_root.hex() if isinstance(mission_root, bytes) else mission_root,
            "verification": {
                "protocol": "SMPC-3Phase",
                "verified": smpc_results.get("verified", False),
                "aggregatedResult": smpc_results.get("aggregated_result"),
                "participantCount": smpc_results.get("participants", 0),
                "privacyMetrics": smpc_results.get("privacy_metrics", {}),
            },
        }

        report_json = json.dumps(report, sort_keys=True, separators=(",", ":"))
        report_hash = Web3.keccak(text=report_json)

        return report_json, report_hash

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tx_params(self) -> dict:
        if not self.account:
            raise RuntimeError("No deployer private key configured")
        return {
            "from": self.account.address,
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "gas": self.config.gas_limit,
            "chainId": self.config.chain_id,
        }

    def _send_tx(self, tx: dict):
        if not self.account:
            raise RuntimeError("No deployer account configured (check vault)")
        signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
