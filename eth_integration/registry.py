"""
ERC-8004 Registry interaction client.

High-level functions for Identity, Reputation, and Validation registry operations.
"""

from typing import Optional, Tuple

from web3 import Web3
from web3.contract import Contract
from eth_account import Account

from eth_integration.config import ChainConfig
from eth_integration.secure_key_vault import SecureKeyVault
from eth_integration.abis import (
    IDENTITY_REGISTRY_ABI,
    REPUTATION_REGISTRY_ABI,
    VALIDATION_REGISTRY_ABI,
)


class RegistryClient:
    """
    Client for interacting with ERC-8004 registries on-chain.

    Wraps Identity, Reputation, and Validation registry contracts
    with high-level Python functions.
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

        # Initialize contract instances
        self._identity: Optional[Contract] = None
        self._reputation: Optional[Contract] = None
        self._validation: Optional[Contract] = None

        if config.identity_registry:
            self._identity = self.w3.eth.contract(
                address=Web3.to_checksum_address(config.identity_registry),
                abi=IDENTITY_REGISTRY_ABI,
            )
        if config.reputation_registry:
            self._reputation = self.w3.eth.contract(
                address=Web3.to_checksum_address(config.reputation_registry),
                abi=REPUTATION_REGISTRY_ABI,
            )
        if config.validation_registry:
            self._validation = self.w3.eth.contract(
                address=Web3.to_checksum_address(config.validation_registry),
                abi=VALIDATION_REGISTRY_ABI,
            )

    # ------------------------------------------------------------------
    # Identity Registry
    # ------------------------------------------------------------------

    def register_swarm_agent(self, agent_uri_json: str) -> int:
        """
        Register a new swarm agent in the Identity Registry.

        Args:
            agent_uri_json: JSON string describing the agent metadata

        Returns:
            agentId (token ID) of the newly registered agent
        """
        if not self._identity:
            raise RuntimeError("Identity registry address not configured")

        tx = self._identity.functions.register(agent_uri_json).build_transaction(
            self._tx_params()
        )
        receipt = self._send_tx(tx)

        # Parse AgentRegistered event to get agentId
        logs = self._identity.events.AgentRegistered().process_receipt(receipt)
        if logs:
            return logs[0]["args"]["agentId"]

        raise RuntimeError("AgentRegistered event not found in receipt")

    def set_agent_wallet(
        self, agent_id: int, wallet: str, deadline: int = 0, sig: bytes = b""
    ) -> str:
        """
        Set the wallet address for an agent.

        Args:
            agent_id: Agent token ID
            wallet: Wallet address to associate
            deadline: Signature expiry timestamp (0 for no expiry in mock)
            sig: EIP-712 signature bytes

        Returns:
            Transaction hash hex string
        """
        if not self._identity:
            raise RuntimeError("Identity registry address not configured")

        tx = self._identity.functions.setAgentWallet(
            agent_id,
            Web3.to_checksum_address(wallet),
            deadline,
            sig,
        ).build_transaction(self._tx_params())
        receipt = self._send_tx(tx)
        return receipt.transactionHash.hex()

    def get_agent_wallet(self, agent_id: int) -> str:
        """Get the wallet address for an agent."""
        if not self._identity:
            raise RuntimeError("Identity registry address not configured")
        return self._identity.functions.agentWallet(agent_id).call()

    # ------------------------------------------------------------------
    # Reputation Registry
    # ------------------------------------------------------------------

    def give_reputation_feedback(
        self,
        agent_id: int,
        value: int,
        decimals: int = 0,
        tag1: str = "",
        tag2: str = "",
        feedback_uri: str = "",
        feedback_hash: bytes = b"\x00" * 32,
    ) -> str:
        """
        Record reputation feedback for an agent.

        Args:
            agent_id: Target agent's token ID
            value: Numeric reputation score
            decimals: Decimal precision of value
            tag1: Primary category tag
            tag2: Secondary category tag
            feedback_uri: URI to detailed feedback
            feedback_hash: keccak256 of feedback payload

        Returns:
            Transaction hash hex string
        """
        if not self._reputation:
            raise RuntimeError("Reputation registry address not configured")

        tx = self._reputation.functions.giveFeedback(
            agent_id, value, decimals, tag1, tag2, feedback_uri, feedback_hash
        ).build_transaction(self._tx_params())
        receipt = self._send_tx(tx)
        return receipt.transactionHash.hex()

    # ------------------------------------------------------------------
    # Validation Registry
    # ------------------------------------------------------------------

    def open_validation_request(
        self,
        validator: str,
        agent_id: int,
        request_uri: str,
        request_hash: bytes,
    ) -> str:
        """
        Create a validation request.

        Args:
            validator: Designated validator address
            agent_id: Agent requesting validation
            request_uri: URI to validation request payload
            request_hash: keccak256 commitment

        Returns:
            Transaction hash hex string
        """
        if not self._validation:
            raise RuntimeError("Validation registry address not configured")

        tx = self._validation.functions.validationRequest(
            Web3.to_checksum_address(validator),
            agent_id,
            request_uri,
            request_hash,
        ).build_transaction(self._tx_params())
        receipt = self._send_tx(tx)
        return receipt.transactionHash.hex()

    def submit_validation_response(
        self,
        request_hash: bytes,
        value: int,
        response_uri: str,
        response_hash: bytes,
        tag1: str = "",
    ) -> str:
        """
        Submit a validation response.

        Args:
            request_hash: Original request hash
            value: Numeric validation result
            response_uri: URI to response payload
            response_hash: keccak256 of response
            tag1: Category tag

        Returns:
            Transaction hash hex string
        """
        if not self._validation:
            raise RuntimeError("Validation registry address not configured")

        tx = self._validation.functions.validationResponse(
            request_hash, value, response_uri, response_hash, tag1
        ).build_transaction(self._tx_params())
        receipt = self._send_tx(tx)
        return receipt.transactionHash.hex()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tx_params(self) -> dict:
        """Build common transaction parameters."""
        if not self.account:
            raise RuntimeError("No deployer private key configured")
        return {
            "from": self.account.address,
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "gas": self.config.gas_limit,
            "chainId": self.config.chain_id,
        }

    def _send_tx(self, tx: dict):
        """Sign and send a transaction, wait for receipt."""
        if not self.account:
            raise RuntimeError("No deployer account configured (check vault)")
        signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
