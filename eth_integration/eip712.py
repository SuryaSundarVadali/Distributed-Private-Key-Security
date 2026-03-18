"""
EIP-712 typed data signing utilities.

Provides helpers for constructing and signing EIP-712 typed data structures
used in ERC-8004 operations (e.g. setAgentWallet authorization).
"""

import hashlib
from typing import Any, Dict, Tuple

from eth_account import Account
from eth_account.messages import encode_defunct, encode_typed_data


def build_set_wallet_message(
    chain_id: int,
    verifying_contract: str,
    agent_id: int,
    wallet: str,
    deadline: int,
    nonce: int,
) -> Dict[str, Any]:
    """
    Build an EIP-712 typed data structure for setAgentWallet.

    Args:
        chain_id: EIP-155 chain ID
        verifying_contract: Address of the contract verifying the signature
        agent_id: Agent token ID
        wallet: New wallet address to set
        deadline: Signature expiry timestamp
        nonce: Replay protection nonce

    Returns:
        EIP-712 structured data dict ready for signing
    """
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "SetAgentWallet": [
                {"name": "agentId", "type": "uint256"},
                {"name": "wallet", "type": "address"},
                {"name": "deadline", "type": "uint256"},
                {"name": "nonce", "type": "uint256"},
            ],
        },
        "primaryType": "SetAgentWallet",
        "domain": {
            "name": "SwarmMissionController",
            "version": "1",
            "chainId": chain_id,
            "verifyingContract": verifying_contract,
        },
        "message": {
            "agentId": agent_id,
            "wallet": wallet,
            "deadline": deadline,
            "nonce": nonce,
        },
    }


def sign_typed_data(private_key: str, structured_data: Dict[str, Any]) -> Tuple[bytes, str]:
    """
    Sign EIP-712 typed data with a private key.

    Args:
        private_key: Hex-encoded private key (with or without 0x prefix)
        structured_data: EIP-712 structured data dict

    Returns:
        Tuple of (signature_bytes, hex_signature)
    """
    signable = encode_typed_data(structured_data)
    signed = Account.sign_message(signable, private_key=private_key)
    return signed.signature, signed.signature.hex()


def sign_message(private_key: str, message: bytes) -> Tuple[bytes, str]:
    """
    Sign a raw message with EIP-191 personal sign.

    Args:
        private_key: Hex-encoded private key
        message: Raw message bytes

    Returns:
        Tuple of (signature_bytes, hex_signature)
    """
    signable = encode_defunct(message)
    signed = Account.sign_message(signable, private_key=private_key)
    return signed.signature, signed.signature.hex()


def compute_request_hash(payload: bytes) -> bytes:
    """
    Compute keccak256 hash of a payload (for validation request/response hashing).

    Args:
        payload: Raw bytes to hash

    Returns:
        32-byte keccak256 digest
    """
    from web3 import Web3
    return Web3.keccak(payload)
