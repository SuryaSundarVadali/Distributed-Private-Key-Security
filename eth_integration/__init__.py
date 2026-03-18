"""
eth_integration - Ethereum / ERC-8004 integration for Distributed Private Key Security

Bridges the off-chain cryptographic swarm architecture (HKDF, Shamir SSS, Merkle,
HOTP, HEFT, SMPC) with on-chain ERC-8004 Trustless Agents registries via web3.py.
"""

from eth_integration.config import ChainConfig, load_config
from eth_integration.mission import MissionManager
from eth_integration.registry import RegistryClient

__all__ = [
    "ChainConfig",
    "load_config",
    "MissionManager",
    "RegistryClient",
]
