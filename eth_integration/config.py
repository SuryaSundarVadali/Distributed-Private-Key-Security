"""
Chain configuration for Ethereum / ERC-8004 integration.

Loads settings from environment variables or .env file.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChainConfig:
    """Ethereum chain and contract configuration."""

    # RPC
    rpc_url: str = "http://127.0.0.1:8545"
    chain_id: int = 31337  # Anvil / Hardhat default

    # Contract addresses (set after deployment)
    identity_registry: str = ""
    reputation_registry: str = ""
    validation_registry: str = ""
    swarm_controller: str = ""

    # Default validator for validation requests
    default_validator: str = ""

    # Key Vault settings
    vault_path: str = "vault_keys/deployer.enc"
    deployer_key_name: str = "deployer"

    # Gas settings
    gas_limit: int = 3_000_000
    max_fee_per_gas: Optional[int] = None

    def validate(self) -> list[str]:
        """Return list of missing required fields."""
        missing = []
        if not self.rpc_url:
            missing.append("rpc_url / ETH_RPC_URL")
        return missing


def load_config(env_file: Optional[str] = None) -> ChainConfig:
    """
    Load chain config from environment variables.

    Optionally reads from a .env file first (requires python-dotenv).

    Environment variables:
        ETH_RPC_URL, ETH_CHAIN_ID,
        IDENTITY_REGISTRY_ADDRESS, REPUTATION_REGISTRY_ADDRESS,
        VALIDATION_REGISTRY_ADDRESS, SWARM_CONTROLLER_ADDRESS,
        DEFAULT_VALIDATOR_ADDRESS, DEPLOYER_PRIVATE_KEY,
        GAS_LIMIT, MAX_FEE_PER_GAS
    """
    # Try loading .env if path given or default exists
    if env_file or os.path.exists(".env"):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file or ".env")
        except ImportError:
            pass  # python-dotenv not installed, rely on real env vars

    return ChainConfig(
        rpc_url=os.getenv("ETH_RPC_URL", "http://127.0.0.1:8545"),
        chain_id=int(os.getenv("ETH_CHAIN_ID", "31337")),
        identity_registry=os.getenv("IDENTITY_REGISTRY_ADDRESS", ""),
        reputation_registry=os.getenv("REPUTATION_REGISTRY_ADDRESS", ""),
        validation_registry=os.getenv("VALIDATION_REGISTRY_ADDRESS", ""),
        swarm_controller=os.getenv("SWARM_CONTROLLER_ADDRESS", ""),
        default_validator=os.getenv("DEFAULT_VALIDATOR_ADDRESS", ""),
        vault_path=os.getenv("VAULT_PATH", "vault_keys/deployer.enc"),
        deployer_key_name=os.getenv("DEPLOYER_KEY_NAME", "deployer"),
        gas_limit=int(os.getenv("GAS_LIMIT", "3000000")),
        max_fee_per_gas=int(os.getenv("MAX_FEE_PER_GAS")) if os.getenv("MAX_FEE_PER_GAS") else None,
    )
