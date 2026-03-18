"""
Tests for the eth_integration module.

Uses mocked web3 providers to test all functionality without a live chain.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ── Config Tests ──

class TestChainConfig:
    def test_default_config(self):
        from eth_integration.config import ChainConfig
        config = ChainConfig()
        assert config.rpc_url == "http://127.0.0.1:8545"
        assert config.chain_id == 31337
        assert config.gas_limit == 3_000_000

    def test_validate_missing_rpc(self):
        from eth_integration.config import ChainConfig
        config = ChainConfig(rpc_url="")
        missing = config.validate()
        assert "rpc_url / ETH_RPC_URL" in missing

    def test_validate_all_present(self):
        from eth_integration.config import ChainConfig
        config = ChainConfig(
            rpc_url="http://localhost:8545",
        )
        assert config.validate() == []

    def test_load_config_from_env(self):
        from eth_integration.config import load_config
        with patch.dict(os.environ, {
            "ETH_RPC_URL": "http://custom:8545",
            "ETH_CHAIN_ID": "1",
            "VAULT_PATH": "env",
            "DEPLOYER_KEY_NAME": "testkey",
        }):
            config = load_config()
            assert config.rpc_url == "http://custom:8545"
            assert config.chain_id == 1
            assert config.vault_path == "env"
            assert config.deployer_key_name == "testkey"


# ── EIP-712 Tests ──

class TestEIP712:
    def test_build_set_wallet_message(self):
        from eth_integration.eip712 import build_set_wallet_message
        msg = build_set_wallet_message(
            chain_id=31337,
            verifying_contract="0x" + "ab" * 20,
            agent_id=1,
            wallet="0x" + "cd" * 20,
            deadline=999999,
            nonce=0,
        )
        assert msg["primaryType"] == "SetAgentWallet"
        assert msg["domain"]["chainId"] == 31337
        assert msg["message"]["agentId"] == 1

    def test_sign_message(self):
        from eth_integration.eip712 import sign_message
        # Use a known test key
        test_key = "0x" + "ab" * 32
        sig_bytes, sig_hex = sign_message(test_key, b"test message")
        assert len(sig_bytes) == 65
        assert isinstance(sig_hex, str)

    def test_compute_request_hash(self):
        from eth_integration.eip712 import compute_request_hash
        h = compute_request_hash(b"test payload")
        assert len(h) == 32


# ── Mission Manager Tests ──

class TestMissionManager:
    def test_generate_validation_report(self):
        from eth_integration.mission import MissionManager

        mission_root = bytes.fromhex("aa" * 32)
        smpc_results = {
            "verified": True,
            "aggregated_result": 0.95,
            "participants": 5,
            "privacy_metrics": {"information_leakage": 0.0},
        }

        report_json, report_hash = MissionManager.generate_validation_report(
            mission_root, smpc_results, agent_id=1, mission_id=0
        )

        report = json.loads(report_json)
        assert report["version"] == "1.0"
        assert report["agentId"] == 1
        assert report["verification"]["verified"] is True
        assert report["verification"]["participantCount"] == 5
        assert len(report_hash) == 32

    def test_generate_validation_report_deterministic(self):
        from eth_integration.mission import MissionManager

        root = b"\x00" * 32
        results = {"verified": True, "participants": 3}

        json1, hash1 = MissionManager.generate_validation_report(root, results)
        json2, hash2 = MissionManager.generate_validation_report(root, results)

        assert json1 == json2
        assert hash1 == hash2


# ── ABI Tests ──

class TestABIs:
    def test_abis_are_valid_lists(self):
        from eth_integration.abis import (
            IDENTITY_REGISTRY_ABI,
            REPUTATION_REGISTRY_ABI,
            VALIDATION_REGISTRY_ABI,
            SWARM_MISSION_CONTROLLER_ABI,
        )
        for abi in [IDENTITY_REGISTRY_ABI, REPUTATION_REGISTRY_ABI,
                    VALIDATION_REGISTRY_ABI, SWARM_MISSION_CONTROLLER_ABI]:
            assert isinstance(abi, list)
            assert len(abi) > 0
            # Each entry should have a "type" field
            for entry in abi:
                assert "type" in entry

    def test_controller_abi_has_required_functions(self):
        from eth_integration.abis import SWARM_MISSION_CONTROLLER_ABI
        function_names = {
            e["name"] for e in SWARM_MISSION_CONTROLLER_ABI if e["type"] == "function"
        }
        assert "startMission" in function_names
        assert "markMissionCompleted" in function_names
        assert "getMission" in function_names
        assert "isValidSignature" in function_names


# ── Integration: Config → Registry Client init ──

class TestRegistryClientInit:
    def test_init_without_addresses(self):
        from eth_integration.config import ChainConfig
        from eth_integration.registry import RegistryClient

        config = ChainConfig(vault_path="env", deployer_key_name="deployer")
        with patch.dict(os.environ, {"VAULT_KEY_DEPLOYER": "ab" * 32}):
            client = RegistryClient(config)
        assert client._identity is None
        assert client._reputation is None
        assert client._validation is None

    def test_register_raises_without_address(self):
        from eth_integration.config import ChainConfig
        from eth_integration.registry import RegistryClient

        config = ChainConfig(vault_path="env", deployer_key_name="deployer")
        with patch.dict(os.environ, {"VAULT_KEY_DEPLOYER": "ab" * 32}):
            client = RegistryClient(config)
        with pytest.raises(RuntimeError, match="Identity registry address not configured"):
            client.register_swarm_agent('{"name": "test"}')


class TestMissionManagerInit:
    def test_init_without_controller(self):
        from eth_integration.config import ChainConfig
        from eth_integration.mission import MissionManager

        config = ChainConfig(vault_path="env", deployer_key_name="deployer")
        with patch.dict(os.environ, {"VAULT_KEY_DEPLOYER": "ab" * 32}):
            mgr = MissionManager(config)
        assert mgr._controller is None

    def test_start_mission_raises_without_controller(self):
        from eth_integration.config import ChainConfig
        from eth_integration.mission import MissionManager

        config = ChainConfig(vault_path="env", deployer_key_name="deployer")
        with patch.dict(os.environ, {"VAULT_KEY_DEPLOYER": "ab" * 32}):
            mgr = MissionManager(config)
        with pytest.raises(RuntimeError, match="SwarmMissionController address not configured"):
            mgr.start_mission(1, b"\x00" * 32, "uri")
