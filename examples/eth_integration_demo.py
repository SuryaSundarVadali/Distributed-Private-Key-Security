#!/usr/bin/env python3
"""
ERC-8004 Integration Demo — End-to-End Mission Lifecycle

Demonstrates the full flow from agent registration through mission completion
and validation, bridging off-chain cryptographic protocols with on-chain
ERC-8004 Trustless Agents registries.

Prerequisites:
    1. Deploy contracts to a local Anvil node:
           anvil &
           forge script contracts/deploy.s.sol --broadcast --rpc-url http://127.0.0.1:8545
       OR use the mock mode below for a fully offline demo.

    2. Install Python dependencies:
           pip install web3 eth-account

    3. Set environment variables or create .env (see .env.example)

Usage:
    # Offline demo (simulates all on-chain interactions):
    python examples/eth_integration_demo.py

    # Live demo against local Anvil:
    python examples/eth_integration_demo.py --live
"""

import json
import os
import sys
import hashlib
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_stage(number: int, title: str):
    """Print a formatted stage header."""
    print(f"\n{'='*60}")
    print(f"  Stage {number}: {title}")
    print(f"{'='*60}")


def run_offline_demo():
    """
    Run a fully offline demo simulating the ERC-8004 integration flow.
    No blockchain connection required.
    """
    print("\n" + "🤖 " * 20)
    print("  ERC-8004 Trustless Agents — Offline Demo")
    print("🤖 " * 20)

    # ── Stage 1: Register Swarm Agent ──
    print_stage(1, "Register Swarm Agent Identity")

    agent_metadata = {
        "name": "SwarmAlpha",
        "type": "precision_farming",
        "capabilities": ["navigation", "soil_sampling", "crop_monitoring"],
        "swarm_size": 5,
        "threshold": 3,
    }
    print(f"  Agent metadata: {json.dumps(agent_metadata, indent=4)}")
    agent_id = 1  # Mock
    print(f"  ✅ Agent registered with agentId = {agent_id}")

    # ── Stage 2: Derive Mission Key & Build Merkle Tree ──
    print_stage(2, "Off-Chain Crypto: Key Derivation + Merkle Tree")

    # Simulate HKDF key derivation
    from Crypto.Protocol.KDF import HKDF   # type: ignore
    from Crypto.Hash import SHA256          # type: ignore
    import secrets

    ikm = secrets.token_bytes(32)
    salt = secrets.token_bytes(16)
    mission_key = HKDF(ikm, 32, salt, SHA256, context=b"swarm-mission-001")
    print(f"  Mission key derived (HKDF): {mission_key.hex()[:32]}...")

    # Simulate Merkle tree from task commitments
    tasks = [
        "navigate_to_field_A",
        "collect_soil_sample_1",
        "collect_soil_sample_2",
        "monitor_crop_section_B",
        "return_to_base",
    ]
    task_hashes = [hashlib.sha256(t.encode()).digest() for t in tasks]

    # Build minimal merkle root
    def merkle_root(leaves):
        if len(leaves) == 1:
            return leaves[0]
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        parents = []
        for i in range(0, len(leaves), 2):
            parents.append(hashlib.sha256(leaves[i] + leaves[i + 1]).digest())
        return merkle_root(parents)

    mission_root = merkle_root(task_hashes[:])
    print(f"  Merkle root: 0x{mission_root.hex()[:32]}...")
    print(f"  Tasks committed: {len(tasks)}")

    # ── Stage 3: Publish Mission On-Chain ──
    print_stage(3, "Publish Mission On-Chain (SwarmMissionController)")

    mission_uri = "ipfs://QmFakeHashForDemoMissionMetadata"
    mission_id = 0  # First mission
    print(f"  agentId:     {agent_id}")
    print(f"  missionRoot: 0x{mission_root.hex()[:32]}...")
    print(f"  missionURI:  {mission_uri}")
    print(f"  ✅ Mission #{mission_id} started (simulated tx)")

    # ── Stage 4: SMPC Verification ──
    print_stage(4, "Off-Chain Crypto: SMPC Task Verification")

    smpc_results = {
        "verified": True,
        "aggregated_result": 0.97,
        "participants": 5,
        "phases_completed": 3,
        "privacy_metrics": {
            "information_leakage": 0.0,
            "byzantine_tolerance": 0.4,
        },
    }
    print(f"  Protocol: 3-Phase SMPC")
    print(f"  Participants: {smpc_results['participants']}")
    print(f"  Verified: {smpc_results['verified']}")
    print(f"  Aggregated result: {smpc_results['aggregated_result']}")
    print(f"  Privacy: zero information leakage ✓")

    # ── Stage 5: Generate Validation Report & Complete Mission ──
    print_stage(5, "Complete Mission + Open Validation Request")

    report = {
        "version": "1.0",
        "agentId": agent_id,
        "missionId": mission_id,
        "missionRoot": mission_root.hex(),
        "verification": {
            "protocol": "SMPC-3Phase",
            "verified": smpc_results["verified"],
            "aggregatedResult": smpc_results["aggregated_result"],
            "participantCount": smpc_results["participants"],
            "privacyMetrics": smpc_results["privacy_metrics"],
        },
    }
    report_json = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report_hash = hashlib.sha256(report_json.encode()).hexdigest()
    report_uri = "ipfs://QmFakeHashForValidationReport"
    print(f"  Report hash: 0x{report_hash[:32]}...")
    print(f"  Report URI:  {report_uri}")
    print(f"  ✅ Mission #{mission_id} completed (simulated tx)")
    print(f"  ✅ Validation request opened in Validation Registry")

    # ── Stage 6: Validator Response + Reputation Feedback ──
    print_stage(6, "Validation Response + Reputation Feedback")

    print(f"  Validator responds with score: 95")
    print(f"  ✅ Validation response recorded (simulated tx)")
    print(f"  Reputation feedback: value=95, tag='mission_success'")
    print(f"  ✅ Reputation feedback recorded (simulated tx)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  🎉 Demo Complete — All 6 Stages Passed")
    print(f"{'='*60}")
    print(f"\n  Off-chain modules used:")
    print(f"    • HKDF entropy aggregation  → mission key derivation")
    print(f"    • Merkle tree               → task commitment root")
    print(f"    • SMPC 3-phase verification  → collective task verification")
    print(f"\n  On-chain ERC-8004 registries:")
    print(f"    • Identity Registry   → agent registration (agentId={agent_id})")
    print(f"    • SwarmMissionController → mission lifecycle (missionId={mission_id})")
    print(f"    • Validation Registry → request/response")
    print(f"    • Reputation Registry → feedback recording")
    print(f"\n  Integration bridge:")
    print(f"    • Merkle root published on-chain before execution")
    print(f"    • SMPC results → validation report → on-chain validation request")
    print(f"    • ERC-1271 smart contract wallet for collective signing")


def run_live_demo():
    """
    Run a live demo against a local Anvil node with deployed contracts.
    Requires: anvil running, contracts deployed, .env configured.
    """
    from eth_integration import load_config, MissionManager, RegistryClient

    print("\n" + "🤖 " * 20)
    print("  ERC-8004 Trustless Agents — Live Demo")
    print("🤖 " * 20)

    config = load_config()
    missing = config.validate()
    if missing:
        print(f"\n❌ Missing config: {', '.join(missing)}")
        print("   Set env vars or create .env (see .env.example)")
        sys.exit(1)

    registry = RegistryClient(config)
    missions = MissionManager(config)

    # Stage 1: Register
    print_stage(1, "Register Swarm Agent")
    agent_uri = json.dumps({"name": "SwarmAlpha", "type": "precision_farming"})
    agent_id = registry.register_swarm_agent(agent_uri)
    print(f"  ✅ Registered agentId = {agent_id}")

    # Stage 2: Derive key + Merkle tree
    print_stage(2, "Off-Chain Crypto")
    import secrets
    mission_root = secrets.token_bytes(32)
    print(f"  Mission root: 0x{mission_root.hex()[:32]}...")

    # Stage 3: Start mission
    print_stage(3, "Start Mission On-Chain")
    mission_id, tx_hash = missions.start_mission(agent_id, mission_root, "ipfs://demo")
    print(f"  ✅ Mission #{mission_id} started (tx: {tx_hash[:16]}...)")

    # Stage 4: Simulated SMPC
    print_stage(4, "SMPC Verification (simulated)")
    smpc_results = {"verified": True, "aggregated_result": 0.97, "participants": 5}
    print(f"  Verified: {smpc_results['verified']}")

    # Stage 5: Complete mission
    print_stage(5, "Complete Mission")
    report_json, report_hash = MissionManager.generate_validation_report(
        mission_root, smpc_results, agent_id, mission_id
    )
    tx_hash = missions.complete_mission(mission_id, "ipfs://report", report_hash)
    print(f"  ✅ Completed (tx: {tx_hash[:16]}...)")

    # Stage 6: Query final state
    print_stage(6, "Query Mission State")
    state = missions.get_mission(mission_id)
    print(f"  Status: {state['status']}")
    print(f"  Completed at block: {state['completedAt']}")

    print(f"\n{'='*60}")
    print("  🎉 Live Demo Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERC-8004 Integration Demo")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run against a live Anvil node (requires deployed contracts)",
    )
    args = parser.parse_args()

    if args.live:
        run_live_demo()
    else:
        run_offline_demo()
