# Architecture: Distributed Private Key Security with ERC-8004 Trust Layer

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    ON-CHAIN (Ethereum)                   │
│                                                         │
│  ┌──────────────────┐  ┌────────────────────────────┐   │
│  │ Identity Registry │  │  SwarmMissionController     │   │
│  │ (ERC-721 NFTs)    │  │  • startMission(root, uri) │   │
│  │ • register()      │  │  • markMissionCompleted()  │   │
│  │ • setAgentWallet() │  │  • isValidSignature()     │   │
│  └──────────────────┘  │    (ERC-1271)               │   │
│                         └──────────┬─────────────────┘   │
│  ┌──────────────────┐              │                     │
│  │ Reputation Reg.   │  ┌──────────▼─────────────────┐   │
│  │ • giveFeedback()  │  │  Validation Registry       │   │
│  └──────────────────┘  │  • validationRequest()      │   │
│                         │  • validationResponse()     │   │
│                         └────────────────────────────┘   │
│                    ERC-8004 Registries                   │
└─────────────────────────────┬───────────────────────────┘
                              │ eth_integration/
                              │ (web3.py bridge)
┌─────────────────────────────▼───────────────────────────┐
│                  OFF-CHAIN (Python)                      │
│                                                         │
│  ┌─────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │ HKDF Entropy │  │ Shamir SSS    │  │ Merkle Tree   │  │
│  │ Key Derivation│  │ + Feldman VSS │  │ Task Commits  │──┼── missionRoot
│  └──────┬──────┘  └───────┬───────┘  └───────────────┘  │   published on-chain
│         │                 │                              │
│  ┌──────▼──────┐  ┌───────▼───────┐  ┌───────────────┐  │
│  │ MFKDF       │  │ HOTP Sync     │  │ SMPC Verify   │──┼── validation report
│  │ Key Gen     │  │ Phase Mgmt    │  │ 3-Phase       │  │   → on-chain validation
│  └─────────────┘  └───────────────┘  └───────────────┘  │
│                                                         │
│  ┌─────────────┐  ┌───────────────┐                     │
│  │ HEFT        │  │ Distributed   │                     │
│  │ Scheduler   │  │ Nodes         │                     │
│  └─────────────┘  └───────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

## Mission Lifecycle

```
1. Agent Registration     Identity Registry ← register(agentURI)
2. Key Derivation         HKDF + multi-source entropy → mission key
3. Share Distribution     Shamir (t,n) + Feldman VSS → shares to robots
4. Task Commitment        Merkle tree → missionRoot
5. Mission Start          SwarmMissionController ← startMission(root, uri)
6. Phase Synchronization  HOTP counters (no clock sync needed)
7. Task Execution         HEFT scheduler + distributed nodes
8. Collective Verify      SMPC 3-phase → zero-knowledge proof of success
9. Mission Complete       SwarmMissionController ← markMissionCompleted()
                          → auto-opens Validation Registry request
10. External Validation   Validator → validationResponse()
11. Reputation Update     Reputation Registry ← giveFeedback()
```

## Module Reference

| Layer | Module | Purpose |
|-------|--------|---------|
| On-chain | `contracts/SwarmMissionController.sol` | Mission lifecycle + ERC-1271 wallet |
| On-chain | `contracts/interfaces/I*Registry.sol` | ERC-8004 registry interfaces |
| Bridge | `eth_integration/config.py` | Chain & contract configuration |
| Bridge | `eth_integration/registry.py` | Identity, Reputation, Validation clients |
| Bridge | `eth_integration/mission.py` | Mission lifecycle + report generation |
| Bridge | `eth_integration/eip712.py` | EIP-712 typed data signing |
| Off-chain | `Cryptographic Modules/hkdf_entropy.py` | Key derivation from multi-source entropy |
| Off-chain | `Cryptographic Modules/shamir_secret_sharing.py` | (t,n) secret sharing + Feldman VSS |
| Off-chain | `Cryptographic Modules/merkle_tree.py` | Task commitment tree |
| Off-chain | `Cryptographic Modules/hotp_synchronization.py` | Event-based phase sync |
| Off-chain | `Cryptographic Modules/smpc_verification.py` | 3-phase SMPC task verification |
| Off-chain | `Core System/task_scheduler.py` | HEFT scheduling |
| Off-chain | `Core System/distributed_node.py` | Worker node logic |

## Standards Compliance

| Standard | Usage |
|----------|-------|
| **ERC-8004** | Identity, Reputation, Validation registries |
| **ERC-721** | Agent identity as NFT token |
| **ERC-1271** | SwarmMissionController as smart contract wallet |
| **EIP-712** | Typed data signing for authorized operations |
| **EIP-155** | Chain-ID aware transaction signing |
