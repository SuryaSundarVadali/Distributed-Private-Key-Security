"""
Contract ABIs for ERC-8004 registry interfaces and SwarmMissionController.

These are minimal ABIs containing only the functions we call from Python.
Generated from the Solidity interfaces in contracts/interfaces/.
"""

IDENTITY_REGISTRY_ABI = [
    {
        "inputs": [{"name": "agentURI", "type": "string"}],
        "name": "register",
        "outputs": [{"name": "agentId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "wallet", "type": "address"},
            {"name": "deadline", "type": "uint256"},
            {"name": "sig", "type": "bytes"},
        ],
        "name": "setAgentWallet",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "name": "ownerOf",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "name": "agentURI",
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "name": "agentWallet",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "agentId", "type": "uint256"},
            {"indexed": True, "name": "owner", "type": "address"},
            {"indexed": False, "name": "agentURI", "type": "string"},
        ],
        "name": "AgentRegistered",
        "type": "event",
    },
]

REPUTATION_REGISTRY_ABI = [
    {
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "value", "type": "uint256"},
            {"name": "valueDecimals", "type": "uint8"},
            {"name": "tag1", "type": "string"},
            {"name": "tag2", "type": "string"},
            {"name": "feedbackURI", "type": "string"},
            {"name": "feedbackHash", "type": "bytes32"},
        ],
        "name": "giveFeedback",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "agentId", "type": "uint256"},
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
            {"indexed": False, "name": "tag1", "type": "string"},
            {"indexed": False, "name": "tag2", "type": "string"},
        ],
        "name": "FeedbackGiven",
        "type": "event",
    },
]

VALIDATION_REGISTRY_ABI = [
    {
        "inputs": [
            {"name": "validator", "type": "address"},
            {"name": "agentId", "type": "uint256"},
            {"name": "requestURI", "type": "string"},
            {"name": "requestHash", "type": "bytes32"},
        ],
        "name": "validationRequest",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "requestHash", "type": "bytes32"},
            {"name": "value", "type": "uint256"},
            {"name": "responseURI", "type": "string"},
            {"name": "responseHash", "type": "bytes32"},
            {"name": "tag1", "type": "string"},
        ],
        "name": "validationResponse",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "requestHash", "type": "bytes32"},
            {"indexed": True, "name": "agentId", "type": "uint256"},
            {"indexed": True, "name": "validator", "type": "address"},
            {"indexed": False, "name": "requestURI", "type": "string"},
        ],
        "name": "ValidationRequested",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "requestHash", "type": "bytes32"},
            {"indexed": False, "name": "value", "type": "uint256"},
            {"indexed": False, "name": "responseURI", "type": "string"},
        ],
        "name": "ValidationResponded",
        "type": "event",
    },
]

SWARM_MISSION_CONTROLLER_ABI = [
    {
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "missionRoot", "type": "bytes32"},
            {"name": "missionURI", "type": "string"},
        ],
        "name": "startMission",
        "outputs": [{"name": "missionId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "missionId", "type": "uint256"},
            {"name": "requestURI", "type": "string"},
            {"name": "requestHash", "type": "bytes32"},
        ],
        "name": "markMissionCompleted",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "missionId", "type": "uint256"}],
        "name": "getMission",
        "outputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "missionRoot", "type": "bytes32"},
            {"name": "missionURI", "type": "string"},
            {"name": "status", "type": "uint8"},
            {"name": "startedAt", "type": "uint256"},
            {"name": "completedAt", "type": "uint256"},
            {"name": "validationRequestHash", "type": "bytes32"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "missionCount",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "signer", "type": "address"},
            {"name": "authorized", "type": "bool"},
        ],
        "name": "setAuthorizedSigner",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "hash", "type": "bytes32"},
            {"name": "signature", "type": "bytes"},
        ],
        "name": "isValidSignature",
        "outputs": [{"name": "", "type": "bytes4"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "owner",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "missionId", "type": "uint256"},
            {"indexed": True, "name": "agentId", "type": "uint256"},
            {"indexed": False, "name": "missionRoot", "type": "bytes32"},
            {"indexed": False, "name": "missionURI", "type": "string"},
        ],
        "name": "MissionStarted",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "missionId", "type": "uint256"},
            {"indexed": False, "name": "requestHash", "type": "bytes32"},
        ],
        "name": "MissionCompleted",
        "type": "event",
    },
]
