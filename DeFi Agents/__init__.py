"""
DeFi Agent Coordination System
Entry point for autonomous yield farming and DeFi protocol coordination

Author: Venkata Surya Sundar Vadali
Date: January 11, 2026
"""

__version__ = "1.0.0"
__author__ = "Venkata Surya Sundar Vadali"

from typing import Dict, List

# Module-level exports
__all__ = [
    "YieldFarmingAgent",
    "DeFiPoolVerifier",
    "RewardDistributor",
    "create_defi_agent"
]


def create_defi_agent(agent_id: str, config: Dict) -> "YieldFarmingAgent":
    """
    Factory function to create a DeFi yield farming agent.
    
    Args:
        agent_id: Unique identifier for the agent
        config: Configuration dictionary with agent parameters
        
    Returns:
        Configured YieldFarmingAgent instance
    """
    from .yield_farming_agent import YieldFarmingAgent
    return YieldFarmingAgent(agent_id=agent_id, config=config)


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "YieldFarmingAgent":
        from .yield_farming_agent import YieldFarmingAgent
        return YieldFarmingAgent
    elif name == "DeFiPoolVerifier":
        from .pool_verifier import DeFiPoolVerifier
        return DeFiPoolVerifier
    elif name == "RewardDistributor":
        from .reward_distributor import RewardDistributor
        return RewardDistributor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
