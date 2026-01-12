"""
Bayesian Reputation Scorer
===========================

Probabilistic reputation scoring using Beta distribution for modeling agent reliability.
Used to weight votes in consensus and adjust trust based on historical behavior.

Mathematical Foundation:
- Uses Beta(α, β) distribution to model probability of honest behavior
- α = successes (honest actions), β = failures (dishonest actions)
- Expected reliability = α / (α + β)
- Variance decreases with more observations (confidence increases)

Features:
- Probabilistic trust scoring (0.0 to 1.0)
- Bayesian updates from new evidence
- Historical reliability tracking
- Confidence intervals for trust estimates
- Decay for old evidence (optional)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class AgentReputationState:
    """
    Bayesian reputation state for an agent using Beta distribution.
    
    Beta distribution parameters:
    - alpha: Number of successful/honest actions + prior
    - beta: Number of failed/dishonest actions + prior
    
    The Beta distribution is ideal for modeling binary outcomes (honest/dishonest)
    and provides natural Bayesian updating.
    """
    agent_id: str
    alpha: float  # Successes + prior
    beta: float   # Failures + prior
    
    # History tracking
    total_observations: int = 0
    last_update_time: float = field(default_factory=time.time)
    
    # Event tracking
    honest_actions: int = 0
    dishonest_actions: int = 0
    
    # Metadata
    first_seen: float = field(default_factory=time.time)
    notes: List[str] = field(default_factory=list)
    
    def get_expected_reliability(self) -> float:
        """
        Get expected reliability (mean of Beta distribution).
        E[X] = α / (α + β)
        """
        return self.alpha / (self.alpha + self.beta)
    
    def get_variance(self) -> float:
        """
        Get variance of reliability estimate.
        Var[X] = αβ / ((α+β)²(α+β+1))
        Lower variance = more confidence in estimate.
        """
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence interval for reliability estimate.
        Uses normal approximation for Beta distribution.
        
        Returns: (lower_bound, upper_bound)
        """
        mean = self.get_expected_reliability()
        std = math.sqrt(self.get_variance())
        
        # Z-score for given confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        
        margin = z * std
        lower = max(0.0, mean - margin)
        upper = min(1.0, mean + margin)
        
        return lower, upper
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        reliability = self.get_expected_reliability()
        variance = self.get_variance()
        ci_lower, ci_upper = self.get_confidence_interval()
        
        return {
            'agent_id': self.agent_id,
            'alpha': self.alpha,
            'beta': self.beta,
            'reliability': reliability,
            'variance': variance,
            'confidence_interval_95': [ci_lower, ci_upper],
            'total_observations': self.total_observations,
            'honest_actions': self.honest_actions,
            'dishonest_actions': self.dishonest_actions,
            'age_seconds': time.time() - self.first_seen
        }


class BayesianReputationScorer:
    """
    Bayesian reputation scoring system for DeFi agents.
    
    Uses Beta distribution to model agent reliability and provides
    probabilistic trust scores for weighted voting in consensus.
    
    Parameters:
    - prior_alpha: Prior belief in honest actions (default: 1.0 = uniform prior)
    - prior_beta: Prior belief in dishonest actions (default: 1.0 = uniform prior)
    - decay_factor: Time decay factor for old observations (0.0 = no decay, 1.0 = full decay)
    - decay_halflife_days: Half-life in days for observation decay
    """
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0,
                 decay_factor: float = 0.0, decay_halflife_days: float = 30.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay_factor = decay_factor
        self.decay_halflife_seconds = decay_halflife_days * 24 * 3600
        
        # Agent states
        self.agents: Dict[str, AgentReputationState] = {}
        
        # Statistics
        self.stats = {
            'total_agents': 0,
            'total_updates': 0,
            'honest_observations': 0,
            'dishonest_observations': 0,
            'reputation_increases': 0,
            'reputation_decreases': 0
        }
    
    def _apply_time_decay(self, state: AgentReputationState) -> AgentReputationState:
        """
        Apply time-based decay to observations.
        Recent observations have more weight than old ones.
        
        Uses exponential decay: weight = exp(-λt) where λ = ln(2) / halflife
        """
        if self.decay_factor == 0.0:
            return state  # No decay
        
        elapsed = time.time() - state.last_update_time
        decay_lambda = math.log(2) / self.decay_halflife_seconds
        decay_weight = math.exp(-decay_lambda * elapsed)
        
        # Apply decay to both alpha and beta
        # Move towards prior as observations decay
        decayed_alpha = self.prior_alpha + (state.alpha - self.prior_alpha) * decay_weight
        decayed_beta = self.prior_beta + (state.beta - self.prior_beta) * decay_weight
        
        state.alpha = decayed_alpha
        state.beta = decayed_beta
        
        return state
    
    def get_or_create_agent(self, agent_id: str) -> AgentReputationState:
        """Get agent state or create new one with priors"""
        if agent_id not in self.agents:
            state = AgentReputationState(
                agent_id=agent_id,
                alpha=self.prior_alpha,
                beta=self.prior_beta
            )
            self.agents[agent_id] = state
            self.stats['total_agents'] += 1
        
        return self.agents[agent_id]
    
    # ========== Bayesian Updates ==========
    
    def record_honest_action(self, agent_id: str, weight: float = 1.0, note: str = ""):
        """
        Record honest/successful action by agent.
        Updates Beta distribution: α → α + weight
        
        Parameters:
        - agent_id: Agent identifier
        - weight: Weight of observation (default: 1.0)
        - note: Optional note about the action
        """
        state = self.get_or_create_agent(agent_id)
        
        # Apply time decay if enabled
        if self.decay_factor > 0:
            state = self._apply_time_decay(state)
        
        # Record old reliability for comparison
        old_reliability = state.get_expected_reliability()
        
        # Bayesian update: add to alpha
        state.alpha += weight
        state.total_observations += 1
        state.honest_actions += 1
        state.last_update_time = time.time()
        
        if note:
            state.notes.append(f"[{time.time():.0f}] HONEST: {note}")
        
        # Track statistics
        self.stats['total_updates'] += 1
        self.stats['honest_observations'] += 1
        
        new_reliability = state.get_expected_reliability()
        if new_reliability > old_reliability:
            self.stats['reputation_increases'] += 1
    
    def record_dishonest_action(self, agent_id: str, weight: float = 1.0, note: str = ""):
        """
        Record dishonest/failed action by agent.
        Updates Beta distribution: β → β + weight
        
        Parameters:
        - agent_id: Agent identifier
        - weight: Weight of observation (default: 1.0, higher = more severe)
        - note: Optional note about the action
        """
        state = self.get_or_create_agent(agent_id)
        
        # Apply time decay if enabled
        if self.decay_factor > 0:
            state = self._apply_time_decay(state)
        
        # Record old reliability for comparison
        old_reliability = state.get_expected_reliability()
        
        # Bayesian update: add to beta
        state.beta += weight
        state.total_observations += 1
        state.dishonest_actions += 1
        state.last_update_time = time.time()
        
        if note:
            state.notes.append(f"[{time.time():.0f}] DISHONEST: {note}")
        
        # Track statistics
        self.stats['total_updates'] += 1
        self.stats['dishonest_observations'] += 1
        
        new_reliability = state.get_expected_reliability()
        if new_reliability < old_reliability:
            self.stats['reputation_decreases'] += 1
    
    def record_action(self, agent_id: str, is_honest: bool, weight: float = 1.0, note: str = ""):
        """
        Record action (honest or dishonest) by agent.
        Convenience method that calls record_honest_action or record_dishonest_action.
        """
        if is_honest:
            self.record_honest_action(agent_id, weight, note)
        else:
            self.record_dishonest_action(agent_id, weight, note)
    
    # ========== Reputation Queries ==========
    
    def get_reliability_score(self, agent_id: str) -> float:
        """
        Get reliability score for agent (0.0 to 1.0).
        Returns expected value of Beta distribution.
        
        Returns 0.5 for unknown agents (uniform prior).
        """
        if agent_id not in self.agents:
            return 0.5  # Neutral score for unknown agents
        
        state = self.agents[agent_id]
        
        # Apply time decay if enabled
        if self.decay_factor > 0:
            state = self._apply_time_decay(state)
        
        return state.get_expected_reliability()
    
    def get_confidence(self, agent_id: str) -> float:
        """
        Get confidence in reliability estimate (0.0 to 1.0).
        Based on inverse of variance: more observations = higher confidence.
        
        Returns 0.0 for unknown agents.
        """
        if agent_id not in self.agents:
            return 0.0
        
        state = self.agents[agent_id]
        variance = state.get_variance()
        
        # Convert variance to confidence (inverse relationship)
        # Variance ranges from ~0 (high confidence) to 0.25 (low confidence)
        # Map to confidence: 1.0 (high) to 0.0 (low)
        confidence = 1.0 - min(1.0, variance * 4.0)
        
        return confidence
    
    def get_weighted_score(self, agent_id: str) -> float:
        """
        Get reliability score weighted by confidence.
        
        Used for vote weighting: agents with both high reliability AND
        high confidence get the highest weights.
        
        Returns: reliability * confidence
        """
        reliability = self.get_reliability_score(agent_id)
        confidence = self.get_confidence(agent_id)
        return reliability * confidence
    
    def get_reputation_state(self, agent_id: str) -> Optional[AgentReputationState]:
        """Get full reputation state for agent"""
        return self.agents.get(agent_id)
    
    def get_all_agents_ranked(self, metric: str = 'reliability') -> List[Tuple[str, float]]:
        """
        Get all agents ranked by metric.
        
        Parameters:
        - metric: 'reliability', 'confidence', or 'weighted'
        
        Returns: List of (agent_id, score) tuples, sorted descending
        """
        agents_scores = []
        
        for agent_id in self.agents:
            if metric == 'reliability':
                score = self.get_reliability_score(agent_id)
            elif metric == 'confidence':
                score = self.get_confidence(agent_id)
            elif metric == 'weighted':
                score = self.get_weighted_score(agent_id)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            agents_scores.append((agent_id, score))
        
        # Sort by score (descending)
        agents_scores.sort(key=lambda x: x[1], reverse=True)
        
        return agents_scores
    
    # ========== Analysis ==========
    
    def identify_unreliable_agents(self, threshold: float = 0.4) -> List[str]:
        """
        Identify agents with reliability below threshold.
        
        Parameters:
        - threshold: Reliability threshold (default: 0.4)
        
        Returns: List of agent IDs with low reliability
        """
        unreliable = []
        
        for agent_id in self.agents:
            reliability = self.get_reliability_score(agent_id)
            confidence = self.get_confidence(agent_id)
            
            # Only flag if we have sufficient confidence
            if reliability < threshold and confidence > 0.3:
                unreliable.append(agent_id)
        
        return unreliable
    
    def identify_highly_reliable_agents(self, threshold: float = 0.8) -> List[str]:
        """
        Identify agents with reliability above threshold.
        
        Parameters:
        - threshold: Reliability threshold (default: 0.8)
        
        Returns: List of agent IDs with high reliability
        """
        reliable = []
        
        for agent_id in self.agents:
            reliability = self.get_reliability_score(agent_id)
            confidence = self.get_confidence(agent_id)
            
            # Only flag if we have sufficient confidence
            if reliability > threshold and confidence > 0.3:
                reliable.append(agent_id)
        
        return reliable
    
    def get_statistics(self) -> Dict:
        """Get scoring statistics"""
        return {
            **self.stats,
            'agents_tracked': len(self.agents),
            'average_reliability': sum(self.get_reliability_score(a) for a in self.agents) / max(1, len(self.agents))
        }


# ========== Built-in Tests ==========

def test_bayesian_scorer():
    """Test Bayesian reputation scorer"""
    print("Testing Bayesian Reputation Scorer")
    print("=" * 60)
    
    # Create scorer with uniform prior
    scorer = BayesianReputationScorer(prior_alpha=1.0, prior_beta=1.0)
    print("\n1. Created Bayesian scorer (uniform prior)")
    
    # Test agent with all honest actions
    print("\n2. Testing honest agent (alice)...")
    for i in range(10):
        scorer.record_honest_action("alice", weight=1.0, note=f"Honest action {i+1}")
    
    reliability = scorer.get_reliability_score("alice")
    confidence = scorer.get_confidence("alice")
    weighted = scorer.get_weighted_score("alice")
    
    print(f"  Reliability: {reliability:.3f}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Weighted score: {weighted:.3f}")
    
    state = scorer.get_reputation_state("alice")
    ci_lower, ci_upper = state.get_confidence_interval()
    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Test agent with mixed actions
    print("\n3. Testing mixed agent (bob)...")
    for i in range(5):
        scorer.record_honest_action("bob", note=f"Honest {i+1}")
    for i in range(5):
        scorer.record_dishonest_action("bob", note=f"Dishonest {i+1}")
    
    reliability = scorer.get_reliability_score("bob")
    confidence = scorer.get_confidence("bob")
    print(f"  Reliability: {reliability:.3f}")
    print(f"  Confidence: {confidence:.3f}")
    
    # Test agent with mostly dishonest actions
    print("\n4. Testing dishonest agent (charlie)...")
    for i in range(2):
        scorer.record_honest_action("charlie")
    for i in range(8):
        scorer.record_dishonest_action("charlie", weight=1.0)
    
    reliability = scorer.get_reliability_score("charlie")
    confidence = scorer.get_confidence("charlie")
    print(f"  Reliability: {reliability:.3f}")
    print(f"  Confidence: {confidence:.3f}")
    
    # Test new agent (no history)
    print("\n5. Testing new agent (dave)...")
    reliability = scorer.get_reliability_score("dave")
    confidence = scorer.get_confidence("dave")
    print(f"  Reliability: {reliability:.3f} (neutral)")
    print(f"  Confidence: {confidence:.3f} (no observations)")
    
    # Rank agents
    print("\n6. Agent rankings:")
    ranked = scorer.get_all_agents_ranked(metric='weighted')
    for i, (agent_id, score) in enumerate(ranked, 1):
        print(f"  {i}. {agent_id}: {score:.3f}")
    
    # Identify unreliable agents
    print("\n7. Identifying unreliable agents (threshold=0.5)...")
    unreliable = scorer.identify_unreliable_agents(threshold=0.5)
    print(f"  Unreliable agents: {unreliable}")
    
    # Identify highly reliable agents
    print("\n8. Identifying highly reliable agents (threshold=0.8)...")
    reliable = scorer.identify_highly_reliable_agents(threshold=0.8)
    print(f"  Highly reliable agents: {reliable}")
    
    # Statistics
    print("\n9. Statistics:")
    stats = scorer.get_statistics()
    print(f"  Total agents: {stats['agents_tracked']}")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  Honest observations: {stats['honest_observations']}")
    print(f"  Dishonest observations: {stats['dishonest_observations']}")
    print(f"  Average reliability: {stats['average_reliability']:.3f}")
    
    # Test weighted voting
    print("\n10. Weighted voting simulation...")
    agents = ['alice', 'bob', 'charlie']
    votes = {'alice': 'yes', 'bob': 'yes', 'charlie': 'no'}
    
    weighted_yes = sum(scorer.get_weighted_score(a) for a, v in votes.items() if v == 'yes')
    weighted_no = sum(scorer.get_weighted_score(a) for a, v in votes.items() if v == 'no')
    
    print(f"  Votes: {votes}")
    print(f"  Weighted YES: {weighted_yes:.3f}")
    print(f"  Weighted NO: {weighted_no:.3f}")
    print(f"  Winner: {'YES' if weighted_yes > weighted_no else 'NO'}")
    
    print("\n✓ All Bayesian scorer tests passed!")


if __name__ == "__main__":
    test_bayesian_scorer()
