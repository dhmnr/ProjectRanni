"""Random agent for sanity testing the evaluation pipeline."""

import numpy as np
from typing import Any, Dict, Optional


class RandomAgent:
    """Random policy agent for testing evaluation pipeline.

    Outputs random multi-binary actions regardless of observation.
    Useful for:
    - Verifying the evaluation loop works correctly
    - Establishing a baseline performance
    - Testing environment integration
    """

    def __init__(
        self,
        num_actions: int = 13,  # 13 semantic actions from keybinds_v2.json
        action_prob: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Initialize random agent.

        Args:
            num_actions: Number of binary actions (default: 13 semantic actions)
            action_prob: Probability of each action being 1 (default: 0.1)
            seed: Random seed for reproducibility
        """
        self.num_actions = num_actions
        self.action_prob = action_prob
        self.rng = np.random.default_rng(seed)

        # For compatibility with BC agents
        self.is_temporal = False
        self.use_state = False

    def reset(self):
        """Reset agent state (no-op for random agent)."""
        pass

    def act(self, observation: Dict[str, Any]) -> np.ndarray:
        """Select random action.

        Args:
            observation: Environment observation (ignored)

        Returns:
            Multi-binary action array [num_actions]
        """
        # Random binary actions with given probability
        # Use int8 to match MultiBinary action space (env.action_space.sample() returns int8)
        action = (self.rng.random(self.num_actions) < self.action_prob).astype(
            np.int8
        )
        return action

    def get_action_probs(self, observation: Dict[str, Any]) -> np.ndarray:
        """Get action probabilities (constant for random agent).

        Args:
            observation: Environment observation (ignored)

        Returns:
            Action probabilities [num_actions]
        """
        return np.full(self.num_actions, self.action_prob, dtype=np.float32)

    def __repr__(self) -> str:
        return f"RandomAgent(num_actions={self.num_actions}, action_prob={self.action_prob})"


