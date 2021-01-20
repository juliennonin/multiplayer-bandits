"""
    Different classes of indices, all of them have a compute_index(player) method
    which compute indices for each arm using the history of plays of a given player.
"""

import numpy as np

class IndexPolicy():
    """Abstract base class for index policies"""
    
    def __init__(self):
        pass

    def compute_index(self, player):
        raise NotImplementedError("Must be implemented.")


class UCB1Policy(IndexPolicy):
    """UCB1 index 

    Args:
        alpha (float): Parameter of UCB1
    """
    
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def compute_index(self, player):
        """Compute UCB1 index for each arm using the history of plays of the given player"""
        means = player.cum_rewards / player.nb_draws
        bonus = np.sqrt(self.alpha * np.log(player.t) / player.nb_draws)
        return means + bonus
    
    def name(self):
        return f"UCB1({self.alpha})"

    def __repr__(self):
        return f"UCB1Policy({self.alpha})"


class KlUCBPolicy(IndexPolicy):
    """kl-UCB index

    Args:
        arm_types (list of Arm): Reward distribution of each arm of the bandit
    """
    
    def __init__(self, arms_types):
        self.arms = arms_types
    
    def compute_index(self, player):
        """Compute kl-UCB index for each arm using the history of plays of the given player"""
        assert len(self.arms) == player.nb_arms, "Number of arms of the policy and the player doesn't match"
        means = player.cum_rewards / player.nb_draws
        levels = np.log(player.t) / player.nb_draws
        return np.array([arm.kl_ucb(µ, level) for (arm, µ, level) in zip(self.arms, means, levels)])
    
    @classmethod
    def name(cls):
        return "KlUCB"