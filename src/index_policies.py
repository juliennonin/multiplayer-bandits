import numpy as np

class IndexPolicy():
    
    def __init__(self):
        pass

    def compute_index(self, player):
        raise NotImplementedError("Must be implemented.")


class UCB1Policy(IndexPolicy):
    
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def compute_index(self, player):
        means = player.cum_rewards / player.nb_draws
        bonus = np.sqrt(self.alpha * np.log(player.t) / player.nb_draws)
        return means + bonus
    
    def name(self):
        return f"UCB1({self.alpha})"

    def __repr__(self):
        return f"UCB1Policy({self.alpha})"


class KlUCBPolicy(IndexPolicy):
    
    def __init__(self, arms_types):
        self.arms = arms_types
    
    def compute_index(self, player):
        assert len(self.arms) == player.nb_arms, "Number of arms of the policy and the player doesn't match"
        means = player.cum_rewards / player.nb_draws
        levels = np.log(player.t) / player.nb_draws
        return np.array([arm.kl_ucb(µ, level) for (arm, µ, level) in zip(self.arms, means, levels)])
    
    @classmethod
    def name(cls):
        return "KlUCB"