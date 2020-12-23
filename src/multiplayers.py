import numpy as np
from src.utils import randmax


# -------- Experiments --------

def multiplayer_env(bandit, players, max_time):
    """Run a multiplayer multi-armed bandit strategy

    Args:
        bandit (MAB): multi-armed bandit
        players (list of Player): 
        max_time (int): time horizon
    """

    M = len(players)
    K = bandit.nb_arms
    selections = np.zeros((M, max_time), dtype=int)
    collisions = np.zeros((M, max_time), dtype=bool)
    sensing_infos = np.zeros((M, max_time))

    for t in range(max_time):
        chosen_arm_by_player = [player.choose_arm_to_play() for player in players]
        arms_counts = dict(zip(*np.unique(chosen_arm_by_player, return_counts=True)))  # arm -> number of player that have chosen it
        rewards = {arm: bandit.generate_reward(arm) for arm in arms_counts}  # arm -> reward

        for j, player in enumerate(players):
            arm = chosen_arm_by_player[j]
            reward, collision = rewards[arm], arms_counts[arm] != 1
            
            player.receive_reward(reward, collision)
            
            selections[j][t] = arm
            collisions[j][t] = collision         
            sensing_infos[j][t] = reward
    
    return selections, collisions, sensing_infos


# -------- Index Policies --------
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


class KlUCBPolicy(IndexPolicy):
    
    def __init__(self, arms_types):
        self.arms = arms_types
    
    def compute_index(self, player):
        assert len(self.arms) == player.nb_arms, "Number of arms of the policy and the player doesn't match"
        means = player.cum_rewards / player.nb_draws
        levels = np.log(player.t) / player.nb_draws
        return np.array([arm.kl_ucb(µ, level) for (arm, µ, level) in zip(self.arms, means, levels)])

    
# -------- Strategies --------

class Player:
    def __init__(self):
        pass


class PlayerRandTop(Player):
    """One Player using RandTopM with UCB policy

    Args:
        nb_arms (int):                  number of arms (K)
        nb_players (int):               number of players (M)
        alpha (int):                    UCB parameter

    Attributes:
        nb_arms (int):                  number of arms (K)
        nb_players (int):               number of players (M)
        policy (IndexPolicy):           policy (UCB1, klUCB)

        nb_draws (array of size K):     number of selections of each arm k
        cum_rewards (array of size K):  cumulative rewards of each arm k
                                        !! this is not the cumulative rewards of the player !!
        t (int): current time stamp

        best_arms (array of size M):    index of the M best arms
        my_arm (int):                   currently chosen arm
        ucbs (array of size K):         UCB1 of each arm k
        has_collided (bool):            was there a collision?
    """

    def __init__(self, nb_arms, nb_players, policy):
        self.nb_arms = nb_arms
        self.nb_players = nb_players
        self.policy = policy
        self.clear()

    def clear(self):
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_rewards = np.zeros(self.nb_arms)
        self.t = 0
        
        self.best_arms = np.zeros(self.nb_players)
        self.ucbs = np.zeros(self.nb_arms)
        self.my_arm = None
        self.has_collided = False

    def choose_arm_to_play(self):
        if np.any(self.nb_draws == 0):
            self.my_arm = randmax(-self.nb_draws)
            return self.my_arm

        ucbs_new = self.policy.compute_index(self)
        best_arms = np.argsort(ucbs_new)[:self.nb_players:-1]  # best arms
        
        if self.my_arm not in best_arms:
            ## if my arm doesn't belong to the M best arms anymore
            if self.has_collided:
                ## if there was a collision, randomly choose a new arm
                self.my_arm = np.random.choice(best_arms)
            else:
                ## my arm is no more a good choice
                # arms_previously_worse = set(np.where(self.ucbs <= self.ucbs[self.my_arm])[0])
                # new_arms_to_choose = set(best_arms) & arms_previously_worse
                min_ucb_of_best_arms = ucbs_new[best_arms[-1]]
                new_arms_to_choose = np.where((self.ucbs <= self.ucbs[self.my_arm]) & (ucbs_new >= min_ucb_of_best_arms))[0]
                self.my_arm = np.random.choice(new_arms_to_choose)

        self.ucbs = ucbs_new
        return self.my_arm
        
    def receive_reward(self, reward, collision):
        self.cum_rewards[self.my_arm] += reward
        self.nb_draws[self.my_arm] += 1
        self.has_collided = collision

        self.t += 1

    def name(self):
        return "Player"

