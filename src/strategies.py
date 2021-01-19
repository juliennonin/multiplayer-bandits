import numpy as np
from src.utils import randmax

class Player:
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
        self.is_on_chair=False
            
    def choose_arm_to_play(self):
        raise NotImplementedError("Must be implemented.")
        
    def receive_reward(self, reward, collision):
        self.cum_rewards[self.my_arm] += reward
        self.nb_draws[self.my_arm] += 1
        self.has_collided = collision

        self.t += 1

    def name(self):
        return "Player"


def strategy(strategy, nb_arms, nb_players, *args, **kwargs):
    assert issubclass(strategy, Player), "strategy should be a sub-class of Player"
    return [strategy(nb_arms, nb_players, *args, **kwargs) for _ in range(nb_players)]


class PlayerRandTopOld(Player):
    def choose_arm_to_play(self):
        if np.any(self.nb_draws == 0):
            self.my_arm = randmax(-self.nb_draws)
            return self.my_arm

        ucbs_new = self.policy.compute_index(self)
        best_arms = np.argsort(ucbs_new)[::-1][:self.nb_players]  # best arms
        
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

    @classmethod
    def name(cls):
        return "RandTopMOld"


class PlayerRandTop(Player):
    def choose_arm_to_play(self):
        if np.any(self.nb_draws == 0):
            self.my_arm = randmax(-self.nb_draws)
            return self.my_arm

        ucbs_new = self.policy.compute_index(self)
        best_arms = np.argsort(ucbs_new)[::-1][:self.nb_players]  # best arms
        
        if self.my_arm not in best_arms:
            ## my arm is no more a good choice
            # arms_previously_worse = set(np.where(self.ucbs <= self.ucbs[self.my_arm])[0])
            # new_arms_to_choose = set(best_arms) & arms_previously_worse
            min_ucb_of_best_arms = ucbs_new[best_arms[-1]]
            new_arms_to_choose = np.where((self.ucbs <= self.ucbs[self.my_arm]) & (ucbs_new >= min_ucb_of_best_arms))[0]
            self.my_arm = np.random.choice(new_arms_to_choose)
        elif self.has_collided:
            ## if there was a collision, randomly choose a new arm
            self.my_arm = np.random.choice(best_arms)

        self.ucbs = ucbs_new
        return self.my_arm
    
    @classmethod
    def name(cls):
        return "RandTopM"


class PlayerMCTop(Player):

    def choose_arm_to_play(self):
        if np.any(self.nb_draws == 0):
            self.my_arm = randmax(-self.nb_draws)
            return self.my_arm

        ucbs_new = self.policy.compute_index(self)
        best_arms = np.argsort(ucbs_new)[::-1][:self.nb_players]  # best arms
        
        if self.my_arm not in best_arms:
            ## if my arm doesn't belong to the M best arms anymore
        
            # arms_previously_worse = set(np.where(self.ucbs <= self.ucbs[self.my_arm])[0])
            # new_arms_to_choose = set(best_arms) & arms_previously_worse
            min_ucb_of_best_arms = ucbs_new[best_arms[-1]]
            new_arms_to_choose = np.where((self.ucbs <= self.ucbs[self.my_arm]) & (ucbs_new >= min_ucb_of_best_arms))[0]
            self.my_arm = np.random.choice(new_arms_to_choose)
            self.is_on_chair=False
        else:
            ## if my arm  belongs to the M best arms 
            if self.has_collided and not self.is_on_chair:
                ## if there was a collision and my arm is not marked as a chair, 
                # randomly choose a new arm and the chosen arm is not a chair
                self.my_arm = np.random.choice(best_arms)
                self.is_on_chair=False
            else:
                ## if there wasn't a collision, 
                #my arm remains marked as a chair and choose the same arm
                self.is_on_chair=True

        self.ucbs = ucbs_new

        return self.my_arm
    @classmethod
    def name(cls):
        return "MCTopM"


class PlayerSelfish(Player):

    def choose_arm_to_play(self):
        if np.any(self.nb_draws == 0):
            self.my_arm = randmax(-self.nb_draws)
            return self.my_arm
         
        self.my_arm=randmax(self.ucbs)     # my arm is the best arm among all arms
        self.ucbs = self.policy.compute_index(self)
    
        return self.my_arm

    def receive_reward(self, reward, collision):
        reward_no_sensing = 0 if collision else reward
        return super().receive_reward(reward_no_sensing, collision)

    
    @classmethod
    def name(cls):
        return "Selfish"
