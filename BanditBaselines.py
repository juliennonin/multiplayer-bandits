import numpy as np
from math import log,sqrt
from scipy.stats import beta as pi
from BanditTools import *
from math import log,sqrt


class Player:
    def __init__(self, nb_arms, nb_players,alpha=0.1):
        self.alpha=alpha
        self.nb_arms= nb_arms
        self.nb_players=nb_players
        self.clear()

    def clear(self):
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_rewards = np.zeros(self.nb_arms)
        self.best_arms=np.zeros(self.nb_players)
        self.ucbs=np.zeros(self.nb_arms)
        self.my_arm=None
        self.t = 0
        self.has_collided=False

    def choose_arm_to_play(self):
        self.t=self.t+1
        
        if min(self.nb_draws)==0:
             return randmax(-self.nb_draws)
        else:
            ucbs_new = self.cum_rewards/self.nb_draws + np.sqrt(self.alpha*log(self.t)/self.nb_draws) # use formula
        
        best_arms = np.argsort(ucbs_new)[:self.nb_players:-1]  # M best arms
        if self.my_arm in ucbs_new:  # if my arm is still a best arm
            return self.my_arm
        elif self.has_collided:
            return np.random.choice(best_arms)
        else:  # no collision
            # arms_previously_worse = set(np.where(self.ucbs <= self.ucbs[self.my_arm])[0])
            # new_arms_to_choose = set(best_arms) & arms_previously_worse
            new_arms_to_choose = np.where((self.ucbs <= self.ucbs[self.my_arm]) & (ucbs_new >= ucbs_new[best_arms[-1]]))[0]
            self.ucbs = ucbs_new
            return np.random.choice(new_arms_to_choose)
        

    def receive_reward(self,arm,reward, collision=False):
        self.cum_rewards[arm] = self.cum_rewards[arm]+reward
        self.nb_draws[arm] = self.nb_draws[arm] +1
        self.has_collided=collision
        self.my_arm=arm


    def name(self):
        return "Player"

class RandTopM:
    def __init__(self, nb_arms, nb_players,alpha=0.1):
        self.alpha=alpha
        self.nb_arms= nb_arms
        self.nb_players=nb_players
        self.clear()

    def clear(self):
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_rewards = np.zeros(self.nb_arms)
        self.best_arms=np.zeros(self.nb_players)
        self.ucbs=np.zeros(self.nb_arms)
        self.my_arm=None
        self.t = 0
        self.has_collided=False

    def choose_arm_to_play(self):
        self.t=self.t+1
        if min(self.nb_draws)==0:
             return randmax(-self.nb_draws)
        else:
            ucbs_new = self.cum_rewards/self.nb_draws + np.sqrt(self.alpha*log(self.t)/self.nb_draws) # use formula
        best_arms = np.argsort(ucbs_new)[:self.nb_players:-1]  # M best arms

        if self.my_arm in ucbs_new:  # if my arm is still a best arm
            return self.my_arm

        elif self.has_collided:
            return np.random.choice(best_arms)

        else:  
            print(self.my_arm)
            new_arms_to_choose = np.where((self.ucbs <= self.ucbs[self.my_arm]) & (ucbs_new >= ucbs_new[best_arms[-1]]))[0]
            self.ucbs = ucbs_new
            return np.random.choice(new_arms_to_choose)

    def receive_reward(self,arm,reward, collision):
        self.cum_rewards[arm] = self.cum_rewards[arm]+reward
        self.nb_draws[arm] = self.nb_draws[arm] +1
        self.has_collided=collision
        self.my_arm=arm

    def name(self):
        return "RandTopM"


class Players2:
    def __init__(self, nb_arms, nb_players,strategy):
        self.nb_arms= nb_arms
        self.nb_players=nb_players
        self.strategy=strategy(self.nb_arms,self.nb_players)
        self.clear()
    def clear(self):
        self.strategy.clear()
    def choose_arm_to_play(self):
        return self.strategy.choose_arm_to_play()
    def receive_reward(self,arm,reward,collision):
        self.strategy.receive_reward(arm,reward, collision)

