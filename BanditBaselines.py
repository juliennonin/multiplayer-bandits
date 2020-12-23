import numpy as np
from math import log,sqrt
from scipy.stats import beta as pi
from BanditTools import *
from math import log,sqrt

class UCB:
    """UCB1 with parameter alpha"""
    def __init__(self,nbArms,alpha):
        self.nbArms=nbArms
        self.alpha=alpha
        self.nam_e= "UCB1( alpha =" +str(self.alpha) + " )"
        self.clear()
    def set_name(self,name):
        self.nam_e=name
    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.t = 0
        self.Explore = True 

    def choose_arm_to_play(self):
        self.t=self.t+1
        if min(self.nbDraws)==0:
             return randmax(-self.nbDraws)
        else:
            ucb=self.cumRewards/self.nbDraws + np.sqrt(self.alpha*log(self.t)/self.nbDraws)
            return randmax(ucb)
       
    def receive_reward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1


    def name(self):
        return str(self.nam_e)
class klUCB:
    """klUCB (Bernoulli divergence by default)"""
    def __init__(self,nbArms):
        self.nbArms=nbArms
        self.name="klUCB"
        self.clear()
    
    def clear(self):
        self.nbDraws = np.zeros(self.nbArms,dtype=int)
        self.cumRewards = np.zeros(self.nbArms)
        self.t = 0
        
    
    def choose_arm_to_play(self):
        self.t=self.t+1
        if min(self.nbDraws)==0 or min(self.cumRewards)==0 :
            return randmax(-self.nbDraws)
        else:
            mu_hat=self.cumRewards/self.nbDraws
            ucb=[klucbBern(mu_hat[arm],np.log(self.t)/self.nbDraws[arm], precision=1e-6)  for arm in range(self.nbArms)]
            return randmax(ucb)
    def receive_reward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1

    def name(self):
        return self.name


class ThompsonSampling:
    """Thompson Sampling with Beta(a,b) prior and Bernoulli likelihood"""
    def __init__(self,nbArms,alpha,beta):
        self.nbArms=nbArms
        self.alpha=alpha
        self.beta=beta
        self.clear()
    
    def clear(self):
        self.nbDraws = np.zeros(self.nbArms,dtype=int)
        self.cumRewards = np.zeros(self.nbArms)
        self.t = 0
        # self.Explore = True 
    
    def choose_arm_to_play(self):
        self.t=self.t+1
        ucb=[pi.rvs(self.alpha+self.cumRewards[arm],self.beta+self.nbDraws[arm]- self.cumRewards[arm])              for arm in range(self.nbArms)]
        return randmax(ucb)

    def receive_reward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1

    def name(self):
        return "Thompson Sampling"

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
        self.Explore = True 

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

