import numpy as np
from src import arms
import math


class MAB:
    """Multi-armed bandits environnement
    
    Args:
        arms (list of arms.Arm)
    """

    def __init__(self, arms,m):

        self.arms = arms
        self.nb_arms = len(arms)
        self.m=m
        self.means = [arm.mean for arm in self.arms]
        self.m_best_arms_means=np.sort(self.means)[::-1][:self.m]
        
    def generate_reward(self, arm):
        return self.arms[arm].sample()
    
    def __repr__(self):
        return f"MAB({self.arms})"

class BernoulliMAB(MAB):
    """Bernoulli MAB


    Args:
        means (list of float): vector of Bernoulli's means
    """

    def __init__(self,means,m):
        super().__init__([arms.Bernoulli(p) for p in means],m)
    
    def __repr__(self):
        return f"BernoulliMAB({self.means})"

def RandomBernoulliBandit(Delta,K):
    """generates a K-armed Bernoulli instance at random where Delta is the gap between the best and second best arm"""
    maxMean = Delta + np.random.rand()*(1.-Delta)
    secondmaxMean= maxMean-Delta
    means = secondmaxMean*np.random.random(K)
    bestarm = np.random.randint(0,K)
    secondbestarm = np.random.randint(0,K)
    while (secondbestarm==bestarm):
        secondbestarm = np.random.randint(0,K)
    means[bestarm]=maxMean
    means[secondbestarm]=secondmaxMean
    return BernoulliBandit(means)
