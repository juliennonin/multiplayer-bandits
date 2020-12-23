"""
    different classes of arms, all of them have a sample() method which produce rewards
"""

import numpy as np
from random import random
from math import sqrt, log, exp


class Arm:
    """Abstract class to be implemented by an arm"""
    eps = 1e-15

    def __init__(self):
        self.mean = None
        self.variance = None

    def sample(self):
        """Generate a reward"""
        raise NotImplementedError("Must be implemented.")

    @classmethod
    def kl(cls, x, y):
        """Kullback-Leibler divergence"""
        # print("Abstract kl")
        raise NotImplementedError("Must be implemented.")

    @classmethod
    def kl_ucb(cls, x, level, upper_bound, lower_bound=float('inf'), precision=1e-6):
        """Return u > x such that kl(x, u) = level (using binary search)"""
        lower, upper = max(x, lower_bound), upper_bound
        while upper - lower > precision:
            m = (lower + upper) / 2
            if cls.kl(x, m) > level:
                upper = m
            else:
                lower = m
        return (lower + upper) / 2


class Bernoulli(Arm):
    """Bernoulli arm with mean p"""
    
    def __init__(self, p):
        assert 0 <= p <= 1, "The parameter of Bernoulli's distribution must lie between 0 and 1."
        self.mean = p
        self.variance = p * (1-p)

    def sample(self):
        return float(random() < self.mean)

    def __repr__(self):
        return f"Bernoulli({self.mean})"

    @classmethod
    def kl(cls, x, y):
        """@author: Émilie Kaufmann"""
        # print("Bernoulli kl")
        x = min(max(x, cls.eps), 1-cls.eps)
        y = min(max(y, cls.eps), 1-cls.eps)
        return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))

    @classmethod
    def kl_ucb(cls, x, level, precision=1e-6):
        # [TODO] !! precision is used for the lower_bound ...
        upper_bound = min(1, x + np.sqrt(level / 2))
        return super().kl_ucb(x, level, upper_bound, precision)


class Gaussian(Arm):
    """Gaussian arm with specified mean and variance"""
    
    def __init__(self, mu, var=1):
        self.mean = mu
        self.variance = var

    def sample(self):
        return self.mean + sqrt(self.variance) * np.random.normal()

    @classmethod
    def kl(cls, x, y, sig2=1):
        return (x - y) ** 2 / (2 * sig2)

    @classmethod
    def kl_ucb(cls, x, level, precision=1e-6, sig2=1.):
        return x + sqrt(2 * sig2 * level)
        

class Exponential(Arm):
    """Exponential arm with parameter p"""

    def __init__(self, p):
        self.mean = 1 / p
        self.variance = 1 / (p * p)

    def sample(self):
        return - self.mean * log(random())

    @classmethod
    def kl(cls, x, y):
        x, y = max(x, cls.eps), max(y, cls.eps)
        return (x / y - 1 - log(x / y))

    @classmethod
    def kl_ucb(cls, x, level, precision=1e-6):
        upper_bound = x / (1 + 2/3 * level - sqrt(4/9 * level**2 + 2 * level)) if level < 0.77 else x * exp(level + 1)
        lower_bound = x * exp(level) if level > 1.61 else x / (1 + level - sqrt(level**2 + 2 * level))
        return super().kl_ucb(x, level, upper_bound, lower_bound, precision)


class TruncatedExponential(Arm):
    """Truncated Exponential arm with parameter p"""

    def __init__(self, p, trunc):
        self.p = p
        self.trunc = trunc
        self.mean = (1. - exp(-p * trunc)) / p
        self.variance = 0
        
    def sample(self):
        return min(-(1 / self.p) * log(random()), self.trunc)
