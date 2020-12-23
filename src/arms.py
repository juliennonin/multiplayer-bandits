"""
    different classes of arms, all of them have a sample() method which produce rewards
"""

import numpy as np
from random import random
from math import sqrt, log, exp


class Arm:
    """Abstract class to be implemented by an arm"""

    def __init__(self):
        self.mean = None
        self.variance = None

    def sample(self):
        """Generate a reward"""
        raise NotImplementedError("Must be implemented.")


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


class Gaussian(Arm):
    """Gaussian arm with specified mean and variance"""
    
    def __init__(self, mu, var=1):
        self.mean = mu
        self.variance = var

    def sample(self):
        return self.mean + sqrt(self.variance) * np.random.normal()
        

class Exponential(Arm):
    """Exponential arm with parameter p"""

    def __init__(self, p):
        self.mean = 1 / p
        self.variance = 1 / (p * p)

    def sample(self):
        return - self.mean * log(random())


class TruncatedExponential(Arm):
    """Truncated Exponential arm with parameter p"""

    def __init__(self, p, trunc):
        self.p = p
        self.trunc = trunc
        self.mean = (1. - exp(-p * trunc)) / p
        self.variance = 0
        
    def sample(self):
        return min(-(1 / self.p) * log(random()), self.trunc)
