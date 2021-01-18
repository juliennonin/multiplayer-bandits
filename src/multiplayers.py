import numpy as np
from src.utils import randmax


# -------- Experiments --------

class MultiplayerEnv():
    """Run a multiplayer multi-armed bandit strategy

    Args:
        bandit (MAB): multi-armed bandit
        players (list of Player): 
        time_horizon (int): time horizon
    """

    def __init__(self, bandit, players, time_horizon):
        self.bandit = bandit
        self.players = players
        self.time_horizon = time_horizon
        self.K, self.M = self.bandit.nb_arms, len(players)
        self.clear()

    def clear(self):
        self.selections = np.zeros((self.M, self.time_horizon), dtype=int)
        self.collisions = np.zeros((self.M, self.time_horizon), dtype=bool)
        # self.chairs = np.zeros((self.M, self.time_horizon), dtype=bool)
        self.sensing_infos = np.zeros((self.M, self.time_horizon))
        for player in self.players:
            player.clear()

    @property
    def rewards(self):
        return self.sensing_infos * (~self.collisions)

    def cumulative_reward(self):
        return np.cumsum(self.rewards.sum(0))

    def cumulative_nb_of_colliding_players(self, arm):
        return np.cumsum((self.collisions & (self.selections == arm)).sum(0))

    def cumulative_nb_of_selections(self, arm):
        return np.cumsum((self.selections == arm).sum(0))

    def run(self):
        for t in range(self.time_horizon):
            chosen_arm_by_player = [player.choose_arm_to_play() for player in self.players]
            arms_counts = dict(zip(*np.unique(chosen_arm_by_player, return_counts=True)))  # arm -> number of player that have chosen it
            rewards = {arm: self.bandit.generate_reward(arm) for arm in arms_counts}  # arm -> reward

            for j, player in enumerate(self.players):
                arm = chosen_arm_by_player[j]
                reward, collision = rewards[arm], arms_counts[arm] != 1
                
                player.receive_reward(reward, collision)
                self.selections[j][t] = arm
                self.collisions[j][t] = collision 
                # self.chairs[j][t] = player.is_on_chair      
                self.sensing_infos[j][t] = reward




#-----------Plot experiment------------
def run_experiments(n_random_arm,nb_arms,strategies,policy,nb_players,max_time):
    
    """
    Parameters: 
    n_random_arm (int) : number of averaging times (random geeration on arms)
    nb_arms (int) : number of bandit  arms
    strategies (list of palyer class ):  players strategy type ex: PlayerMCTop, PlayerRandTop, PlayersSelfish 
    policy (policy class) :  the policy to be used  ex: KlUCBPolicy, UCB1Policy
    nb_players (int) : numbers of players
    max_time (int):            experimnent time horizon

    Output:
    Plot the cumulative centralised regret of each strategy average on bandits n_random_arm instances times
    """
    bandits=[BernoulliMAB(np.random.uniform(0,1,nb_arms),m=nb_players) for i in range(n_random_arm)]
    names=[strategies[i].name() + "_" + policy.name() for i in range(len(strategies))]
    strategy_cum_r=[]
    for strategy in strategies:
        r=[]
        for i in range(n_random_arm):
            bandit=bandits[i]
            if policy.name()=="KlUCB":
                players=[strategy(nb_arms=3, nb_players=2,policy=policy(bandit.arms))]
                s,_,_,_=multiplayer_env(bandit, players, max_time)
                r.append(cumulative_centralised_regret(bandit,s))
            else:
                players=[strategy(nb_arms=3, nb_players=2,policy=policy(alpha=0.1))]
                s,_,_,_=multiplayer_env(bandit, players, max_time)
                r.append(cumulative_centralised_regret(bandit,s))
        strategy_cum_r.append(np.mean(np.array(r),axis=0))

    for i in range(len(strategies)):
        plt.plot(strategy_cum_r[i],label=names[i])
    plt.legend()
    plt.show()

#----- Cumulative Centralised Pseudo regret---

def cumulative_centralised_regret(bandit,selections):
    """Compute the cumulative centralised pseudo-regret associated to players sequence of arm selections"""
    return np.sum(np.cumsum(bandit.m_best_arms_means.reshape(-1,1)*np.ones(selections.shape)-np.array(bandit.means)[selections],axis=1),axis=0)


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

    
# -------- Strategies --------

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


class PlayerRandTop(Player):
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

