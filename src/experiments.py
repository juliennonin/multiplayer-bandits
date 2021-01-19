import numpy as np
from src.utils import print_loading
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from math import sqrt


# -------- Experiments --------

class MultiplayerExp():
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

    def animate(self):
        M, K, T = self.M, self.K, self.time_horizon
        MAX_MARKER_SIZE = 2500
        fig = plt.figure(figsize=(7, M))

        nb_draws_scatters = [None] * M
        annotations = [None] * (K * M)
        for j in range(M):
            nb_draws_scatters[j] = plt.scatter(np.arange(K), np.zeros(K) + j, s=50*np.ones(K), color="plum", zorder=10, alpha=.8)
            for k in range(K):
                annotations[j*K + k] = plt.annotate(0, xy=(k, j), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", zorder=11)
            
        for k in range(K):
            plt.axvline(k, c="whitesmoke", lw=1, zorder=0)
        # plt.axis("off")

        plt.ylim(-0.5, M-.25)
        plt.xticks(range(K), self.bandit.means)
        plt.yticks(range(M))
        plt.xlabel("Arms")
        plt.ylabel("Players")
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.gca().xaxis.set_label_position('top')
        plt.tick_params(color="lightgrey", labelbottom=False, labeltop=True, top=True, bottom=False)
        
        def update_plot(t):
            for j in range(M):
                nb_draws, _ = np.histogram(self.selections[j, 0:t], range=(0, K), bins=K)
                sizes = MAX_MARKER_SIZE * nb_draws / T
                nb_draws_scatters[j].set_sizes(sizes)
                nb_draws_scatters[j].set_color(["plum" if self.selections[j, t-1] != k else "firebrick" if self.collisions[j, t-1] else "darkorchid" for k in range(K)])
                
                for k in range(K):
                    annotations[j*K + k].set_text(nb_draws[k])
                    annotations[j*K + k].set_position((0, sqrt(sizes[k])/2 + 2))
                    
            return (*nb_draws_scatters, *annotations)

        return animation.FuncAnimation(fig, update_plot, frames=T+1, blit=True)


def multiple_runs(env, N_exp, return_end_regrets=False):
    time_horizon = env.time_horizon
    M, K = env.M, env.K
    T = np.arange(1, time_horizon + 1)
    oracle_cum_reward = env.bandit.m_best_arms_means(M).sum() * T

    ## Initialization
    # avg_cum_reward = np.zeros(time_horizon)
    # Average number of colliding players on arm k up to time t, E[C_k(t)] 
    avg_nb_colliding_players = np.zeros((K, time_horizon))
    # Average number of selections of arm k up to time t, E[N_k(t)]
    avg_nb_selections = np.zeros((K, time_horizon))
    # Final empirical regret for each simulation, i.e. oracle_reward - cum_reward
    end_regrets = np.zeros(N_exp)

    for i in range(N_exp):
        env.clear()
        env.run()
        # avg_cum_reward += env.cumulative_reward()
        for k in range(K):
            avg_nb_colliding_players[k] += env.cumulative_nb_of_colliding_players(k)
            avg_nb_selections[k] += env.cumulative_nb_of_selections(k)
        print_loading(i+1, N_exp)
        end_regrets[i] = oracle_cum_reward[-1] - env.cumulative_reward()[-1]

    # avg_cum_reward /= N_exp
    avg_nb_colliding_players /= N_exp
    avg_nb_selections /= N_exp

    ## Compute the expected regret using the regret definition
    # cum_regret1 = oracle_cum_reward - avg_cum_reward

    ## Compute the expected regret using the regret decomposition formula
    gaps = (env.bandit.means - env.bandit.last_best_arm_mean(M))[:, np.newaxis]  # gaps[k] is µ_k - µ_M^*
    decomp_c = (env.bandit.means[:, np.newaxis] * avg_nb_colliding_players).sum(0)  # c-term in the decomposition
    decomp_ab = np.where(gaps < 0,  - gaps * avg_nb_selections, gaps * (T - avg_nb_selections)).sum(0)  # a & b terms
    cum_regret = decomp_ab + decomp_c

    if return_end_regrets:
        return cum_regret, end_regrets
    else:
        return cum_regret
