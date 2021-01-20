"""Useful tools to test a strategy by running experiments and estimating expected regret."""


import numpy as np
from src.utils import print_loading
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from math import sqrt


class MultiplayerExp():
    """Run a multiplayer multi-armed bandit strategy

    Args:
        bandit (MAB): multi-armed bandit
        players (list of Player): list of individual strategies of each player
        time_horizon (int): time horizon

    Attributes & Properties:
        bandit (MAB): multi-armed bandit environment
        players (list of Player): list of players
        time_horizon (int): time horizon
        K (int): number of arms in the MAB
        M (int): number of players

        selections (array of int btw 0 and K of shape (M, time_horizon)):
            selections[j,t] is the index of the arm chosen by player j at time t.
        collisions (array of bool of shape (M, time_horizon)):
            collisions[j,t] is True if player j collided at time t, False otherwise.
        sensing_infos (array of float of shape (M, time_horizon)):
            sensing_infos[j,t] is the reward produced by arm selections[j,t] for player j at time t.
        rewards (array of float of shape (M, time_horizon)):
            rewards[j,t] is the reward received by player j at time t.
    """

    def __init__(self, bandit, players, time_horizon):
        self.bandit = bandit
        self.players = players
        self.time_horizon = time_horizon
        self.K, self.M = self.bandit.nb_arms, len(players)
        self.clear()

    def clear(self):
        """(Re)initialize the experiment"""
        self.selections = np.zeros((self.M, self.time_horizon), dtype=int)
        self.collisions = np.zeros((self.M, self.time_horizon), dtype=bool)
        # self.chairs = np.zeros((self.M, self.time_horizon), dtype=bool)
        self.sensing_infos = np.zeros((self.M, self.time_horizon))
        for player in self.players:
            player.clear()

    @property
    def rewards(self):
        """History of rewards for each player

        Returns:
            (array of shape (M, time_horizon)): coefficient [j,t] is the reward received by player j at time t.
        """
        return self.sensing_infos * (~self.collisions)

    def cumulative_reward(self):
        """History of cumulated (centralized) reward."""
        return np.cumsum(self.rewards.sum(0))

    def cumulative_nb_of_colliding_players(self, arm):
        """History of the number of colliding players on the given arm."""
        return np.cumsum((self.collisions & (self.selections == arm)).sum(0))

    def cumulative_nb_of_selections(self, arm):
        """History of the number of selections of the given arm."""
        return np.cumsum((self.selections == arm).sum(0))

    def run(self):
        """Run the experiment (Let the players play on the arms which generate rewards.)"""
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
        """Produce an animation of the history of selections"""
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
        plt.gca().invert_yaxis()
        plt.tight_layout()

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


def multiple_runs(env, N_exp, return_end_regrets=False, return_regret_decomposition=False):
    """Run a multiplayer environment N_exp times to estimate the cumulated reward.

    Args:
        env (MultiplayerExp): a multiplayer experiment to be run
        N_exp (int): number of runs
        return_end_regrets (bool, optional): If set to True, return the list of end cumulated reward of each run.
        return_regret_decomposition (bool, optional): If set to True return the three terms of the regret decomposition

    Returns:
        (array of size env.time_horizon): Evolution of expected cumulated reward
        (array of size N_exp, optional): final cumulated reward of each run (if return_end_regrets is set to True)
        (tuple of 3 arrays of size env.time_horizon, optional): three terms of the regret decomposition
    """
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
    decomp_a = np.where(gaps < 0, - gaps * avg_nb_selections, 0).sum(0)  # a-terù in the decomposition
    decomp_b = np.where(gaps >= 0, gaps * (T - avg_nb_selections), 0).sum(0)  # b-term
    decomp_c = (env.bandit.means[:, np.newaxis] * avg_nb_colliding_players).sum(0)  # c-term in the decomposition
    cum_regret = decomp_a + decomp_b + decomp_c

    to_return = [cum_regret]
    if return_end_regrets:
        to_return.append(end_regrets)
    if return_regret_decomposition:
        to_return.append((decomp_a, decomp_b, decomp_c))
    return to_return[0] if len(to_return) == 1 else to_return 