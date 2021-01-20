import matplotlib.pyplot as plt


def plot_regret_decomposition(exp, cum_regret, cum_regret_terms, N_runs):
    plt.figure(figsize=(13, 7))
    K, M = exp.K, exp.M
    
    plt.plot(cum_regret, c='crimson', lw=2, label="cumulated centralized regret")
    plt.plot(cum_regret_terms[0], c="cornflowerblue", lw=2, label=fr"$(a)$ term: pulls of {K-M} suboptimal arms")
    plt.plot(cum_regret_terms[1], c="yellowgreen", lw=2, label=fr'$(b)$ term: non-pulls of {M} optimal arms')
    plt.plot(cum_regret_terms[2], c="darkorange", lw=2, label=r'$(c)$ term: weigted count of collisions')
    
    plt.title(fr"Multi-players $M={{{M}}}$: Cumulated centralized regret,"
        fr"averaged {N_runs} times for {exp.players[0].name()}-{exp.players[0].policy.name()}."
        "\n" fr"{K} arms: " + exp.bandit.to_latex())
    plt.xlabel("Time")
    plt.ylabel("Cumulative centralized regret")
    plt.grid(color="lightgrey")
    plt.legend()