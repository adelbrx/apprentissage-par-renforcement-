import sys
import random

import matplotlib.pyplot as plt
import seaborn as sns

from bandit import (Bandit, Ban10, MovingBan10, GreedyPlayer,
                    OptimistGreedyPlayer, UCBGreedyPlayer)

STEPS = 1000
RUNS = 2000


def ex1():
    b = Bandit()
    vals = [b.play() for _ in range(1000)]
    plt.figure()
    sns.kdeplot(vals, fill=True)
    plt.hist(vals, bins=40, density=True, alpha=0.4)
    plt.title(f"1 bandit, avg = {b.avg:.2f}")
    plt.savefig("ex1_un_bandit.png")
    plt.close()

    plt.figure()
    for i in range(3):
        b = Bandit()
        vals = [b.play() for _ in range(1000)]
        sns.kdeplot(vals, label=f"bandit {i} (avg={b.avg:.2f})")
    plt.legend()
    plt.title("3 bandits")
    plt.savefig("ex1_trois_bandits.png")
    plt.close()


def ex3(verbose=True):
    ban = Ban10()
    p = GreedyPlayer(10, 0.1)
    rewards = []
    for i in range(STEPS):
        a = p.get_action()
        r = ban.play(a)
        p.reward(a, r)
        rewards.append(r)
        if verbose:
            print(i, round(r, 3), "optimal" if a == ban.best else "pas optimal")
    return rewards


def simulate(make_player, make_ban=Ban10, bans=None, steps=STEPS, runs=RUNS):
    if bans is None:
        bans = [make_ban() for _ in range(runs)]
    players = [make_player() for _ in range(len(bans))]
    mean_rewards = []
    optimal = []
    for t in range(steps):
        total = 0
        good = 0
        for p, ban in zip(players, bans):
            a = p.get_action()
            r = ban.play(a)
            p.reward(a, r)
            total += r
            if a == ban.best:
                good += 1
        mean_rewards.append(total / len(bans))
        optimal.append(100 * good / len(bans))
    return mean_rewards, optimal


def plot_results(results, name, title=""):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for label, (rew, opt) in results.items():
        ax1.plot(rew, label=label)
        ax2.plot(opt, label=label)
    ax1.set_ylabel("récompense moyenne")
    ax2.set_ylabel("% action optimale")
    ax2.set_xlabel("étapes")
    ax1.legend()
    ax2.legend()
    if title:
        ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def ex4():
    rewards = ex3(verbose=False)
    plt.figure()
    plt.plot(rewards)
    plt.title("récompense d'un seul joueur (eps=0.1)")
    plt.savefig("ex4_un_joueur.png")
    plt.close()

    res = {}
    for eps in [0, 0.01, 0.1]:
        print("eps =", eps)
        res[f"eps={eps}"] = simulate(lambda: GreedyPlayer(10, eps))
    plot_results(res, "ex4_epsilons.png", "epsilon-greedy, moyenne sur 2000 parties")


def ex5():
    bans = [Ban10() for _ in range(RUNS)]
    res = {}
    res["greedy eps=0.1"] = simulate(lambda: GreedyPlayer(10, 0.1), bans=bans)
    res["optimiste Q0=5, eps=0"] = simulate(lambda: OptimistGreedyPlayer(10), bans=bans)
    plot_results(res, "ex5_optimiste.png", "initialisation optimiste")


def ex6():
    bans = [Ban10() for _ in range(RUNS)]
    res = {}
    res["greedy eps=0.1"] = simulate(lambda: GreedyPlayer(10, 0.1), bans=bans)
    res["UCB c=2"] = simulate(lambda: UCBGreedyPlayer(10, 2), bans=bans)
    plot_results(res, "ex6_ucb.png", "UCB vs epsilon-greedy")


def ex7():
    steps = 3000
    res = {}
    for eps in [0, 0.01, 0.1]:
        print("moving eps =", eps)
        res[f"eps={eps}"] = simulate(lambda: GreedyPlayer(10, eps), make_ban=MovingBan10, steps=steps, runs=500)
    plot_results(res, "ex7_moving_eps.png", "non-stationnaire, pas 1/n")

    res = {}
    res["eps=0.1 pas 1/n"] = simulate(lambda: GreedyPlayer(10, 0.1), make_ban=MovingBan10, steps=steps, runs=500)
    res["eps=0.1 pas 0.1"] = simulate(lambda: GreedyPlayer(10, 0.1, step=0.1), make_ban=MovingBan10, steps=steps, runs=500)
    res["optimiste pas 0.1"] = simulate(lambda: OptimistGreedyPlayer(10), make_ban=MovingBan10, steps=steps, runs=500)
    res["UCB c=2"] = simulate(lambda: UCBGreedyPlayer(10, 2), make_ban=MovingBan10, steps=steps, runs=500)
    res["UCB c=2 pas 0.1"] = simulate(lambda: UCBGreedyPlayer(10, 2, step=0.1), make_ban=MovingBan10, steps=steps, runs=500)
    plot_results(res, "ex7_moving_compare.png", "non-stationnaire")


if __name__ == "__main__":
    exos = {"1": ex1, "3": ex3, "4": ex4, "5": ex5, "6": ex6, "7": ex7}
    todo = sys.argv[1:] or ["1", "3", "4", "5", "6", "7"]
    for e in todo:
        print("=== exercice", e)
        exos[e]()
