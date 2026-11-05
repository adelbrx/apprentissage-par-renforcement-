import sys
import time

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from snake import ShortSnake, UP, DOWN, LEFT, RIGHT, NAMES
from agent import GreedyPlayer

SIZE = 5
BIG = 10
STEPS = 1000
RUNS = 300
GAMMAS = [0, 0.1, 0.5, 0.9, 1]


def run(steps, gamma=0.5, length=1, eps=0.1, alpha=0.1, size=None):
    game = ShortSnake(size or SIZE, length=length)
    player = GreedyPlayer(game, eps=eps, gamma=gamma, alpha=alpha)
    rewards = []
    for _ in range(steps):
        a = player.get_action()
        r = game.move(a)
        player.reward(a, r)
        rewards.append(r)
    return rewards


def mean_curve(steps, runs, **kw):
    total = [0] * steps
    for _ in range(runs):
        for i, r in enumerate(run(steps, **kw)):
            total[i] += r
    return [t / runs for t in total]


def smooth(values, w):
    out = []
    s = 0
    for i, v in enumerate(values):
        s += v
        if i >= w:
            s -= values[i - w]
        out.append(s / min(i + 1, w))
    return out


def ex1():
    g = ShortSnake(BIG)
    print(g, "\n")
    for d in [None, UP, DOWN, LEFT, None, None, None, None, None, None]:
        r = g.move(d)
        print("action :", NAMES.get(d, "rien"), "| recompense :", r)
        print(g, "\n")


def draw_state(ax, x0, y0, head, d, fruit, size=3, cell=0.5):
    for i in range(size):
        for j in range(size):
            color = "white"
            if (i, j) == fruit:
                color = "tomato"
            if (i, j) == head:
                color = "seagreen"
            ax.add_patch(Rectangle((x0 + i * cell, y0 - (j + 1) * cell), cell, cell, facecolor=color, edgecolor="black"))
    hx = x0 + head[0] * cell + cell / 2
    hy = y0 - head[1] * cell - cell / 2
    ax.annotate("", xy=(hx + d[0] * cell * 0.4, hy - d[1] * cell * 0.4), xytext=(hx, hy),
                arrowprops=dict(arrowstyle="->", color="white", lw=2))


def ex2():
    states = {
        "A": ((0, 1), RIGHT, (2, 2), (0.5, 6)),
        "B": ((1, 1), RIGHT, (2, 2), (4.5, 6)),
        "C": ((2, 1), RIGHT, (2, 2), (8.5, 6)),
        "D": ((1, 2), DOWN, (2, 2), (4.5, 2.5)),
        "E": ((2, 2), RIGHT, (0, 0), (9.5, 3.6)),
        "F": ((2, 2), RIGHT, (1, 0), (9.5, 1.4)),
    }
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, (head, d, fruit, (x, y)) in states.items():
        draw_state(ax, x, y, head, d, fruit)
        ax.text(x + 0.75, y + 0.3, name, ha="center", fontsize=14, weight="bold")

    def arrow(p1, p2, label, lx, ly, rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=18, lw=1.5))
        ax.text(lx, ly, label, ha="center", va="center", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))

    arrow((2.1, 5.25), (4.4, 5.25), "droite / 0", 3.25, 5.55)
    arrow((6.1, 5.25), (8.4, 5.25), "droite / 0", 7.25, 5.55)
    arrow((9.25, 6.75), (1.25, 6.75), "droite / -10 (traverse le bord)", 5.25, 8.1, rad=0.3)
    arrow((5.25, 4.4), (5.25, 3.0), "bas / 0", 5.85, 3.5)
    arrow((6.1, 1.75), (9.4, 2.85), "droite / +10\np = 1/8", 7.7, 2.75)
    arrow((6.1, 1.75), (9.4, 0.65), "droite / +10\np = 1/8", 7.7, 0.75)
    ax.text(11.3, 1.75, "... et 6 autres états possibles\n(le nouveau fruit peut être\nsur n'importe quelle case libre)",
            fontsize=9.5, va="center")
    ax.text(0.3, 1.6, "vert : tête du serpent\n(la flèche donne la direction)\nrouge : fruit\n\nsur les flèches :\naction / récompense",
            fontsize=10, va="center")
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.2, 8.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Quelques états du MDP sur une grille 3x3 (cycle A -> B -> C -> A)", pad=10)
    plt.savefig("ex2_mdp.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot(curves, name, title, w=50, ylabel="récompense moyenne par coup"):
    plt.figure(figsize=(9, 5))
    for label, c in curves.items():
        plt.plot(smooth(c, w), label=label)
    plt.xlabel("étapes")
    plt.ylabel(ylabel + (f"\n(moyenne glissante sur {w} coups)" if w > 1 else ""))
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def ex3():
    r = run(STEPS)
    plt.figure(figsize=(9, 5))
    plt.plot(r, ".", markersize=3, label="récompense")
    plt.plot(smooth(r, 50), label="moyenne glissante (50)")
    plt.xlabel("étapes")
    plt.title("Une seule partie, grille 5x5, gamma = 0.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ex3_une_partie.png")
    plt.close()

    curves = {}
    for n in [10, 100, 300, 1000]:
        t = time.time()
        curves[f"{n} parties"] = mean_curve(STEPS, n)
        print(f"{n} parties : {time.time() - t:.1f}s")
    plot(curves, "ex3_nb_parties.png", "Choix du nombre de parties (grille 5x5, gamma = 0.5)", w=1)

    for size in [BIG, SIZE]:
        curves = {}
        for gamma in GAMMAS:
            curves[f"gamma = {gamma}"] = mean_curve(STEPS, RUNS, gamma=gamma, size=size)
            print(f"{size}x{size} gamma", gamma, "moyenne :", round(sum(curves[f"gamma = {gamma}"]) / STEPS, 3))
        plot(curves, f"ex3_gammas_1000_{size}x{size}.png",
             f"Q-learning, grille {size}x{size}, 1000 étapes, moyenne sur {RUNS} parties")


def search(steps, runs):
    res = {}
    for i in range(21):
        gamma = i / 20
        c = mean_curve(steps, runs, gamma=gamma)
        res[gamma] = sum(c) / steps
        print(f"  gamma = {gamma:.2f} : {res[gamma]:.3f}")
    best = max(res, key=res.get)
    print("  meilleur gamma :", best)
    return res


def ex3_search():
    res1 = search(STEPS, RUNS)
    res2 = search(10000, 50)
    plt.figure(figsize=(9, 5))
    plt.plot(list(res1), list(res1.values()), "o-", label="1000 étapes")
    plt.plot(list(res2), list(res2.values()), "o-", label="10000 étapes")
    plt.xlabel("gamma")
    plt.ylabel("récompense moyenne par coup")
    plt.title("Recherche du meilleur facteur d'actualisation (grille 5x5)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ex3_recherche_gamma.png")
    plt.close()


def ex3_long():
    curves = {}
    for gamma in GAMMAS:
        c = mean_curve(10000, 100, gamma=gamma)
        curves[f"gamma = {gamma}"] = c
        print("gamma", gamma, "moyenne 10000 :", round(sum(c) / 10000, 3), "fin :", round(sum(c[-2000:]) / 2000, 3))
    plot(curves, "ex3_gammas_10000.png", "Q-learning, grille 5x5, 10000 étapes, moyenne sur 100 parties", w=200)

    curves = {}
    for gamma in [0, 0.5, 0.9]:
        curves[f"gamma = {gamma}"] = mean_curve(300000, 5, gamma=gamma, size=BIG)
        print(f"10x10 gamma {gamma} : fin {sum(curves[f'gamma = {gamma}'][-20000:]) / 20000:.3f}")
    plot(curves, "ex3_gammas_300000_10x10.png", "Grille 10x10, 300000 étapes, moyenne sur 5 parties", w=5000)


def ex4():
    curves = {}
    for length in [1, 2, 3, 4, 5]:
        t = time.time()
        c = mean_curve(300000, 5, gamma=0.9, length=length, size=BIG)
        curves[f"longueur {length}"] = c
        print(f"longueur {length} : debut {sum(c[:20000]) / 20000:.3f}, fin {sum(c[-20000:]) / 20000:.3f}"
              f" ({time.time() - t:.0f}s)")
    plot(curves, "ex4_longueurs.png", "Grille 10x10, gamma = 0.9, 300000 étapes, moyenne sur 5 parties", w=5000)

    curves = {}
    for length in [1, 2]:
        c = mean_curve(10000, 100, gamma=0.9, length=length)
        curves[f"longueur {length}"] = c
        print(f"5x5 longueur {length} : fin {sum(c[-2000:]) / 2000:.3f}")
    plot(curves, "ex4_longueur2_10000.png", "Grille 5x5, longueurs 1 et 2, 10000 étapes, moyenne sur 100 parties", w=200)


if __name__ == "__main__":
    exos = {"1": ex1, "2": ex2, "3": ex3, "recherche": ex3_search, "long": ex3_long, "4": ex4}
    todo = sys.argv[1:] or ["1", "2", "3", "recherche", "long", "4"]
    for e in todo:
        print("=== exercice", e)
        exos[e]()
