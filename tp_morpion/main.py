import sys
import time
import pickle
import random

from tictactoe import TicTacToe
from rl import (new_value_function, opponent_random, opponent_next,
                greedy_move, play_game, play_game2)


def stats(results):
    n = len(results)
    return {r: round(100 * results.count(r) / n, 1) for r in ["X", "O", "nul"]}


def match(ia_x, ia_o, verbose=False):
    g = TicTacToe()
    while not g.is_over():
        if g.player == "X":
            g.play(ia_x(g))
        else:
            g.play(ia_o(g))
        if verbose:
            print(g, "\n")
    w = g.winner()
    return w if w else "nul"


def evaluate(V, opponent, n=5000, sym=False):
    res = [play_game(V, opponent, eps=0, learn=False, sym=sym) for _ in range(n)]
    return stats(res)


def train(V, opponent, n, every=10000, sym=False, learn_explore=True):
    for i in range(1, n + 1):
        play_game(V, opponent, sym=sym, learn_explore=learn_explore)
        if i % every == 0:
            print(f"  {i} parties :", evaluate(V, opponent, 2000, sym), "etats:", len(V))


def ex2():
    print("random vs next :")
    match(opponent_random, opponent_next, verbose=True)
    print(stats([match(opponent_random, opponent_next) for _ in range(10000)]))
    print("next vs random :", stats([match(opponent_next, opponent_random) for _ in range(10000)]))

    V = new_value_function()
    res = [play_game(V, opponent_random, eps=0, learn=False) for _ in range(20000)]
    print("greedy sans apprentissage vs random :", stats(res))


def ex3():
    for name, opp in [("random", opponent_random), ("next", opponent_next)]:
        print("entrainement contre", name)
        V = new_value_function()
        train(V, opp, 50000)
        with open(f"value_{name}.pkl", "wb") as f:
            pickle.dump(dict(V), f)

    print("pur greedy (eps=0) contre random")
    V = new_value_function()
    for _ in range(50000):
        play_game(V, opponent_random, eps=0)
    print(" ", evaluate(V, opponent_random))

    print("sans mise a jour sur les coups exploratoires, contre random")
    V = new_value_function()
    train(V, opponent_random, 50000, every=25000, learn_explore=False)


def ex3_selfplay():
    V1 = new_value_function()
    V2 = new_value_function()
    for i in range(1, 100001):
        play_game2(V1, V2)
        if i % 20000 == 0:
            res = [play_game2(V1, V2, eps=0, learn=False) for _ in range(1000)]
            print(f"  {i} parties self-play :", stats(res))
    print("V1 (appris en self-play) contre random :", evaluate(V1, opponent_random))
    print("V1 contre next :", evaluate(V1, opponent_next))


def ex3_ameliorations():
    for sym in [False, True]:
        print("symetries" if sym else "sans symetries")
        V = new_value_function()
        t = time.time()
        train(V, opponent_random, 20000, every=5000, sym=sym)
        print(f"  temps : {time.time() - t:.1f}s")


def jouer():
    try:
        with open("value_next.pkl", "rb") as f:
            V = new_value_function()
            V.update(pickle.load(f))
    except FileNotFoundError:
        print("pas de fichier value_next.pkl, lance d'abord : python main.py 3")
        return
    g = TicTacToe()
    while not g.is_over():
        print(g, "\n")
        if g.player == "X":
            m = greedy_move(V, g)
            print("l'IA joue", m)
            g.play(m)
        else:
            try:
                g.play(int(input("ta case (0-8) : ")))
            except (ValueError, IndexError) as e:
                print("coup invalide", e)
    print(g)
    print("gagnant :", g.winner() or "personne")


if __name__ == "__main__":
    random.seed()
    cmds = {"2": ex2, "3": ex3, "selfplay": ex3_selfplay, "sym": ex3_ameliorations, "jouer": jouer}
    todo = sys.argv[1:] or ["2", "3", "selfplay", "sym"]
    for c in todo:
        print("===", c)
        cmds[c]()
