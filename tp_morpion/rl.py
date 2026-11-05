import random
from collections import defaultdict

from tictactoe import TicTacToe

SYMS = [
    (0, 1, 2, 3, 4, 5, 6, 7, 8),
    (6, 3, 0, 7, 4, 1, 8, 5, 2),
    (8, 7, 6, 5, 4, 3, 2, 1, 0),
    (2, 5, 8, 1, 4, 7, 0, 3, 6),
    (2, 1, 0, 5, 4, 3, 8, 7, 6),
    (6, 7, 8, 3, 4, 5, 0, 1, 2),
    (0, 3, 6, 1, 4, 7, 2, 5, 8),
    (8, 5, 2, 7, 4, 1, 6, 3, 0),
]
VAL = {" ": 0, "X": 1, "O": 2}


def new_value_function():
    return defaultdict(lambda: 0.5)


def key(game, sym=False):
    if not sym:
        return hash(game)
    g = game.grid
    best = None
    for s in SYMS:
        k = 0
        for i in range(9):
            k = k * 3 + VAL[g[s[i]]]
        if best is None or k < best:
            best = k
    return best


def opponent_random(game):
    return random.choice(game.allowed_moves)


def opponent_next(game):
    moves = game.allowed_moves
    for m in moves:
        game.play(m)
        win = game.has_winner()
        game.undo(m)
        if win:
            return m
    return random.choice(moves)


def greedy_move(V, game, sym=False):
    best_val = None
    best = []
    for m in game.allowed_moves:
        game.play(m)
        v = V[key(game, sym)]
        game.undo(m)
        if best_val is None or v > best_val:
            best_val = v
            best = [m]
        elif v == best_val:
            best.append(m)
    return random.choice(best)


def play_game(V, opponent, alpha=0.1, eps=0.05, learn=True, learn_explore=True, sym=False):
    game = TicTacToe()
    prev = None
    while True:
        explored = random.random() < eps
        if explored:
            m = random.choice(game.allowed_moves)
        else:
            m = greedy_move(V, game, sym)
        game.play(m)
        s = key(game, sym)

        if game.has_winner():
            res = "X"
        elif game.is_draw():
            res = "nul"
        else:
            res = None

        if learn:
            if res == "X":
                V[s] = 1.0
            elif res == "nul":
                V[s] = 0.0
            if prev is not None and (learn_explore or not explored):
                V[prev] += alpha * (V[s] - V[prev])
        if res:
            return res
        prev = s

        game.play(opponent(game))
        if game.has_winner():
            if learn:
                s = key(game, sym)
                V[s] = 0.0
                V[prev] += alpha * (V[s] - V[prev])
            return "O"


def play_game2(V1, V2, alpha=0.1, eps=0.05, learn=True, sym=False, draw=0.5):
    game = TicTacToe()
    values = {"X": V1, "O": V2}
    prev = {"X": None, "O": None}
    while True:
        p = game.player
        V = values[p]
        if random.random() < eps:
            m = random.choice(game.allowed_moves)
        else:
            m = greedy_move(V, game, sym)
        game.play(m)
        s = key(game, sym)

        if game.has_winner():
            if learn:
                V[s] = 1.0
                if prev[p] is not None:
                    V[prev[p]] += alpha * (1.0 - V[prev[p]])
                q = game.other(p)
                values[q][prev[q]] += alpha * (0.0 - values[q][prev[q]])
            return p
        if game.is_draw():
            if learn:
                V[s] = draw
                for q in "XO":
                    if prev[q] is not None:
                        values[q][prev[q]] += alpha * (draw - values[q][prev[q]])
            return "nul"

        if learn and prev[p] is not None:
            V[prev[p]] += alpha * (V[s] - V[prev[p]])
        prev[p] = s
