# TP Morpion - Rapport

Groupe : Ahmed Adel BEREKSI REGUIG, Ismail ELBOUCH, Anis GRIGAH

## Répartition du travail

| Membre | Travail réalisé |
|---|---|
| Ahmed Adel BEREKSI REGUIG | Exercice 1 (classe TicTacToe, questions 1 à 8), exercice 3 questions 9 et 10 (symétries, hash incrémental et autres améliorations) |
| Ismail ELBOUCH | Exercice 2 (opponent_random, opponent_next, value function, __hash__, greedy_move, play_game, vérification des 59% / 29% / 12%), exercice 3 questions 7 et 8 (self-play avec play_game2) |
| Anis GRIGAH | Exercice 3 questions 1 à 6 (mise à jour temporal-difference, coups exploratoires, tests contre random et next, sauvegarde avec pickle, mode jouer) |

Chacun a écrit le code, fait les tests et les graphiques et rédigé dans ce rapport les réponses des questions qui lui étaient attribuées. Le rapport complet a ensuite été relu et corrigé par les trois membres du groupe.

## Reproduction des résultats

Le code est en Python 3 et n'utilise que la bibliothèque standard.

- tictactoe.py : la classe TicTacToe
- rl.py : les adversaires, la value function, greedy_move, play_game et play_game2
- main.py : les expériences

Commandes :

- python main.py : lance tout (exercice 2, exercice 3, self-play, symétries), environ 15 secondes
- python main.py 2 : seulement l'exercice 2
- python main.py 3 : entraînement contre opponent_random et opponent_next, sauvegarde value_random.pkl et value_next.pkl
- python main.py selfplay : deux IA qui apprennent l'une contre l'autre
- python main.py sym : comparaison avec et sans symétries
- python main.py jouer : jouer contre l'IA (à lancer après python main.py 3)

Notre IA joue toujours X (elle commence). Pour évaluer une value function on joue 2000 à 5000 parties en mode totalement greedy (eps = 0) et sans apprentissage. Il n'y a pas de graine fixée, les pourcentages varient donc d'un ou deux points d'un lancement à l'autre.


Code (main.py) :

```python
import sys
import time
import pickle
import random

from tictactoe import TicTacToe
from rl import (new_value_function, opponent_random, opponent_next,
                greedy_move, play_game, play_game2)


if __name__ == "__main__":
    random.seed()
    cmds = {"2": ex2, "3": ex3, "selfplay": ex3_selfplay, "sym": ex3_ameliorations, "jouer": jouer}
    todo = sys.argv[1:] or ["2", "3", "selfplay", "sym"]
    for c in todo:
        print("===", c)
        cmds[c]()
```

## Exercice 1 - Une classe TicTacToe

1. La grille est une liste de 9 cases qui contiennent " ", "X" ou "O". L'attribut player indique qui doit jouer, X au départ.
2. play(i) lève une ValueError si la case est déjà prise, sinon pose le symbole et passe au joueur suivant.
3. __str__ affiche la grille sur 3 lignes avec des séparateurs.
4. has_winner teste les 8 alignements (3 lignes, 3 colonnes, 2 diagonales). On a aussi une méthode winner qui renvoie "X", "O" ou None.
5. is_draw renvoie True si les 9 cases sont remplies et qu'il n'y a pas de gagnant.
6. reset remet la grille vide et redonne la main à X.
7. undo(i) vide la case i et redonne la main au joueur qui avait joué ce coup.
8. allowed_moves est une propriété qui renvoie la liste des cases vides.


Code (tictactoe.py) :

```python
class TicTacToe:
    LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
             (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]
    POW = [3 ** i for i in range(9)]

    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = [" "] * 9
        self.player = "X"
        self.code = 0
        self.nb_moves = 0

    def other(self, p=None):
        if p is None:
            p = self.player
        return "O" if p == "X" else "X"

    def play(self, i):
        if self.grid[i] != " ":
            raise ValueError(f"case {i} deja prise")
        self.grid[i] = self.player
        self.code += self.POW[i] * (1 if self.player == "X" else 2)
        self.nb_moves += 1
        self.player = self.other()

    def undo(self, i):
        if self.grid[i] == " ":
            raise ValueError(f"case {i} vide")
        p = self.grid[i]
        self.code -= self.POW[i] * (1 if p == "X" else 2)
        self.grid[i] = " "
        self.nb_moves -= 1
        self.player = p

    def winner(self):
        g = self.grid
        for a, b, c in self.LINES:
            if g[a] != " " and g[a] == g[b] == g[c]:
                return g[a]
        return None

    def has_winner(self):
        return self.winner() is not None

    def is_draw(self):
        return self.nb_moves == 9 and not self.has_winner()

    def is_over(self):
        return self.nb_moves == 9 or self.has_winner()

    @property
    def allowed_moves(self):
        return [i for i in range(9) if self.grid[i] == " "]

    def __hash__(self):
        return self.code

    def __eq__(self, other):
        return isinstance(other, TicTacToe) and self.grid == other.grid

    def __str__(self):
        rows = []
        for r in range(3):
            rows.append(" " + " | ".join(self.grid[3 * r:3 * r + 3]) + " ")
        return "\n---+---+---\n".join(rows)
```

## Exercice 2 - Jouer !

1. opponent_random renvoie random.choice(game.allowed_moves).
2. opponent_next essaye chaque coup possible avec play puis undo. S'il y a un coup gagnant il le joue, sinon il joue au hasard.
3. Sur 10000 parties, random (X) contre next (O) : X 39%, O 52%, nuls 8%. Next (X) contre random (O) : X 81%, O 12%, nuls 7%. Le joueur next est donc bien plus fort que le joueur aléatoire.
4. La value function est un defaultdict(lambda: 0.5), donc un état jamais vu vaut 0.5.
5. __hash__ renvoie un entier unique pour chaque grille : la grille est vue comme un nombre en base 3 (case vide = 0, X = 1, O = 2). Pour aller plus vite cet entier est mis à jour directement dans play et undo au lieu d'être recalculé. On a aussi défini __eq__. Comme l'objet TicTacToe change au cours de la partie, on indexe le dictionnaire avec hash(game) et pas avec l'objet lui-même, sinon la clé changerait après avoir été insérée.
6. greedy_move essaye chaque coup possible (play, lecture de la valeur de l'état obtenu, undo) et choisit au hasard parmi les coups qui ont la valeur maximale.
7. play_game(V, opponent) joue une partie complète : notre IA joue avec greedy_move, l'adversaire avec la fonction donnée. Elle renvoie "X", "O" ou "nul".
8. Sans apprentissage toutes les valeurs restent à 0.5, donc greedy_move joue au hasard. Sur 20000 parties contre opponent_random on obtient X 58.3%, O 29.0%, nuls 12.7%, ce qui correspond aux 59%, 29% et 12% attendus.


Code (rl.py) :

```python
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


def new_value_function():
    return defaultdict(lambda: 0.5)


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
```

Code (main.py) :

```python
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


def ex2():
    print("random vs next :")
    match(opponent_random, opponent_next, verbose=True)
    print(stats([match(opponent_random, opponent_next) for _ in range(10000)]))
    print("next vs random :", stats([match(opponent_next, opponent_random) for _ in range(10000)]))

    V = new_value_function()
    res = [play_game(V, opponent_random, eps=0, learn=False) for _ in range(20000)]
    print("greedy sans apprentissage vs random :", stats(res))
```

## Exercice 3 - Temporal-Difference

1. Quand la partie est finie on met la valeur de l'état final à 1 si notre IA a gagné, 0 si elle a perdu ou fait match nul.
2. Après chaque coup de notre IA on applique la formule du cours : V(s) = V(s) + alpha * (V(s') - V(s)), où s est l'état après notre coup précédent et s' l'état après notre coup actuel. alpha = 0.1. Quand l'adversaire gagne, l'état après notre dernier coup est mis à jour vers 0.

Code (rl.py) :

```python
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
```


3. Question : si on joue uniquement de manière greedy, l'IA ne visite que les états qu'elle pense déjà bons. Dès qu'un coup lui a donné une victoire elle va le rejouer et ne testera jamais les autres, même s'ils sont meilleurs. Elle ne joue pas forcément moins bien : dans notre test (50000 parties en pur greedy contre random) elle gagne quand même environ 98% du temps, parce qu'un joueur aléatoire est facile à battre. Mais elle n'est pas forcément meilleure non plus, rien ne garantit qu'elle trouve la stratégie optimale, ça dépend beaucoup des premières parties. Une valeur initiale plus élevée (par exemple 1) règle en partie le problème : les états jamais visités ont l'air meilleurs que ceux déjà testés dont la valeur a baissé, donc l'IA est poussée à les essayer. C'est le même principe que l'initialisation optimiste du TP bandit.

4. Avec une probabilité de 5% notre IA joue un coup au hasard au lieu du coup greedy (paramètre eps de play_game). Résultats en mode greedy après entraînement :

| Entraînement | Parties | X gagne | O gagne | Nuls |
|---|---|---|---|---|
| contre random | 10000 | 98.3% | 0.8% | 0.9% |
| contre random | 50000 | 98.5% | 0.1% | 1.4% |
| contre next | 10000 | 97.0% | 0.5% | 2.5% |
| contre next | 50000 | 99.8% | 0.2% | 0.0% |

Au départ (sans apprentissage) on avait 58% de victoires contre random, l'apprentissage marche donc très bien. La value function contient environ 2300 états à la fin.

5. Question : si on ne met pas à jour la value function après un coup exploratoire (paramètre learn_explore=False), elle estime la probabilité de gagner en jouant ensuite de manière greedy, c'est-à-dire avec la meilleure stratégie trouvée. Si on apprend aussi pendant l'exploration, elle estime la probabilité de gagner avec la stratégie qu'on suit vraiment, exploration comprise. Les valeurs sont alors un peu plus basses près des états où un mauvais coup coûte cher. La première version est meilleure si on arrête d'explorer après l'entraînement (on joue alors vraiment greedy), la deuxième est plus prudente pendant l'entraînement. Dans nos tests contre random les deux donnent à peu près le même résultat (98.9% de victoires sans mise à jour pendant l'exploration).

6. La value function est sauvegardée avec pickle dans value_random.pkl et value_next.pkl, et python main.py jouer recharge value_next.pkl pour jouer contre un humain.

Code (main.py) :

```python
def evaluate(V, opponent, n=5000, sym=False):
    res = [play_game(V, opponent, eps=0, learn=False, sym=sym) for _ in range(n)]
    return stats(res)


def train(V, opponent, n, every=10000, sym=False, learn_explore=True):
    for i in range(1, n + 1):
        play_game(V, opponent, sym=sym, learn_explore=learn_explore)
        if i % every == 0:
            print(f"  {i} parties :", evaluate(V, opponent, 2000, sym), "etats:", len(V))


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
```


7. Question : si les deux joueurs utilisent la même méthode d'apprentissage, l'adversaire n'est plus fixe : chacun s'adapte aux faiblesses de l'autre, qui les corrige à son tour. Les deux devraient converger vers un jeu optimal, et au morpion un jeu optimal des deux côtés donne toujours match nul. La value function apprise est donc différente : contre un joueur aléatoire l'IA apprend à profiter de ses erreurs, en self-play elle apprend surtout à ne jamais perdre.

8. play_game2(V1, V2) fait jouer deux IA qui ont chacune leur value function (V1 pour X, V2 pour O). Chacune évalue les états après ses propres coups et fait sa mise à jour TD. Premier essai avec un nul valant 0 comme une défaite : X gagnait 100% des parties, parce que O n'avait aucun intérêt à bloquer (perdre ou faire nul lui donnait la même valeur). En donnant la valeur 0.5 au match nul on obtient bien le résultat attendu :

| Parties de self-play | X gagne | O gagne | Nuls |
|---|---|---|---|
| 20000 | 0% | 0% | 100% |
| 100000 | 0% | 0% | 100% |

La value function de X apprise en self-play gagne quand même 99% des parties contre random et 98.7% contre next, sans jamais perdre.

Code (rl.py) :

```python
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
```

Code (main.py) :

```python
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
```


9. Question : beaucoup de grilles sont identiques à une rotation ou une symétrie près (8 transformations : 4 rotations, avec ou sans miroir). On peut ramener chaque grille à une forme canonique, par exemple la plus petite des 8 en base 3, et utiliser cette forme comme clé de la value function. On l'a implémenté (option sym=True, fonction key dans rl.py). On passe d'environ 2150 à environ 410 états pour 20000 parties, et chaque partie met à jour la valeur de toutes les grilles symétriques d'un coup, donc l'apprentissage est plus rapide en nombre de parties : 99.8% de victoires après 20000 parties contre 98.5% sans symétries.
Par contre, si l'adversaire n'est pas symétrique (s'il préfère par exemple le côté droit), deux grilles symétriques n'ont pas la même probabilité de gain. Les fusionner ferait perdre de l'information et l'IA ne pourrait pas exploiter cette préférence. Dans ce cas les états symétriques ne devraient pas avoir la même valeur.

Code (rl.py) :

```python
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
```

Code (main.py) :

```python
def ex3_ameliorations():
    for sym in [False, True]:
        print("symetries" if sym else "sans symetries")
        V = new_value_function()
        t = time.time()
        train(V, opponent_random, 20000, every=5000, sym=sym)
        print(f"  temps : {time.time() - t:.1f}s")
```


10. Améliorations proposées et implémentées :
- hash incrémental : la clé en base 3 est mise à jour dans play et undo, on évite de reconstruire une chaîne de caractères à chaque appel (ce qui serait le cas en passant par __str__) ;
- symétries (question 9) : moins de parties nécessaires, mais chaque coup coûte plus cher car on calcule les 8 transformations (2.1 s contre 0.4 s pour 20000 parties), c'est donc plus intéressant quand une partie coûte cher ;
- les états finaux reçoivent directement leur valeur 0 ou 1, donc greedy_move voit tout de suite un coup gagnant dès qu'il a été rencontré une fois.

Autres pistes non implémentées : faire diminuer alpha et eps au fil des parties (le cours indique qu'avec un alpha décroissant on converge vers les vraies probabilités), commencer l'entraînement contre opponent_next puis continuer en self-play, ou initialiser les états de façon optimiste pour explorer plus vite.
