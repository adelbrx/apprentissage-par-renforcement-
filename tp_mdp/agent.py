import random
from collections import defaultdict


class GreedyPlayer:
    def __init__(self, game, eps=0.1, gamma=0.5, alpha=0.1):
        self.game = game
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.action_values = defaultdict(float)
        self.eval_count = defaultdict(int)
        self.state = None

    def get_action(self):
        self.state = self.game.save_state()
        explore = random.random() < self.eps
        if explore:
            return self._random_action()
        return self._greedy_action()

    def _best_value(self, state, moves):
        return max(self.action_values[(state, a)] for a in moves)

    def _greedy_action(self):
        moves = self.game.allowed_moves()
        m = self._best_value(self.state, moves)
        best = [a for a in moves if self.action_values[(self.state, a)] == m]
        return random.choice(best)

    def _random_action(self):
        return random.choice(self.game.allowed_moves())

    def reward(self, action, r):
        new_state = self.game.save_state()
        target = r + self.gamma * self._best_value(new_state, self.game.allowed_moves())
        key = (self.state, action)
        self.eval_count[key] += 1
        self.action_values[key] = (1 - self.alpha) * self.action_values[key] + self.alpha * target
