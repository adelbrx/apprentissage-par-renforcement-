import math
import random


class Bandit:
    def __init__(self):
        self.avg = random.gauss(0, 1)

    def play(self):
        return random.gauss(self.avg, 1)


class MovingBandit(Bandit):
    def __init__(self):
        super().__init__()
        self.period = random.randint(100, 500)
        a = random.gauss(0, 1)
        b = random.gauss(0, 1)
        self.low = min(a, b)
        self.high = max(a, b)
        self.phase = random.uniform(0, 2 * math.pi)
        self.t = 0
        self._update_avg()

    def _update_avg(self):
        mid = (self.low + self.high) / 2
        amp = (self.high - self.low) / 2
        self.avg = mid + amp * math.sin(2 * math.pi * self.t / self.period + self.phase)

    def play(self):
        r = random.gauss(self.avg, 1)
        self.t += 1
        self._update_avg()
        return r


class Ban10:
    bandit_class = Bandit

    def __init__(self):
        self.bandits = [self.bandit_class() for _ in range(10)]
        self.best = self._find_best()

    def _find_best(self):
        best = 0
        for i in range(len(self.bandits)):
            if self.bandits[i].avg > self.bandits[best].avg:
                best = i
        return best

    def play(self, arm):
        r = self.bandits[arm].play()
        return r


class MovingBan10(Ban10):
    bandit_class = MovingBandit

    def play(self, arm):
        self.best = self._find_best()
        r = self.bandits[arm].play()
        return r


class GreedyPlayer:
    def __init__(self, n, eps, step=None):
        self.n = n
        self.eps = eps
        self.step = step
        self.action_values = [0.0] * n
        self.eval_count = [0] * n
        self.t = 0

    def get_action(self):
        explore = random.random() < self.eps
        if explore:
            return self._random_action()
        return self._greedy_action()

    def _greedy_action(self):
        m = max(self.action_values)
        best = [a for a in range(self.n) if self.action_values[a] == m]
        return random.choice(best)

    def _random_action(self):
        return random.randrange(self.n)

    def reward(self, action, r):
        self.t += 1
        self.eval_count[action] += 1
        if self.step is None:
            alpha = 1 / self.eval_count[action]
        else:
            alpha = self.step
        self.action_values[action] += alpha * (r - self.action_values[action])


class OptimistGreedyPlayer(GreedyPlayer):
    def __init__(self, n, eps=0, q0=5, step=0.1):
        super().__init__(n, eps, step)
        self.action_values = [float(q0)] * n


class UCBGreedyPlayer(GreedyPlayer):
    def __init__(self, n, c, eps=0, step=None):
        super().__init__(n, eps, step)
        self.c = c

    def _greedy_action(self):
        untried = [a for a in range(self.n) if self.eval_count[a] == 0]
        if untried:
            return random.choice(untried)
        lt = math.log(self.t)
        scores = [self.action_values[a] + self.c * math.sqrt(lt / self.eval_count[a]) for a in range(self.n)]
        m = max(scores)
        best = [a for a in range(self.n) if scores[a] == m]
        return random.choice(best)
