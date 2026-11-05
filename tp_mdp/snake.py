import random

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
NAMES = {UP: "haut", DOWN: "bas", LEFT: "gauche", RIGHT: "droite"}


class ShortSnake:
    def __init__(self, width, height=None, length=1):
        if height is None:
            height = width
        self.width = width
        self.height = height
        self.grid = [["." for _ in range(width)] for _ in range(height)]
        self.snake_dir = RIGHT
        head = (width // 2, height // 2)
        self.body = [((head[0] - i) % width, head[1]) for i in range(length)]
        self.score = 0
        self.fruit_pos = None
        self._new_fruit()
        self._draw()

    @property
    def snake_pos(self):
        return self.body[0]

    def _new_fruit(self):
        free = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.body]
        self.fruit_pos = random.choice(free)

    def _draw(self):
        for row in self.grid:
            for x in range(self.width):
                row[x] = "."
        fx, fy = self.fruit_pos
        self.grid[fy][fx] = "*"
        for x, y in self.body[1:]:
            self.grid[y][x] = "o"
        x, y = self.body[0]
        self.grid[y][x] = "@"

    def allowed_moves(self):
        back = (-self.snake_dir[0], -self.snake_dir[1])
        return [d for d in DIRECTIONS if d != back]

    def move(self, direction=None):
        if direction is not None and direction != (-self.snake_dir[0], -self.snake_dir[1]):
            self.snake_dir = direction
        reward = 0
        x = self.body[0][0] + self.snake_dir[0]
        y = self.body[0][1] + self.snake_dir[1]
        if not (0 <= x < self.width and 0 <= y < self.height):
            x %= self.width
            y %= self.height
            reward -= 10
        self.body = [(x, y)] + self.body[:-1]
        if (x, y) in self.body[1:]:
            reward -= 10
        if (x, y) == self.fruit_pos:
            reward += 10
            self._new_fruit()
        self.score += reward
        self._draw()
        return reward

    def save_state(self):
        return (tuple(self.body), self.snake_dir, self.fruit_pos)

    def restore_state(self, state):
        body, d, fruit = state
        self.body = list(body)
        self.snake_dir = d
        self.fruit_pos = fruit
        self._draw()

    def __str__(self):
        lines = [" ".join(row) for row in self.grid]
        lines.append(f"score : {self.score}")
        return "\n".join(lines)
