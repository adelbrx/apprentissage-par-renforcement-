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
