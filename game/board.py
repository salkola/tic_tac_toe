"""Core tic-tac-toe game logic."""

from __future__ import annotations

from dataclasses import dataclass

WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True)
class Board:
    """3x3 board. Cells are 0 (empty), 1 (X), or 2 (O)."""

    cells: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    def legal_moves(self) -> list[int]:
        return [index for index, cell in enumerate(self.cells) if cell == 0]

    def apply_move(self, action: int, player: int) -> Board:
        if action not in self.legal_moves():
            msg = f"Illegal move: cell {action} is not empty"
            raise ValueError(msg)
        cells = list(self.cells)
        cells[action] = player
        return Board(cells=tuple(cells))

    def winner(self) -> int | None:
        for a, b, c in WIN_LINES:
            line = self.cells[a]
            if line != 0 and line == self.cells[b] == self.cells[c]:
                return line
        return None

    def is_draw(self) -> bool:
        return self.winner() is None and not self.legal_moves()

    def is_terminal(self) -> bool:
        return self.winner() is not None or self.is_draw()

    def render(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        rows: list[str] = []
        for row in range(3):
            cells = [symbols[self.cells[row * 3 + col]] for col in range(3)]
            rows.append(" ".join(cells))
        return "\n".join(rows)
