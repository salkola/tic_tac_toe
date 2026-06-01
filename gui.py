"""Graphical tic-tac-toe UI against the trained policy-value MCTS agent."""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

from agent.policy_value_agent import PolicyValueAgent
from game.board import Board
from play.logic import agent_move, load_agent, outcome_message, symbol

GRID_PAD = 12
SYMBOLS = {0: "", 1: "X", 2: "O"}
COLORS = {
    "bg": "#1e1e2e",
    "panel": "#313244",
    "text": "#cdd6f4",
    "x_fg": "#ffffff",
    "x_bg": "#0000ff",
    "x_border": "#99ccff",
    "o_fg": "#ffffff",
    "o_bg": "#ff0000",
    "o_border": "#ff9999",
    "empty_bg": "#45475a",
    "empty_fg": "#cdd6f4",
    "empty_active": "#585b70",
    "accent": "#a6e3a1",
}


class BoardCell:
    """Clickable cell using Label (macOS renders custom bg colors reliably)."""

    def __init__(
        self,
        master: tk.Misc,
        index: int,
        on_click: callable,
        cell_font: tkfont.Font,
    ) -> None:
        self.index = index
        self.on_click = on_click
        self.clickable = False

        self.frame = tk.Frame(master, highlightthickness=5, bd=0)
        self.label = tk.Label(
            self.frame,
            text="",
            width=3,
            height=1,
            font=cell_font,
            relief=tk.FLAT,
            bd=0,
        )
        self.label.pack(ipadx=20, ipady=16)

        for widget in (self.frame, self.label):
            widget.bind("<Button-1>", self._handle_click)

    def _handle_click(self, _event: tk.Event) -> None:
        if self.clickable:
            self.on_click(self.index)

    def grid(self, **kwargs: object) -> None:
        self.frame.grid(**kwargs)

    def render(self, player: int, clickable: bool) -> None:
        self.clickable = clickable
        style = _cell_style(player)
        border = style.pop("highlightbackground")
        thickness = 5 if player != 0 else 2

        self.frame.configure(bg=border, highlightbackground=border, highlightthickness=thickness)
        self.label.configure(**style)
        cursor = "hand2" if clickable else "arrow"
        self.frame.configure(cursor=cursor)
        self.label.configure(cursor=cursor)


def _cell_style(player: int) -> dict[str, str]:
    if player == 1:
        return {
            "text": SYMBOLS[1],
            "fg": COLORS["x_fg"],
            "bg": COLORS["x_bg"],
            "highlightbackground": COLORS["x_border"],
        }
    if player == 2:
        return {
            "text": SYMBOLS[2],
            "fg": COLORS["o_fg"],
            "bg": COLORS["o_bg"],
            "highlightbackground": COLORS["o_border"],
        }
    return {
        "text": "",
        "fg": COLORS["empty_fg"],
        "bg": COLORS["empty_bg"],
        "highlightbackground": COLORS["panel"],
    }


class TicTacToeGUI:
    def __init__(self, root: tk.Tk, agent: PolicyValueAgent) -> None:
        self.root = root
        self.agent = agent
        self.human_player = 1
        self.board = Board()
        self.current = 1
        self.game_over = False

        self.root.title("Tic-Tac-Toe vs Agent")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(False, False)

        self.title_font = tkfont.Font(family="Helvetica", size=18, weight="bold")
        self.cell_font = tkfont.Font(family="Helvetica", size=64, weight="bold")
        self.button_font = tkfont.Font(family="Helvetica", size=13)
        self.legend_font = tkfont.Font(family="Helvetica", size=20, weight="bold")

        self.status = tk.Label(
            root,
            text="Choose your side and start playing.",
            font=self.title_font,
            fg=COLORS["text"],
            bg=COLORS["bg"],
            pady=16,
        )
        self.status.pack()

        legend_frame = tk.Frame(root, bg=COLORS["bg"])
        legend_frame.pack(pady=(0, 10))

        tk.Label(
            legend_frame,
            text="X",
            font=self.legend_font,
            fg=COLORS["x_fg"],
            bg=COLORS["x_bg"],
            width=3,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(
            legend_frame,
            text="= blue crosses",
            font=self.button_font,
            fg=COLORS["text"],
            bg=COLORS["bg"],
        ).pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(
            legend_frame,
            text="O",
            font=self.legend_font,
            fg=COLORS["o_fg"],
            bg=COLORS["o_bg"],
            width=3,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(
            legend_frame,
            text="= red circles",
            font=self.button_font,
            fg=COLORS["text"],
            bg=COLORS["bg"],
        ).pack(side=tk.LEFT)

        side_frame = tk.Frame(root, bg=COLORS["bg"])
        side_frame.pack(pady=(0, 8))

        tk.Label(
            side_frame,
            text="Play as:",
            font=self.button_font,
            fg=COLORS["text"],
            bg=COLORS["bg"],
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.side_var = tk.IntVar(value=1)
        for value, label, fg, bg in (
            (1, "X (first)", COLORS["x_fg"], COLORS["x_bg"]),
            (2, "O (second)", COLORS["o_fg"], COLORS["o_bg"]),
        ):
            tk.Radiobutton(
                side_frame,
                text=label,
                variable=self.side_var,
                value=value,
                font=self.button_font,
                fg=fg,
                bg=bg,
                selectcolor=COLORS["panel"],
                activebackground=bg,
                activeforeground=fg,
                padx=10,
                pady=4,
                indicatoron=False,
                command=self.start_new_game,
            ).pack(side=tk.LEFT, padx=4)

        board_frame = tk.Frame(root, bg=COLORS["panel"], padx=GRID_PAD, pady=GRID_PAD)
        board_frame.pack()

        self.cells: list[BoardCell] = []
        for index in range(9):
            cell = BoardCell(board_frame, index, self.on_cell_click, self.cell_font)
            row, col = divmod(index, 3)
            cell.grid(row=row, column=col, padx=8, pady=8)
            self.cells.append(cell)

        controls = tk.Frame(root, bg=COLORS["bg"])
        controls.pack(pady=16)

        tk.Button(
            controls,
            text="New game",
            font=self.button_font,
            bg=COLORS["accent"],
            fg=COLORS["bg"],
            activebackground=COLORS["text"],
            relief=tk.FLAT,
            padx=16,
            pady=8,
            command=self.start_new_game,
        ).pack()

        self.start_new_game()

    def start_new_game(self) -> None:
        self.human_player = self.side_var.get()
        self.board = Board()
        self.current = 1
        self.game_over = False
        self.refresh_board()
        self.set_status(f"You are {symbol(self.human_player)}. Click a cell to play.")

        if self.current != self.human_player:
            self.root.after(300, self.agent_turn)

    def set_status(self, message: str) -> None:
        self.status.configure(text=message)

    def refresh_board(self) -> None:
        for index, cell in enumerate(self.cells):
            player = self.board.cells[index]
            clickable = not self.game_over and player == 0 and self.current == self.human_player
            cell.render(player, clickable)

    def on_cell_click(self, action: int) -> None:
        if self.game_over or self.current != self.human_player:
            return
        if action not in self.board.legal_moves():
            return

        self.apply_move(action)
        if not self.game_over:
            self.set_status("Agent is thinking...")
            self.root.after(250, self.agent_turn)

    def agent_turn(self) -> None:
        if self.game_over or self.current == self.human_player:
            return
        action = agent_move(self.agent, self.board, self.current)
        self.apply_move(action)

    def apply_move(self, action: int) -> None:
        self.board = self.board.apply_move(action, self.current)
        self.current = 2 if self.current == 1 else 1
        self.refresh_board()

        if self.board.is_terminal():
            self.game_over = True
            self.refresh_board()
            self.set_status(outcome_message(self.board, self.human_player))
        elif self.current == self.human_player:
            self.set_status("Your turn.")


def main() -> None:
    agent = load_agent()
    root = tk.Tk()
    TicTacToeGUI(root, agent)
    root.mainloop()


if __name__ == "__main__":
    main()
