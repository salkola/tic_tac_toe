"""RL environments for tic-tac-toe."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from game.board import Board


def observation(board: Board, player: int) -> np.ndarray:
    obs = np.zeros(9, dtype=np.float32)
    for index, cell in enumerate(board.cells):
        if cell == 0:
            obs[index] = 0.0
        elif cell == player:
            obs[index] = 1.0
        else:
            obs[index] = -1.0
    return obs


def action_mask(board: Board) -> np.ndarray:
    mask = np.zeros(9, dtype=bool)
    for move in board.legal_moves():
        mask[move] = True
    return mask


class Opponent(Protocol):
    def select_action(self, board: Board, player: int) -> int: ...


class TicTacToeEnv:
    """Self-play environment with player-relative observations."""

    def __init__(self) -> None:
        self.board = Board()
        self.current_player = 1

    def reset(self) -> np.ndarray:
        self.board = Board()
        self.current_player = 1
        return observation(self.board, self.current_player)

    def action_mask(self) -> np.ndarray:
        return action_mask(self.board)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if action not in self.board.legal_moves():
            msg = f"Illegal action {action}"
            raise ValueError(msg)

        player = self.current_player
        self.board = self.board.apply_move(action, player)

        winner = self.board.winner()
        if winner == player:
            return observation(self.board, self.current_player), 1.0, True, {}

        if self.board.is_terminal():
            return observation(self.board, self.current_player), 0.0, True, {}

        self.current_player = 2 if player == 1 else 1
        return observation(self.board, self.current_player), 0.0, False, {}


class VsOpponentEnv:
    """Agent trains against a fixed opponent from the agent's perspective."""

    def __init__(self, agent_player: int, opponent: Opponent) -> None:
        self.agent_player = agent_player
        self.opponent = opponent
        self.board = Board()
        self.current = 1

    def reset(self) -> np.ndarray:
        self.board = Board()
        self.current = 1
        if self.current != self.agent_player:
            self._opponent_move()
        return observation(self.board, self.agent_player)

    def action_mask(self) -> np.ndarray:
        return action_mask(self.board)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if action not in self.board.legal_moves():
            msg = f"Illegal action {action}"
            raise ValueError(msg)

        self.board = self.board.apply_move(action, self.agent_player)
        reward, done = self._terminal_outcome()
        if done:
            return observation(self.board, self.agent_player), reward, True, {}

        self.current = 2 if self.agent_player == 1 else 1
        self._opponent_move()
        reward, done = self._terminal_outcome()
        return observation(self.board, self.agent_player), reward, done, {}

    def _opponent_move(self) -> None:
        opponent_player = 2 if self.agent_player == 1 else 1
        action = self.opponent.select_action(self.board, opponent_player)
        self.board = self.board.apply_move(action, self.current)
        self.current = 2 if self.current == 1 else 1

    def _terminal_outcome(self) -> tuple[float, bool]:
        winner = self.board.winner()
        if winner == self.agent_player:
            return 1.0, True
        if winner is not None:
            return -1.0, True
        if self.board.is_draw():
            return 0.0, True
        return 0.0, False
