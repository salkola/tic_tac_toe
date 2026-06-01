"""Fixed opponents for evaluation."""

from typing import Protocol

import numpy as np

from agent.policy_value_agent import PolicyValueAgent
from game.board import Board


class Opponent(Protocol):
    def select_action(self, board: Board, player: int) -> int: ...


class RandomOpponent:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def select_action(self, board: Board, player: int) -> int:
        del player
        moves = board.legal_moves()
        return int(self.rng.choice(moves))


class MinimaxOpponent:
    def select_action(self, board: Board, player: int) -> int:
        _, action = self._minimax(board, player, player, True)
        return action

    def _minimax(
        self,
        board: Board,
        player: int,
        root_player: int,
        maximizing: bool,
    ) -> tuple[float, int]:
        winner = board.winner()
        if winner == root_player:
            return 1.0, -1
        if winner is not None:
            return -1.0, -1
        if board.is_draw():
            return 0.0, -1

        best_action = board.legal_moves()[0]
        if maximizing:
            best_score = -np.inf
            for action in board.legal_moves():
                next_board = board.apply_move(action, player)
                next_player = 2 if player == 1 else 1
                score, _ = self._minimax(next_board, next_player, root_player, False)
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_score, best_action

        best_score = np.inf
        for action in board.legal_moves():
            next_board = board.apply_move(action, player)
            next_player = 2 if player == 1 else 1
            score, _ = self._minimax(next_board, next_player, root_player, True)
            if score < best_score:
                best_score = score
                best_action = action
        return best_score, best_action


def play_agent_vs_opponent(
    agent: PolicyValueAgent,
    opponent: Opponent,
    *,
    agent_player: int,
    agent_first: bool,
    mcts_simulations: int | None = None,
) -> int:
    """Return 1 if agent won, -1 if agent lost, 0 if draw."""

    board = Board()
    current = 1 if agent_first else 2

    while not board.is_terminal():
        if current == agent_player:
            action = agent.select_action(
                board,
                agent_player,
                num_simulations=mcts_simulations,
            )
        else:
            opponent_player = 2 if agent_player == 1 else 1
            action = opponent.select_action(board, opponent_player)

        board = board.apply_move(action, current)
        current = 2 if current == 1 else 1

    winner = board.winner()
    if winner == agent_player:
        return 1
    if winner is None:
        return 0
    return -1


def evaluate_agent(
    agent: PolicyValueAgent,
    opponent: Opponent,
    *,
    games: int,
    rng: np.random.Generator,
    mcts_simulations: int | None = None,
) -> tuple[float, float, float]:
    wins = draws = losses = 0
    for game_index in range(games):
        agent_player = 1 if game_index % 2 == 0 else 2
        agent_first = agent_player == 1
        outcome = play_agent_vs_opponent(
            agent,
            opponent,
            agent_player=agent_player,
            agent_first=agent_first,
            mcts_simulations=mcts_simulations,
        )
        if outcome == 1:
            wins += 1
        elif outcome == 0:
            draws += 1
        else:
            losses += 1

    total = float(games)
    return wins / total, draws / total, losses / total
