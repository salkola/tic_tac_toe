"""Shared helpers for human vs agent play."""

import sys

import numpy as np

from agent.policy_value_agent import PolicyValueAgent
from config import BEST_MODEL_PATH, MODEL_PATH
from game.board import Board


def load_agent() -> PolicyValueAgent:
    model_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else MODEL_PATH
    if not model_path.exists():
        print(
            f"No trained model found at {model_path}. Run tic-tac-toe-train first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rng = np.random.default_rng(0)
    return PolicyValueAgent.load(model_path, rng)


def agent_move(agent: PolicyValueAgent, board: Board, player: int) -> int:
    return agent.select_action(board, player)


def outcome_message(board: Board, human_player: int) -> str:
    winner = board.winner()
    if winner is None:
        return "Draw."
    if winner == human_player:
        return "You win!"
    return "Agent wins."


def symbol(player: int) -> str:
    return "X" if player == 1 else "O"
