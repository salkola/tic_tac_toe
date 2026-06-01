"""Perfect minimax labels for all reachable tic-tac-toe positions."""

from __future__ import annotations

import numpy as np

from agent.replay_buffer import TrainingExample
from eval.opponents import MinimaxOpponent
from game.board import Board
from game.env import observation

_SOLVER = MinimaxOpponent()


def _move_scores(board: Board, player: int) -> list[tuple[int, float, Board]]:
    root_player = player
    scored: list[tuple[int, float, Board]] = []
    for action in board.legal_moves():
        next_board = board.apply_move(action, player)
        next_player = 2 if player == 1 else 1
        score, _ = _SOLVER._minimax(next_board, next_player, root_player, False)
        scored.append((action, score, next_board))
    return scored


def _opponent_winning_moves(board: Board, opponent: int) -> list[int]:
    wins: list[int] = []
    for action in board.legal_moves():
        next_board = board.apply_move(action, opponent)
        if next_board.winner() == opponent:
            wins.append(action)
    return wins


def optimal_policy(board: Board, player: int) -> tuple[float, np.ndarray]:
    """Minimax value plus a policy that prefers immediate wins and blocks."""
    policy = np.zeros(9, dtype=np.float32)
    scored = _move_scores(board, player)
    best_score = max(score for _, score, _ in scored)
    best = [
        (action, next_board) for action, score, next_board in scored if score >= best_score - 1e-9
    ]

    immediate_wins = [action for action, next_board in best if next_board.winner() == player]
    if immediate_wins:
        chosen = immediate_wins
    else:
        opponent = 2 if player == 1 else 1
        threats = _opponent_winning_moves(board, opponent)
        if threats:
            blocks = [action for action, _ in best if action in threats]
            chosen = blocks if blocks else [action for action, _ in best]
        else:
            chosen = [action for action, _ in best]

    weight = 1.0 / len(chosen)
    for action in chosen:
        policy[action] = weight
    return best_score, policy


def generate_expert_examples() -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    stack = [(Board(), 1)]

    while stack:
        board, player = stack.pop()
        key = (board.cells, player)
        if key in seen:
            continue
        seen.add(key)

        if board.is_terminal():
            continue

        value, policy = optimal_policy(board, player)
        examples.append(
            TrainingExample(
                state=observation(board, player),
                policy=policy,
                value=value,
            )
        )

        for action in board.legal_moves():
            next_board = board.apply_move(action, player)
            next_player = 2 if player == 1 else 1
            stack.append((next_board, next_player))

    return examples


def sample_expert_examples(
    examples: list[TrainingExample],
    ratio: float,
    rng: np.random.Generator,
) -> list[TrainingExample]:
    if ratio >= 1.0:
        return list(examples)
    if ratio <= 0.0:
        return []

    count = max(1, int(len(examples) * ratio))
    indices = rng.choice(len(examples), size=count, replace=False)
    return [examples[index] for index in indices]
