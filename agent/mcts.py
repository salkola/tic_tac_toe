"""Monte Carlo Tree Search guided by a policy-value network."""

from __future__ import annotations

import math

import numpy as np
import torch

from agent.model import PolicyValueNet
from game.board import Board
from game.env import action_mask, observation


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, logits, -1e9)
    masked = masked - np.max(masked)
    exp = np.exp(masked) * mask
    total = exp.sum()
    if total <= 0:
        uniform = mask.astype(np.float64)
        return uniform / uniform.sum()
    return exp / total


def terminal_value(board: Board, player: int) -> float:
    winner = board.winner()
    if winner == player:
        return 1.0
    if winner is None:
        return 0.0
    return -1.0


class _Node:
    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, _Node] = {}

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        # Child value is from the opponent's perspective; negate for parent Q.
        return -self.value() + exploration

    def select_child(self, c_puct: float) -> tuple[int, _Node]:
        action, child = max(
            self.children.items(),
            key=lambda item: item[1].ucb_score(self.visit_count, c_puct),
        )
        return action, child

    def expand(self, priors: dict[int, float]) -> None:
        for action, prior in priors.items():
            self.children[action] = _Node(prior)

    def backup(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    def __init__(
        self,
        *,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(
        self,
        board: Board,
        player: int,
        net: PolicyValueNet,
        *,
        num_simulations: int,
        device: torch.device,
        add_root_noise: bool = False,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        root = _Node(prior=1.0)
        mask = action_mask(board)
        cache: dict[tuple[tuple[int, ...], int], tuple[np.ndarray, float]] = {}

        if board.is_terminal():
            return mask.astype(np.float64) / mask.sum()

        priors, _ = self._priors_and_value(board, player, net, device, cache)
        root.expand(priors)

        root_policy = np.zeros(9, dtype=np.float64)
        for action, prior in priors.items():
            root_policy[action] = prior

        if num_simulations <= 0:
            total = root_policy.sum()
            if total <= 0:
                return mask.astype(np.float64) / mask.sum()
            return root_policy / total

        if add_root_noise and rng is not None:
            legal = board.legal_moves()
            noise = rng.dirichlet([self.dirichlet_alpha] * len(legal))
            for index, action in enumerate(legal):
                child = root.children[action]
                child.prior = (
                    1 - self.dirichlet_epsilon
                ) * child.prior + self.dirichlet_epsilon * noise[index]

        for _ in range(num_simulations):
            node = root
            sim_board = board
            sim_player = player
            path = [node]

            while node.children and not sim_board.is_terminal():
                action, node = node.select_child(self.c_puct)
                sim_board = sim_board.apply_move(action, sim_player)
                sim_player = 2 if sim_player == 1 else 1
                path.append(node)

            if sim_board.is_terminal():
                value = terminal_value(sim_board, sim_player)
            else:
                priors, value = self._priors_and_value(sim_board, sim_player, net, device, cache)
                node.expand(priors)

            for path_node in reversed(path):
                path_node.backup(value)
                value = -value

        visits = np.zeros(9, dtype=np.float64)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        total = visits.sum()
        if total <= 0:
            root_total = root_policy.sum()
            if root_total > 0:
                return root_policy / root_total
            return mask.astype(np.float64) / mask.sum()
        return visits / total

    def _priors_and_value(
        self,
        board: Board,
        player: int,
        net: PolicyValueNet,
        device: torch.device,
        cache: dict[tuple[tuple[int, ...], int], tuple[np.ndarray, float]],
    ) -> tuple[dict[int, float], float]:
        key = (board.cells, player)
        if key not in cache:
            obs = observation(board, player)
            state = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.inference_mode():
                policy_logits, value = net(state)
            logits = policy_logits.squeeze(0).cpu().numpy()
            cache[key] = (logits, float(value.item()))

        logits, value = cache[key]
        mask = action_mask(board)
        probs = masked_softmax(logits, mask)
        priors = {action: float(probs[action]) for action in board.legal_moves()}
        return priors, value
