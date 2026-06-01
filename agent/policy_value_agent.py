"""Policy-value agent with MCTS search and optional minimax pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from agent.mcts import MCTS
from agent.model import PolicyValueNet
from agent.replay_buffer import ReplayBuffer, TrainingExample
from game.board import Board
from game.env import observation
from training.symmetry import augment_example

ARCHITECTURE_NAME = "policy_value_mcts"


@dataclass(frozen=True)
class TrajectoryStep:
    state: np.ndarray
    policy: np.ndarray
    player: int


class PolicyValueAgent:
    def __init__(
        self,
        *,
        hidden_dim: int,
        learning_rate: float,
        replay_size: int,
        batch_size: int,
        mcts_simulations: int,
        mcts_eval_simulations: int,
        mcts_play_simulations: int,
        c_puct: float,
        symmetry_augment: bool,
        temperature_moves: int,
        grad_clip: float,
        ema_decay: float,
        rng: np.random.Generator,
        device: torch.device | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.mcts_simulations = mcts_simulations
        self.mcts_eval_simulations = mcts_eval_simulations
        self.mcts_play_simulations = mcts_play_simulations
        self.symmetry_augment = symmetry_augment
        self.temperature_moves = temperature_moves
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        self.rng = rng
        self.device = device or torch.device("cpu")
        self.recent_policy_losses: list[float] = []
        self.recent_value_losses: list[float] = []
        self.recent_total_losses: list[float] = []

        self.net = PolicyValueNet(hidden_dim).to(self.device)
        self.inference_net = PolicyValueNet(hidden_dim).to(self.device)
        self.inference_net.load_state_dict(self.net.state_dict())
        self.net.eval()
        self.inference_net.eval()
        self.mcts = MCTS(c_puct=c_puct)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(replay_size)

    def play_self_play_game(self) -> tuple[list[TrajectoryStep], Board]:
        return self._play_game(opponent=None)

    def play_opponent_game(
        self,
        opponent,
        *,
        agent_player: int,
    ) -> tuple[list[TrajectoryStep], Board]:
        return self._play_game(opponent=opponent, agent_player=agent_player)

    def _play_game(
        self,
        opponent,
        *,
        agent_player: int = 1,
    ) -> tuple[list[TrajectoryStep], Board]:
        board = Board()
        current = 1
        move_index = 0
        trajectory: list[TrajectoryStep] = []

        while not board.is_terminal():
            if opponent is not None and current != agent_player:
                action = opponent.select_action(board, current)
            else:
                policy = self.mcts.search(
                    board,
                    current,
                    self.inference_net,
                    num_simulations=self.mcts_simulations,
                    device=self.device,
                    add_root_noise=opponent is None,
                    rng=self.rng,
                )
                trajectory.append(
                    TrajectoryStep(
                        state=observation(board, current).copy(),
                        policy=policy.copy(),
                        player=current,
                    )
                )

                if move_index < self.temperature_moves:
                    legal = board.legal_moves()
                    probs = policy[legal]
                    probs = probs / probs.sum()
                    action = int(self.rng.choice(legal, p=probs))
                else:
                    action = int(np.argmax(policy))

            board = board.apply_move(action, current)
            current = 2 if current == 1 else 1
            move_index += 1

        return trajectory, board

    def store_game(self, trajectory: list[TrajectoryStep], board: Board) -> None:
        winner = board.winner()
        for step in trajectory:
            if winner is None:
                value = 0.0
            elif winner == step.player:
                value = 1.0
            else:
                value = -1.0
            example = TrainingExample(state=step.state, policy=step.policy, value=value)
            if self.symmetry_augment:
                for augmented in augment_example(example):
                    self.buffer.push(augmented)
            else:
                self.buffer.push(example)

    def train_step(self) -> tuple[float, float, float] | None:
        if len(self.buffer) < self.batch_size:
            return None

        self.net.train()
        batch = self.buffer.sample(self.batch_size, self.rng)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        target_policies = torch.as_tensor(
            np.stack([item.policy for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        target_values = torch.as_tensor(
            [item.value for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        policy_logits, values = self.net(states)
        legal_mask = states == 0
        masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        log_probs = torch.log_softmax(masked_logits, dim=1)
        policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
        value_loss = nn.functional.mse_loss(values, target_values)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()
        self.net.eval()
        self._update_ema()

        policy_value = float(policy_loss.item())
        value_value = float(value_loss.item())
        total_value = float(loss.item())
        self.recent_policy_losses.append(policy_value)
        self.recent_value_losses.append(value_value)
        self.recent_total_losses.append(total_value)
        return policy_value, value_value, total_value

    def _update_ema(self) -> None:
        with torch.no_grad():
            for ema_param, param in zip(
                self.inference_net.parameters(),
                self.net.parameters(),
                strict=True,
            ):
                ema_param.mul_(self.ema_decay).add_(param, alpha=1.0 - self.ema_decay)

    def pretrain(
        self,
        examples: list[TrainingExample],
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        policy_weight: float = 1.0,
    ) -> None:
        if not examples:
            return

        dataset = list(examples)
        if self.symmetry_augment:
            augmented: list[TrainingExample] = []
            for example in examples:
                augmented.extend(augment_example(example))
            dataset = augmented

        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        indices = np.arange(len(dataset))

        for _epoch in range(epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(dataset), batch_size):
                batch_indices = indices[start : start + batch_size]
                if len(batch_indices) < 2:
                    continue

                batch = [dataset[index] for index in batch_indices]
                states = torch.as_tensor(
                    np.stack([item.state for item in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                target_policies = torch.as_tensor(
                    np.stack([item.policy for item in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                target_values = torch.as_tensor(
                    [item.value for item in batch],
                    dtype=torch.float32,
                    device=self.device,
                )

                self.net.train()
                policy_logits, values = self.net(states)
                legal_mask = states == 0
                masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
                log_probs = torch.log_softmax(masked_logits, dim=1)
                policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
                value_loss = nn.functional.mse_loss(values, target_values)
                loss = policy_weight * policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                optimizer.step()

        self.net.eval()
        self.inference_net.load_state_dict(self.net.state_dict())
        self.inference_net.eval()
        for example in examples:
            self.buffer.push(example)

    def mean_recent_losses(self, window: int) -> tuple[float, float, float]:
        if not self.recent_total_losses:
            return 0.0, 0.0, 0.0
        policy = float(np.mean(self.recent_policy_losses[-window:]))
        value = float(np.mean(self.recent_value_losses[-window:]))
        total = float(np.mean(self.recent_total_losses[-window:]))
        return policy, value, total

    def select_action(
        self,
        board: Board,
        player: int,
        *,
        num_simulations: int | None = None,
    ) -> int:
        sims = num_simulations if num_simulations is not None else self.mcts_play_simulations
        policy = self.mcts.search(
            board,
            player,
            self.inference_net,
            num_simulations=sims,
            device=self.device,
            add_root_noise=False,
        )
        legal = board.legal_moves()
        return int(max(legal, key=lambda action: policy[action]))

    def save(
        self,
        path: Path,
        *,
        episode: int | None = None,
        score: float | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "policy_value_state_dict": self.inference_net.state_dict(),
            "hidden_dim": self.inference_net.trunk[0].out_features,
            "architecture": ARCHITECTURE_NAME,
            "mcts_play_simulations": self.mcts_play_simulations,
            "c_puct": self.mcts.c_puct,
        }
        if episode is not None:
            payload["best_episode"] = episode
        if score is not None:
            payload["best_score"] = score
        torch.save(payload, path)

    def reload_from_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = checkpoint["policy_value_state_dict"]
        self.net.load_state_dict(state_dict)
        self.inference_net.load_state_dict(state_dict)
        self.net.eval()
        self.inference_net.eval()

    @classmethod
    def load(cls, path: Path, rng: np.random.Generator) -> PolicyValueAgent:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        hidden_dim = int(checkpoint["hidden_dim"])
        agent = cls(
            hidden_dim=hidden_dim,
            learning_rate=1e-4,
            replay_size=1,
            batch_size=1,
            mcts_simulations=50,
            mcts_eval_simulations=75,
            mcts_play_simulations=int(checkpoint.get("mcts_play_simulations", 100)),
            c_puct=float(checkpoint.get("c_puct", 1.25)),
            symmetry_augment=False,
            temperature_moves=0,
            grad_clip=1.0,
            ema_decay=0.995,
            rng=rng,
        )
        state_dict = checkpoint["policy_value_state_dict"]
        agent.net.load_state_dict(state_dict)
        agent.inference_net.load_state_dict(state_dict)
        agent.net.eval()
        agent.inference_net.eval()
        return agent
