"""Training metrics logging and plotting."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import LEARNING_CURVES_PATH, METRICS_CSV_PATH


@dataclass
class MetricsRow:
    episode: int
    train_reward_mean: float
    policy_loss_mean: float
    value_loss_mean: float
    total_loss_mean: float
    win_rate_random: float
    draw_rate_random: float
    loss_rate_random: float
    win_rate_minimax: float
    draw_rate_minimax: float
    loss_rate_minimax: float


FIELD_NAMES = [
    "episode",
    "train_reward_mean",
    "policy_loss_mean",
    "value_loss_mean",
    "total_loss_mean",
    "win_rate_random",
    "draw_rate_random",
    "loss_rate_random",
    "win_rate_minimax",
    "draw_rate_minimax",
    "loss_rate_minimax",
]


@dataclass
class MetricsTracker:
    csv_path: Path = METRICS_CSV_PATH
    plot_path: Path = LEARNING_CURVES_PATH
    rows: list[MetricsRow] = field(default_factory=list)

    def record(self, row: MetricsRow) -> None:
        self.rows.append(row)

    def save_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(
                    {
                        "episode": row.episode,
                        "train_reward_mean": row.train_reward_mean,
                        "policy_loss_mean": row.policy_loss_mean,
                        "value_loss_mean": row.value_loss_mean,
                        "total_loss_mean": row.total_loss_mean,
                        "win_rate_random": row.win_rate_random,
                        "draw_rate_random": row.draw_rate_random,
                        "loss_rate_random": row.loss_rate_random,
                        "win_rate_minimax": row.win_rate_minimax,
                        "draw_rate_minimax": row.draw_rate_minimax,
                        "loss_rate_minimax": row.loss_rate_minimax,
                    }
                )

    def save_plot(self) -> None:
        if not self.rows:
            return

        episodes = [row.episode for row in self.rows]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Policy-Value MCTS Training Curves")

        axes[0, 0].plot(
            episodes,
            [row.win_rate_random for row in self.rows],
            label="win",
        )
        axes[0, 0].plot(
            episodes,
            [row.draw_rate_random for row in self.rows],
            label="draw",
        )
        axes[0, 0].plot(
            episodes,
            [row.loss_rate_random for row in self.rows],
            label="loss",
        )
        axes[0, 0].set_title("Model vs random")
        axes[0, 0].set_xlabel("Game")
        axes[0, 0].set_ylim(0.0, 1.0)
        axes[0, 0].legend()

        axes[0, 1].plot(
            episodes,
            [row.win_rate_minimax for row in self.rows],
            label="win",
        )
        axes[0, 1].plot(
            episodes,
            [row.draw_rate_minimax for row in self.rows],
            label="draw",
        )
        axes[0, 1].plot(
            episodes,
            [row.loss_rate_minimax for row in self.rows],
            label="loss",
        )
        axes[0, 1].set_title("Model vs minimax")
        axes[0, 1].set_xlabel("Game")
        axes[0, 1].set_ylim(0.0, 1.0)
        axes[0, 1].legend()

        axes[1, 0].plot(episodes, [row.policy_loss_mean for row in self.rows], label="policy")
        axes[1, 0].plot(episodes, [row.value_loss_mean for row in self.rows], label="value")
        axes[1, 0].set_title("Training losses")
        axes[1, 0].set_xlabel("Game")
        axes[1, 0].legend()

        axes[1, 1].plot(episodes, [row.total_loss_mean for row in self.rows])
        axes[1, 1].set_title("Total loss")
        axes[1, 1].set_xlabel("Game")

        fig.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
