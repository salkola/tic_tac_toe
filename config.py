"""Hyperparameters, paths, and seeds."""

from dataclasses import dataclass
from pathlib import Path


def _find_project_root() -> Path:
    """Resolve repo root even when config.py is imported from site-packages."""
    candidates = [Path(__file__).resolve().parent, Path.cwd(), *Path.cwd().parents]
    seen: set[Path] = set()
    for start in candidates:
        path = start.resolve()
        for _ in range(6):
            if path in seen:
                break
            seen.add(path)
            if (path / "pyproject.toml").is_file():
                return path
            if path.parent == path:
                break
            path = path.parent
    return Path(__file__).resolve().parent


PROJECT_ROOT = _find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "model.pt"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
METRICS_CSV_PATH = OUTPUT_DIR / "metrics.csv"
LEARNING_CURVES_PATH = OUTPUT_DIR / "learning_curves.png"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    num_episodes: int = 3000
    pretrain_epochs: int = 200
    pretrain_policy_weight: float = 2.0
    pretrain_batch_size: int = 32
    pretrain_learning_rate: float = 1e-3
    expert_sample_ratio: float = 0.0
    mcts_simulations: int = 15
    mcts_eval_simulations: int = 0
    mcts_play_simulations: int = 0
    c_puct: float = 1.25
    replay_size: int = 1_000
    batch_size: int = 32
    learning_rate: float = 1e-3
    train_steps_per_game: int = 1
    temperature_moves: int = 2
    symmetry_augment: bool = True
    warmup_episodes: int = 0
    opponent_game_ratio: float = 0.1
    eval_every: int = 100
    eval_games: int = 100
    rolling_window: int = 100
    hidden_dim: int = 128
    dropout: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    early_stop_evals: int = 10000
    revert_on_regression: bool = False


DEFAULT_CONFIG = TrainConfig()


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
