"""Train a policy-value MCTS agent on tic-tac-toe."""

import numpy as np
import torch

from agent.policy_value_agent import PolicyValueAgent
from config import (
    BEST_MODEL_PATH,
    DEFAULT_CONFIG,
    MODEL_PATH,
    OUTPUT_DIR,
    TrainConfig,
    ensure_output_dir,
)
from eval.opponents import MinimaxOpponent, RandomOpponent, evaluate_agent
from training.expert import generate_expert_examples, sample_expert_examples
from training.metrics import MetricsRow, MetricsTracker


def _evaluate(
    agent: PolicyValueAgent,
    random_opponent: RandomOpponent,
    minimax_opponent: MinimaxOpponent,
    rng: np.random.Generator,
    *,
    games: int,
) -> tuple[float, float, float, float, float, float]:
    win_r, draw_r, loss_r = evaluate_agent(
        agent,
        random_opponent,
        games=games,
        rng=rng,
        mcts_simulations=agent.mcts_eval_simulations,
    )
    win_m, draw_m, loss_m = evaluate_agent(
        agent,
        minimax_opponent,
        games=games,
        rng=rng,
        mcts_simulations=agent.mcts_eval_simulations,
    )
    return win_r, draw_r, loss_r, win_m, draw_m, loss_m


def _record_eval(
    metrics: MetricsTracker,
    *,
    episode: int,
    agent: PolicyValueAgent,
    recent_rewards: list[float],
    rolling_window: int,
    random_opponent: RandomOpponent,
    minimax_opponent: MinimaxOpponent,
    rng: np.random.Generator,
    games: int,
) -> MetricsRow:
    reward_window = recent_rewards[-rolling_window:]
    policy_loss, value_loss, total_loss = agent.mean_recent_losses(rolling_window)
    win_r, draw_r, loss_r, win_m, draw_m, loss_m = _evaluate(
        agent,
        random_opponent,
        minimax_opponent,
        rng,
        games=games,
    )
    return MetricsRow(
        episode=episode,
        train_reward_mean=float(np.mean(reward_window)) if reward_window else 0.0,
        policy_loss_mean=policy_loss,
        value_loss_mean=value_loss,
        total_loss_mean=total_loss,
        win_rate_random=win_r,
        draw_rate_random=draw_r,
        loss_rate_random=loss_r,
        win_rate_minimax=win_m,
        draw_rate_minimax=draw_m,
        loss_rate_minimax=loss_m,
    )


def _print_row(row: MetricsRow) -> None:
    print(
        f"game={row.episode:4d} reward={row.train_reward_mean:.3f} "
        f"loss={row.total_loss_mean:.4f} "
        f"win_random={row.win_rate_random:.2f} draw_random={row.draw_rate_random:.2f} "
        f"loss_random={row.loss_rate_random:.2f} "
        f"draw_minimax={row.draw_rate_minimax:.2f} loss_minimax={row.loss_rate_minimax:.2f}",
        flush=True,
    )


def _score(row: MetricsRow) -> float:
    return row.draw_rate_minimax - row.loss_rate_minimax + 0.5 * row.win_rate_random


def _checkpoint_label(path, checkpoint: dict) -> str:
    episode = checkpoint.get("best_episode")
    score = checkpoint.get("best_score")
    details: list[str] = []
    if episode is not None:
        details.append(f"saved at game={episode}")
    if score is not None:
        details.append(f"score={float(score):.2f}")
    if details:
        return f"{path} ({', '.join(details)})"
    return str(path)


def train(config: TrainConfig = DEFAULT_CONFIG) -> None:
    ensure_output_dir()
    print(f"Writing outputs to {OUTPUT_DIR.resolve()}", flush=True)
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    random_opponent = RandomOpponent(rng)
    minimax_opponent = MinimaxOpponent()
    agent = PolicyValueAgent(
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        replay_size=config.replay_size,
        batch_size=config.batch_size,
        mcts_simulations=config.mcts_simulations,
        mcts_eval_simulations=config.mcts_eval_simulations,
        mcts_play_simulations=config.mcts_play_simulations,
        c_puct=config.c_puct,
        symmetry_augment=config.symmetry_augment,
        temperature_moves=config.temperature_moves,
        grad_clip=config.grad_clip,
        ema_decay=config.ema_decay,
        rng=rng,
    )
    metrics = MetricsTracker()
    recent_rewards: list[float] = []
    best_score = -2.0
    best_episode = 0
    stale_evals = 0

    if config.expert_sample_ratio <= 0.0:
        print("Skipping expert pretraining (expert_sample_ratio=0).", flush=True)
    else:
        all_expert_examples = generate_expert_examples()
        expert_examples = sample_expert_examples(
            all_expert_examples,
            config.expert_sample_ratio,
            rng,
        )
        print(
            f"Pretraining on {len(expert_examples)} of {len(all_expert_examples)} "
            f"expert positions ({config.expert_sample_ratio:.0%})...",
            flush=True,
        )
        agent.pretrain(
            expert_examples,
            epochs=config.pretrain_epochs,
            batch_size=config.pretrain_batch_size,
            learning_rate=config.pretrain_learning_rate,
            policy_weight=config.pretrain_policy_weight,
        )

    pretrain_row = _record_eval(
        metrics,
        episode=0,
        agent=agent,
        recent_rewards=recent_rewards,
        rolling_window=config.rolling_window,
        random_opponent=random_opponent,
        minimax_opponent=minimax_opponent,
        rng=rng,
        games=config.eval_games,
    )
    metrics.record(pretrain_row)
    _print_row(pretrain_row)
    metrics.save_csv()
    metrics.save_plot()
    best_score = _score(pretrain_row)
    best_episode = 0
    agent.save(BEST_MODEL_PATH, episode=best_episode, score=best_score)

    for episode in range(1, config.num_episodes + 1):
        if config.warmup_episodes > 0 and episode <= config.warmup_episodes:
            agent_player = 1 if episode % 2 == 0 else 2
            trajectory, board = agent.play_opponent_game(random_opponent, agent_player=agent_player)
        elif config.opponent_game_ratio > 0 and rng.random() < config.opponent_game_ratio:
            agent_player = 1 if episode % 2 == 0 else 2
            trajectory, board = agent.play_opponent_game(random_opponent, agent_player=agent_player)
        else:
            trajectory, board = agent.play_self_play_game()
        agent.store_game(trajectory, board)

        winner = board.winner()
        if winner == 1:
            recent_rewards.append(1.0)
        elif winner == 2:
            recent_rewards.append(-1.0)
        else:
            recent_rewards.append(0.0)

        for _ in range(config.train_steps_per_game):
            agent.train_step()

        if episode % config.eval_every == 0 or episode == config.num_episodes:
            row = _record_eval(
                metrics,
                episode=episode,
                agent=agent,
                recent_rewards=recent_rewards,
                rolling_window=config.rolling_window,
                random_opponent=random_opponent,
                minimax_opponent=minimax_opponent,
                rng=rng,
                games=config.eval_games,
            )
            metrics.record(row)
            score = _score(row)
            if score >= best_score:
                best_score = score
                best_episode = episode
                agent.save(BEST_MODEL_PATH, episode=best_episode, score=best_score)
                stale_evals = 0
            else:
                stale_evals += 1
                if config.revert_on_regression and BEST_MODEL_PATH.exists():
                    agent.reload_from_checkpoint(BEST_MODEL_PATH)
                    print(
                        f"Reverted to best checkpoint from game={best_episode} "
                        f"(score={best_score:.2f}) at {BEST_MODEL_PATH}",
                        flush=True,
                    )

            _print_row(row)

            metrics.save_csv()
            metrics.save_plot()

            if stale_evals >= config.early_stop_evals:
                print(
                    f"Early stop at game={episode} "
                    f"(best game={best_episode}, score={best_score:.2f})",
                    flush=True,
                )
                break

    if BEST_MODEL_PATH.exists():
        checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
        best_agent = PolicyValueAgent.load(BEST_MODEL_PATH, rng)
        best_agent.save(MODEL_PATH)
        source = _checkpoint_label(BEST_MODEL_PATH, checkpoint)
        print(f"Restored best checkpoint from {source} -> {MODEL_PATH}", flush=True)
    else:
        agent.save(MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}", flush=True)

    metrics.save_csv()
    metrics.save_plot()


def main() -> None:
    train()


if __name__ == "__main__":
    main()
