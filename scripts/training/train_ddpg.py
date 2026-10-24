"""Train Stable-Baselines3 DDPG on the Lite6 reaching environment."""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.control.lite6_env import Lite6ReachEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--goal-mode", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name")
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--save-replay-buffer", action="store_true")
    return parser.parse_args()


def make_env(goal_mode: str, seed: int, monitor_path: Path) -> Monitor:
    env = Lite6ReachEnv(goal_mode=goal_mode)
    env.reset(seed=seed)
    return Monitor(env, filename=str(monitor_path), info_keywords=("is_success", "distance"))


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_run_config(args: argparse.Namespace, env: Lite6ReachEnv) -> dict:
    return {
        "arguments": vars(args),
        "environment": {
            "class": type(env).__name__,
            "urdf_path": str(env.urdf_path.relative_to(PROJECT_ROOT)),
            "action_scale_m": env.action_scale,
            "success_threshold_m": env.success_threshold,
            "max_episode_steps": env.max_episode_steps,
            "physics_steps_per_action": env.physics_steps_per_action,
            "physics_timestep_s": env.physics_timestep,
        },
        "software": {
            "python": platform.python_version(),
            "gymnasium": version("gymnasium"),
            "pybullet": version("pybullet"),
            "stable_baselines3": version("stable-baselines3"),
        },
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "has_tracked_changes": bool(
                git_value("status", "--porcelain", "--untracked-files=no")
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0 or args.learning_starts < 0:
        raise ValueError("timesteps must be positive and learning-starts non-negative")
    if args.action_noise < 0:
        raise ValueError("action-noise cannot be negative")
    if min(
        args.buffer_size,
        args.batch_size,
        args.checkpoint_freq,
        args.eval_freq,
        args.eval_episodes,
        args.log_interval,
    ) <= 0:
        raise ValueError("buffer, batch, checkpoint, and evaluation values must be positive")

    run_name = args.run_name or datetime.now().strftime(
        f"ddpg_lite6_{args.goal_mode}_%Y%m%d_%H%M%S"
    )
    run_dir = PROJECT_ROOT / "runs/ddpg" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    checkpoints_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
    checkpoints_dir.mkdir(parents=True)
    best_model_dir.mkdir()

    train_env: Monitor | None = None
    eval_env: Monitor | None = None
    try:
        train_env = make_env(
            args.goal_mode, args.seed, run_dir / "train_monitor.csv"
        )
        eval_env = make_env(
            args.goal_mode, args.seed + 10_000, run_dir / "eval_monitor.csv"
        )
        config = build_run_config(args, train_env.unwrapped)
        config["arguments"]["run_name"] = run_name
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        action_noise = NormalActionNoise(
            mean=np.zeros(3), sigma=args.action_noise * np.ones(3)
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoints_dir),
            name_prefix="ddpg_lite6",
            save_replay_buffer=args.save_replay_buffer,
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(run_dir / "evaluation"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        )
        model = DDPG(
            "MlpPolicy",
            train_env,
            learning_rate=1e-3,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            action_noise=action_noise,
            policy_kwargs={"net_arch": [256, 256]},
            seed=args.seed,
            verbose=1,
            device=args.device,
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            log_interval=args.log_interval,
        )
        final_path = run_dir / "final_model"
        model.save(final_path)
        print(f"Saved final model: {final_path}.zip")
        print(f"Run artifacts: {run_dir}")
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
