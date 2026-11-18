"""Train an archive-inspired SB3 DDPG+HER Lite6 reaching policy."""

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.envs import Lite6GoalReachEnv  # noqa: E402


SOURCE_FILES = (
    Path("scripts/rl/train_ddpg_her.py"),
    Path("scripts/rl/evaluate_ddpg_goal.py"),
    Path("src/rl/envs/lite6_goal_reach.py"),
    Path("src/control/lite6_env/reach_env.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument(
        "--reward-type",
        choices=("sparse", "dense", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    return parser.parse_args()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def make_env(seed: int, reward_type: str, monitor_path: Path) -> Monitor:
    env = Lite6GoalReachEnv(goal_mode="random", reward_type=reward_type)
    env.reset(seed=seed)
    return Monitor(
        env,
        filename=str(monitor_path),
        info_keywords=("is_success", "distance"),
    )


def snapshot_sources(run_dir: Path) -> dict[str, str]:
    """Copy and hash the exact local source used by an uncommitted run."""
    snapshot_dir = run_dir / "source_snapshot"
    hashes: dict[str, str] = {}
    for relative_path in SOURCE_FILES:
        source = PROJECT_ROOT / relative_path
        destination = snapshot_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative_path)] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def build_config(
    args: argparse.Namespace,
    run_name: str,
    env: Lite6GoalReachEnv,
    source_hashes: dict[str, str],
) -> dict:
    return {
        "experiment": {
            "name": "archive_inspired_ddpg_her",
            "run_name": run_name,
            "claim_boundary": (
                "Uses archived algorithm settings with the new official URDF and "
                "validated reachable target region; it is not an exact reproduction."
            ),
        },
        "arguments": vars(args),
        "environment": {
            "class": type(env).__name__,
            "urdf_path": str(env.urdf_path.relative_to(PROJECT_ROOT)),
            "goal_mode": env.goal_mode,
            "reward_type": env.reward_type,
            "action_scale_m": env.action_scale,
            "success_threshold_m": env.success_threshold,
            "max_episode_steps": env.max_episode_steps,
            "physics_steps_per_action": env.physics_steps_per_action,
            "random_goal_low_m": env.random_goal_low.tolist(),
            "random_goal_high_m": env.random_goal_high.tolist(),
        },
        "algorithm": {
            "policy": "MultiInputPolicy",
            "batch_size": 2048,
            "buffer_size": 1_000_000,
            "gamma": 0.95,
            "learning_rate": 0.001,
            "learning_starts": 100,
            "net_arch": [512, 512, 512],
            "n_critics": 2,
            "tau": 0.05,
            "action_noise": None,
            "replay_buffer": "HerReplayBuffer",
            "goal_selection_strategy": "future",
            "n_sampled_goal": 4,
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
            "has_worktree_changes": bool(git_value("status", "--porcelain")),
        },
        "source_sha256": source_hashes,
    }


def main() -> None:
    args = parse_args()
    if min(
        args.timesteps,
        args.checkpoint_freq,
        args.eval_freq,
        args.eval_episodes,
        args.log_interval,
    ) <= 0:
        raise ValueError("training, checkpoint, and evaluation values must be positive")

    run_name = args.run_name or datetime.now().strftime(
        f"ddpg_her_lite6_{args.reward_type}_%Y%m%d_%H%M%S"
    )
    run_dir = PROJECT_ROOT / "runs/ddpg_her" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    checkpoints_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
    checkpoints_dir.mkdir(parents=True)
    best_model_dir.mkdir()

    train_env: Monitor | None = None
    eval_env: Monitor | None = None
    try:
        train_env = make_env(args.seed, args.reward_type, run_dir / "train_monitor.csv")
        eval_env = make_env(
            args.seed + 10_000,
            args.reward_type,
            run_dir / "eval_monitor.csv",
        )
        source_hashes = snapshot_sources(run_dir)
        config = build_config(
            args,
            run_name,
            train_env.unwrapped,
            source_hashes,
        )
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoints_dir),
            name_prefix="ddpg_her_lite6",
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
            "MultiInputPolicy",
            train_env,
            learning_rate=0.001,
            buffer_size=1_000_000,
            learning_starts=100,
            batch_size=2048,
            tau=0.05,
            gamma=0.95,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "goal_selection_strategy": "future",
                "n_sampled_goal": 4,
            },
            policy_kwargs={"net_arch": [512, 512, 512], "n_critics": 2},
            action_noise=None,
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
