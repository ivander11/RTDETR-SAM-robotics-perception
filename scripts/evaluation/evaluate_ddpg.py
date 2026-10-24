"""Evaluate a trained Lite6 DDPG model with success and distance metrics."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import DDPG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.control.lite6_env import Lite6ReachEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--goal-mode", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    env = Lite6ReachEnv(
        render_mode="human" if args.gui else None,
        goal_mode=args.goal_mode,
    )
    model = DDPG.load(args.model_path, env=env, device=args.device)
    returns: list[float] = []
    final_distances: list[float] = []
    successes: list[bool] = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            episode_return = 0.0
            info = {"distance": float("nan"), "is_success": False}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
            returns.append(episode_return)
            final_distances.append(float(info["distance"]))
            successes.append(bool(info["is_success"]))
    finally:
        env.close()

    metrics = {
        "episodes": args.episodes,
        "goal_mode": args.goal_mode,
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "mean_final_distance_m": float(np.mean(final_distances)),
        "final_distance_std_m": float(np.std(final_distances)),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
