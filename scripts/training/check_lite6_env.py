"""Validate the Lite6 environment and run a short random-action rollout."""

from pathlib import Path
import sys

import numpy as np
from stable_baselines3.common.env_checker import check_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.control.lite6_env import Lite6ReachEnv  # noqa: E402


def main() -> None:
    env = Lite6ReachEnv(goal_mode="fixed")
    try:
        check_env(env, warn=True)
        observation, info = env.reset(seed=0)
        start_distance = info["distance"]
        direction = observation[6:9] - observation[:3]
        action_toward_goal = direction / np.max(np.abs(direction))
        for _ in range(5):
            observation, _, terminated, truncated, info = env.step(
                action_toward_goal
            )
            if terminated or truncated:
                break
        assert info["distance"] < start_distance, (
            "An action toward the fixed goal should reduce the distance"
        )

        observation, info = env.reset(seed=0)
        episode_finished = False
        for _ in range(100):
            observation, _, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                episode_finished = True
                observation, info = env.reset()
        assert observation.shape == (9,)
        assert observation.dtype == np.float32
        assert np.isfinite(observation).all()
        assert episode_finished, "Expected the 50-step episode limit to be reached"
        print("PASS: SB3 interface, Cartesian motion, and episode limit checks.")
        print(f"Last distance: {info['distance']:.4f} m")
    finally:
        env.close()

    random_env = Lite6ReachEnv(goal_mode="random")
    try:
        first, _ = random_env.reset(seed=123)
        repeated, _ = random_env.reset(seed=123)
        different, _ = random_env.reset(seed=124)
        assert np.array_equal(first[6:9], repeated[6:9])
        assert not np.array_equal(first[6:9], different[6:9])
        print("PASS: random goals are reproducible from the reset seed.")
    finally:
        random_env.close()


if __name__ == "__main__":
    main()
