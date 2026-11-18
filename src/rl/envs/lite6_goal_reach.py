"""Goal-conditioned wrapper around the Lite6 Cartesian reaching task."""

from typing import Any, Literal

from gymnasium import spaces
import numpy as np

from src.control.lite6_env.reach_env import Lite6ReachEnv


RewardType = Literal["sparse", "dense", "hybrid"]


class Lite6GoalReachEnv(Lite6ReachEnv):
    """Expose Lite6 reaching in the dictionary format required by SB3 HER.

    The six-value ``observation`` contains end-effector position and linear
    velocity. ``achieved_goal`` is the current end-effector position, and
    ``desired_goal`` is the sampled target position.
    """

    def __init__(self, reward_type: RewardType = "hybrid", **kwargs: Any) -> None:
        if reward_type not in ("sparse", "dense", "hybrid"):
            raise ValueError("reward_type must be sparse, dense, or hybrid")
        kwargs.setdefault("action_scale", 0.025)
        super().__init__(**kwargs)
        self.reward_type = reward_type
        unbounded = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),
                    dtype=np.float32,
                ),
                "achieved_goal": unbounded,
                "desired_goal": unbounded,
            }
        )

    def _get_observation(self) -> dict[str, np.ndarray]:
        flat_observation = super()._get_observation()
        return {
            "observation": flat_observation[:6].copy(),
            "achieved_goal": flat_observation[:3].copy(),
            "desired_goal": flat_observation[6:9].copy(),
        }

    def _distance_to_goal(self, observation: dict[str, np.ndarray]) -> float:
        return float(
            np.linalg.norm(
                observation["achieved_goal"] - observation["desired_goal"]
            )
        )

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict | np.ndarray | None,
    ) -> np.ndarray:
        """Compute scalar or vectorized rewards for normal steps and HER."""
        del info
        distance = np.linalg.norm(
            np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1
        )
        if self.reward_type == "sparse":
            reward = -(distance > self.success_threshold).astype(np.float32)
        elif self.reward_type == "dense":
            reward = -distance
        else:
            success_bonus = (distance <= self.success_threshold).astype(np.float32)
            reward = -distance + success_bonus
        return np.asarray(reward, dtype=np.float32)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        observation, _, terminated, truncated, info = super().step(action)
        reward = self.compute_reward(
            observation["achieved_goal"], observation["desired_goal"], info
        )
        distance = float(info["distance"])
        info.update(
            {
                "reward_sparse": float(
                    -(distance > self.success_threshold)
                ),
                "reward_dense": -distance,
                "reward_hybrid": -distance
                + float(distance <= self.success_threshold),
            }
        )
        return observation, float(reward), terminated, truncated, info
