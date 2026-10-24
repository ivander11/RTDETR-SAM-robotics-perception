"""A small coordinate-based Lite6 reaching environment."""

from pathlib import Path
import time
from typing import Literal

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF = PROJECT_ROOT / "ufactory_lite6/urdf/lite6.urdf"


class Lite6ReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Move the Lite6 end effector to a 3-D target coordinate.

    Actions are normalized Cartesian displacements ``[dx, dy, dz]``. The
    environment scales each component by ``action_scale`` metres, converts the
    resulting end-effector target to six joint targets with inverse kinematics,
    and advances the position-controlled robot in PyBullet.

    Observations contain end-effector position, linear velocity, and goal
    position (nine float32 values). This first baseline uses a dense reward:
    the negative Euclidean distance from the end effector to the goal.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        goal_mode: Literal["fixed", "random"] = "fixed",
        urdf_path: str | Path | None = None,
        action_scale: float = 0.02,
        success_threshold: float = 0.01,
        max_episode_steps: int = 50,
        physics_steps_per_action: int = 24,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if goal_mode not in ("fixed", "random"):
            raise ValueError("goal_mode must be 'fixed' or 'random'")
        if action_scale <= 0 or success_threshold <= 0:
            raise ValueError("action_scale and success_threshold must be positive")
        if max_episode_steps <= 0 or physics_steps_per_action <= 0:
            raise ValueError("episode and physics step counts must be positive")

        self.render_mode = render_mode
        self.goal_mode = goal_mode
        self.urdf_path = Path(urdf_path or DEFAULT_URDF).resolve()
        self.action_scale = float(action_scale)
        self.success_threshold = float(success_threshold)
        self.max_episode_steps = int(max_episode_steps)
        self.physics_steps_per_action = int(physics_steps_per_action)
        self.physics_timestep = 1.0 / 240.0
        self.step_count = 0

        self.neutral_joint_positions = np.zeros(6, dtype=np.float64)
        self.fixed_goal = np.array([0.137, 0.0, 0.204], dtype=np.float32)
        self.random_goal_low = np.array([0.11, -0.05, 0.18], dtype=np.float32)
        self.random_goal_high = np.array([0.18, 0.05, 0.24], dtype=np.float32)
        self.workspace_low = np.array([0.02, -0.25, 0.05], dtype=np.float64)
        self.workspace_high = np.array([0.35, 0.25, 0.42], dtype=np.float64)
        self.goal = self.fixed_goal.copy()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        connection_mode = p.GUI if render_mode == "human" else p.DIRECT
        self.client_id = p.connect(connection_mode)
        if self.client_id < 0:
            raise RuntimeError("Could not connect to PyBullet")

        try:
            if not self.urdf_path.is_file():
                raise FileNotFoundError(f"Lite6 URDF not found: {self.urdf_path}")
            p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
            p.setTimeStep(self.physics_timestep, physicsClientId=self.client_id)
            self.robot_id = p.loadURDF(
                str(self.urdf_path),
                useFixedBase=True,
                physicsClientId=self.client_id,
            )
            self.joint_indices = self._find_revolute_joints()
            if len(self.joint_indices) != 6:
                raise RuntimeError(
                    f"Expected 6 revolute joints, found {len(self.joint_indices)}"
                )
            self.ee_link_index = self.joint_indices[-1]
            joint_info = [
                p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)
                for index in self.joint_indices
            ]
            self.joint_lower_limits = np.array([info[8] for info in joint_info])
            self.joint_upper_limits = np.array([info[9] for info in joint_info])
            self.joint_ranges = self.joint_upper_limits - self.joint_lower_limits
            self.joint_forces = np.array(
                [max(float(info[10]), 1.0) for info in joint_info]
            )
            self.goal_marker_id = self._create_goal_marker()
            if self.render_mode == "human":
                p.resetDebugVisualizerCamera(
                    1.0, 45, -25, [0.1, 0.0, 0.2], physicsClientId=self.client_id
                )
        except Exception:
            p.disconnect(self.client_id)
            self.client_id = -1
            raise

    def _find_revolute_joints(self) -> list[int]:
        return [
            index
            for index in range(
                p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
            )
            if p.getJointInfo(
                self.robot_id, index, physicsClientId=self.client_id
            )[2]
            == p.JOINT_REVOLUTE
        ]

    def _create_goal_marker(self) -> int:
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.success_threshold,
            rgbaColor=[0.1, 0.9, 0.1, 0.7],
            physicsClientId=self.client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.goal.tolist(),
            physicsClientId=self.client_id,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        del options  # Reserved by the Gymnasium API for future reset options.
        self.step_count = 0

        for joint_index, joint_position in zip(
            self.joint_indices, self.neutral_joint_positions
        ):
            p.resetJointState(
                self.robot_id,
                joint_index,
                targetValue=float(joint_position),
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )
        self._command_joint_positions(self.neutral_joint_positions)

        if self.goal_mode == "fixed":
            self.goal = self.fixed_goal.copy()
        else:
            self.goal = self.np_random.uniform(
                self.random_goal_low, self.random_goal_high
            ).astype(np.float32)
        p.resetBasePositionAndOrientation(
            self.goal_marker_id,
            self.goal.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )

        observation = self._get_observation()
        distance = self._distance_to_goal(observation)
        return observation, {"distance": distance, "is_success": False}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ee_position, _ = self._get_ee_state()
        target_position = np.clip(
            ee_position + action.astype(np.float64) * self.action_scale,
            self.workspace_low,
            self.workspace_high,
        )
        current_joints = np.array(
            [
                p.getJointState(
                    self.robot_id, index, physicsClientId=self.client_id
                )[0]
                for index in self.joint_indices
            ]
        )
        ik_solution = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            targetPosition=target_position.tolist(),
            lowerLimits=self.joint_lower_limits.tolist(),
            upperLimits=self.joint_upper_limits.tolist(),
            jointRanges=self.joint_ranges.tolist(),
            restPoses=current_joints.tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self.client_id,
        )
        self._command_joint_positions(np.asarray(ik_solution[:6]))

        for _ in range(self.physics_steps_per_action):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.render_mode == "human":
                time.sleep(self.physics_timestep)

        self.step_count += 1
        observation = self._get_observation()
        distance = self._distance_to_goal(observation)
        success = distance <= self.success_threshold
        terminated = bool(success)
        truncated = bool(
            self.step_count >= self.max_episode_steps and not terminated
        )
        reward = -distance
        info = {
            "distance": distance,
            "is_success": bool(success),
            "step_count": self.step_count,
        }
        return observation, float(reward), terminated, truncated, info

    def _command_joint_positions(self, target_positions: np.ndarray) -> None:
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.POSITION_CONTROL,
            targetPositions=np.asarray(target_positions, dtype=float).tolist(),
            forces=self.joint_forces.tolist(),
            physicsClientId=self.client_id,
        )

    def _get_ee_state(self) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return np.asarray(state[4]), np.asarray(state[6])

    def _get_observation(self) -> np.ndarray:
        ee_position, ee_velocity = self._get_ee_state()
        return np.concatenate((ee_position, ee_velocity, self.goal)).astype(np.float32)

    def _distance_to_goal(self, observation: np.ndarray) -> float:
        return float(np.linalg.norm(observation[:3] - observation[6:9]))

    def render(self) -> None:
        if self.render_mode != "human":
            return

    def close(self) -> None:
        if self.client_id >= 0 and p.isConnected(self.client_id):
            p.disconnect(self.client_id)
        self.client_id = -1
