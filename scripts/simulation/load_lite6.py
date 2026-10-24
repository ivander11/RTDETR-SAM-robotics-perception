"""Load and inspect the standalone Lite6 model (headless by default)."""

import argparse
import math
from pathlib import Path
import time

import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf-path", type=Path,
                        default=PROJECT_ROOT / "ufactory_lite6/urdf/lite6.urdf")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--steps", type=int, default=2400)
    args = parser.parse_args()
    model = args.urdf_path.resolve()
    if not model.is_file():
        parser.error(f"Missing {model}; run export_lite6.py first.")
    client = p.connect(p.GUI if args.gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect to PyBullet.")
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(1 / 240, physicsClientId=client)
        robot = p.loadURDF(str(model), useFixedBase=True, physicsClientId=client)
        joints = range(p.getNumJoints(robot, physicsClientId=client))
        assert len(joints) == 6, "Expected six joints."
        for joint in joints:
            info = p.getJointInfo(robot, joint, physicsClientId=client)
            assert info[2] == p.JOINT_REVOLUTE
            print(f"{joint}: {info[1].decode()} -> {info[12].decode()}")
        p.setJointMotorControlArray(robot, list(joints), p.POSITION_CONTROL,
                                    targetPositions=[0] * 6,
                                    forces=[p.getJointInfo(robot, j, physicsClientId=client)[10]
                                            for j in joints], physicsClientId=client)
        if args.gui:
            p.resetDebugVisualizerCamera(1.2, 45, -25, [0, 0, 0.3], physicsClientId=client)
        for _ in range(args.steps):
            if not p.isConnected(client):
                break
            p.stepSimulation(physicsClientId=client)
            if args.gui:
                time.sleep(1 / 240)
        if p.isConnected(client):
            states = p.getJointStates(robot, list(joints), physicsClientId=client)
            assert all(math.isfinite(v) for state in states for v in state[:2])
            print("PASS: model loaded and simulation remained finite.")
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()
