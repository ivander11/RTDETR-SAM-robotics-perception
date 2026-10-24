import time
import pybullet as p
import pybullet_data

client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 1])

try:
    for _ in range(1200):
        p.stepSimulation()
        time.sleep(1 / 240)

finally:
    p.disconnect()
