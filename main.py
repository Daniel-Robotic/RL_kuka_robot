import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym

from gymnasium import spaces
from typing import Optional, Tuple, List


physicalClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0, -9.8)

planeId = p.loadURDF("plane.urdf")
start_pose = [0, 0, 1]
start_orientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("kuka_iiwa/model.urdf", start_pose, start_orientation, useFixedBase=True)

for i in range(1_000_000):
    p.stepSimulation()
    time.sleep(1/100)

robo_pose, robo_orn = p.getBasePositionAndOrientation(robotId)
print(robo_pose, robo_orn)

p.disconnect()
