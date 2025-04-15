import time

import os
import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data

from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

from gymnasium.core import ObsType


@dataclass
class WorldConfig:
    plane_urdf_path: str = "plane.urdf"
    table_urdf_path: str = ""
    robo_urdf_path: str = "kuka_iiwa/model.urdf"

    table_position = [0, 0, 0]
    table_orientation = [0, 0, 0]
    robo_position = [0,0,0]
    robo_orientation = [0,0,0]




class KukaEnv(gym.Env):

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 60
    }

    def __init__(self,
                 urdf_root: str = pybullet_data.getDataPath(),
                 world_config: WorldConfig = WorldConfig,
                 is_enable_self_collision: bool = True,
                 renders: bool = False,
                 dt: float = 0.01,
                 max_steps: int = 1000):

        self._dt = dt
        self._info = {}
        self._observation = {}
        self._renders = renders
        self._urdf_root = urdf_root
        self._max_steps = max_steps
        self._world_config = world_config
        self._is_enable_self_collision = is_enable_self_collision

        self.terminated = 0
        self._envStepCounter = 0

        self.__renders_setting()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0, -9.8)
        p.setTimeStep(self._dt)

        # Генерация первоначальных объектов
        self._generate_world_entities(plane_urdf_path=self._world_config.plane_urdf_path,
                                      table_urdf_path=self._world_config.table_urdf_path,
                                      robo_urdf_path=self._world_config.robo_urdf_path,
                                      base_table_position=self._world_config.table_position,
                                      base_table_orientation=self._world_config.table_orientation,
                                      base_robo_position=self._world_config.robo_position,
                                      base_robo_orientation=self._world_config.robo_orientation)
    #     Получение ограничений на поворот суставов
        joint_limits = self.__get_joint_limits()

        # Установка пространсво действий
        self.action_space = spaces.Box(low=joint_limits[0],
                                       high=joint_limits[1],
                                       shape=(self._numJoints,),
                                       dtype=np.float32)

        # Установка пространства наблюдений
        self.observation_space = spaces.Dict(
            {
                "joint_positions": spaces.Box(low=joint_limits[0], high=joint_limits[1], shape=(self._numJoints,), dtype=np.float32),
                "end_effector_positions": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "target_positions": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            }
        )

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:

        super().reset(seed=seed, options=options)

        self.terminated = 0
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0, -9.8)
        p.setTimeStep(self._dt)

        self._generate_world_entities(plane_urdf_path=self._world_config.plane_urdf_path,
                                      table_urdf_path=self._world_config.table_urdf_path,
                                      robo_urdf_path=self._world_config.robo_urdf_path,
                                      base_table_position=self._world_config.table_position,
                                      base_table_orientation=self._world_config.table_orientation,
                                      base_robo_position=self._world_config.robo_position,
                                      base_robo_orientation=self._world_config.robo_orientation)
        self.__generate_target_positions()

        self._envStepCounter = 0
        p.stepSimulation()

        self._observation = self._get_observations()
        self._info = self._get_info()

        return self._observation, self._info

    def close(self):
        p.disconnect()

    def _get_observations(self):
        joint_states = p.getJointStates(self._robotId, range(self._numJoints))
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)

        return {
            "joint_positions": joint_positions,
            "end_effector_positions": self.__get_end_effector_position(),
            "target_positions": self._target_positions,
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self.__get_end_effector_position() - self._target_positions, ord=1)
        }

    def _generate_world_entities(self,
                                 plane_urdf_path: str,
                                 table_urdf_path: str,
                                 robo_urdf_path: str,
                                 base_table_position:List[int],
                                 base_table_orientation: List[int],
                                 base_robo_position: List[int],
                                 base_robo_orientation: List[int]) -> None:

        if base_robo_position is None:
            base_robo_position = [0, 0, 0]

        flag = p.URDF_USE_SELF_COLLISION if self._is_enable_self_collision else 0
        p.loadURDF(fileName=plane_urdf_path)
        self._robotId = p.loadURDF(fileName=robo_urdf_path,
                                  basePosition=base_robo_position,
                                  baseOrientation=p.getQuaternionFromEuler(base_robo_orientation),
                                  useFixedBase=True,
                                  flags=flag)

        self._numJoints = p.getNumJoints(self._robotId)
        self._jointIndices = [p.getJointInfo(self._robotId, i)[0] for i in range(self._numJoints)]
        self._endEffectorIndex = self._numJoints - 1

        # Отслеживание коннечного эффектора
        end_effector_visual_shaped_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                            radius=0.05,
                                                            rgbaColor=[0,0,1,1])
        self._end_effector_shaped_id = p.createMultiBody(baseVisualShapeIndex=end_effector_visual_shaped_id,
                                                         basePosition=self.__get_end_effector_position())


        # TODO: Добавление стола
        # if table_urdf_path != "":
        #     self.tableid = p.loadURDF()

    def __generate_target_positions(self):
        self._target_positions = np.array([0.7, 0.0, 0.5], dtype=np.float32)
        target_visual_shaped_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                      radius=0.05,
                                                      rgbaColor=[0,1,0,1])

        self._targetId = p.createMultiBody(baseVisualShapeIndex=target_visual_shaped_id,
                                           basePosition=self._target_positions)

    def __get_end_effector_position(self) -> np.ndarray:
        link_state = p.getLinkState(self._robotId, self._endEffectorIndex)
        return np.array(link_state[0], dtype=np.float32)

    def __get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        low_joints = []
        high_joints = []
        for jointIndex in range(self._numJoints):
            joint_info = p.getJointInfo(self._robotId, jointIndex)
            joint_type = joint_info[2]

            if joint_type == p.JOINT_REVOLUTE:
                low_joints.append(joint_info[8])
                high_joints.append(joint_info[9])

        return np.array(low_joints, dtype=np.float32), np.array(high_joints, dtype=np.float32)

    def __renders_setting(self) -> None:
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=1.6,
                                         cameraYaw=-123,
                                         cameraPitch=217.7,
                                         cameraTargetPosition=[0.13, -0.09, 0.54])
        else:
            p.connect(p.DIRECT)


if __name__ == "__main__":
    env = KukaEnv(renders=True,
                  is_enable_self_collision=False)

    observation, info = env.reset()

    print(observation)
    print(info)

    # time.sleep(10)