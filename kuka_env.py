import time

import os
import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data

from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Tuple, Any, SupportsFloat

from gymnasium.core import ObsType, RenderFrame, ActType


@dataclass
class WorldConfig:
    table_position = [0, 0, 0]
    table_orientation = [0, 0, 0]
    robo_position = [0, 0, 0]
    robo_orientation = [0, 0, 0]

    plane_urdf_path: str = "plane.urdf"
    table_urdf_path: str = ""
    robo_urdf_path: str = "kuka_iiwa/model.urdf"


class KukaEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 60
    }

    RENDER_WIDTH = 640
    RENDER_HEIGHT = 480

    def __init__(self,
                 urdf_root: str = pybullet_data.getDataPath(),
                 world_config: WorldConfig = WorldConfig,
                 is_enable_self_collision: bool = True,
                 renders: bool = False,
                 max_distance: float = 0.2,
                 dt: float = 0.01,
                 max_steps: int = 1000):

        self._dt = dt
        self._info = {}
        self._observation = {}
        self._last_action = None
        self._renders = renders
        self._urdf_root = urdf_root
        self._max_steps = max_steps
        self._world_config = world_config
        self._is_enable_self_collision = is_enable_self_collision

        self._max_distance = max_distance
        self._envStepCounter = 0

        self._steps_after_goal = 0  # Счетчик шагов после достижения цели
        self._goal_reached = False  # Флаг достижения цели

        self.__renders_setting()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
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
        # Генерация углов в доступном диапазоне
        # self.action_space = spaces.Box(low=joint_limits[0],
        #                                high=joint_limits[1],
        #                                shape=(self._numJoints,),
        #                                dtype=np.float32)

        # Генерация угла смещения
        self.action_space = spaces.Box(low=-0.01, high=0.01, shape=(self._numJoints,), dtype=np.float32)

        # Старое пространство наблюдений
        # Установка пространства наблюдений
        # self.observation_space = spaces.Dict(
        #     {
        #         "joint_positions": spaces.Box(low=joint_limits[0], high=joint_limits[1], shape=(self._numJoints,),
        #                                       dtype=np.float32),
        #         "end_effector_positions": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        #         "target_positions": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        #     }
        # )

        #     Новое пространство наблюдений
        # Установка пространства наблюдений
        low_limits = np.concatenate([
            joint_limits[0],                   # Минимальные значения для углов суставов
            [-np.inf, -np.inf, -np.inf],       # Минимальные значения для позиции конечного эффектора
            [-np.inf, -np.inf, -np.inf]        # Минимальные значения для целевой позиции
        ])
        high_limits = np.concatenate([
            joint_limits[1],                   # Максимальные значения для углов суставов
            [np.inf, np.inf, np.inf],          # Максимальные значения для позиции конечного эффектора
            [np.inf, np.inf, np.inf]           # Максимальные значения для целевой позиции
        ])

        self.observation_space = spaces.Box(low=low_limits, high=high_limits, dtype=np.float64)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._last_action = action
        # Изначальный метод управления роботом
        # for i in range(self._numJoints):
        #     p.setJointMotorControl2(bodyIndex=self._robotId,
        #                             jointIndex=i,
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPosition=action[i],
        #                             force=800)

        # Новый метод управления роботом
        # Получаем текущие углы суставов
        joint_states = p.getJointStates(self._robotId, range(self._numJoints))
        current_joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)

        # Получаем ограничения на углы суставов
        joint_limits_low, joint_limits_high = self.__get_joint_limits()

        # Обновляем углы суставов с учетом смещений
        new_joint_positions = []
        for i in range(self._numJoints):
            # Вычисляем новое значение угла
            new_angle = current_joint_positions[i] + action[i]

            # Ограничиваем новое значение угла пределами joint_limits
            new_angle = np.clip(new_angle, joint_limits_low[i], joint_limits_high[i])

            # Сохраняем новое значение угла
            new_joint_positions.append(new_angle)

            # Устанавливаем новое целевое положение для сустава
            p.setJointMotorControl2(bodyIndex=self._robotId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=new_angle,
                                    force=800)

        # ПРОДОЛЖЕНИЕ старого кода
        end_effector_pos = self.__get_end_effector_position()
        p.resetBasePositionAndOrientation(bodyUniqueId=self._end_effector_shaped_id,
                                          posObj=end_effector_pos,
                                          ornObj=[0, 0, 0, 1])
        self._envStepCounter += 1
        p.stepSimulation()

        if self._renders:
            time.sleep(self._dt)

        self._observation = self._get_observations()
        self._info = self._get_info()
        reward = self.__calculate_reward()
        truncated = False

        # Проверяем достижение цели
        distance = self.__get_distance()
        if distance <= self._max_distance and not self._goal_reached:
            self._goal_reached = True
            self._steps_after_goal = 0

        # Увеличиваем счетчик, если цель достигнута
        if self._goal_reached:
            self._steps_after_goal += 1

        # Условие завершения эпизода
        terminated = (self._envStepCounter >= self._max_steps or
                      (self._goal_reached and self._steps_after_goal >= 10))

        # terminated = (self._envStepCounter >= self._max_steps or self.__get_distance() <= self._max_distance)

        return self._observation, reward, terminated, truncated, self._info

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:

        super().reset(seed=seed, options=options)

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self._dt)

        self._generate_world_entities(plane_urdf_path=self._world_config.plane_urdf_path,
                                      table_urdf_path=self._world_config.table_urdf_path,
                                      robo_urdf_path=self._world_config.robo_urdf_path,
                                      base_table_position=self._world_config.table_position,
                                      base_table_orientation=self._world_config.table_orientation,
                                      base_robo_position=self._world_config.robo_position,
                                      base_robo_orientation=self._world_config.robo_orientation)
        self.__generate_target_positions()

        self._goal_reached = False
        self._steps_after_goal = 0
        self._envStepCounter = 0
        p.stepSimulation()

        self._observation = self._get_observations()
        self._info = self._get_info()

        return self._observation, self._info

    def render(self, mode="rgb_array") -> RenderFrame | list[RenderFrame] | None:
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = p.getBasePositionAndOrientation(self._robotId)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=1.6,
            yaw=-123,
            pitch=217.7,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.RENDER_WIDTH) / self.RENDER_HEIGHT,
            nearVal=0.1,
            farVal=100.0
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=self.RENDER_WIDTH,
            height=self.RENDER_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.RENDER_HEIGHT, self.RENDER_WIDTH, 4))

        return rgb_array[:, :, :3]

    def close(self) -> None:
        p.disconnect()

    # def _get_observations(self) -> dict:
    #     # Старый метод получения наблюдений
    #     joint_states = p.getJointStates(self._robotId, range(self._numJoints))
    #     joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
    #
    #     return {
    #         "joint_positions": joint_positions,
    #         "end_effector_positions": self.__get_end_effector_position(),
    #         "target_positions": self._target_positions,
    #     }

    def _get_observations(self) -> np.ndarray:
        # Новый метод получения наблюдений
        joint_states = p.getJointStates(self._robotId, range(self._numJoints))
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        end_effector_position = self.__get_end_effector_position()
        target_position = self._target_positions

        # Объединяем все данные в один массив
        observation = np.concatenate([
            joint_positions,         # Углы суставов
            end_effector_position,   # Позиция конечного эффектора
            target_position          # Целевая позиция
        ])

        return observation

    def _get_info(self) -> dict:
        return {
            "distance": self.__get_distance()
        }

    def _generate_world_entities(self,
                                 plane_urdf_path: str,
                                 table_urdf_path: str,
                                 robo_urdf_path: str,
                                 base_table_position: List[int],
                                 base_table_orientation: List[int],
                                 base_robo_position: List[int],
                                 base_robo_orientation: List[int]) -> None:

        if base_robo_position is None:
            base_robo_position = [0, 0, 0]

        flag = p.URDF_USE_SELF_COLLISION if self._is_enable_self_collision else 0
        self.planeid = p.loadURDF(fileName=plane_urdf_path)
        self._robotId = p.loadURDF(fileName=robo_urdf_path,
                                   basePosition=base_robo_position,
                                   baseOrientation=p.getQuaternionFromEuler(base_robo_orientation),
                                   useFixedBase=True,
                                   flags=flag)

        self._numJoints = p.getNumJoints(self._robotId)
        self._jointIndices = [p.getJointInfo(self._robotId, i)[0] for i in range(self._numJoints)]
        self._endEffectorIndex = self._numJoints - 1

        # Отслеживание конечного эффектора
        end_effector_visual_shaped_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                            radius=0.05,
                                                            rgbaColor=[0, 0, 1, 1])
        self._end_effector_shaped_id = p.createMultiBody(baseVisualShapeIndex=end_effector_visual_shaped_id,
                                                         basePosition=self.__get_end_effector_position())

        # TODO: Добавление стола
        # if table_urdf_path != "":
        #     self.tableid = p.loadURDF()

    def __calculate_reward(self) -> float:
        distance_to_target = self.__get_distance()

        # Основная награда/штраф за расстояние
        distance_reward = -distance_to_target * 10  # Меньший множитель для баланса

        # Награда за достижение цели
        goal_reward = 0
        if distance_to_target <= self._max_distance:
            goal_reward = 1000  # Большой бонус за достижение

        # Штраф за движение после достижения цели
        stabilization_penalty = 0
        if distance_to_target <= self._max_distance:
            joint_states = p.getJointStates(self._robotId, range(self._numJoints))
            joint_velocities = np.array([state[1] for state in joint_states])
            stabilization_penalty = -np.sum(np.square(joint_velocities)) * 100  # Квадратичный штраф

        # Штраф за большие действия (чтобы движения были плавнее)
        action_penalty = -np.sum(np.square(self._last_action)) * 0.1 if hasattr(self, '_last_action') else 0

        # Комбинированная награда
        reward = (distance_reward
                  + goal_reward
                  + stabilization_penalty
                  + action_penalty)

        return reward

    def __generate_target_positions(self) -> None:
        # self._target_positions = np.array([0.7, 0.0, 0.5], dtype=np.float32)

        # Генерация случайных координат в указанных диапазонах
        x = np.random.uniform(0.4, 0.7)  # Случайное значение по X от 0.5 до 0.7
        y = np.random.uniform(-0.5, 0.5)  # Случайное значение по Y от -0.5 до 0.5
        z = np.random.uniform(0.2, 0.7)  # Случайное значение по Z от 0.2 до 0.7
        self._target_positions = np.array([x, y, z], dtype=np.float32)

        target_visual_shaped_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                      radius=0.05,
                                                      rgbaColor=[0, 1, 0, 1])

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

    def __get_distance(self) -> np.float32:
        return np.linalg.norm(self.__get_end_effector_position() - self._target_positions, ord=1)

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