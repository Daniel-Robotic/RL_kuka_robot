import os
import time
import numpy as np
import pybullet as p
import gymnasium as gym

from gymnasium import spaces
from typing import Tuple, Any, SupportsFloat
from gymnasium.core import RenderFrame, ObsType, ActType

from shemas import EnvConfig


class KukaEnv(gym.Env):
    def __init__(
            self,
            renders: bool = False,
            **kwargs
    ):

        super(KukaEnv, self).__init__()

        self._renders = renders
        self._config = EnvConfig(**kwargs)

        # Spawn components
        self._robot_id = None
        self._table_id = None
        self._target_id = None
        self._capture_object_id = None

        # Robot params
        self._num_joints = 7
        self._joint_indices = []
        self._target_position = []
        self._target_orientation = []
        self._end_effector_index = None

        # Observation params
        self._angel_error = 0
        self._distance_target_to_ee_pos = 0
        self._artificial_potential_field = 0
        self._step_counter = 0
        self._max_duration_seconds = self._config.max_duration_seconds  # максимум 30 секунд симуляции

        # Rewards params
        self._K_D = 10      # вес расстояния
        self._K_A = 100     # вес ориентации
        self._K1 = 2000     # штраф за сближение звеньев
        self._K2 = 20
        self._last_U = 0    # потенциал на предыдущем шаге

        # Rendering configurations
        self._render_setting()

        # Action space
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(self._num_joints, ), dtype=np.float32)

        # Observation space
        # theta joint angels = 7
        # ee_pos = 3
        # ee_orn = 4
        # target_pos = 3
        # target_orn = 4
        # distance_target_to_ee_pos = 1
        # angel_error = 1
        # U = 1
        # SUM = 24
        shape_obs = 24
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(shape_obs, ), dtype=np.float32)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        # Применить действие к суставам
        for joint_id in range(self._num_joints):
            joint_state = p.getJointState(self._robot_id, joint_id)
            target_pos = joint_state[0] + action[joint_id]
            p.setJointMotorControl2(
                self._robot_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=target_pos
            )
        p.stepSimulation()
        time.sleep(self._config.time_step) if self._renders else None

        # Вычислить вознаграждение
        reward = self._calc_reward()

        # Обновить метрики
        self._calc_distance()
        self._calc_angel_error()

        # Визуализация
        if self._renders:
            ee_pos, ee_orn = self._get_end_effector_position()
            self._draw_axes(ee_pos, ee_orn, length=0.15)
            self._draw_axes(self._target_position, self._target_orientation, length=0.15)

        # Условия завершения

        terminated = (self._distance_target_to_ee_pos < self._config.max_check_distance and
                      self._angel_error < np.deg2rad(self._config.max_check_angle_deg))

        sim_time = self._step_counter * self._config.time_step
        truncated = sim_time >= self._max_duration_seconds

        if terminated or truncated:
            self._step_counter = 0  # сброс счётчика эпизода

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        p.resetSimulation()

        self._load_environment()
        self._calc_distance()
        self._calc_angel_error()
        self._artificial_potential_field = 0

        for joint_id in range(self._num_joints):
            p.resetJointState(self._robot_id, joint_id, targetValue=0)

        return self._get_obs(), self._get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self) -> None:
        p.disconnect()

    def _render_setting(self) -> None:
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            p.connect(p.GUI) if cid < 0 else None
            p.resetDebugVisualizerCamera(**self._config.visualizer_camera.to_dict())
        else:
            p.connect(p.DIRECT)

    def _draw_axes(self, position, orientation, length=0.1, duration=0.1) -> None:
        """Отрисовка локальных осей объекта."""
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

        origin = np.array(position)
        x_axis = origin + rot_matrix[:, 0] * length
        y_axis = origin + rot_matrix[:, 1] * length
        z_axis = origin + rot_matrix[:, 2] * length

        p.addUserDebugLine(origin, x_axis, [1, 0, 0], lineWidth=2, lifeTime=duration)  # X - Red
        p.addUserDebugLine(origin, y_axis, [0, 1, 0], lineWidth=2, lifeTime=duration)  # Y - Green
        p.addUserDebugLine(origin, z_axis, [0, 0, 1], lineWidth=2, lifeTime=duration)  # Z - Blue

    def _load_environment(self) -> None:
        # Установка параметров среды
        p.setAdditionalSearchPath(self._config.urdf_path)
        p.setTimeStep(self._config.time_step)
        p.setGravity(*self._config.gravity)

        # Create physical and graphical objects

        # Plane loading
        p.loadURDF(**self._config.word_objects.plane_description.to_dict())

        # Robot configuration
        self._robot_id = p.loadURDF(**self._config.word_objects.robot_description.to_dict())

        self._num_joints = p.getNumJoints(self._robot_id)
        self._joint_indices = [p.getJointInfo(self._robot_id, i)[0] for i in range(self._num_joints)]
        self._end_effector_index = self._num_joints - 1

        # Target object configuration
        shape_cfg = self._config.target_shape
        if shape_cfg.render_shape or shape_cfg.render_collision:
            visual_id = p.createVisualShape(**shape_cfg.visual_config.to_dict()) if shape_cfg.render_shape else -1
            collision_id = p.createCollisionShape(**shape_cfg.collision_config.to_dict()) if shape_cfg.render_collision else -1

            # Позиция
            if shape_cfg.random_position:
                if shape_cfg.position_range is None:
                    raise ValueError("При randomPosition=True необходимо указать positionRange.")
                self._target_position = self._sample_random_position()
            else:
                self._target_position = shape_cfg.position or shape_cfg.multi_body_config.base_position or (0, 0, 0)

            # Ориентация
            if shape_cfg.random_orientation:
                rand_euler = np.random.uniform(-np.pi, np.pi, size=3)
                self._target_orientation = p.getQuaternionFromEuler(rand_euler.tolist())
            else:
                # "вниз" — по Z
                self._target_orientation = [0, 0, 0, 1]

            self._target_id = p.createMultiBody(
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=self._target_position,
                baseOrientation=self._target_orientation,
                **shape_cfg.multi_body_config.to_dict()
            )

        # Table load
        if self._config.word_objects.table_description.file_name != "":
            self._table_id = p.loadURDF(**self._config.word_objects.table_description.to_dict())

        # Capture object loading
        if self._config.word_objects.capture_object.file_name != "":
            self._capture_object_id = p.loadURDF(**self._config.word_objects.capture_object.to_dict())

    def _get_obs(self) -> np.ndarray:
        joint_states = p.getJointStates(self._robot_id, range(self._num_joints))
        joint_angels = np.array([state[0] for state in joint_states], dtype=np.float32)
        ee_pos, ee_orn = self._get_end_effector_position()

        return np.concatenate([
            joint_angels,
            ee_pos,
            ee_orn,
            np.array(self._target_position, dtype=np.float32),
            np.array(self._target_orientation, dtype=np.float32),
            np.array([self._distance_target_to_ee_pos], dtype=np.float32),
            np.array([self._angel_error], dtype=np.float32),
            np.array([self._artificial_potential_field], dtype=np.float32)
        ])

    def _get_info(self) -> dict[str, Any]:

        ee_pos, ee_orn = self._get_end_effector_position()

        return {
            "joint_angels": self._get_obs()[:self._num_joints],
            "end_effector_position": ee_pos,
            "end_effector_orientation": ee_orn,
            "target_position": self._target_position,
            "target_orientation": self._target_orientation,
            "distance_target_to_ee_pos": self._distance_target_to_ee_pos,
            "distance_target_to_ee_orn": self._angel_error,
            "artificial_potential_field": self._artificial_potential_field,
            "reward": self._calc_reward()
        }

    def _sample_random_position(self) -> tuple:
        ranges = np.array(self._config.target_shape.position_range)
        return tuple(np.random.uniform(ranges[:, 0], ranges[:, 1]))

    def _get_end_effector_position(self) -> Tuple[np.ndarray, np.ndarray]:
        link_state = p.getLinkState(self._robot_id, self._end_effector_index)
        position = np.array(link_state[0], dtype=np.float32)
        orientation = np.array(link_state[1], dtype=np.float32)
        return position, orientation

    def _calc_reward(self) -> float:
        # Потенциал текущий
        self._calc_artificial_potential_field()
        U_t = self._artificial_potential_field
        U_prev = self._last_U
        self._last_U = U_t  # сохранить для следующего шага

        # Расчет штрафа за возможную самоколлизию
        d_min = self._calc_min_link_distance()
        if d_min <= 0.1:
            r_d = -0.1
        elif d_min <= 0.2:
            r_d = -1 / (self._K1 * d_min**2 + self._K2)
        else:
            r_d = 0.0

        return (U_t - U_prev) + r_d

    def _calc_artificial_potential_field(self) -> None:
        d = self._distance_target_to_ee_pos
        a = self._angel_error
        self._artificial_potential_field = -self._K_D * d + self._K_A / ((d + 1) * (a + 1))

    def _calc_min_link_distance(self) -> float:
        """
        Вычисляет минимальное расстояние между звеньями робота
        с использованием pybullet.getClosestPoints.
        """
        min_distance = float('inf')

        for i in range(self._num_joints):
            for j in range(i + 1, self._num_joints):
                points = p.getClosestPoints(
                    bodyA=self._robot_id,
                    bodyB=self._robot_id,
                    distance=self._config.max_check_distance_link,
                    linkIndexA=i,
                    linkIndexB=j
                )

                if points:
                    # Берем минимальное расстояние среди найденных точек
                    closest = min(point[8] for point in points)  # index 8 = distance
                    if closest < min_distance:
                        min_distance = closest

        return min_distance if min_distance != float('inf') else self._config.max_check_distance_link

    def _calc_distance(self) -> None:
        self._distance_target_to_ee_pos = np.linalg.norm(self._get_end_effector_position()[0] - self._target_position)

    def _calc_angel_error(self) -> None:
        """Расчет угловой ошибки (aet) между ориентацией энд-эффектора и целевой ориентацией.
        Возвращает угол в радианах.
        """
        _, ee_quat = self._get_end_effector_position()
        target_quat = self._target_orientation

        # Преобразуем к единичным quaternion'ам
        ee_quat = ee_quat / np.linalg.norm(ee_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)

        # Скалярное произведение кватернионов
        dot_product = np.clip(np.abs(np.dot(ee_quat, target_quat)), -1.0, 1.0)

        # Угловая ошибка через acos — это половина угла вращения между ориентациями
        self._angel_error = 2 * np.arccos(dot_product)


if __name__ == "__main__":
    params = {
        "n_steps": 500,
        "word_objects": {
            "plane_description": {
                "file_name": "plane.urdf"
            },
            "robot_description": {
                "file_name": "kuka_iiwa/model.urdf",
                "use_fixed_base": True,
                "flags": p.URDF_USE_SELF_COLLISION
            }
        },
        "target_shape": {
            "renderShape": True,
            "renderCollision": False,
            "randomPosition": True,
            "randomOrientation": True,
            "position_range": [(0.5, 0.6), (-0.3, 0.3), (0.2, 0.7)],
            "visual_config": {
                "shapeType": p.GEOM_SPHERE,
                "radius": 0.05,
                "rgbaColor": [0, 1, 0, 1],
            }
        }
    }

    env = KukaEnv(renders=True, **params)
    obs, inf = env.reset()

    for i in range(1000):
        action = env.action_space.sample()
        observ, rew, term, trunc, info = env.step(action)

        print(f"Step: {i}", info)

        if term or trunc:
            observ, info = env.reset()

    env.close()

