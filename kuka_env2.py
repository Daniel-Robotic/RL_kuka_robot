import os
import time

import pybullet as p
import pybullet_data
import gymnasium as gym

from gymnasium import spaces
from pydantic import BaseModel, Field
from typing import Tuple, Any, SupportsFloat, List, Optional

from gymnasium.core import RenderFrame, ObsType, ActType



class ObjectDescriptionConfig(BaseModel):
    file_name: str = Field(default="", alias="fileName")
    base_position: List[float] = Field(default_factory=lambda: [0, 0, 0], alias="basePosition")
    base_orientation: List[float] = Field(default_factory=lambda: [0, 0, 0, 1], alias="baseOrientation")
    use_fixed_base: bool = Field(default=False, alias="useFixedBase")
    flags: Optional[int] = Field(default=0, alias="flags")

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True, by_alias=True)

    model_config = dict(populate_by_name=True)


class WordObjectsConfig(BaseModel):
    plane_description: ObjectDescriptionConfig = Field(default_factory=ObjectDescriptionConfig, alias="planeDescription")
    robot_description: ObjectDescriptionConfig = Field(default_factory=ObjectDescriptionConfig, alias="robotDescription")
    table_description: ObjectDescriptionConfig = Field(default_factory=ObjectDescriptionConfig, alias="tableDescription")
    capture_object: ObjectDescriptionConfig = Field(default_factory=ObjectDescriptionConfig, alias="captureObject")

    model_config = dict(populate_by_name=True)


class VisualCollisionShapeConfig(BaseModel):
    shape_type: int = Field(default=p.GEOM_SPHERE, alias="shapeType")
    radius: float = Field(default=0.5, alias="radius")
    half_extents: Optional[Tuple[float, float, float]] = Field(default=None, alias="halfExtents")
    height: float = Field(default=1.0, alias="height")
    mesh_scale: Tuple[float, float, float] = Field(default=(1.0, 1.0, 1.0), alias="meshScale")
    plane_normal: Tuple[float, float, float] = Field(default=(0.0, 0.0, 1.0), alias="planeNormal")
    file_name: Optional[str] = Field(default=None, alias="fileName")
    vertices: Optional[List[Tuple[float, float, float]]] = Field(default=None, alias="vertices")
    indices: Optional[List[int]] = Field(default=None, alias="indices")
    heightfield_texture_scaling: Optional[float] = Field(default=None, alias="heightfieldTextureScaling")
    num_heightfield_rows: Optional[int] = Field(default=None, alias="numHeightfieldRows")
    num_heightfield_columns: Optional[int] = Field(default=None, alias="numHeightfieldColumns")
    replace_heightfield_index: Optional[int] = Field(default=None, alias="replaceHeightfieldIndex")
    collision_frame_position: Optional[Tuple[float, float, float]] = Field(default=None, alias="collisionFramePosition")
    collision_frame_orientation: Optional[Tuple[float, float, float, float]] = Field(default=None, alias="collisionFrameOrientation")
    rgba_color: Tuple[float, float, float, float] = Field(default=(1.0, 0.0, 0.0, 1.0), alias="rgbaColor")
    flags: Optional[int] = Field(default=None, alias="flags")
    physics_client_id: Optional[int] = Field(default=None, alias="physicsClientId")

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True, by_alias=True)

    model_config = dict(populate_by_name=True)


class MultiBodyConfig(BaseModel):
    base_mass: Optional[float] = Field(default=None, alias="baseMass")
    base_position: Optional[Tuple[float, float, float]] = Field(default=None, alias="basePosition")
    base_orientation: Optional[Tuple[float, float, float, float]] = Field(default=None, alias="baseOrientation")
    base_inertial_frame_position: Optional[Tuple[float, float, float]] = Field(default=None, alias="baseInertialFramePosition")
    base_inertial_frame_orientation: Optional[Tuple[float, float, float, float]] = Field(default=None, alias="baseInertialFrameOrientation")
    link_masses: Optional[List[float]] = Field(default=None, alias="linkMasses")
    link_collision_shape_indices: Optional[List[int]] = Field(default=None, alias="linkCollisionShapeIndices")
    link_visual_shape_indices: Optional[List[int]] = Field(default=None, alias="linkVisualShapeIndices")
    link_positions: Optional[List[Tuple[float, float, float]]] = Field(default=None, alias="linkPositions")
    link_orientations: Optional[List[Tuple[float, float, float, float]]] = Field(default=None, alias="linkOrientations")
    link_inertial_frame_positions: Optional[List[Tuple[float, float, float]]] = Field(default=None, alias="linkInertialFramePositions")
    link_inertial_frame_orientations: Optional[List[Tuple[float, float, float, float]]] = Field(default=None, alias="linkInertialFrameOrientations")
    link_parent_indices: Optional[List[int]] = Field(default=None, alias="linkParentIndices")
    link_joint_types: Optional[List[int]] = Field(default=None, alias="linkJointTypes")
    link_joint_axes: Optional[List[Tuple[float, float, float]]] = Field(default=None, alias="linkJointAxes")
    use_maximal_coordinates: Optional[int] = Field(default=None, alias="useMaximalCoordinates")
    flags: Optional[int] = Field(default=None, alias="flags")
    batch_positions: Optional[List[Tuple[float, float, float]]] = Field(default=None, alias="batchPositions")
    physics_client_id: Optional[int] = Field(default=None, alias="physicsClientId")

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True, by_alias=True)

    model_config = dict(populate_by_name=True)


class VisualCollisionObjectConfig(BaseModel):
    render_shape: bool = Field(default=True, alias="renderShape")
    render_collision: bool = Field(default=True, alias="renderCollision")
    random_position: bool = Field(default=False, alias="randomPosition")
    random_orientation: bool = Field(default=False, alias="randomOrientation")

    visual_config: VisualCollisionShapeConfig = Field(default_factory=VisualCollisionShapeConfig, alias="visualConfig")
    collision_config: VisualCollisionShapeConfig = Field(
        default_factory=VisualCollisionShapeConfig,
        alias="collisionConfig"
    )
    multi_body_config: MultiBodyConfig = Field(default_factory=MultiBodyConfig, alias="multiBodyConfig")

    model_config = dict(populate_by_name=True)

class VisualizerCameraConfig(BaseModel):
    camera_distance: float = Field(default=5.0, alias="cameraDistance")
    camera_yaw: float = Field(default=50.0, alias="cameraYaw")
    camera_pitch: float = Field(default=-35.0, alias="cameraPitch")
    camera_target_position: List[float] = Field(default_factory=lambda: [0, 0, 0], alias="cameraTargetPosition")

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True, by_alias=True)

    model_config = dict(populate_by_name=True)


class EnvConfig(BaseModel):
    urdf_path: str = Field(default=pybullet_data.getDataPath(), alias="urdfPath")
    time_step: float = Field(default=1 / 240, alias="timeStep")
    n_steps: int = Field(default=1000, alias="nSteps")
    gravity: Tuple[float, float, float] = Field(default=(0, 0, -9.81), alias="gravity")

    word_objects: WordObjectsConfig = Field(default_factory=WordObjectsConfig, alias="wordObjects")
    end_effector_shape: VisualCollisionObjectConfig = Field(default_factory=VisualCollisionObjectConfig, alias="endEffectorShape")
    target_shape: VisualCollisionObjectConfig = Field(default_factory=VisualCollisionObjectConfig, alias="targetShape")
    visualizer_camera: VisualizerCameraConfig = Field(default_factory=VisualizerCameraConfig, alias="visualizerCamera")

    model_config = dict(populate_by_name=True)


class KukaEnv(gym.Env):
    def __init__(
            self,
             renders:bool=False,
             **kwargs
    ):

        super(KukaEnv, self).__init__()

        self._renders = renders
        self._config = EnvConfig(**kwargs)

        # Spawn components
        self._robot_id = None
        self._table_id = None
        self.target_id = None
        self._capture_object_id = None
        self._end_effector_shape_id = None

        # Robot params
        self._num_joints = None
        self._joint_indices = []
        self._end_effector_index = None

        # Rendering configurations
        self._render_setting()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        return {}, 0, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        p.resetSimulation()
        self._load_environment()

        return {}, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self):
        p.disconnect()

    def _load_environment(self):
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

        # Create visual end effector shape
        shape_cfg = self._config.end_effector_shape
        if shape_cfg.render_shape or shape_cfg.render_collision:
            visual_id = (
                p.createVisualShape(**shape_cfg.visual_config.to_dict())
                if shape_cfg.render_shape else -1
            )
            collision_id = (
                p.createCollisionShape(**shape_cfg.collision_config.to_dict())
                if shape_cfg.render_collision else -1
            )

            self._end_effector_shape_id = p.createMultiBody(
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                **shape_cfg.multi_body_config.to_dict()
        )

        # Table load
        if self._config.word_objects.table_description.file_name != "":
            self._table_id = p.loadURDF(**self._config.word_objects.table_description.to_dict())

        # Capture object loading
        if self._config.word_objects.capture_object.file_name != "":
            self._capture_object_id = p.loadURDF(**self._config.word_objects.capture_object.to_dict())


    def _render_setting(self):
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            p.connect(p.GUI) if cid < 0 else None
            p.resetDebugVisualizerCamera(**self._config.visualizer_camera.to_dict())
        else:
            p.connect(p.DIRECT)

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
        }
    }

    env = KukaEnv(renders=False, **params)
    env.reset()
    # time.sleep(100)
    env.close()


