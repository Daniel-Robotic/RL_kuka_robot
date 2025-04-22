import pybullet_data

from typing import Tuple
from pydantic import BaseModel, Field
from . import WordObjectsConfig, VisualCollisionObjectConfig, VisualizerCameraConfig


class EnvConfig(BaseModel):
    urdf_path: str = Field(default=pybullet_data.getDataPath(), alias="urdfPath")
    time_step: float = Field(default=1 / 240, alias="timeStep")
    n_steps: int = Field(default=1000, alias="nSteps")
    gravity: Tuple[float, float, float] = Field(default=(0, 0, -9.81), alias="gravity")
    max_check_distance_link: float = Field(default=0.2, alias="maxCheckDistanceLink")
    max_check_distance: float = Field(default=0.1, alias="maxCheckDistance")
    max_check_angle_deg: float = Field(default=10, alias="maxCheckAngleDeg")
    max_duration_seconds: float = Field(default=30, alias="maxDurationSeconds")
    word_objects: WordObjectsConfig = Field(default_factory=WordObjectsConfig, alias="wordObjects")
    target_shape: VisualCollisionObjectConfig = Field(default_factory=VisualCollisionObjectConfig, alias="targetShape")
    visualizer_camera: VisualizerCameraConfig = Field(default_factory=VisualizerCameraConfig, alias="visualizerCamera")

    model_config = dict(populate_by_name=True)