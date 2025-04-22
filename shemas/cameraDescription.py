from pydantic import BaseModel, Field
from typing import List


class VisualizerCameraConfig(BaseModel):
    camera_distance: float = Field(default=5.0, alias="cameraDistance")
    camera_yaw: float = Field(default=50.0, alias="cameraYaw")
    camera_pitch: float = Field(default=-35.0, alias="cameraPitch")
    camera_target_position: List[float] = Field(default_factory=lambda: [0, 0, 0], alias="cameraTargetPosition")

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True, by_alias=True)

    model_config = dict(populate_by_name=True)