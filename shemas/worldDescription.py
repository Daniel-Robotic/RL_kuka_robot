from pydantic import BaseModel, Field
from typing import List, Optional


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