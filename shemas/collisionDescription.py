import pybullet as p
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional


class VisualCollisionShapeConfig(BaseModel):
    shape_type: int = Field(default=p.GEOM_SPHERE, alias="shapeType")
    radius: float = Field(default=0.5, alias="radius")
    half_extents: Optional[Tuple[float, float, float]] = Field(default=None, alias="halfExtents")
    height: Optional[float] = Field(default=None, alias="height")
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
    position_range: Optional[List[Tuple[float, float]]] = Field(default=None, alias="positionRange")  # [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
    random_orientation: bool = Field(default=False, alias="randomOrientation")
    position: Optional[Tuple[float, float, float]] = Field(default=None, alias="position")
    orientation: Optional[Tuple[float, float, float, float]] = Field(default=None, alias="orientation")
    visual_config: VisualCollisionShapeConfig = Field(default_factory=VisualCollisionShapeConfig, alias="visualConfig")
    collision_config: VisualCollisionShapeConfig = Field(default_factory=VisualCollisionShapeConfig, alias="collisionConfig")
    multi_body_config: MultiBodyConfig = Field(default_factory=MultiBodyConfig, alias="multiBodyConfig")

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True, by_alias=True)

    model_config = dict(populate_by_name=True)