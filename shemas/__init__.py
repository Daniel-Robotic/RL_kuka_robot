from enum import Enum
from .collisionDescription import (VisualCollisionShapeConfig,
                                   VisualCollisionObjectConfig,
                                   MultiBodyConfig
                                   )
from .cameraDescription import VisualizerCameraConfig
from .worldDescription import ObjectDescriptionConfig, WordObjectsConfig
from .envDescription import EnvConfig


class DistanceCalculate(Enum):
    POSITION = 0
    ORIENTATION = 1
