"""FloodNet defaults used by the built-in dataset profile."""

from __future__ import annotations

SUPERVISED_ROOT_NAME = "FloodNet-Supervised_v1.0"

CLASS_NAMES = (
    "Background",
    "Building-flooded",
    "Building-non-flooded",
    "Road-flooded",
    "Road-non-flooded",
    "Water",
    "Tree",
    "Vehicle",
    "Pool",
    "Grass",
)
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255
