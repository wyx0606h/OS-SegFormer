"""Public OS-SegFormer training, evaluation, and model utilities."""

from .constants import CLASS_NAMES, NUM_CLASSES
from .dataset_spec import DatasetSpec, dataset_spec_to_dict, resolve_dataset_spec

__all__ = [
    "CLASS_NAMES",
    "NUM_CLASSES",
    "DatasetSpec",
    "dataset_spec_to_dict",
    "resolve_dataset_spec",
]
