"""Dataset profiles shared by configuration, data loading, and metrics.

The training pipeline depends on this small schema instead of embedding
FloodNet class IDs throughout the implementation.  Existing configurations
without ``dataset.profile`` intentionally resolve to the built-in FloodNet
profile for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


class DatasetSpecError(ValueError):
    """Raised when a dataset profile is missing or internally inconsistent."""


@dataclass(frozen=True)
class StateMetricSpec:
    positive_class_ids: tuple[int, ...]
    negative_class_ids: tuple[int, ...]
    positive_name: str = "positive"


@dataclass(frozen=True)
class StateGroupSpec:
    """One object-conditioned semantic state group.

    ``semantic_class_ids`` is ordered exactly like the corresponding slice of
    the conditional state head.  Keeping that order in the dataset profile is
    what lets FloodNet preserve its historical non-flooded/flooded channel
    convention while RescueNet uses four Building states and two Road states.
    """

    name: str
    object_class_id: int
    semantic_class_ids: tuple[int, ...]
    state_names: tuple[str, ...]


@dataclass(frozen=True)
class FactorizationSpec:
    """Dataset-owned object/state hierarchy used by OS-SegFormer."""

    object_class_names: tuple[str, ...]
    semantic_to_object: tuple[int, ...]
    state_groups: tuple[StateGroupSpec, ...]

    @property
    def num_object_classes(self) -> int:
        return len(self.object_class_names)

    @property
    def num_state_channels(self) -> int:
        return sum(len(group.semantic_class_ids) for group in self.state_groups)

    @property
    def state_object_class_ids(self) -> tuple[int, ...]:
        return tuple(group.object_class_id for group in self.state_groups)


@dataclass(frozen=True)
class DatasetSpec:
    """Runtime description of one semantic-segmentation label space."""

    name: str
    class_names: tuple[str, ...]
    ignore_index: int = 255
    root_resolver: str = "direct"
    allowed_protocols: tuple[str, ...] = ()
    canonical_crop_size: int | None = None
    primary_metric: str = "mean_iou"
    excluded_class_ids: tuple[int, ...] = ()
    mean_iou_aliases: tuple[str, ...] = ()
    excluded_mean_iou_aliases: tuple[str, ...] = ()
    class_mean_groups: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    binary_iou_groups: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    state: StateMetricSpec | None = None
    factorization: FactorizationSpec | None = None
    source: str = "<inline>"

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def metric_names(self) -> frozenset[str]:
        names = {
            "mean_iou",
            "mean_iou_excluding",
            "macro_f1",
            "pixel_accuracy",
            "mean_boundary_f1",
            *self.mean_iou_aliases,
            *self.excluded_mean_iou_aliases,
            *self.class_mean_groups,
            *(f"mean_{name}" for name in self.binary_iou_groups),
        }
        if self.state is not None:
            names.update(
                {
                    "mean_state_accuracy",
                    "mean_state_macro_f1",
                    f"mean_{self.state.positive_name}_precision",
                    f"mean_{self.state.positive_name}_recall",
                }
            )
        if self.factorization is not None:
            for group in self.factorization.state_groups:
                names.update(
                    {
                        f"{group.name}_state_accuracy",
                        f"{group.name}_state_macro_f1",
                    }
                )
        return frozenset(names)


def _integer_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise DatasetSpecError(f"{field_name} must be a list of integer class IDs")
    return tuple(int(item) for item in value)


def _group_mapping(value: Any, *, field_name: str) -> dict[str, tuple[int, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DatasetSpecError(f"{field_name} must be a mapping")
    groups: dict[str, tuple[int, ...]] = {}
    for raw_name, raw_ids in value.items():
        name = str(raw_name).strip()
        if not name:
            raise DatasetSpecError(f"{field_name} contains an empty metric name")
        ids = _integer_tuple(raw_ids, field_name=f"{field_name}.{name}")
        if not ids:
            raise DatasetSpecError(f"{field_name}.{name} must not be empty")
        if len(set(ids)) != len(ids):
            raise DatasetSpecError(f"{field_name}.{name} contains duplicate class IDs")
        groups[name] = ids
    return groups


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise DatasetSpecError(f"{field_name} must be a list of metric names")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise DatasetSpecError(f"{field_name} values must be non-empty and unique")
    return result


def _factorization_spec(
    value: Any,
    *,
    class_names: tuple[str, ...],
) -> FactorizationSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DatasetSpecError("factorization must be a mapping or null")
    raw_object_names = value.get("object_class_names")
    if not isinstance(raw_object_names, (list, tuple)) or not raw_object_names:
        raise DatasetSpecError("factorization.object_class_names must be a non-empty list")
    object_names = tuple(str(item).strip() for item in raw_object_names)
    if any(not item for item in object_names) or len(set(object_names)) != len(object_names):
        raise DatasetSpecError(
            "factorization.object_class_names must be non-empty and unique"
        )
    semantic_to_object = _integer_tuple(
        value.get("semantic_to_object"),
        field_name="factorization.semantic_to_object",
    )
    if len(semantic_to_object) != len(class_names):
        raise DatasetSpecError(
            "factorization.semantic_to_object must contain one entry per semantic class"
        )
    invalid_objects = sorted(
        {item for item in semantic_to_object if item < 0 or item >= len(object_names)}
    )
    if invalid_objects:
        raise DatasetSpecError(
            f"factorization.semantic_to_object references invalid object IDs: {invalid_objects}"
        )

    raw_groups = value.get("state_groups")
    if not isinstance(raw_groups, (list, tuple)) or not raw_groups:
        raise DatasetSpecError("factorization.state_groups must be a non-empty list")
    groups: list[StateGroupSpec] = []
    used_semantic_ids: set[int] = set()
    used_object_ids: set[int] = set()
    used_names: set[str] = set()
    for index, raw_group in enumerate(raw_groups):
        field_name = f"factorization.state_groups[{index}]"
        if not isinstance(raw_group, Mapping):
            raise DatasetSpecError(f"{field_name} must be a mapping")
        name = str(raw_group.get("name", "")).strip()
        if not name or name in used_names:
            raise DatasetSpecError(f"{field_name}.name must be non-empty and unique")
        object_id = raw_group.get("object_class_id")
        if (
            not isinstance(object_id, int)
            or isinstance(object_id, bool)
            or object_id < 0
            or object_id >= len(object_names)
            or object_id in used_object_ids
        ):
            raise DatasetSpecError(
                f"{field_name}.object_class_id must be a unique valid object ID"
            )
        semantic_ids = _integer_tuple(
            raw_group.get("semantic_class_ids"),
            field_name=f"{field_name}.semantic_class_ids",
        )
        state_names = _string_tuple(
            raw_group.get("state_names"),
            field_name=f"{field_name}.state_names",
        )
        if len(semantic_ids) < 2 or len(state_names) != len(semantic_ids):
            raise DatasetSpecError(
                f"{field_name} requires at least two semantic IDs and matching state names"
            )
        if len(set(semantic_ids)) != len(semantic_ids):
            raise DatasetSpecError(f"{field_name}.semantic_class_ids contains duplicates")
        invalid_semantic = sorted(
            {item for item in semantic_ids if item < 0 or item >= len(class_names)}
        )
        if invalid_semantic:
            raise DatasetSpecError(
                f"{field_name} references invalid semantic IDs: {invalid_semantic}"
            )
        overlap = sorted(set(semantic_ids) & used_semantic_ids)
        if overlap:
            raise DatasetSpecError(
                f"factorization state groups overlap on semantic IDs: {overlap}"
            )
        mismatched = sorted(
            item for item in semantic_ids if semantic_to_object[item] != object_id
        )
        if mismatched:
            raise DatasetSpecError(
                f"{field_name} semantic IDs do not map to object_class_id {object_id}: "
                f"{mismatched}"
            )
        groups.append(
            StateGroupSpec(
                name=name,
                object_class_id=object_id,
                semantic_class_ids=semantic_ids,
                state_names=state_names,
            )
        )
        used_names.add(name)
        used_object_ids.add(object_id)
        used_semantic_ids.update(semantic_ids)
    for object_id in range(len(object_names)):
        mapped = {
            semantic_id
            for semantic_id, mapped_object_id in enumerate(semantic_to_object)
            if mapped_object_id == object_id
        }
        if object_id in used_object_ids:
            grouped = {
                semantic_id
                for group in groups
                if group.object_class_id == object_id
                for semantic_id in group.semantic_class_ids
            }
            if mapped != grouped:
                raise DatasetSpecError(
                    f"factorization state object {object_id} must group all and only its "
                    f"semantic IDs; mapped={sorted(mapped)}, grouped={sorted(grouped)}"
                )
        elif len(mapped) != 1:
            raise DatasetSpecError(
                f"factorization non-state object {object_id} must map to exactly one "
                f"semantic class, got {sorted(mapped)}"
            )
    return FactorizationSpec(
        object_class_names=object_names,
        semantic_to_object=semantic_to_object,
        state_groups=tuple(groups),
    )


def dataset_spec_from_mapping(
    value: Mapping[str, Any], *, source: str = "<inline>"
) -> DatasetSpec:
    """Parse and validate a dataset profile mapping."""

    name = str(value.get("name", "")).strip()
    if not name:
        raise DatasetSpecError("dataset profile name is required")
    raw_names = value.get("class_names")
    if not isinstance(raw_names, (list, tuple)) or not raw_names:
        raise DatasetSpecError("class_names must be a non-empty list")
    class_names = tuple(str(item).strip() for item in raw_names)
    if any(not item for item in class_names) or len(set(class_names)) != len(class_names):
        raise DatasetSpecError("class_names must be non-empty and unique")

    ignore_index = value.get("ignore_index", 255)
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise DatasetSpecError("ignore_index must be an integer")
    if 0 <= ignore_index < len(class_names):
        raise DatasetSpecError("ignore_index must not overlap a valid class ID")
    root_resolver = str(value.get("root_resolver", "direct")).strip().casefold()
    if root_resolver not in {"direct", "floodnet"}:
        raise DatasetSpecError("root_resolver must be 'direct' or 'floodnet'")

    raw_protocols = value.get("allowed_protocols", ())
    if not isinstance(raw_protocols, (list, tuple)):
        raise DatasetSpecError("allowed_protocols must be a list")
    allowed_protocols = tuple(str(item).strip() for item in raw_protocols)
    if any(not item for item in allowed_protocols):
        raise DatasetSpecError("allowed_protocols must not contain empty values")

    crop_size = value.get("canonical_crop_size")
    if crop_size is not None and (
        not isinstance(crop_size, int) or isinstance(crop_size, bool) or crop_size <= 0
    ):
        raise DatasetSpecError("canonical_crop_size must be a positive integer or null")

    metrics = value.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise DatasetSpecError("metrics must be a mapping")
    excluded_ids = _integer_tuple(
        metrics.get("excluded_class_ids"), field_name="metrics.excluded_class_ids"
    )
    mean_aliases = _string_tuple(
        metrics.get("mean_iou_aliases"), field_name="metrics.mean_iou_aliases"
    )
    excluded_aliases = _string_tuple(
        metrics.get("excluded_mean_iou_aliases"),
        field_name="metrics.excluded_mean_iou_aliases",
    )
    class_mean_groups = _group_mapping(
        metrics.get("class_mean_groups"), field_name="metrics.class_mean_groups"
    )
    binary_iou_groups = _group_mapping(
        metrics.get("binary_iou_groups"), field_name="metrics.binary_iou_groups"
    )

    state_value = metrics.get("state")
    state = None
    if state_value is not None:
        if not isinstance(state_value, Mapping):
            raise DatasetSpecError("metrics.state must be a mapping or null")
        state = StateMetricSpec(
            positive_class_ids=_integer_tuple(
                state_value.get("positive_class_ids"),
                field_name="metrics.state.positive_class_ids",
            ),
            negative_class_ids=_integer_tuple(
                state_value.get("negative_class_ids"),
                field_name="metrics.state.negative_class_ids",
            ),
            positive_name=str(state_value.get("positive_name", "positive")).strip(),
        )
        if not state.positive_class_ids or not state.negative_class_ids or not state.positive_name:
            raise DatasetSpecError("metrics.state requires non-empty positive/negative IDs and positive_name")

    factorization = _factorization_spec(
        value.get("factorization"), class_names=class_names
    )

    all_groups = [excluded_ids, *class_mean_groups.values(), *binary_iou_groups.values()]
    if state is not None:
        all_groups.extend((state.positive_class_ids, state.negative_class_ids))
        if set(state.positive_class_ids) & set(state.negative_class_ids):
            raise DatasetSpecError("metrics.state positive and negative class IDs overlap")
    for ids in all_groups:
        invalid = sorted({item for item in ids if item < 0 or item >= len(class_names)})
        if invalid:
            raise DatasetSpecError(
                f"dataset profile references class IDs outside [0, {len(class_names) - 1}]: {invalid}"
            )

    primary_metric = str(metrics.get("primary_metric", "mean_iou")).strip()
    if not primary_metric:
        raise DatasetSpecError("metrics.primary_metric must not be empty")
    available = {
        "mean_iou",
        "mean_iou_excluding",
        "macro_f1",
        "pixel_accuracy",
        *mean_aliases,
        *excluded_aliases,
        *class_mean_groups,
    }
    if primary_metric not in available:
        raise DatasetSpecError(
            f"metrics.primary_metric {primary_metric!r} is not produced by this profile"
        )

    return DatasetSpec(
        name=name,
        class_names=class_names,
        ignore_index=ignore_index,
        root_resolver=root_resolver,
        allowed_protocols=allowed_protocols,
        canonical_crop_size=crop_size,
        primary_metric=primary_metric,
        excluded_class_ids=excluded_ids,
        mean_iou_aliases=mean_aliases,
        excluded_mean_iou_aliases=excluded_aliases,
        class_mean_groups=class_mean_groups,
        binary_iou_groups=binary_iou_groups,
        state=state,
        factorization=factorization,
        source=source,
    )


@lru_cache(maxsize=32)
def load_dataset_spec(path: str | Path) -> DatasetSpec:
    profile_path = Path(path).expanduser().resolve()
    with profile_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise DatasetSpecError(f"dataset profile must contain a mapping: {profile_path}")
    return dataset_spec_from_mapping(loaded, source=str(profile_path))


def _builtin_profile_path(name: str) -> Path:
    key = name.strip().casefold()
    aliases = {"floodnet": "floodnet.yaml"}
    if key not in aliases:
        raise DatasetSpecError(
            f"unknown built-in dataset profile {name!r}; use a YAML file path for custom datasets"
        )
    return Path(__file__).resolve().parents[1] / "configs" / "datasets" / aliases[key]


def resolve_dataset_spec(config: Mapping[str, Any]) -> DatasetSpec:
    """Resolve ``dataset.profile`` relative to the experiment configuration."""

    dataset = config.get("dataset", {})
    if dataset is None:
        dataset = {}
    if not isinstance(dataset, Mapping):
        raise DatasetSpecError("config section 'dataset' must be a mapping")
    reference = dataset.get("profile", "floodnet")
    if isinstance(reference, Mapping):
        return dataset_spec_from_mapping(reference, source="dataset.profile (inline)")
    reference_text = str(reference).strip()
    if not reference_text:
        raise DatasetSpecError("dataset.profile must not be empty")
    if reference_text.casefold() == "floodnet":
        return load_dataset_spec(_builtin_profile_path(reference_text))
    profile_path = Path(reference_text).expanduser()
    if not profile_path.is_absolute():
        config_path = config.get("_config_path")
        base = Path(str(config_path)).resolve().parent if config_path else Path.cwd()
        profile_path = base / profile_path
    return load_dataset_spec(profile_path)


def default_floodnet_spec() -> DatasetSpec:
    return load_dataset_spec(_builtin_profile_path("floodnet"))


def dataset_spec_to_dict(spec: DatasetSpec) -> dict[str, Any]:
    """Return a stable YAML/JSON-ready profile snapshot for run provenance."""

    state = None
    if spec.state is not None:
        state = {
            "positive_name": spec.state.positive_name,
            "positive_class_ids": list(spec.state.positive_class_ids),
            "negative_class_ids": list(spec.state.negative_class_ids),
        }
    factorization = None
    if spec.factorization is not None:
        factorization = {
            "object_class_names": list(spec.factorization.object_class_names),
            "semantic_to_object": list(spec.factorization.semantic_to_object),
            "state_groups": [
                {
                    "name": group.name,
                    "object_class_id": group.object_class_id,
                    "semantic_class_ids": list(group.semantic_class_ids),
                    "state_names": list(group.state_names),
                }
                for group in spec.factorization.state_groups
            ],
        }
    return {
        "name": spec.name,
        "source": spec.source,
        "root_resolver": spec.root_resolver,
        "class_names": list(spec.class_names),
        "ignore_index": spec.ignore_index,
        "allowed_protocols": list(spec.allowed_protocols),
        "canonical_crop_size": spec.canonical_crop_size,
        "factorization": factorization,
        "metrics": {
            "primary_metric": spec.primary_metric,
            "mean_iou_aliases": list(spec.mean_iou_aliases),
            "excluded_class_ids": list(spec.excluded_class_ids),
            "excluded_mean_iou_aliases": list(spec.excluded_mean_iou_aliases),
            "class_mean_groups": {
                name: list(ids) for name, ids in spec.class_mean_groups.items()
            },
            "binary_iou_groups": {
                name: list(ids) for name, ids in spec.binary_iou_groups.items()
            },
            "state": state,
        },
    }
