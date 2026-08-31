"""Configuration loading and validation for released training recipes."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .dataset_spec import DatasetSpecError, resolve_dataset_spec


class ConfigError(ValueError):
    """Raised when a release configuration is incomplete or inconsistent."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {config_path}")
    config = _expand_environment(copy.deepcopy(loaded))
    config["_config_path"] = str(config_path)
    validate_supervised_config(config)
    return config


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ConfigError(f"Unresolved environment variable in value: {value}")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config section '{key}' must be a mapping")
    return value


def _positive_int(section: Mapping[str, Any], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive integer, got {value!r}")
    return value


def _nonnegative_number(section: Mapping[str, Any], key: str) -> float:
    value = section.get(key, 0.0)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"'{key}' must be a non-negative number")
    return float(value)


def validate_supervised_config(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config, "experiment")
    dataset = _mapping(config, "dataset")
    data = _mapping(config, "data")
    model = _mapping(config, "model")
    loss = _mapping(config, "loss")
    training = _mapping(config, "training")
    evaluation = _mapping(config, "evaluation")
    modules = config.get("modules", {})
    if not isinstance(modules, Mapping):
        raise ConfigError("Config section 'modules' must be a mapping")

    try:
        dataset_spec = resolve_dataset_spec(config)
    except (DatasetSpecError, OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Invalid dataset profile: {error}") from error

    for key in ("name", "run_id", "output_dir"):
        if not str(experiment.get(key, "")).strip():
            raise ConfigError(f"experiment.{key} is required")
    if experiment.get("kind") != "supervised":
        raise ConfigError("experiment.kind must be 'supervised'")

    protocol = str(dataset.get("protocol", "")).strip()
    if dataset_spec.allowed_protocols and protocol not in dataset_spec.allowed_protocols:
        raise ConfigError(
            f"dataset.protocol must be one of {list(dataset_spec.allowed_protocols)}"
        )
    for key in ("data_root", "manifest"):
        if not str(data.get(key, "")).strip():
            raise ConfigError(f"data.{key} is required")
    crop_size = _positive_int(data, "crop_size")
    if (
        dataset_spec.canonical_crop_size is not None
        and crop_size != dataset_spec.canonical_crop_size
    ):
        raise ConfigError(
            f"{dataset_spec.name} requires crop_size "
            f"{dataset_spec.canonical_crop_size}"
        )
    for key in ("image_mean", "image_std"):
        values = data.get(key)
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 3
            or any(not isinstance(value, (int, float)) for value in values)
        ):
            raise ConfigError(f"data.{key} must contain three numeric values")
    if any(float(value) <= 0 for value in data["image_std"]):
        raise ConfigError("data.image_std values must be positive")
    scale_range = data.get("scale_range")
    if (
        not isinstance(scale_range, (list, tuple))
        or len(scale_range) != 2
        or any(not isinstance(value, (int, float)) or value <= 0 for value in scale_range)
        or float(scale_range[0]) > float(scale_range[1])
    ):
        raise ConfigError("data.scale_range must contain two increasing positive values")
    workers = data.get("num_workers", 0)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ConfigError("data.num_workers must be a non-negative integer")

    if str(model.get("name", "")) != "segformer_b0":
        raise ConfigError("model.name must be 'segformer_b0'")
    if int(model.get("num_labels", -1)) != dataset_spec.num_classes:
        raise ConfigError(
            f"model.num_labels must be {dataset_spec.num_classes} for "
            f"{dataset_spec.name}"
        )
    if bool(model.get("pretrained", True)) and not str(
        model.get("pretrained_model_name_or_path", "")
    ).strip():
        raise ConfigError(
            "model.pretrained_model_name_or_path is required when pretrained=true"
        )

    factor_model = model.get("state_factorization", {})
    if factor_model and not isinstance(factor_model, Mapping):
        raise ConfigError("model.state_factorization must be a mapping")
    factor_enabled = bool((factor_model or {}).get("enabled", False))
    auxiliary_heads = tuple(model.get("auxiliary_heads", ()))
    if factor_enabled:
        if dataset_spec.factorization is None:
            raise ConfigError("OS-SegFormer requires dataset.factorization")
        if auxiliary_heads != ("object", "state"):
            raise ConfigError(
                "OS-SegFormer requires auxiliary_heads: [object, state]"
            )
        if str(factor_model.get("feature_source")) != "encoder_multiscale":
            raise ConfigError("feature_source must be 'encoder_multiscale'")
        if str(factor_model.get("state_mode")) != "conditional":
            raise ConfigError("state_mode must be 'conditional'")
        if str(factor_model.get("conditioning_operator")) != "additive":
            raise ConfigError("conditioning_operator must be 'additive'")
        if str(factor_model.get("fusion_scope")) != "grouped_state":
            raise ConfigError("fusion_scope must be 'grouped_state'")
        decoder_channels = _positive_int(factor_model, "decoder_channels")
        if decoder_channels != 64:
            raise ConfigError("The released recipe uses decoder_channels=64")
        dropout = factor_model.get("dropout")
        if not isinstance(dropout, (int, float)) or not 0 <= float(dropout) < 1:
            raise ConfigError("model.state_factorization.dropout must be in [0, 1)")
        fusion_weight = factor_model.get("fusion_weight")
        if (
            not isinstance(fusion_weight, (int, float))
            or not 0 <= float(fusion_weight) <= 1
        ):
            raise ConfigError("fusion_weight must be in [0, 1]")
    elif auxiliary_heads:
        raise ConfigError("Flat SegFormer-B0 must not enable auxiliary heads")

    if str(loss.get("name", "")).casefold() != "ce_dice":
        raise ConfigError("loss.name must be 'ce_dice'")
    _nonnegative_number(loss, "ce_weight")
    _nonnegative_number(loss, "dice_weight")
    semantic_source = str(loss.get("semantic_source", "semantic_direct"))
    if factor_enabled and semantic_source != "semantic_direct":
        raise ConfigError("OS-SegFormer directly supervises semantic_direct")

    factor_loss = modules.get("state_factorization", {})
    if factor_loss and not isinstance(factor_loss, Mapping):
        raise ConfigError("modules.state_factorization must be a mapping")
    if factor_enabled:
        if not bool((factor_loss or {}).get("enabled", False)):
            raise ConfigError("OS-SegFormer requires its object-state auxiliary losses")
        for key in (
            "object_weight",
            "state_weight",
            "object_dice_weight",
            "state_dice_weight",
        ):
            _nonnegative_number(factor_loss, key)
    elif factor_loss:
        raise ConfigError("Flat SegFormer-B0 must not configure factorization losses")

    max_iterations = _positive_int(training, "max_iterations")
    _positive_int(training, "batch_size")
    _positive_int(training, "gradient_accumulation_steps")
    _positive_int(training, "val_interval")
    if str(training.get("optimizer", "")).casefold() != "adamw":
        raise ConfigError("training.optimizer must be 'adamw'")
    learning_rate = training.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ConfigError("training.learning_rate must be positive")
    _nonnegative_number(training, "weight_decay")
    if str(training.get("scheduler", "")).casefold() != "poly":
        raise ConfigError("training.scheduler must be 'poly'")
    warmup = training.get("warmup_iterations", 0)
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ConfigError("training.warmup_iterations must be non-negative")
    if warmup >= max_iterations:
        raise ConfigError("warmup_iterations must be less than max_iterations")
    if "gradient_clip_norm" in training:
        _nonnegative_number(training, "gradient_clip_norm")

    for key in ("tile_size", "stride", "tile_batch_size"):
        _positive_int(evaluation, key)
    selection_metric = str(
        evaluation.get("selection_metric", dataset_spec.primary_metric)
    )
    if selection_metric not in dataset_spec.metric_names:
        raise ConfigError(
            f"evaluation.selection_metric {selection_metric!r} is not produced "
            f"by {dataset_spec.name}"
        )
    prediction_source = str(evaluation.get("prediction_source", "fused"))
    allowed_sources = {"fused", "semantic_direct", "hierarchical"}
    if prediction_source not in allowed_sources:
        raise ConfigError(
            f"evaluation.prediction_source must be one of {sorted(allowed_sources)}"
        )
