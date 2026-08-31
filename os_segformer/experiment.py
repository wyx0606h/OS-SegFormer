"""Shared training/evaluation helpers for manifest-driven experiments."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from .dataset import ManifestSegmentationDataset
from .dataset_spec import resolve_dataset_spec
from .inference import sliding_window_predict, sliding_window_predict_sources
from .metrics import (
    SegmentationMeter,
    boundary_f1,
    grouped_object_iou,
    state_group_metrics_from_semantic,
    state_metrics_from_semantic,
)
from .models import extract_logits
from .training import checkpoint_state_dict, collect_runtime_metadata
from .transforms import CenterCrop, build_supervised_train_transform


def apply_path_overrides(
    config: dict[str, Any],
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if data_root is not None:
        config.setdefault("data", {})["data_root"] = str(Path(data_root).expanduser())
    if output_dir is not None:
        config.setdefault("experiment", {})["output_dir"] = str(Path(output_dir).expanduser())
    return config


def ensure_run_layout(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "curves": output_dir / "curves",
        "predictions": output_dir / "predictions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_resolved_yaml(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("os_segformer")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def make_transform(config: Mapping[str, Any], *, split: str, training: bool):
    data = config["data"]
    dataset_spec = resolve_dataset_spec(config)
    seed = int(config["experiment"].get("seed", 0))
    crop_size = int(data.get("crop_size", 512))
    if training:
        default_class_ids = tuple(range(1, dataset_spec.num_classes)) or (0,)
        return build_supervised_train_transform(
            crop_size=crop_size,
            seed=seed,
            scale_range=tuple(float(v) for v in data.get("scale_range", (0.75, 1.25))),
            class_ids=tuple(
                int(v) for v in data.get("class_aware_ids", default_class_ids)
            ),
            class_aware_probability=float(data.get("class_aware_probability", 0.5)),
            mask_fill=dataset_spec.ignore_index,
        )
    if bool(data.get("center_crop_eval", False)):
        return CenterCrop(crop_size, mask_fill=dataset_spec.ignore_index)
    return None


def make_dataset(
    config: Mapping[str, Any], split: str, *, training: bool = False
) -> ManifestSegmentationDataset:
    data = config["data"]
    dataset_spec = resolve_dataset_spec(config)
    return ManifestSegmentationDataset(
        data["data_root"],
        data["manifest"],
        split=split,
        transform=make_transform(config, split=split, training=training),
        image_mean=tuple(float(value) for value in data["image_mean"]),
        image_std=tuple(float(value) for value in data["image_std"]),
        ignore_index=dataset_spec.ignore_index,
        num_classes=dataset_spec.num_classes,
        root_resolver=dataset_spec.root_resolver,
    )


def make_loader(
    config: Mapping[str, Any],
    split: str,
    *,
    training: bool,
    sampler: Sampler[Any] | None = None,
) -> DataLoader:
    dataset = make_dataset(config, split, training=training)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=int(config["data"].get("num_workers", 0)),
        drop_last=bool(training and config["data"].get("drop_last", False)),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if not np.isnan(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def write_metrics_files(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = payload["metrics"]
    (output_dir / "metrics.json").write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    matrix = np.asarray(metrics["confusion_matrix"])
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    with (output_dir / "class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class_name", "iou", "precision", "recall", "f1"])
        writer.writeheader()
        class_names = metrics["class_names"]
        for class_id, class_name in enumerate(class_names):
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": metrics["iou_per_class"][class_id],
                    "precision": metrics["precision_per_class"][class_id],
                    "recall": metrics["recall_per_class"][class_id],
                    "f1": metrics["f1_per_class"][class_id],
                }
            )
    total_pixels = int(matrix.sum())
    ground_truth_counts = matrix.sum(axis=1)
    prediction_counts = matrix.sum(axis=0)
    with (output_dir / "class_distribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_id",
                "class_name",
                "ground_truth_pixels",
                "ground_truth_fraction",
                "predicted_pixels",
                "prediction_fraction",
            ],
        )
        writer.writeheader()
        for class_id, class_name in enumerate(class_names):
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "ground_truth_pixels": int(ground_truth_counts[class_id]),
                    "ground_truth_fraction": (
                        float(ground_truth_counts[class_id] / total_pixels)
                        if total_pixels
                        else float("nan")
                    ),
                    "predicted_pixels": int(prediction_counts[class_id]),
                    "prediction_fraction": (
                        float(prediction_counts[class_id] / total_pixels)
                        if total_pixels
                        else float("nan")
                    ),
                }
            )
    rows = payload.get("per_sample", [])
    if rows:
        with (output_dir / "per_sample_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: ManifestSegmentationDataset,
    *,
    config: Mapping[str, Any],
    split: str,
    device: torch.device,
    checkpoint: str | None = None,
    max_samples: int | None = None,
    save_predictions_dir: Path | None = None,
    prediction_source: str | None = None,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    dataset_spec = resolve_dataset_spec(config)
    tile_size = int(evaluation.get("tile_size", 512))
    stride = int(evaluation.get("stride", 384))
    tile_batch_size = int(evaluation.get("tile_batch_size", 4))
    use_amp = bool(evaluation.get("use_amp", config.get("training", {}).get("use_amp", False)))
    resolved_prediction_source = str(
        prediction_source or evaluation.get("prediction_source", "fused")
    )
    meter = SegmentationMeter(ignore_index=dataset_spec.ignore_index, spec=dataset_spec)
    per_sample: list[dict[str, Any]] = []
    boundary_values: list[float] = []
    per_sample_metric_values: dict[str, list[float]] = {}

    sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)
    for index in range(sample_count):
        sample = dataset[index]
        target = sample["mask"]
        if target is None:
            raise ValueError(f"Cannot evaluate sample without mask: {sample['id']}")
        probabilities = sliding_window_predict(
            model,
            sample["image"],
            tile_size=tile_size,
            stride=stride,
            tile_batch_size=tile_batch_size,
            device=device,
            use_amp=use_amp,
            prediction_source=resolved_prediction_source,
        )
        prediction = probabilities.argmax(dim=1).squeeze(0).cpu()
        target_cpu = target.cpu()
        meter.update(prediction, target_cpu)
        grouped = grouped_object_iou(
            prediction, target_cpu, ignore_index=dataset_spec.ignore_index, spec=dataset_spec
        )
        state = state_metrics_from_semantic(
            prediction, target_cpu, ignore_index=dataset_spec.ignore_index, spec=dataset_spec
        )
        grouped_state = state_group_metrics_from_semantic(
            prediction,
            target_cpu,
            ignore_index=dataset_spec.ignore_index,
            spec=dataset_spec,
        )
        bf1 = boundary_f1(
            prediction,
            target_cpu,
            tolerance=int(evaluation.get("boundary_tolerance", 3)),
            ignore_index=dataset_spec.ignore_index,
        )
        boundary_values.append(bf1)
        sample_metrics = {"boundary_f1": bf1, **grouped, **state, **grouped_state}
        for name, value in {**grouped, **state, **grouped_state}.items():
            if isinstance(value, (int, float, np.generic)):
                per_sample_metric_values.setdefault(name, []).append(float(value))
        per_sample.append({"sample_id": sample["id"], "split": split, **sample_metrics})
        if save_predictions_dir is not None:
            from PIL import Image
            Image.fromarray(prediction.numpy().astype(np.uint8), mode="L").save(save_predictions_dir / f"{sample['id']}.png")

    metrics = meter.compute()
    metrics["mean_boundary_f1"] = _mean(boundary_values)
    metrics.update(
        {f"mean_{name}": _mean(values) for name, values in per_sample_metric_values.items()}
    )
    return {
        "experiment_name": config["experiment"].get("name", config["experiment"].get("run_id")),
        "dataset": dataset_spec.name,
        "dataset_profile": dataset_spec.source,
        "protocol": config.get("dataset", {}).get("protocol"),
        "split": split,
        "checkpoint": checkpoint,
        "num_samples": sample_count,
        "tile_size": tile_size,
        "stride": stride,
        "tile_batch_size": tile_batch_size,
        "prediction_source": resolved_prediction_source,
        "metrics": _json_ready(metrics),
        "per_sample": _json_ready(per_sample),
        "runtime": collect_runtime_metadata(device),
    }


@torch.no_grad()
def evaluate_model_sources(
    model: torch.nn.Module,
    dataset: ManifestSegmentationDataset,
    *,
    config: Mapping[str, Any],
    split: str,
    device: torch.device,
    prediction_sources: Iterable[str],
    checkpoint: str | None = None,
    max_samples: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate several posteriors while sharing each sliding-window forward."""

    sources = tuple(str(source).casefold() for source in prediction_sources)
    if not sources:
        raise ValueError("At least one prediction source is required")
    if len(set(sources)) != len(sources):
        raise ValueError("Prediction sources must be unique")

    evaluation = config["evaluation"]
    dataset_spec = resolve_dataset_spec(config)
    tile_size = int(evaluation.get("tile_size", 512))
    stride = int(evaluation.get("stride", 384))
    tile_batch_size = int(evaluation.get("tile_batch_size", 4))
    use_amp = bool(
        evaluation.get("use_amp", config.get("training", {}).get("use_amp", False))
    )
    tolerance = int(evaluation.get("boundary_tolerance", 3))
    sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    meters = {
        source: SegmentationMeter(ignore_index=dataset_spec.ignore_index, spec=dataset_spec)
        for source in sources
    }
    per_sample: dict[str, list[dict[str, Any]]] = {source: [] for source in sources}
    boundary_values: dict[str, list[float]] = {source: [] for source in sources}
    per_sample_metric_values: dict[str, dict[str, list[float]]] = {
        source: {} for source in sources
    }

    for index in range(sample_count):
        sample = dataset[index]
        target = sample["mask"]
        if target is None:
            raise ValueError(f"Cannot evaluate sample without mask: {sample['id']}")
        probabilities_by_source = sliding_window_predict_sources(
            model,
            sample["image"],
            prediction_sources=sources,
            tile_size=tile_size,
            stride=stride,
            tile_batch_size=tile_batch_size,
            device=device,
            use_amp=use_amp,
        )
        target_cpu = target.cpu()
        for source in sources:
            prediction = probabilities_by_source[source].argmax(dim=1).squeeze(0).cpu()
            meters[source].update(prediction, target_cpu)
            grouped = grouped_object_iou(
                prediction,
                target_cpu,
                ignore_index=dataset_spec.ignore_index,
                spec=dataset_spec,
            )
            state = state_metrics_from_semantic(
                prediction,
                target_cpu,
                ignore_index=dataset_spec.ignore_index,
                spec=dataset_spec,
            )
            grouped_state = state_group_metrics_from_semantic(
                prediction,
                target_cpu,
                ignore_index=dataset_spec.ignore_index,
                spec=dataset_spec,
            )
            bf1 = boundary_f1(
                prediction,
                target_cpu,
                tolerance=tolerance,
                ignore_index=dataset_spec.ignore_index,
            )
            boundary_values[source].append(bf1)
            sample_metrics = {**grouped, **state, **grouped_state}
            for name, value in sample_metrics.items():
                if isinstance(value, (int, float, np.generic)):
                    per_sample_metric_values[source].setdefault(name, []).append(
                        float(value)
                    )
            per_sample[source].append(
                {
                    "sample_id": sample["id"],
                    "split": split,
                    "boundary_f1": bf1,
                    **sample_metrics,
                }
            )

    runtime = collect_runtime_metadata(device)
    payloads: dict[str, dict[str, Any]] = {}
    for source in sources:
        metrics = meters[source].compute()
        metrics.update(
            {"mean_boundary_f1": _mean(boundary_values[source])}
        )
        metrics.update(
            {
                f"mean_{name}": _mean(values)
                for name, values in per_sample_metric_values[source].items()
            }
        )
        payloads[source] = {
            "experiment_name": config["experiment"].get(
                "name", config["experiment"].get("run_id")
            ),
            "dataset": dataset_spec.name,
            "dataset_profile": dataset_spec.source,
            "protocol": config.get("dataset", {}).get("protocol"),
            "split": split,
            "checkpoint": checkpoint,
            "num_samples": sample_count,
            "tile_size": tile_size,
            "stride": stride,
            "tile_batch_size": tile_batch_size,
            "prediction_source": source,
            "metrics": _json_ready(metrics),
            "per_sample": _json_ready(per_sample[source]),
            "runtime": runtime,
        }
    return payloads


def append_history_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_miou: float,
    history: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": checkpoint_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
            "history": list(history),
            "config": dict(config),
        },
        path,
    )
