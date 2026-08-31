"""NumPy reference metrics driven by a dataset label-space profile."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .constants import IGNORE_INDEX, NUM_CLASSES
from .dataset_spec import DatasetSpec, default_floodnet_spec


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def confusion_matrix(
    prediction: object,
    target: object,
    *,
    num_classes: int = NUM_CLASSES,
    ignore_index: int | None = IGNORE_INDEX,
) -> np.ndarray:
    prediction_array = _as_numpy(prediction).astype(np.int64, copy=False)
    target_array = _as_numpy(target).astype(np.int64, copy=False)
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"Prediction and target shapes differ: "
            f"{prediction_array.shape} versus {target_array.shape}"
        )
    prediction_flat = prediction_array.reshape(-1)
    target_flat = target_array.reshape(-1)
    valid = np.ones_like(target_flat, dtype=bool)
    if ignore_index is not None:
        valid &= target_flat != ignore_index
    valid &= (target_flat >= 0) & (target_flat < num_classes)
    prediction_flat = prediction_flat[valid]
    target_flat = target_flat[valid]
    if np.any((prediction_flat < 0) | (prediction_flat >= num_classes)):
        invalid = np.unique(
            prediction_flat[
                (prediction_flat < 0) | (prediction_flat >= num_classes)
            ]
        )
        raise ValueError(f"Prediction contains invalid class IDs: {invalid.tolist()}")
    encoded = target_flat * num_classes + prediction_flat
    return np.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def _nanmean(values: object) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[~np.isnan(array)]
    return float(finite.mean()) if finite.size else float("nan")


def metrics_from_confusion(
    matrix: object, *, spec: DatasetSpec | None = None
) -> dict[str, object]:
    spec = spec or default_floodnet_spec()
    num_classes = spec.num_classes
    confusion = _as_numpy(matrix).astype(np.float64, copy=False)
    if confusion.shape != (num_classes, num_classes):
        raise ValueError(
            f"Expected {num_classes}x{num_classes} confusion matrix, got {confusion.shape}"
        )
    true_positive = np.diag(confusion)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    iou_denominator = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    iou = np.divide(
        true_positive,
        iou_denominator,
        out=np.full(num_classes, np.nan),
        where=iou_denominator > 0,
    )
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.full(num_classes, np.nan),
        where=precision_denominator > 0,
    )
    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.full(num_classes, np.nan),
        where=recall_denominator > 0,
    )
    f1 = np.divide(
        2 * true_positive,
        f1_denominator,
        out=np.full(num_classes, np.nan),
        where=f1_denominator > 0,
    )
    total = confusion.sum()
    pixel_accuracy = float(true_positive.sum() / total) if total else float("nan")
    excluded = set(spec.excluded_class_ids)
    included_iou = [value for class_id, value in enumerate(iou) if class_id not in excluded]
    mean_iou = _nanmean(iou)
    excluded_mean_iou = _nanmean(included_iou)
    result: dict[str, object] = {
        "class_names": spec.class_names,
        "iou_per_class": iou,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "mean_iou": mean_iou,
        "mean_iou_excluding": excluded_mean_iou,
        "macro_f1": _nanmean(f1),
        "pixel_accuracy": pixel_accuracy,
    }
    for alias in spec.mean_iou_aliases:
        result[alias] = mean_iou
    for alias in spec.excluded_mean_iou_aliases:
        result[alias] = excluded_mean_iou
    for name, class_ids in spec.class_mean_groups.items():
        result[name] = _nanmean([iou[class_id] for class_id in class_ids])
    result.update(_state_group_metrics_from_confusion(confusion, spec=spec))
    return result


def _state_group_metrics_from_confusion(
    confusion: np.ndarray, *, spec: DatasetSpec
) -> dict[str, object]:
    """Compute conditional state metrics for every profile-defined object group."""

    if spec.factorization is None:
        return {}
    result: dict[str, object] = {}
    for group in spec.factorization.state_groups:
        ids = np.asarray(group.semantic_class_ids, dtype=np.int64)
        true_counts = confusion[ids, :].sum(axis=1)
        conditional = confusion[np.ix_(ids, ids)]
        true_positive = np.diag(conditional)
        false_negative = true_counts - true_positive
        false_positive = conditional.sum(axis=0) - true_positive
        iou_denominator = true_positive + false_positive + false_negative
        f1_denominator = 2 * true_positive + false_positive + false_negative
        state_iou = np.divide(
            true_positive,
            iou_denominator,
            out=np.full(len(ids), np.nan),
            where=iou_denominator > 0,
        )
        state_f1 = np.divide(
            2 * true_positive,
            f1_denominator,
            out=np.full(len(ids), np.nan),
            where=f1_denominator > 0,
        )
        total = true_counts.sum()
        prefix = group.name
        result[f"{prefix}_state_accuracy"] = (
            float(true_positive.sum() / total) if total else float("nan")
        )
        result[f"{prefix}_state_macro_f1"] = _nanmean(state_f1)
        result[f"{prefix}_state_iou_per_class"] = state_iou
        for state_name, value in zip(group.state_names, state_iou):
            result[f"{prefix}_{state_name}_iou"] = float(value)
    return result


def segmentation_metrics(
    prediction: object,
    target: object,
    *,
    ignore_index: int | None = IGNORE_INDEX,
    spec: DatasetSpec | None = None,
) -> dict[str, object]:
    spec = spec or default_floodnet_spec()
    matrix = confusion_matrix(
        prediction, target, num_classes=spec.num_classes, ignore_index=ignore_index
    )
    result = metrics_from_confusion(matrix, spec=spec)
    result["confusion_matrix"] = matrix
    result.update(grouped_object_iou(prediction, target, ignore_index=ignore_index, spec=spec))
    result.update(
        state_metrics_from_semantic(prediction, target, ignore_index=ignore_index, spec=spec)
    )
    result.update(
        state_group_metrics_from_semantic(
            prediction, target, ignore_index=ignore_index, spec=spec
        )
    )
    return result


def _binary_iou(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    prediction = prediction & valid
    target = target & valid
    intersection = np.count_nonzero(prediction & target)
    union = np.count_nonzero(prediction | target)
    return float(intersection / union) if union else float("nan")


def grouped_object_iou(
    prediction: object,
    target: object,
    *,
    ignore_index: int | None = IGNORE_INDEX,
    spec: DatasetSpec | None = None,
) -> dict[str, float]:
    spec = spec or default_floodnet_spec()
    pred = _as_numpy(prediction)
    truth = _as_numpy(target)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and target shapes differ")
    valid = np.ones(truth.shape, dtype=bool)
    if ignore_index is not None:
        valid &= truth != ignore_index
    return {
        name: _binary_iou(np.isin(pred, ids), np.isin(truth, ids), valid)
        for name, ids in spec.binary_iou_groups.items()
    }


def state_metrics_from_semantic(
    prediction: object,
    target: object,
    *,
    ignore_index: int | None = IGNORE_INDEX,
    spec: DatasetSpec | None = None,
) -> dict[str, float]:
    spec = spec or default_floodnet_spec()
    if spec.state is None:
        return {}
    positive_ids = spec.state.positive_class_ids
    negative_ids = spec.state.negative_class_ids
    pred = _as_numpy(prediction)
    truth = _as_numpy(target)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and target shapes differ")
    valid = np.isin(truth, (*positive_ids, *negative_ids))
    if ignore_index is not None:
        valid &= truth != ignore_index
    if not np.any(valid):
        return {
            "state_accuracy": float("nan"),
            "state_macro_f1": float("nan"),
            f"{spec.state.positive_name}_precision": float("nan"),
            f"{spec.state.positive_name}_recall": float("nan"),
        }
    true_state = np.where(np.isin(truth, positive_ids), 1, 0)
    predicted_state = np.full(pred.shape, -1, dtype=np.int8)
    predicted_state[np.isin(pred, positive_ids)] = 1
    predicted_state[np.isin(pred, negative_ids)] = 0

    state_f1: list[float] = []
    precision_by_state: dict[int, float] = {}
    recall_by_state: dict[int, float] = {}
    for state in (0, 1):
        true_positive = np.count_nonzero(
            valid & (true_state == state) & (predicted_state == state)
        )
        false_positive = np.count_nonzero(
            valid & (true_state != state) & (predicted_state == state)
        )
        false_negative = np.count_nonzero(
            valid & (true_state == state) & (predicted_state != state)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else float("nan")
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else float("nan")
        )
        f1 = (
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
            if 2 * true_positive + false_positive + false_negative
            else float("nan")
        )
        precision_by_state[state] = float(precision)
        recall_by_state[state] = float(recall)
        state_f1.append(float(f1))
    accuracy = np.count_nonzero(
        valid & (predicted_state == true_state)
    ) / np.count_nonzero(valid)
    return {
        "state_accuracy": float(accuracy),
        "state_macro_f1": float(np.nanmean(state_f1)),
        f"{spec.state.positive_name}_precision": precision_by_state[1],
        f"{spec.state.positive_name}_recall": recall_by_state[1],
    }


def state_group_metrics_from_semantic(
    prediction: object,
    target: object,
    *,
    ignore_index: int | None = IGNORE_INDEX,
    spec: DatasetSpec | None = None,
) -> dict[str, object]:
    spec = spec or default_floodnet_spec()
    if spec.factorization is None:
        return {}
    matrix = confusion_matrix(
        prediction,
        target,
        num_classes=spec.num_classes,
        ignore_index=ignore_index,
    )
    return _state_group_metrics_from_confusion(matrix, spec=spec)


def semantic_boundary(
    labels: object,
    *,
    class_ids: Iterable[int] | None = None,
    ignore_index: int | None = IGNORE_INDEX,
) -> np.ndarray:
    array = _as_numpy(labels)
    if array.ndim != 2:
        raise ValueError(f"Boundary extraction expects a 2D mask, got {array.shape}")
    valid = np.ones(array.shape, dtype=bool)
    if ignore_index is not None:
        valid &= array != ignore_index
    values = np.isin(array, tuple(class_ids)) if class_ids is not None else array
    boundary = np.zeros(array.shape, dtype=bool)

    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    horizontal_difference = (values[:, 1:] != values[:, :-1]) & horizontal_valid
    boundary[:, 1:] |= horizontal_difference
    boundary[:, :-1] |= horizontal_difference

    vertical_valid = valid[1:, :] & valid[:-1, :]
    vertical_difference = (values[1:, :] != values[:-1, :]) & vertical_valid
    boundary[1:, :] |= vertical_difference
    boundary[:-1, :] |= vertical_difference
    return boundary


def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("Dilation radius must be non-negative")
    if radius == 0:
        return binary.copy()
    height, width = binary.shape
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(binary, dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            dilated |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return dilated


def boundary_f1(
    prediction: object,
    target: object,
    *,
    tolerance: int = 3,
    class_ids: Iterable[int] | None = None,
    ignore_index: int | None = IGNORE_INDEX,
) -> float:
    predicted_boundary = semantic_boundary(
        prediction, class_ids=class_ids, ignore_index=ignore_index
    )
    target_boundary = semantic_boundary(
        target, class_ids=class_ids, ignore_index=ignore_index
    )
    predicted_count = np.count_nonzero(predicted_boundary)
    target_count = np.count_nonzero(target_boundary)
    if predicted_count == 0 and target_count == 0:
        return 1.0
    if predicted_count == 0 or target_count == 0:
        return 0.0
    matched_prediction = np.count_nonzero(
        predicted_boundary & _dilate(target_boundary, tolerance)
    )
    matched_target = np.count_nonzero(
        target_boundary & _dilate(predicted_boundary, tolerance)
    )
    precision = matched_prediction / predicted_count
    recall = matched_target / target_count
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


class SegmentationMeter:
    """Accumulate a confusion matrix for an arbitrary dataset profile."""

    def __init__(
        self,
        ignore_index: int | None = IGNORE_INDEX,
        *,
        spec: DatasetSpec | None = None,
    ) -> None:
        self.spec = spec or default_floodnet_spec()
        self.ignore_index = ignore_index
        self.matrix = np.zeros(
            (self.spec.num_classes, self.spec.num_classes), dtype=np.int64
        )

    def update(self, prediction: object, target: object) -> None:
        self.matrix += confusion_matrix(
            prediction,
            target,
            num_classes=self.spec.num_classes,
            ignore_index=self.ignore_index,
        )

    def compute(self) -> dict[str, object]:
        result = metrics_from_confusion(self.matrix, spec=self.spec)
        result["confusion_matrix"] = self.matrix.copy()
        return result

    def reset(self) -> None:
        self.matrix.fill(0)
