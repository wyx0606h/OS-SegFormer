"""Supervised semantic and object-state objectives."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .constants import IGNORE_INDEX, NUM_CLASSES
from .dataset_spec import DatasetSpec
from .models import SegmentationModelOutput, extract_posterior_logits
from .state_factorization import state_factorization_terms


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    num_classes: int = NUM_CLASSES,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss for mutually exclusive semantic classes."""

    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError(
            f"Expected logits [B,C,H,W] and target [B,H,W], "
            f"got {logits.shape}, {target.shape}"
        )
    valid = target != ignore_index
    safe_target = target.clamp(min=0, max=num_classes - 1)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(
        safe_target, num_classes=num_classes
    ).permute(0, 3, 1, 2).to(probabilities.dtype)
    valid_float = valid.unsqueeze(1).to(probabilities.dtype)
    probabilities = probabilities * valid_float
    one_hot = one_hot * valid_float
    dimensions = (0, 2, 3)
    intersection = torch.sum(probabilities * one_hot, dim=dimensions)
    denominator = torch.sum(probabilities + one_hot, dim=dimensions)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    present = torch.sum(one_hot, dim=dimensions) > 0
    if not torch.any(present):
        return logits.sum() * 0.0
    return 1.0 - dice[present].mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, Any] | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute the paper's CE+Dice semantic objective."""

    config = dict(config or {"name": "ce_dice"})
    if str(config.get("name", "ce_dice")).casefold() != "ce_dice":
        raise ValueError("This release supports loss.name=ce_dice")
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index)
    dice = multiclass_dice_loss(
        logits,
        target,
        ignore_index=ignore_index,
        num_classes=int(logits.shape[1]),
    )
    return (
        float(config.get("ce_weight", 1.0)) * ce
        + float(config.get("dice_weight", 1.0)) * dice
    )


def supervised_objective(
    output: object,
    target: torch.Tensor,
    config: Mapping[str, Any] | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
    dataset_spec: DatasetSpec | None = None,
) -> torch.Tensor:
    return supervised_objective_components(
        output,
        target,
        config,
        ignore_index=ignore_index,
        dataset_spec=dataset_spec,
    )["total"]


def supervised_objective_components(
    output: object,
    target: torch.Tensor,
    config: Mapping[str, Any] | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
    dataset_spec: DatasetSpec | None = None,
) -> dict[str, torch.Tensor]:
    """Return semantic, object, and grouped-state losses separately."""

    config = dict(config or {})
    loss_config = dict(config.get("loss") or {})
    semantic_source = str(
        loss_config.get("semantic_source", "semantic_direct")
    ).casefold()
    logits = extract_posterior_logits(output, source=semantic_source)
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    semantic = segmentation_loss(
        logits,
        target,
        loss_config,
        ignore_index=ignore_index,
    )
    modules = config.get("modules", {})
    factor_config = (
        modules.get("state_factorization", {})
        if isinstance(modules, Mapping)
        else {}
    )
    auxiliary = output.auxiliary if isinstance(output, SegmentationModelOutput) else {}
    factorization = state_factorization_terms(
        semantic_logits=logits,
        auxiliary=auxiliary,
        target=target,
        config=factor_config,
        ignore_index=ignore_index,
        factorization_spec=(dataset_spec.factorization if dataset_spec else None),
    )
    return {
        "total": semantic + factorization["total"],
        "semantic": semantic,
        "factorization": factorization["total"],
        "object": factorization["object"],
        "state": factorization["state"],
    }
