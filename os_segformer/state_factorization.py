"""Dataset-profile-driven object-state targets, composition, fusion, and losses."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .constants import IGNORE_INDEX
from .dataset_spec import FactorizationSpec, default_floodnet_spec


def _default_factorization() -> FactorizationSpec:
    factorization = default_floodnet_spec().factorization
    if factorization is None:  # pragma: no cover - release invariant
        raise RuntimeError("The built-in FloodNet profile has no factorization schema")
    return factorization


def _resolve_factorization(spec: FactorizationSpec | None) -> FactorizationSpec:
    return spec or _default_factorization()


def semantic_to_object_target(
    target: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    factorization_spec: FactorizationSpec | None = None,
) -> torch.Tensor:
    """Map flat semantic labels to the profile-defined object identities."""

    spec = _resolve_factorization(factorization_spec)
    mapping = torch.tensor(
        spec.semantic_to_object, dtype=torch.long, device=target.device
    )
    result = torch.full_like(target, ignore_index)
    valid = (target >= 0) & (target < len(spec.semantic_to_object))
    valid &= target != ignore_index
    result[valid] = mapping[target[valid]]
    return result


def semantic_to_state_target(
    target: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    factorization_spec: FactorizationSpec | None = None,
) -> torch.Tensor:
    """Map each stateful label to its group-local state index."""

    spec = _resolve_factorization(factorization_spec)
    result = torch.full_like(target, ignore_index)
    for group in spec.state_groups:
        for state_index, semantic_id in enumerate(group.semantic_class_ids):
            result[target == semantic_id] = state_index
    return result


def _state_probability_groups(
    state_logits: torch.Tensor,
    spec: FactorizationSpec,
) -> list[torch.Tensor]:
    """Normalize each object's state slice independently."""

    if state_logits.ndim != 4 or state_logits.shape[1] != spec.num_state_channels:
        raise ValueError(
            f"Expected state logits [B,{spec.num_state_channels},H,W], "
            f"got {state_logits.shape}"
        )
    groups: list[torch.Tensor] = []
    offset = 0
    for group in spec.state_groups:
        size = len(group.semantic_class_ids)
        groups.append(torch.softmax(state_logits[:, offset : offset + size], dim=1))
        offset += size
    return groups


def compose_hierarchical_probabilities(
    object_logits: torch.Tensor,
    state_logits: torch.Tensor,
    *,
    factorization_spec: FactorizationSpec | None = None,
) -> torch.Tensor:
    """Compose P(object|x) P(state|object,x) into semantic probabilities."""

    spec = _resolve_factorization(factorization_spec)
    if object_logits.ndim != 4 or object_logits.shape[1] != spec.num_object_classes:
        raise ValueError(
            f"Expected object logits [B,{spec.num_object_classes},H,W], "
            f"got {object_logits.shape}"
        )
    if (
        object_logits.shape[0] != state_logits.shape[0]
        or object_logits.shape[-2:] != state_logits.shape[-2:]
    ):
        raise ValueError("Object and state logits must share batch and spatial dimensions")

    object_probability = torch.softmax(object_logits, dim=1)
    state_probability_groups = _state_probability_groups(state_logits, spec)
    composed = object_probability.new_zeros(
        (
            object_probability.shape[0],
            len(spec.semantic_to_object),
            object_probability.shape[2],
            object_probability.shape[3],
        )
    )
    grouped_semantic_ids: set[int] = set()
    for group, state_probability in zip(
        spec.state_groups, state_probability_groups
    ):
        grouped_semantic_ids.update(group.semantic_class_ids)
        object_mass = object_probability[:, group.object_class_id]
        for state_index, semantic_id in enumerate(group.semantic_class_ids):
            composed[:, semantic_id] = (
                object_mass * state_probability[:, state_index]
            )
    for semantic_id, object_id in enumerate(spec.semantic_to_object):
        if semantic_id not in grouped_semantic_ids:
            composed[:, semantic_id] = object_probability[:, object_id]
    return composed


def fuse_semantic_and_hierarchical_logits(
    semantic_logits: torch.Tensor,
    hierarchical_probabilities: torch.Tensor,
    *,
    fusion_weight: float,
    epsilon: float = 1e-8,
    factorization_spec: FactorizationSpec | None = None,
) -> torch.Tensor:
    """Refine state allocation while preserving flat group mass.

    For every stateful object group G_o, the total mass
    sum(c in G_o) P_flat(c|x) is retained exactly. Only the conditional
    distribution inside G_o is blended with the factorized posterior.
    Non-stateful classes are copied from the flat posterior.
    """

    if not 0.0 <= fusion_weight <= 1.0:
        raise ValueError("fusion_weight must be in [0, 1]")
    if semantic_logits.shape != hierarchical_probabilities.shape:
        raise ValueError("Flat logits and factorized probabilities must match")
    if fusion_weight == 0.0:
        return semantic_logits

    spec = _resolve_factorization(factorization_spec)
    flat_log_probability = torch.log_softmax(semantic_logits, dim=1)
    factor_log_probability = hierarchical_probabilities.clamp_min(epsilon).log()
    fused_log_probability = flat_log_probability.clone()

    for group in spec.state_groups:
        ids = list(group.semantic_class_ids)
        flat_group = flat_log_probability[:, ids]
        flat_conditional = flat_group - torch.logsumexp(
            flat_group, dim=1, keepdim=True
        )
        factor_group = factor_log_probability[:, ids]
        factor_conditional = factor_group - torch.logsumexp(
            factor_group, dim=1, keepdim=True
        )
        conditional_logits = (
            (1.0 - fusion_weight) * flat_conditional
            + fusion_weight * factor_conditional
        )
        conditional_log_probability = conditional_logits - torch.logsumexp(
            conditional_logits, dim=1, keepdim=True
        )
        flat_group_log_mass = torch.logsumexp(
            flat_group, dim=1, keepdim=True
        )
        fused_log_probability[:, ids] = (
            flat_group_log_mass + conditional_log_probability
        )
    return fused_log_probability


def _safe_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, *, ignore_index: int
) -> torch.Tensor:
    if not torch.any(target != ignore_index):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def _masked_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    valid = target != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0
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
    present = torch.sum(one_hot, dim=dimensions) > 0
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice[present].mean()


def _conditional_state_loss(
    state_logits: torch.Tensor,
    semantic_target: torch.Tensor,
    *,
    dice_weight: float,
    ignore_index: int,
    factorization_spec: FactorizationSpec,
) -> torch.Tensor:
    _state_probability_groups(state_logits, factorization_spec)
    losses: list[torch.Tensor] = []
    offset = 0
    for group in factorization_spec.state_groups:
        size = len(group.semantic_class_ids)
        group_logits = state_logits[:, offset : offset + size]
        offset += size
        group_target = torch.full_like(semantic_target, ignore_index)
        for state_index, semantic_id in enumerate(group.semantic_class_ids):
            group_target[semantic_target == semantic_id] = state_index
        if torch.any(group_target != ignore_index):
            losses.append(
                _safe_cross_entropy(
                    group_logits, group_target, ignore_index=ignore_index
                )
                + dice_weight
                * _masked_dice_loss(
                    group_logits,
                    group_target,
                    num_classes=size,
                    ignore_index=ignore_index,
                )
            )
    if not losses:
        return state_logits.sum() * 0.0
    return torch.stack(losses).mean()


def state_factorization_terms(
    *,
    semantic_logits: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    config: Mapping[str, Any] | None,
    ignore_index: int = IGNORE_INDEX,
    factorization_spec: FactorizationSpec | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the released object and grouped-state auxiliary objectives."""

    module_config = dict(config or {})
    zero = semantic_logits.sum() * 0.0
    if not bool(module_config.get("enabled", False)):
        return {"total": zero, "object": zero, "state": zero}
    if "object" not in auxiliary or "state" not in auxiliary:
        raise ValueError("OS-SegFormer requires object and state auxiliary logits")

    spec = _resolve_factorization(factorization_spec)
    object_logits = auxiliary["object"]
    state_logits = auxiliary["state"]
    target_size = target.shape[-2:]
    if object_logits.shape[-2:] != target_size:
        object_logits = F.interpolate(
            object_logits, size=target_size, mode="bilinear", align_corners=False
        )
    if state_logits.shape[-2:] != target_size:
        state_logits = F.interpolate(
            state_logits, size=target_size, mode="bilinear", align_corners=False
        )

    object_target = semantic_to_object_target(
        target, ignore_index=ignore_index, factorization_spec=spec
    )
    object_loss = _safe_cross_entropy(
        object_logits, object_target, ignore_index=ignore_index
    ) + float(module_config.get("object_dice_weight", 1.0)) * _masked_dice_loss(
        object_logits,
        object_target,
        num_classes=spec.num_object_classes,
        ignore_index=ignore_index,
    )
    state_loss = _conditional_state_loss(
        state_logits,
        target,
        dice_weight=float(module_config.get("state_dice_weight", 1.0)),
        ignore_index=ignore_index,
        factorization_spec=spec,
    )
    total = (
        float(module_config.get("object_weight", 0.25)) * object_loss
        + float(module_config.get("state_weight", 0.25)) * state_loss
    )
    return {"total": total, "object": object_loss, "state": state_loss}


def state_factorization_loss(
    *,
    semantic_logits: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    config: Mapping[str, Any] | None,
    ignore_index: int = IGNORE_INDEX,
    factorization_spec: FactorizationSpec | None = None,
) -> torch.Tensor:
    return state_factorization_terms(
        semantic_logits=semantic_logits,
        auxiliary=auxiliary,
        target=target,
        config=config,
        ignore_index=ignore_index,
        factorization_spec=factorization_spec,
    )["total"]
