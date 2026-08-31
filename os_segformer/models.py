"""SegFormer-B0 and the paper's object-state factorized extension."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .constants import CLASS_NAMES, IGNORE_INDEX
from .dataset_spec import DatasetSpec, FactorizationSpec, default_floodnet_spec
from .state_factorization import (
    compose_hierarchical_probabilities,
    fuse_semantic_and_hierarchical_logits,
)


@dataclass(frozen=True)
class SegmentationModelOutput:
    """Semantic logits plus inspectable object-state branch outputs."""

    logits: torch.Tensor
    auxiliary: Mapping[str, torch.Tensor] = field(default_factory=dict)


class MissingSegFormerDependency(RuntimeError):
    """Raised when the Hugging Face SegFormer backend is unavailable."""


class UnsupportedModelError(ValueError):
    """Raised when a release config requests an unsupported model."""


class MultiScaleFactorDecoder(torch.nn.Module):
    """Fuse all four SegFormer encoder stages without changing the flat decoder."""

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        *,
        decoder_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(hidden_sizes) != 4:
            raise ValueError("OS-SegFormer expects four SegFormer encoder stages")
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive")
        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(int(channels), decoder_channels, kernel_size=1),
                    torch.nn.BatchNorm2d(decoder_channels),
                    torch.nn.GELU(),
                )
                for channels in hidden_sizes
            ]
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Conv2d(
                decoder_channels * len(hidden_sizes),
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(decoder_channels),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
        )

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(hidden_states) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} encoder stages, got {len(hidden_states)}"
            )
        if any(state.ndim != 4 for state in hidden_states):
            raise ValueError("Encoder hidden states must be NCHW tensors")
        target_size = hidden_states[0].shape[-2:]
        projected = []
        for state, projection in zip(hidden_states, self.projections):
            feature = projection(state)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(feature)
        return self.fusion(torch.cat(projected, dim=1))


class ObjectConditionedStateHead(torch.nn.Module):
    """Predict grouped object-dependent states with additive object conditioning."""

    def __init__(
        self,
        channels: int,
        *,
        state_object_class_ids: Sequence[int],
        conditional_state_channels: int,
        detach_object_posterior: bool = False,
    ) -> None:
        super().__init__()
        self.state_object_class_ids = tuple(int(value) for value in state_object_class_ids)
        if not self.state_object_class_ids:
            raise ValueError("At least one stateful object group is required")
        if conditional_state_channels <= 0:
            raise ValueError("conditional_state_channels must be positive")
        self.detach_object_posterior = bool(detach_object_posterior)
        self.object_conditioner = torch.nn.Conv2d(
            len(self.state_object_class_ids), channels, kernel_size=1
        )
        self.refinement = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.GELU(),
        )
        self.classifier = torch.nn.Conv2d(
            channels, conditional_state_channels, kernel_size=1
        )

    def forward(
        self, features: torch.Tensor, object_logits: torch.Tensor
    ) -> torch.Tensor:
        object_posterior = torch.softmax(object_logits, dim=1)[
            :, list(self.state_object_class_ids)
        ]
        if self.detach_object_posterior:
            object_posterior = object_posterior.detach()
        conditioned = features + self.object_conditioner(object_posterior)
        return self.classifier(self.refinement(conditioned))


class OSSegFormer(torch.nn.Module):
    """SegFormer-B0 with the paper's multi-scale object-state factor branch."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        hidden_sizes: Sequence[int],
        factorization_spec: FactorizationSpec,
        decoder_channels: int = 64,
        dropout: float = 0.1,
        detach_object_posterior: bool = False,
        fusion_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if not hasattr(base_model, "segformer") or not hasattr(base_model, "decode_head"):
            raise TypeError("OS-SegFormer requires a SegFormer segmentation model")
        if not 0.0 <= fusion_weight <= 1.0:
            raise ValueError("fusion_weight must be in [0, 1]")
        self.factorization_spec = factorization_spec
        self.base_model = base_model
        self.factor_decoder = MultiScaleFactorDecoder(
            hidden_sizes,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )
        self.object_head = torch.nn.Conv2d(
            decoder_channels,
            factorization_spec.num_object_classes,
            kernel_size=1,
        )
        self.state_head = ObjectConditionedStateHead(
            decoder_channels,
            state_object_class_ids=factorization_spec.state_object_class_ids,
            conditional_state_channels=factorization_spec.num_state_channels,
            detach_object_posterior=detach_object_posterior,
        )
        self.fusion_weight = float(fusion_weight)

    def forward(self, image: torch.Tensor) -> SegmentationModelOutput:
        encoder_output = self.base_model.segformer(
            pixel_values=image,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = encoder_output.hidden_states
        if hidden_states is None:
            raise RuntimeError("SegFormer encoder did not return hidden states")

        # The flat semantic path is kept intact and directly supervised.
        semantic_direct = self.base_model.decode_head(hidden_states)

        factor_features = self.factor_decoder(hidden_states)
        object_logits = self.object_head(factor_features)
        state_logits = self.state_head(factor_features, object_logits)
        if object_logits.shape[-2:] != semantic_direct.shape[-2:]:
            object_for_composition = F.interpolate(
                object_logits,
                size=semantic_direct.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            state_for_composition = F.interpolate(
                state_logits,
                size=semantic_direct.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            object_for_composition = object_logits
            state_for_composition = state_logits

        hierarchical = compose_hierarchical_probabilities(
            object_for_composition,
            state_for_composition,
            factorization_spec=self.factorization_spec,
        )
        fused_logits = fuse_semantic_and_hierarchical_logits(
            semantic_direct,
            hierarchical,
            fusion_weight=self.fusion_weight,
            factorization_spec=self.factorization_spec,
        )
        return SegmentationModelOutput(
            logits=fused_logits,
            auxiliary={
                "semantic_direct": semantic_direct,
                "object": object_logits,
                "state": state_logits,
                "hierarchical": hierarchical,
            },
        )


def segformer_dependency_status() -> dict[str, bool]:
    return {
        package: importlib.util.find_spec(package) is not None
        for package in ("transformers", "safetensors")
    }


def require_segformer_dependencies(*, pretrained: bool = True) -> None:
    status = segformer_dependency_status()
    required = ("transformers", "safetensors") if pretrained else ("transformers",)
    missing = [name for name in required if not status[name]]
    if missing:
        raise MissingSegFormerDependency(
            "Missing SegFormer dependencies: " + ", ".join(missing)
        )


def build_segformer_b0(model_config: Mapping[str, Any]) -> torch.nn.Module:
    """Build the same Hugging Face SegFormer-B0 used by both released methods."""

    pretrained = bool(model_config.get("pretrained", True))
    require_segformer_dependencies(pretrained=pretrained)
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    num_labels = int(model_config.get("num_labels", len(CLASS_NAMES)))
    class_names = tuple(model_config.get("class_names", CLASS_NAMES))
    if len(class_names) != num_labels:
        raise ValueError("model class_names length must match num_labels")
    id2label = {index: name for index, name in enumerate(class_names)}
    label2id = {name: index for index, name in id2label.items()}
    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            str(model_config["pretrained_model_name_or_path"]),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            local_files_only=bool(model_config.get("local_files_only", False)),
        )

    configuration = SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        decoder_hidden_size=256,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return SegformerForSemanticSegmentation(configuration)


def build_model(
    model_config: Mapping[str, Any],
    *,
    class_names: Sequence[str] | None = None,
    ignore_index: int = IGNORE_INDEX,
    dataset_spec: DatasetSpec | None = None,
) -> torch.nn.Module:
    """Build either flat SegFormer-B0 or OS-SegFormer from one framework."""

    options = dict(model_config)
    if dataset_spec is not None:
        class_names = dataset_spec.class_names
        ignore_index = dataset_spec.ignore_index
    if class_names is not None:
        options["class_names"] = tuple(str(item) for item in class_names)
    options["ignore_index"] = int(ignore_index)
    if str(options.get("name", "segformer_b0")) != "segformer_b0":
        raise UnsupportedModelError("This release supports model.name=segformer_b0")

    model = build_segformer_b0(options)
    factorization = options.get("state_factorization", {})
    if not (factorization and bool(factorization.get("enabled", False))):
        return model

    hierarchy = (
        dataset_spec.factorization
        if dataset_spec is not None
        else default_floodnet_spec().factorization
    )
    if hierarchy is None:
        raise ValueError("OS-SegFormer requires dataset.profile.factorization")
    hidden_sizes = tuple(int(value) for value in model.config.hidden_sizes)
    return OSSegFormer(
        model,
        hidden_sizes=hidden_sizes,
        factorization_spec=hierarchy,
        decoder_channels=int(factorization.get("decoder_channels", 64)),
        dropout=float(factorization.get("dropout", 0.1)),
        detach_object_posterior=bool(
            factorization.get("detach_object_posterior", False)
        ),
        fusion_weight=float(factorization.get("fusion_weight", 0.25)),
    )


def extract_logits(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, SegmentationModelOutput):
        return output.logits
    if hasattr(output, "logits") and torch.is_tensor(output.logits):
        return output.logits
    if isinstance(output, dict) and torch.is_tensor(output.get("logits")):
        return output["logits"]
    raise TypeError("Model output must be a tensor or expose tensor logits")


def extract_posterior_logits(
    output: object,
    source: str = "fused",
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Extract the fused, flat, or composed semantic posterior as logits."""

    normalized_source = str(source).casefold()
    if normalized_source in {"fused", "final", "logits"}:
        return extract_logits(output)
    if not isinstance(output, SegmentationModelOutput):
        raise ValueError(
            f"prediction source '{source}' requires SegmentationModelOutput"
        )
    if normalized_source == "semantic_direct":
        tensor = output.auxiliary.get("semantic_direct")
    elif normalized_source == "hierarchical":
        probabilities = output.auxiliary.get("hierarchical")
        tensor = (
            probabilities.clamp_min(epsilon).log()
            if torch.is_tensor(probabilities)
            else None
        )
    else:
        raise ValueError(
            "prediction source must be fused, semantic_direct, or hierarchical"
        )
    if not torch.is_tensor(tensor):
        raise ValueError(f"model output does not expose prediction source '{source}'")
    return tensor
