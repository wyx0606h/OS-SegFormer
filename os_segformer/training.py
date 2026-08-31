"""Small shared helpers used by the release training entry point."""

from __future__ import annotations

import platform
import random
from typing import Any, Mapping

import numpy as np
import torch


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: torch.nn.Module, training_config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    """Build the frozen AdamW optimizer for either released model."""

    if str(training_config["optimizer"]).casefold() != "adamw":
        raise ValueError("This release supports optimizer=adamw")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        parameters,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )


def checkpoint_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return getattr(model, "module", model).state_dict()


def collect_runtime_metadata(device: torch.device | str) -> dict[str, Any]:
    target_device = torch.device(device)
    cuda_available = torch.cuda.is_available()
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "configured_device": str(target_device),
    }
    if target_device.type == "cuda" and cuda_available:
        device_index = (
            target_device.index
            if target_device.index is not None
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(device_index)
        metadata.update(
            {
                "cuda_device_index": device_index,
                "cuda_device_name": properties.name,
                "cuda_total_memory_gb": properties.total_memory / 1024**3,
            }
        )
    return metadata
