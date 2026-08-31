"""Sliding-window probability fusion for high-resolution segmentation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
import torch.nn.functional as torch_functional

from .models import extract_posterior_logits


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("Window values must be positive")
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected two values, got {value}")
    first, second = int(value[0]), int(value[1])
    if first <= 0 or second <= 0:
        raise ValueError("Window values must be positive")
    return first, second


def sliding_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= 0 or tile <= 0 or stride <= 0:
        raise ValueError("Length, tile, and stride must be positive")
    if stride > tile:
        raise ValueError("Stride must not exceed tile size; gaps are not allowed")
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.no_grad()
def sliding_window_predict_sources(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    prediction_sources: Sequence[str],
    tile_size: int | Sequence[int] = 512,
    stride: int | Sequence[int] = 384,
    tile_batch_size: int = 4,
    device: torch.device | str | None = None,
    use_amp: bool = False,
) -> dict[str, torch.Tensor]:
    """Predict several posteriors with one model forward per tile batch.

    Args:
        image: ``[C,H,W]`` or ``[1,C,H,W]`` tensor. Batch sizes above one are
            intentionally rejected to keep per-image geometry explicit.
    """

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected one image [C,H,W] or [1,C,H,W], got {image.shape}")
    if tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be positive")
    sources = tuple(str(source).casefold() for source in prediction_sources)
    if not sources:
        raise ValueError("At least one prediction source is required")
    if len(set(sources)) != len(sources):
        raise ValueError("Prediction sources must be unique")

    tile_height, tile_width = _pair(tile_size)
    stride_height, stride_width = _pair(stride)
    if stride_height > tile_height or stride_width > tile_width:
        raise ValueError("Stride must not exceed tile size")

    original_height, original_width = image.shape[-2:]
    padded_height = max(original_height, tile_height)
    padded_width = max(original_width, tile_width)
    pad_bottom = padded_height - original_height
    pad_right = padded_width - original_width
    padded = torch_functional.pad(image, (0, pad_right, 0, pad_bottom))

    row_positions = sliding_positions(padded_height, tile_height, stride_height)
    column_positions = sliding_positions(padded_width, tile_width, stride_width)
    coordinates = [
        (top, left) for top in row_positions for left in column_positions
    ]

    target_device = torch.device(device) if device is not None else next(
        model.parameters(), torch.empty(0)
    ).device
    if target_device.type == "meta":
        target_device = image.device
    model_was_training = model.training
    model.eval()

    probability_sums: dict[str, torch.Tensor] = {}
    coverage = torch.zeros(
        (1, 1, padded_height, padded_width),
        dtype=torch.float32,
        device=target_device,
    )
    def amp_context():
        if use_amp and target_device.type in {"cuda", "cpu"}:
            return torch.autocast(device_type=target_device.type, enabled=True)
        return nullcontext()

    try:
        for start in range(0, len(coordinates), tile_batch_size):
            batch_coordinates = coordinates[start : start + tile_batch_size]
            tiles = torch.cat(
                [
                    padded[
                        :,
                        :,
                        top : top + tile_height,
                        left : left + tile_width,
                    ]
                    for top, left in batch_coordinates
                ],
                dim=0,
            ).to(target_device)
            with amp_context():
                output = model(tiles)
                logits_by_source = {
                    source: extract_posterior_logits(output, source=source)
                    for source in sources
                }
            for source, logits in logits_by_source.items():
                if logits.ndim != 4:
                    raise ValueError(
                        f"Expected model logits [B,C,H,W], got {logits.shape}"
                    )
                if logits.shape[0] != len(batch_coordinates):
                    raise ValueError("Model changed the tile batch dimension")
                if logits.shape[-2:] != (tile_height, tile_width):
                    logits = torch_functional.interpolate(
                        logits,
                        size=(tile_height, tile_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                probabilities = torch.softmax(logits.float(), dim=1)
                if source not in probability_sums:
                    probability_sums[source] = torch.zeros(
                        (1, probabilities.shape[1], padded_height, padded_width),
                        dtype=torch.float32,
                        device=target_device,
                    )
                for tile_index, (top, left) in enumerate(batch_coordinates):
                    probability_sums[source][
                        :,
                        :,
                        top : top + tile_height,
                        left : left + tile_width,
                    ] += probabilities[tile_index : tile_index + 1]
            for top, left in batch_coordinates:
                coverage[
                    :,
                    :,
                    top : top + tile_height,
                    left : left + tile_width,
                ] += 1.0
    finally:
        model.train(model_was_training)

    if len(probability_sums) != len(sources) or torch.any(coverage == 0):
        raise RuntimeError("Sliding-window coverage is incomplete")
    return {
        source: (probability_sums[source] / coverage)[
            :, :, :original_height, :original_width
        ]
        for source in sources
    }


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    tile_size: int | Sequence[int] = 512,
    stride: int | Sequence[int] = 384,
    tile_batch_size: int = 4,
    device: torch.device | str | None = None,
    use_amp: bool = False,
    prediction_source: str = "fused",
) -> torch.Tensor:
    """Predict one full-resolution posterior by averaging overlapping tiles."""

    source = str(prediction_source).casefold()
    return sliding_window_predict_sources(
        model,
        image,
        prediction_sources=(source,),
        tile_size=tile_size,
        stride=stride,
        tile_batch_size=tile_batch_size,
        device=device,
        use_amp=use_amp,
    )[source]
