"""Manifest-driven PyTorch dataset for semantic segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import IGNORE_INDEX, NUM_CLASSES
from .layout import read_manifest, resolve_track1_root

PairedTransform = Callable[
    [Image.Image, Optional[Image.Image]],
    Union[Tuple[Any, Optional[Any]], dict[str, Any]],
]


class ManifestSegmentationDataset(Dataset[dict[str, Any]]):
    """Load labeled segmentation samples from a versioned manifest.

    Paths in the manifest are interpreted relative to the resolved dataset
    directory. ``root_resolver='floodnet'`` discovers the official
    FloodNet-Supervised layout; ``'direct'`` treats ``data_root`` as the
    dataset directory used by RescueNet.
    """

    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        transform: PairedTransform | None = None,
        to_tensor: bool = True,
        validate_masks: bool = True,
        ignore_index: int = IGNORE_INDEX,
        num_classes: int = NUM_CLASSES,
        root_resolver: str = "floodnet",
        align_image_to_mask: bool = True,
        image_mean: tuple[float, float, float] | None = None,
        image_std: tuple[float, float, float] | None = None,
    ) -> None:
        root_mode = str(root_resolver).strip().casefold()
        if root_mode == "floodnet":
            self.data_root = resolve_track1_root(data_root)
        elif root_mode == "direct":
            self.data_root = Path(data_root).expanduser().resolve()
            if not self.data_root.is_dir():
                raise FileNotFoundError(f"Dataset root does not exist: {self.data_root}")
        else:
            raise ValueError("root_resolver must be 'direct' or 'floodnet'")
        # Compatibility attribute used by older scripts.
        self.track1_root = self.data_root
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        rows = read_manifest(self.manifest_path)
        if split is not None:
            if "split" not in rows[0]:
                raise ValueError(
                    f"Manifest has no split column but split={split!r} was requested"
                )
            rows = [row for row in rows if row.get("split") == split]
        if not rows:
            raise ValueError(
                f"No samples selected from {self.manifest_path} for split={split!r}"
            )
        self.rows = rows
        self.transform = transform
        self.to_tensor = to_tensor
        self.validate_masks = validate_masks
        self.ignore_index = ignore_index
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        self.num_classes = num_classes
        self.root_resolver = root_mode
        self.align_image_to_mask = align_image_to_mask
        if (image_mean is None) != (image_std is None):
            raise ValueError("image_mean and image_std must be provided together")
        if image_mean is not None and (
            len(image_mean) != 3
            or len(image_std or ()) != 3
            or any(value <= 0 for value in image_std or ())
        ):
            raise ValueError("image_mean/image_std must contain three valid channels")
        self.image_mean = image_mean
        self.image_std = image_std

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_manifest_path(self, value: str, *, required: bool) -> Path | None:
        value = value.strip()
        if not value:
            if required:
                raise ValueError("Required manifest path is empty")
            return None
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError(f"Manifest paths must be relative, got: {value}")
        resolved = (self.data_root / relative).resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError as error:
            raise ValueError(f"Manifest path escapes data root: {value}") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"Manifest file does not exist: {resolved}")
        return resolved

    def _coerce_mask(self, mask: Image.Image, sample_id: str) -> Image.Image:
        """Return a single-channel class-index mask.

        FloodNet masks are expected to be class-index images. Some copies are
        stored as RGB/RGBA images whose first three channels repeat the same
        class index. Those are accepted and collapsed to one channel; true color
        masks are rejected because they need an explicit palette mapping.
        """

        array = np.asarray(mask)
        if array.ndim == 2:
            return Image.fromarray(array.astype(np.uint8, copy=False), mode="L")
        if array.ndim == 3 and array.shape[2] in (3, 4):
            rgb = array[..., :3]
            if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
                raise ValueError(
                    f"RGB/RGBA mask for {sample_id} has non-identical RGB channels; "
                    "expected repeated class-index channels, not a color mask"
                )
            return Image.fromarray(rgb[..., 0].astype(np.uint8, copy=False), mode="L")
        raise ValueError(f"Mask for {sample_id} must be 2D, RGB or RGBA, got {array.shape}")

    def _validate_mask(self, mask: Image.Image, sample_id: str) -> None:
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError(
                f"Mask for {sample_id} must be single-channel after coercion, got {array.shape}"
            )
        valid = ((array >= 0) & (array < self.num_classes)) | (array == self.ignore_index)
        if not np.all(valid):
            invalid = np.unique(array[~valid]).tolist()
            raise ValueError(f"Invalid mask IDs for {sample_id}: {invalid}")

    @staticmethod
    def _load_image_frame_zero(path: Path) -> tuple[Image.Image, int]:
        """Decode only frame 0, including for RescueNet's two-frame MPO files."""

        with Image.open(path) as handle:
            frame_count = int(getattr(handle, "n_frames", 1))
            handle.seek(0)
            image = handle.convert("RGB")
        return image, frame_count

    @staticmethod
    def _image_to_tensor(image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(image):
            return image
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected RGB image array, got shape {array.shape}")
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)

    @staticmethod
    def _mask_to_tensor(
        mask: Image.Image | np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if torch.is_tensor(mask):
            return mask.long()
        return torch.from_numpy(np.asarray(mask).copy()).long()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = row["sample_id"]
        image_path = self._resolve_manifest_path(row["image_path"], required=True)
        mask_path = self._resolve_manifest_path(
            row.get("mask_path", ""), required=False
        )
        assert image_path is not None

        image, image_frame_count = self._load_image_frame_zero(image_path)
        mask: Image.Image | None = None
        original_image_size = image.size
        if mask_path is not None:
            with Image.open(mask_path) as handle:
                mask = self._coerce_mask(handle.copy(), sample_id)
            if image.size != mask.size:
                if not self.align_image_to_mask:
                    raise ValueError(
                        f"Image/mask size mismatch for {sample_id}: "
                        f"{image.size} versus {mask.size}"
                    )
                image = image.resize(mask.size, resample=Image.Resampling.BILINEAR)
            if self.validate_masks:
                self._validate_mask(mask, sample_id)

        crop_values = [
            row.get("crop_top", "").strip(),
            row.get("crop_left", "").strip(),
            row.get("crop_height", "").strip(),
            row.get("crop_width", "").strip(),
        ]
        if any(crop_values):
            if not all(crop_values):
                raise ValueError(
                    f"Manifest crop for {sample_id} must define top/left/height/width"
                )
            top, left, crop_height, crop_width = map(int, crop_values)
            if (
                top < 0
                or left < 0
                or crop_height <= 0
                or crop_width <= 0
                or top + crop_height > image.height
                or left + crop_width > image.width
            ):
                raise ValueError(
                    f"Manifest crop is outside aligned image for {sample_id}: "
                    f"top={top}, left={left}, height={crop_height}, width={crop_width}"
                )
            box = (left, top, left + crop_width, top + crop_height)
            image = image.crop(box)
            if mask is not None:
                mask = mask.crop(box)

        transformed_image: Any = image
        transformed_mask: Any | None = mask
        if self.transform is not None:
            transformed = self.transform(image, mask)
            if isinstance(transformed, dict):
                if "image" not in transformed:
                    raise ValueError("Transform dictionary must contain an image key")
                transformed_image = transformed["image"]
                transformed_mask = transformed.get("mask")
            elif isinstance(transformed, tuple) and len(transformed) == 2:
                transformed_image, transformed_mask = transformed
            else:
                raise TypeError(
                    "Paired transform must return (image, mask) or a dictionary"
                )

        if self.to_tensor:
            transformed_image = self._image_to_tensor(transformed_image)
            transformed_mask = self._mask_to_tensor(transformed_mask)
            if self.image_mean is not None and self.image_std is not None:
                mean = torch.tensor(
                    self.image_mean,
                    dtype=transformed_image.dtype,
                    device=transformed_image.device,
                ).view(3, 1, 1)
                std = torch.tensor(
                    self.image_std,
                    dtype=transformed_image.dtype,
                    device=transformed_image.device,
                ).view(3, 1, 1)
                transformed_image = (transformed_image - mean) / std

        return {
            "id": sample_id,
            "image": transformed_image,
            "mask": transformed_mask,
            "scene_label": row.get("scene_label", ""),
            "split": row.get("split", row.get("official_split", "")),
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path is not None else "",
            "original_image_size": original_image_size,
            "aligned_image_size": image.size,
            "image_frame_index": 0,
            "image_frame_count": image_frame_count,
        }
