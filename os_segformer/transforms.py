"""Paired spatial transforms for FloodNet images and semantic masks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

Size2D = Tuple[int, int]


def _pair(value: int | Sequence[int]) -> Size2D:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("Size must be positive")
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected a two-element size, got {value}")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError("Size must be positive")
    return height, width


def _pad_to_size(
    image: Image.Image,
    mask: Optional[Image.Image],
    size: Size2D,
    *,
    image_fill: int | tuple[int, int, int] = 0,
    mask_fill: int = 255,
) -> tuple[Image.Image, Optional[Image.Image]]:
    target_height, target_width = size
    pad_width = max(target_width - image.width, 0)
    pad_height = max(target_height - image.height, 0)
    if not pad_width and not pad_height:
        return image, mask
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    border = (left, top, right, bottom)
    image = ImageOps.expand(image, border=border, fill=image_fill)
    if mask is not None:
        mask = ImageOps.expand(mask, border=border, fill=mask_fill)
    return image, mask


class PairedCompose:
    def __init__(self, transforms: Iterable[object]) -> None:
        self.transforms = tuple(transforms)

    def __call__(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


@dataclass
class RandomHorizontalFlip:
    probability: float = 0.5
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Flip probability must be in [0, 1]")
        self.rng = self.rng or random.Random()

    def __call__(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        assert self.rng is not None
        if self.rng.random() >= self.probability:
            return image, mask
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if mask is not None:
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image, mask


@dataclass
class RandomVerticalFlip:
    probability: float = 0.5
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Flip probability must be in [0, 1]")
        self.rng = self.rng or random.Random()

    def __call__(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        assert self.rng is not None
        if self.rng.random() >= self.probability:
            return image, mask
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if mask is not None:
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image, mask


@dataclass
class RandomScale:
    minimum: float = 0.75
    maximum: float = 1.25
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        if self.minimum <= 0 or self.maximum < self.minimum:
            raise ValueError("Invalid random scale range")
        self.rng = self.rng or random.Random()

    def __call__(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        assert self.rng is not None
        scale = self.rng.uniform(self.minimum, self.maximum)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        if mask is not None:
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        return image, mask


@dataclass
class RandomCrop:
    size: int | Sequence[int] = 512
    image_fill: int | tuple[int, int, int] = 0
    mask_fill: int = 255
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        self.size = _pair(self.size)
        self.rng = self.rng or random.Random()

    def _sample_top_left(self, width: int, height: int) -> tuple[int, int]:
        crop_height, crop_width = self.size
        assert self.rng is not None
        left = self.rng.randint(0, width - crop_width)
        top = self.rng.randint(0, height - crop_height)
        return left, top

    def __call__(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        image, mask = _pad_to_size(
            image,
            mask,
            self.size,
            image_fill=self.image_fill,
            mask_fill=self.mask_fill,
        )
        crop_height, crop_width = self.size
        left, top = self._sample_top_left(image.width, image.height)
        box = (left, top, left + crop_width, top + crop_height)
        image = image.crop(box)
        if mask is not None:
            mask = mask.crop(box)
        return image, mask


@dataclass
class ClassAwareRandomCrop(RandomCrop):
    class_ids: Sequence[int] = (1, 3)
    class_aware_probability: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        self.class_ids = tuple(int(class_id) for class_id in self.class_ids)
        if not self.class_ids:
            raise ValueError("Class-aware crop requires at least one class ID")
        if not 0.0 <= self.class_aware_probability <= 1.0:
            raise ValueError("Class-aware probability must be in [0, 1]")

    def _class_aware_top_left(
        self, mask: Image.Image, width: int, height: int
    ) -> tuple[int, int] | None:
        assert self.rng is not None
        array = np.asarray(mask)
        coordinates = np.argwhere(np.isin(array, self.class_ids))
        if not len(coordinates):
            return None
        row, column = coordinates[self.rng.randrange(len(coordinates))]
        crop_height, crop_width = self.size
        minimum_left = max(0, int(column) - crop_width + 1)
        maximum_left = min(int(column), width - crop_width)
        minimum_top = max(0, int(row) - crop_height + 1)
        maximum_top = min(int(row), height - crop_height)
        left = self.rng.randint(minimum_left, maximum_left)
        top = self.rng.randint(minimum_top, maximum_top)
        return left, top

    def __call__(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        image, mask = _pad_to_size(
            image,
            mask,
            self.size,
            image_fill=self.image_fill,
            mask_fill=self.mask_fill,
        )
        crop_height, crop_width = self.size
        assert self.rng is not None
        top_left = None
        if mask is not None and self.rng.random() < self.class_aware_probability:
            top_left = self._class_aware_top_left(mask, image.width, image.height)
        if top_left is None:
            top_left = self._sample_top_left(image.width, image.height)
        left, top = top_left
        box = (left, top, left + crop_width, top + crop_height)
        image = image.crop(box)
        if mask is not None:
            mask = mask.crop(box)
        return image, mask


@dataclass
class CenterCrop:
    size: int | Sequence[int] = 512
    image_fill: int | tuple[int, int, int] = 0
    mask_fill: int = 255

    def __post_init__(self) -> None:
        self.size = _pair(self.size)

    def __call__(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        image, mask = _pad_to_size(
            image,
            mask,
            self.size,
            image_fill=self.image_fill,
            mask_fill=self.mask_fill,
        )
        crop_height, crop_width = self.size
        left = (image.width - crop_width) // 2
        top = (image.height - crop_height) // 2
        box = (left, top, left + crop_width, top + crop_height)
        image = image.crop(box)
        if mask is not None:
            mask = mask.crop(box)
        return image, mask


@dataclass
class DeterministicClassCrop:
    """Crop around a deterministic median pixel of the first available class."""

    size: int | Sequence[int] = 512
    class_ids: Sequence[int] = (1, 3)
    image_fill: int | tuple[int, int, int] = 0
    mask_fill: int = 255

    def __post_init__(self) -> None:
        self.size = _pair(self.size)
        self.class_ids = tuple(int(class_id) for class_id in self.class_ids)
        if not self.class_ids:
            raise ValueError("Deterministic class crop requires class IDs")

    def __call__(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        image, mask = _pad_to_size(
            image,
            mask,
            self.size,
            image_fill=self.image_fill,
            mask_fill=self.mask_fill,
        )
        crop_height, crop_width = self.size
        left = (image.width - crop_width) // 2
        top = (image.height - crop_height) // 2
        if mask is not None:
            array = np.asarray(mask)
            for class_id in self.class_ids:
                coordinates = np.argwhere(array == class_id)
                if len(coordinates):
                    row, column = coordinates[len(coordinates) // 2]
                    left = min(
                        max(int(column) - crop_width // 2, 0),
                        image.width - crop_width,
                    )
                    top = min(
                        max(int(row) - crop_height // 2, 0),
                        image.height - crop_height,
                    )
                    break
        box = (left, top, left + crop_width, top + crop_height)
        image = image.crop(box)
        if mask is not None:
            mask = mask.crop(box)
        return image, mask


def build_supervised_train_transform(
    *,
    crop_size: int = 512,
    seed: int = 0,
    scale_range: tuple[float, float] = (0.75, 1.25),
    class_ids: Sequence[int] = (1, 3),
    class_aware_probability: float = 0.5,
    mask_fill: int = 255,
) -> PairedCompose:
    """Build the shared supervised spatial transform."""

    master = random.Random(seed)
    return PairedCompose(
        (
            RandomScale(
                minimum=scale_range[0],
                maximum=scale_range[1],
                rng=random.Random(master.randrange(2**63)),
            ),
            RandomHorizontalFlip(
                probability=0.5,
                rng=random.Random(master.randrange(2**63)),
            ),
            RandomVerticalFlip(
                probability=0.5,
                rng=random.Random(master.randrange(2**63)),
            ),
            ClassAwareRandomCrop(
                size=crop_size,
                class_ids=class_ids,
                class_aware_probability=class_aware_probability,
                mask_fill=mask_fill,
                rng=random.Random(master.randrange(2**63)),
            ),
        )
    )
