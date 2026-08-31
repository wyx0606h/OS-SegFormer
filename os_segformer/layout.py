"""Official FloodNet root discovery and manifest reading."""

from __future__ import annotations

import csv
from pathlib import Path

from .constants import SUPERVISED_ROOT_NAME


def is_supervised_root(path: Path) -> bool:
    return (
        (path / "train" / "train-org-img").is_dir()
        and (path / "train" / "train-label-img").is_dir()
        and (path / "val" / "val-org-img").is_dir()
        and (path / "val" / "val-label-img").is_dir()
        and (path / "test" / "test-org-img").is_dir()
        and (path / "test" / "test-label-img").is_dir()
    )


def resolve_track1_root(data_root: str | Path) -> Path:
    """Resolve `FloodNet-Supervised_v1.0` itself or its parent directory."""

    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"FloodNet root does not exist: {root}")
    if is_supervised_root(root):
        return root
    candidate = root / SUPERVISED_ROOT_NAME
    if is_supervised_root(candidate):
        return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the official FloodNet-Supervised_v1.0 layout below "
        f"{root}"
    )


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    required = {"sample_id", "image_path"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} is missing columns: {sorted(missing)}"
        )
    return rows
