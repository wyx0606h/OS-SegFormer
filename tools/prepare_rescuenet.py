"""Build a reproducible manifest for the official read-only RescueNet split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


RESCUENET_EXPECTED_COUNTS = {"train": 3595, "validation": 449, "test": 450}
RESCUENET_LAYOUT = {
    "train": ("train/train-org-img", "train/train-label-img"),
    "validation": ("val/val-org-img", "val/val-label-img"),
    "test": ("test/test-org-img", "test/test-label-img"),
}
FIELDNAMES = (
    "sample_id",
    "split",
    "scene_label",
    "image_path",
    "mask_path",
    "official_split",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the official 3595/449/450 RescueNet manifest."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _files_by_id(
    directory: Path, *, suffix: str, mask: bool = False
) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required RescueNet directory is missing: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.casefold() != suffix:
            continue
        sample_id = path.stem
        if mask:
            if not sample_id.casefold().endswith("_lab"):
                raise ValueError(f"RescueNet mask must end in _lab: {path}")
            sample_id = sample_id[:-4]
        key = sample_id.casefold()
        if key in files:
            raise ValueError(f"Duplicate RescueNet sample ID {sample_id!r}: {directory}")
        files[key] = path
    return files


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _rows_sha256(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(
        ",".join(row[field] for field in FIELDNAMES) for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_rows(data_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RescueNet root does not exist: {root}")
    rows: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    summary: dict[str, Any] = {
        "dataset": "RescueNet",
        "data_root": str(root),
        "splits": {},
    }
    for split, (image_relative, mask_relative) in RESCUENET_LAYOUT.items():
        images = _files_by_id(root / image_relative, suffix=".jpg")
        masks = _files_by_id(root / mask_relative, suffix=".png", mask=True)
        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        if missing_masks or missing_images:
            raise ValueError(
                f"Image/mask mismatch for {split}: missing_masks={missing_masks[:10]}, "
                f"missing_images={missing_images[:10]}"
            )
        overlap = sorted(set(images) & set(seen))
        if overlap:
            raise ValueError(f"RescueNet sample IDs overlap across splits: {overlap[:10]}")
        split_rows: list[dict[str, str]] = []
        for key in sorted(images):
            seen[key] = split
            row = {
                "sample_id": images[key].stem,
                "split": split,
                "scene_label": "",
                "image_path": _relative(images[key], root),
                "mask_path": _relative(masks[key], root),
                "official_split": split,
            }
            split_rows.append(row)
            rows.append(row)
        expected = RESCUENET_EXPECTED_COUNTS[split]
        if len(split_rows) != expected:
            raise ValueError(
                f"Expected {expected} RescueNet {split} samples, found {len(split_rows)}"
            )
        summary["splits"][split] = {"samples": len(split_rows)}
    summary["total_samples"] = len(rows)
    summary["canonical_rows_sha256"] = _rows_sha256(rows)
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows, summary = build_rows(args.data_root)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "manifest.csv", rows)
    summary_path = output / "split_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {summary_path}")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
