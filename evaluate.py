"""Unified semantic-segmentation checkpoint evaluation entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from os_segformer.config import load_yaml_config  # noqa: E402
from os_segformer.dataset_spec import resolve_dataset_spec  # noqa: E402
from os_segformer.experiment import apply_path_overrides, evaluate_model, make_dataset, write_metrics_files  # noqa: E402
from os_segformer.models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint on a manifest split.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, help="Manifest split value, for example validation or test")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <experiment.output_dir>/metrics/<split>_<checkpoint>")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root")
    parser.add_argument("--device", help="Override training.device")
    parser.add_argument("--max-samples", type=int, help="Limit samples for smoke tests")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve dataset/evaluation plan but do not build model or load checkpoint")
    parser.add_argument(
        "--prediction-source",
        choices=("fused", "semantic_direct", "hierarchical"),
        help="Override evaluation.prediction_source for posterior diagnostics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    dataset_spec = resolve_dataset_spec(config)
    apply_path_overrides(config, data_root=args.data_root)
    device = torch.device(args.device or str(config["training"]["device"]))
    checkpoint_path = args.checkpoint.expanduser().resolve()
    dataset = make_dataset(config, args.split, training=False)
    output_dir = args.output_dir
    if output_dir is None:
        name = checkpoint_path.stem
        output_dir = Path(config["experiment"]["output_dir"]) / "metrics" / f"{args.split}_{name}"
    output_dir = output_dir.expanduser().resolve()
    plan = {
        "mode": "dry-run" if args.dry_run else "execute",
        "experiment": config["experiment"].get("name", config["experiment"].get("run_id")),
        "dataset": dataset_spec.name,
        "dataset_profile": dataset_spec.source,
        "protocol": config.get("dataset", {}).get("protocol"),
        "split": args.split,
        "samples": len(dataset),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "evaluation": config["evaluation"],
        "prediction_source": args.prediction_source
        or config["evaluation"].get("prediction_source", "fused"),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("Dry run only: no model was built and no checkpoint was loaded.")
        return 0
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")

    model = build_model(
        config["model"],
        class_names=dataset_spec.class_names,
        ignore_index=dataset_spec.ignore_index,
        dataset_spec=dataset_spec,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state is None:
        raise ValueError("Checkpoint must contain model_state_dict")
    model.load_state_dict(state)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to write into non-empty evaluation directory: {output_dir}")
    predictions_dir = output_dir / "predictions" if args.save_predictions else None
    payload = evaluate_model(
        model,
        dataset,
        config=config,
        split=args.split,
        device=device,
        checkpoint=str(checkpoint_path),
        max_samples=args.max_samples,
        save_predictions_dir=predictions_dir,
        prediction_source=args.prediction_source,
    )
    write_metrics_files(output_dir, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "per_sample"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
