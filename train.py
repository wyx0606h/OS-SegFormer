"""Unified supervised semantic-segmentation training entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.scatter_gather import gather
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from os_segformer.config import load_yaml_config  # noqa: E402
from os_segformer.dataset_spec import dataset_spec_to_dict, resolve_dataset_spec  # noqa: E402
from os_segformer.experiment import (  # noqa: E402
    append_history_csv,
    apply_path_overrides,
    configure_logger,
    ensure_run_layout,
    evaluate_model,
    make_dataset,
    make_loader,
    save_training_checkpoint,
    write_metrics_files,
    write_resolved_yaml,
)
from os_segformer.losses import supervised_objective_components  # noqa: E402
from os_segformer.metrics import metrics_from_confusion  # noqa: E402
from os_segformer.models import SegmentationModelOutput, build_model, extract_logits  # noqa: E402
from os_segformer.training import build_optimizer, collect_runtime_metadata, set_reproducible_seed  # noqa: E402


class SegmentationDataParallel(torch.nn.DataParallel):
    """DataParallel variant that preserves the repository output dataclass."""

    def gather(self, outputs: list[object], output_device: int) -> object:
        if outputs and isinstance(outputs[0], SegmentationModelOutput):
            segmentation_outputs = [
                output
                for output in outputs
                if isinstance(output, SegmentationModelOutput)
            ]
            if len(segmentation_outputs) != len(outputs):
                raise TypeError("All DataParallel outputs must use SegmentationModelOutput")
            logits = gather(
                [output.logits for output in segmentation_outputs],
                output_device,
                dim=self.dim,
            )
            auxiliary = {
                name: gather(
                    [output.auxiliary[name] for output in segmentation_outputs],
                    output_device,
                    dim=self.dim,
                )
                for name in segmentation_outputs[0].auxiliary
            }
            return SegmentationModelOutput(logits=logits, auxiliary=auxiliary)
        return super().gather(outputs, output_device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OS-SegFormer or the same-backbone flat SegFormer-B0."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root")
    parser.add_argument("--output-dir", type=Path, help="Override experiment.output_dir")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint, usually runs/<exp>/checkpoints/last.pth",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and data counts but do not build model or train")
    parser.add_argument("--max-iterations", type=int, help="Temporary CLI override for smoke tests")
    parser.add_argument("--val-interval", type=int, help="Temporary CLI override for smoke tests")
    parser.add_argument("--max-eval-samples", type=int, help="Limit validation samples, mainly for smoke tests")
    return parser.parse_args()


def _resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def _scheduled_lr(training: dict[str, Any], step: int, max_iterations: int) -> float:
    base_lr = float(training["learning_rate"])
    scheduler = str(training.get("scheduler", "constant")).casefold()
    warmup = int(training.get("warmup_iterations", 0))
    poly_reference = str(
        training.get("poly_reference", "post_warmup")
    ).casefold()
    if scheduler == "poly" and poly_reference == "full_schedule":
        power = float(training.get("poly_power", 1.0))
        progress = min(max(step, 0) / max(max_iterations, 1), 1.0)
        regular_lr = base_lr * ((1.0 - progress) ** power)
        if warmup > 0 and step <= warmup:
            warmup_ratio = float(training.get("warmup_ratio", 0.0))
            warmup_progress = float(step) / float(warmup)
            factor = warmup_ratio + (1.0 - warmup_ratio) * warmup_progress
            return regular_lr * factor
        return regular_lr
    if warmup > 0 and step <= warmup:
        warmup_ratio = float(training.get("warmup_ratio", 0.0))
        progress = float(step) / float(warmup)
        return base_lr * (warmup_ratio + (1.0 - warmup_ratio) * progress)
    if scheduler in {"", "none", "constant"}:
        return base_lr
    if scheduler == "poly":
        power = float(training.get("poly_power", 1.0))
        denominator = max(max_iterations - warmup, 1)
        progress = min(max(step - warmup, 0) / denominator, 1.0)
        return base_lr * ((1.0 - progress) ** power)
    raise ValueError(f"Unsupported training.scheduler: {scheduler}")


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def _infinite_training_batches(loader: Any, sampler: DistributedSampler | None = None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _validation_shard_indices(
    dataset_length: int,
    *,
    rank: int,
    world_size: int,
    max_samples: int | None = None,
) -> list[int]:
    sample_count = dataset_length if max_samples is None else min(dataset_length, max_samples)
    return list(range(rank, sample_count, world_size))


def _empty_validation_payload(
    config: dict[str, Any],
    *,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    dataset_spec = resolve_dataset_spec(config)
    matrix = np.zeros((dataset_spec.num_classes, dataset_spec.num_classes), dtype=np.int64)
    metrics = metrics_from_confusion(matrix, spec=dataset_spec)
    metrics.update(
        {
            "confusion_matrix": matrix,
            "mean_boundary_f1": float("nan"),
            "mean_building_iou": float("nan"),
            "mean_road_iou": float("nan"),
            "mean_state_macro_f1": float("nan"),
        }
    )
    return {
        "experiment_name": config["experiment"].get(
            "name", config["experiment"].get("run_id")
        ),
        "dataset": dataset_spec.name,
        "dataset_profile": dataset_spec.source,
        "protocol": config.get("dataset", {}).get("protocol"),
        "split": split,
        "checkpoint": None,
        "num_samples": 0,
        "tile_size": int(evaluation.get("tile_size", 512)),
        "stride": int(evaluation.get("stride", 384)),
        "tile_batch_size": int(evaluation.get("tile_batch_size", 4)),
        "prediction_source": str(evaluation.get("prediction_source", "fused")),
        "metrics": metrics,
        "per_sample": [],
        "runtime": collect_runtime_metadata(device),
    }


def _merge_validation_payloads(
    payloads: list[dict[str, Any] | None], *, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    available = [payload for payload in payloads if payload is not None]
    if not available:
        raise ValueError("No validation payloads were produced")
    dataset_spec = resolve_dataset_spec(config or {"dataset": {"profile": "floodnet"}})
    matrix = np.zeros((dataset_spec.num_classes, dataset_spec.num_classes), dtype=np.int64)
    total_samples = 0
    per_sample: list[dict[str, Any]] = []
    for payload in available:
        matrix += np.asarray(payload["metrics"]["confusion_matrix"], dtype=np.int64)
        total_samples += int(payload["num_samples"])
        per_sample.extend(payload.get("per_sample", []))

    metrics = metrics_from_confusion(matrix, spec=dataset_spec)
    metrics["confusion_matrix"] = matrix

    def weighted_mean(metric_name: str) -> float:
        numerator = 0.0
        denominator = 0
        for payload in available:
            sample_count = int(payload["num_samples"])
            value = float(payload["metrics"].get(metric_name, float("nan")))
            if sample_count <= 0 or np.isnan(value):
                continue
            numerator += value * sample_count
            denominator += sample_count
        return float(numerator / denominator) if denominator else float("nan")

    for metric_name in (
        "mean_boundary_f1",
        "mean_building_iou",
        "mean_road_iou",
        "mean_state_macro_f1",
    ):
        metrics[metric_name] = weighted_mean(metric_name)

    merged = dict(available[0])
    merged["num_samples"] = total_samples
    merged["metrics"] = metrics
    merged["per_sample"] = per_sample
    return merged


def _evaluate_validation_shard(
    model: torch.nn.Module,
    dataset: Any,
    *,
    config: dict[str, Any],
    split: str,
    device: torch.device,
    max_samples: int | None,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    indices = _validation_shard_indices(
        len(dataset), rank=rank, world_size=world_size, max_samples=max_samples
    )
    if not indices:
        return _empty_validation_payload(config, split=split, device=device)
    return evaluate_model(
        _unwrap_model(model),
        Subset(dataset, indices),
        config=config,
        split=split,
        device=device,
    )


def _resolve_data_parallel_device_ids(training: dict[str, Any]) -> list[int]:
    raw_ids = training.get("data_parallel_device_ids")
    if raw_ids in (None, "", "all"):
        return list(range(torch.cuda.device_count()))
    if not isinstance(raw_ids, (list, tuple)):
        raise ValueError("training.data_parallel_device_ids must be a list or 'all'")
    device_ids = [int(value) for value in raw_ids]
    if not device_ids:
        raise ValueError("training.data_parallel_device_ids cannot be empty")
    visible_count = torch.cuda.device_count()
    if min(device_ids) < 0 or max(device_ids) >= visible_count:
        raise ValueError(
            "training.data_parallel_device_ids must refer to visible CUDA devices"
        )
    return device_ids


def _ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _ddp_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_rank_zero() -> bool:
    return _ddp_rank() == 0


def _distributed_enabled(training: dict[str, Any]) -> bool:
    return bool(training.get("distributed_data_parallel", False)) or _ddp_world_size() > 1


def _initialize_distributed_if_needed(training: dict[str, Any]) -> bool:
    if not _distributed_enabled(training):
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA")
    if _ddp_world_size() < 2:
        raise RuntimeError("Distributed training requires torchrun with WORLD_SIZE >= 2")
    expected_world_size = int(training.get("distributed_world_size", _ddp_world_size()))
    if _ddp_world_size() != expected_world_size:
        raise RuntimeError(
            f"Configured distributed_world_size={expected_world_size}, got WORLD_SIZE={_ddp_world_size()}"
        )
    local_rank = _ddp_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        backend = str(training.get("distributed_backend", "nccl"))
        try:
            dist.init_process_group(
                backend=backend,
                device_id=torch.device("cuda", local_rank),
            )
        except TypeError:
            dist.init_process_group(backend=backend)
    return True


def _initialize_distributed(training: dict[str, Any]) -> bool:
    """Initialize distributed training when requested by the configuration."""

    return _initialize_distributed_if_needed(training)


def _barrier_if_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[_ddp_local_rank()])
        else:
            dist.barrier()


def _cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def _maybe_wrap_data_parallel(
    model: torch.nn.Module,
    training: dict[str, Any],
    *,
    device: torch.device,
    logger: Any,
) -> torch.nn.Module:
    if not bool(training.get("data_parallel", False)):
        return model
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("training.data_parallel requires CUDA")
    device_ids = _resolve_data_parallel_device_ids(training)
    if len(device_ids) < 2:
        raise RuntimeError(
            "training.data_parallel requires at least two visible CUDA devices"
        )
    logger.info("Using DataParallel on visible CUDA device ids: %s", device_ids)
    return SegmentationDataParallel(model, device_ids=device_ids)


def _maybe_wrap_distributed(
    model: torch.nn.Module,
    *,
    enabled: bool,
    logger: Any,
) -> torch.nn.Module:
    if not enabled:
        return model
    local_rank = _ddp_local_rank()
    logger.info(
        "Using DistributedDataParallel rank=%d local_rank=%d world_size=%d",
        _ddp_rank(),
        local_rank,
        _ddp_world_size(),
    )
    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )


def _active_cuda_memory_device_ids(
    training: dict[str, Any], device: torch.device
) -> list[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return []
    if bool(training.get("data_parallel", False)):
        return _resolve_data_parallel_device_ids(training)
    if _distributed_enabled(training):
        return [_ddp_local_rank()]
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _reset_cuda_peak_memory(training: dict[str, Any], device: torch.device) -> None:
    for device_id in _active_cuda_memory_device_ids(training, device):
        torch.cuda.reset_peak_memory_stats(device_id)


def _max_cuda_peak_memory_gb(
    training: dict[str, Any], device: torch.device
) -> float | None:
    device_ids = _active_cuda_memory_device_ids(training, device)
    if not device_ids:
        return None
    return max(
        torch.cuda.max_memory_allocated(device_id) / 1024**3
        for device_id in device_ids
    )


def main() -> int:
    args = parse_args()
    distributed = False
    config = load_yaml_config(args.config)
    dataset_spec = resolve_dataset_spec(config)
    apply_path_overrides(
        config,
        data_root=args.data_root,
        output_dir=args.output_dir,
    )
    if args.max_iterations is not None:
        config["training"]["max_iterations"] = args.max_iterations
    if args.val_interval is not None:
        config["training"]["val_interval"] = args.val_interval
    if args.max_eval_samples is not None:
        config.setdefault("evaluation", {})["max_eval_samples"] = args.max_eval_samples

    experiment = config["experiment"]
    training = config["training"]
    if not args.dry_run:
        distributed = _initialize_distributed_if_needed(training)
    output_dir = Path(experiment["output_dir"]).expanduser().resolve()
    if (
        output_dir.exists()
        and args.resume is None
        and not args.dry_run
        and (not _distributed_enabled(training) or _is_rank_zero())
    ):
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    train_dataset = make_dataset(config, str(config["data"].get("train_split", "train")), training=True)
    val_dataset = make_dataset(config, str(config["data"].get("validation_split", "validation")), training=False)
    plan = {
        "experiment": experiment.get("name", experiment.get("run_id")),
        "dataset": dataset_spec.name,
        "dataset_profile": dataset_spec.source,
        "protocol": config.get("dataset", {}).get("protocol"),
        "output_dir": str(output_dir),
        "data_root": config["data"]["data_root"],
        "manifest": config["data"]["manifest"],
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "model": config["model"],
        "loss": config.get("loss", {"name": "ce_dice"}),
        "modules": config.get("modules", {}),
        "training": training,
    }
    if _is_rank_zero():
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        if _is_rank_zero():
            print("Dry run only: no model was built and no training was started.")
        return 0

    if _is_rank_zero():
        paths = ensure_run_layout(output_dir)
        write_resolved_yaml(output_dir / "config_resolved.yaml", config)
        write_resolved_yaml(
            output_dir / "dataset_profile_resolved.yaml",
            dataset_spec_to_dict(dataset_spec),
        )
        (output_dir / "runtime_metadata.json").write_text(
            json.dumps(collect_runtime_metadata(training["device"]), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    _barrier_if_distributed(distributed)
    paths = (
        ensure_run_layout(output_dir)
        if _is_rank_zero()
        else {
            "root": output_dir,
            "checkpoints": output_dir / "checkpoints",
            "metrics": output_dir / "metrics",
            "curves": output_dir / "curves",
            "predictions": output_dir / "predictions",
        }
    )
    logger = configure_logger(
        output_dir / ("train.log" if _is_rank_zero() else f"train_rank{_ddp_rank()}.log")
    )
    logger.info("Resolved plan: %s", json.dumps(plan, ensure_ascii=False))

    device = torch.device("cuda", _ddp_local_rank()) if distributed else torch.device(str(training["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    set_reproducible_seed(int(experiment.get("seed", 0)))
    model = build_model(
        config["model"],
        class_names=dataset_spec.class_names,
        ignore_index=dataset_spec.ignore_index,
        dataset_spec=dataset_spec,
    ).to(device)
    optimizer = build_optimizer(model, training)
    use_amp = bool(training.get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp and device.type == "cuda",
        init_scale=float(training.get("amp_init_scale", 65536.0)),
    )

    start_iteration = 0
    best_miou = float("-inf")
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume.expanduser().resolve(), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = int(checkpoint.get("iteration", 0))
        best_miou = float(checkpoint.get("best_miou", float("-inf")))
        history = list(checkpoint.get("history", []))
        logger.info("Resumed from %s at iteration %d", args.resume, start_iteration)
    if distributed and bool(training.get("data_parallel", False)):
        raise RuntimeError("Use either DataParallel or DistributedDataParallel, not both")
    model = _maybe_wrap_distributed(model, enabled=distributed, logger=logger)
    model = _maybe_wrap_data_parallel(model, training, device=device, logger=logger)
    _reset_cuda_peak_memory(training, device)

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=_ddp_world_size(),
            rank=_ddp_rank(),
            shuffle=True,
            seed=int(experiment.get("seed", 0)),
            drop_last=False,
        )
        if distributed
        else None
    )
    train_loader = make_loader(
        config,
        str(config["data"].get("train_split", "train")),
        training=True,
        sampler=train_sampler,
    )
    train_iter = _infinite_training_batches(train_loader, train_sampler)
    max_iterations = int(training["max_iterations"])
    # CLI ``--max-iterations`` is a smoke-test override applied after config
    # validation, so clamp a configured diagnostic cap to that temporary
    # schedule instead of accidentally running past the smoke budget.
    stop_iteration = min(
        int(training.get("stop_iteration", max_iterations)), max_iterations
    )
    val_interval = int(training.get("val_interval", max_iterations))
    checkpoint_iterations = {
        int(value) for value in training.get("checkpoint_iterations", [])
    }
    grad_accum = int(training.get("gradient_accumulation_steps", 1))
    max_eval_samples = config.get("evaluation", {}).get("max_eval_samples")
    max_eval_samples = int(max_eval_samples) if max_eval_samples not in (None, "") else None

    for iteration in range(start_iteration + 1, stop_iteration + 1):
        model.train()
        current_lr = _scheduled_lr(training, iteration, max_iterations)
        _set_optimizer_lr(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        component_values = {
            name: 0.0
            for name in ("semantic", "factorization", "object", "state")
        }
        for _ in range(grad_accum):
            batch = next(train_iter)
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["mask"].to(device, non_blocking=True)
            amp_enabled = use_amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                model_output = model(images)
                logits = _resize_logits(extract_logits(model_output), labels)
                loss_output = (
                    SegmentationModelOutput(
                        logits=logits,
                        auxiliary=model_output.auxiliary,
                    )
                    if isinstance(model_output, SegmentationModelOutput)
                    else logits
                )
                objective = supervised_objective_components(
                    loss_output,
                    labels,
                    config,
                    ignore_index=dataset_spec.ignore_index,
                    dataset_spec=dataset_spec,
                )
                loss = objective["total"] / grad_accum
            scaler.scale(loss).backward()
            loss_value += float(loss.detach())
            for name in component_values:
                component_values[name] += float(objective[name].detach()) / grad_accum
        clip_norm = training.get("gradient_clip_norm")
        grad_norm_value: float | None = None
        if clip_norm not in (None, ""):
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(clip_norm)
            )
            grad_norm_value = float(grad_norm.detach().cpu())
        scaler.step(optimizer)
        scaler.update()

        should_eval = (
            iteration == 1
            or iteration % val_interval == 0
            or iteration == stop_iteration
        )
        row: dict[str, Any] = {
            "iteration": iteration,
            "train_loss": loss_value,
            "learning_rate": current_lr,
            "train_grad_norm": grad_norm_value,
            "cuda_peak_memory_gb": _max_cuda_peak_memory_gb(training, device),
            **{f"train_loss_{name}": value for name, value in component_values.items()},
        }
        if should_eval:
            validation_split = str(config["data"].get("validation_split", "validation"))
            val_payload: dict[str, Any] | None = None
            if distributed:
                local_val_payload = _evaluate_validation_shard(
                    model,
                    val_dataset,
                    config=config,
                    split=validation_split,
                    device=device,
                    max_samples=max_eval_samples,
                    rank=_ddp_rank(),
                    world_size=_ddp_world_size(),
                )
                gathered_payloads: list[dict[str, Any] | None] = [
                    None for _ in range(_ddp_world_size())
                ]
                dist.all_gather_object(gathered_payloads, local_val_payload)
                if _is_rank_zero():
                    val_payload = _merge_validation_payloads(gathered_payloads, config=config)
            elif _is_rank_zero():
                val_payload = evaluate_model(
                    _unwrap_model(model),
                    val_dataset,
                    config=config,
                    split=validation_split,
                    device=device,
                    max_samples=max_eval_samples,
                )
            if _is_rank_zero():
                assert val_payload is not None
                val_metrics = val_payload["metrics"]
                selection_metric = str(
                    config.get("evaluation", {}).get(
                        "selection_metric", dataset_spec.primary_metric
                    )
                )
                if selection_metric not in val_metrics:
                    raise KeyError(
                        f"Selection metric {selection_metric!r} is not produced by "
                        f"dataset profile {dataset_spec.name}"
                    )
                selection_value = float(val_metrics[selection_metric])
                row.update(
                    {
                        "validation_miou": selection_value,
                        "validation_selection_metric": selection_metric,
                        "validation_macro_f1": val_metrics["macro_f1"],
                        "validation_pixel_accuracy": val_metrics["pixel_accuracy"],
                        "validation_samples": val_payload["num_samples"],
                    }
                )
                if "flooded_miou" in val_metrics:
                    row["validation_flooded_miou"] = val_metrics["flooded_miou"]
                metrics_dir = paths["metrics"] / f"validation_iter_{iteration:07d}"
                write_metrics_files(metrics_dir, val_payload)
                if selection_value >= best_miou:
                    best_miou = selection_value
                    save_training_checkpoint(
                        paths["checkpoints"] / "best_miou.pth",
                        model=model,
                        optimizer=optimizer,
                        iteration=iteration,
                        best_miou=best_miou,
                        history=history + [row],
                        config=config,
                    )
                    logger.info(
                        "New best %s %.6f at iteration %d",
                        selection_metric,
                        best_miou,
                        iteration,
                    )
        if should_eval:
            _barrier_if_distributed(distributed)
        if _is_rank_zero():
            history.append(row)
            append_history_csv(paths["curves"] / "history.csv", history)
        if _is_rank_zero() and iteration in checkpoint_iterations:
            save_training_checkpoint(
                paths["checkpoints"] / f"iteration_{iteration:07d}.pth",
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                best_miou=best_miou,
                history=history,
                config=config,
            )
        if _is_rank_zero() and (should_eval or iteration % 500 == 0):
            save_training_checkpoint(
                paths["checkpoints"] / "last.pth",
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                best_miou=best_miou,
                history=history,
                config=config,
            )
        if _is_rank_zero():
            logger.info("iteration=%d train_loss=%.6f best_miou=%.6f", iteration, loss_value, best_miou)

    final_peak_vram_gb = _max_cuda_peak_memory_gb(training, device)
    final_peak_vram_by_rank: list[float | None] | None = None
    if distributed:
        final_peak_vram_by_rank = [None for _ in range(_ddp_world_size())]
        dist.all_gather_object(final_peak_vram_by_rank, final_peak_vram_gb)
        finite_peaks = [value for value in final_peak_vram_by_rank if value is not None]
        final_peak_vram_gb = max(finite_peaks) if finite_peaks else None

    if not _is_rank_zero():
        _cleanup_distributed(distributed)
        return 0

    summary = {
        "experiment": experiment.get("name", experiment.get("run_id")),
        "protocol": config.get("dataset", {}).get("protocol"),
        "dataset": dataset_spec.name,
        "selection_metric": config.get("evaluation", {}).get(
            "selection_metric", dataset_spec.primary_metric
        ),
        "max_iterations": max_iterations,
        "stop_iteration": stop_iteration,
        "completed_iteration": history[-1]["iteration"] if history else start_iteration,
        "best_miou": best_miou,
        "best_checkpoint": str(paths["checkpoints"] / "best_miou.pth"),
        "last_checkpoint": str(paths["checkpoints"] / "last.pth"),
        "milestone_checkpoints": [
            str(paths["checkpoints"] / f"iteration_{iteration:07d}.pth")
            for iteration in sorted(checkpoint_iterations)
            if (paths["checkpoints"] / f"iteration_{iteration:07d}.pth").is_file()
        ],
        "history_rows": len(history),
        "peak_vram_gb": final_peak_vram_gb,
        "peak_vram_gb_by_rank": final_peak_vram_by_rank,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Training complete: %s", json.dumps(summary, ensure_ascii=False))
    _cleanup_distributed(distributed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
