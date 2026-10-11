"""DDP training and evaluation entry point."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler

from .config import load_config
from .data import PASTISHDCrops, collate_pastis_hd
from .data_mtlcc import MTLCCS2Tiles, collate_mtlcc
from .losses import build_loss
from .metrics import SegmentationMetrics
from .models import build_model, count_parameters


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation without padding or duplicated examples."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.indices = list(range(rank, size, world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class CycleAlignedDistributedSampler(DistributedSampler):
    """Pad each rank to complete fixed-size optimizer update cycles.

    ``DistributedSampler(drop_last=False)`` already repeats a few samples so
    every rank receives the same number.  That number is not necessarily a
    multiple of the samples consumed by one optimizer update, however.  An
    incomplete final accumulation group then either changes the effective
    global batch or drops samples.  Aligning ``num_samples`` to the local
    update size keeps every original sample, uses a fixed optimizer batch, and
    rotates the small padded tail with the sampler's epoch-wise shuffle.
    """

    def __init__(
        self,
        dataset: PASTISHDCrops | MTLCCS2Tiles,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        samples_per_cycle: int,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        if samples_per_cycle < 1:
            raise ValueError("samples_per_cycle must be positive")
        self.samples_per_cycle = int(samples_per_cycle)
        self.num_samples = (
            math.ceil(len(dataset) / num_replicas / self.samples_per_cycle)
            * self.samples_per_cycle
        )
        self.total_size = self.num_samples * num_replicas


class RepeatingBatchSizeSampler(Sampler[list[int]]):
    """Batch a sampler with an exact repeating sequence of microbatch sizes.

    This makes sample-count accumulation possible without dropping or padding
    examples.  For example, ``[2, 2, 2, 2, 2, 2, 2, 1]`` consumes exactly 15
    samples per rank and therefore forms a true global batch of 60 on four
    ranks, while seven of every eight forwards use the larger microbatch.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        pattern: list[int],
        drop_last: bool = False,
    ) -> None:
        self.sampler = sampler
        self.pattern = tuple(int(size) for size in pattern)
        self.drop_last = bool(drop_last)
        if not self.pattern or min(self.pattern) < 1:
            raise ValueError("microbatch_pattern must contain positive sizes")
        self.samples_per_cycle = sum(self.pattern)
        samples = len(sampler)
        if samples % self.samples_per_cycle and not self.drop_last:
            raise ValueError(
                f"{samples} samples per rank are not divisible by the "
                f"microbatch pattern total {self.samples_per_cycle}",
            )
        self.cycles = samples // self.samples_per_cycle

    def __iter__(self) -> Iterator[list[int]]:
        indices = iter(self.sampler)
        for _ in range(self.cycles):
            for size in self.pattern:
                batch = list(itertools.islice(indices, size))
                if len(batch) != size:
                    raise RuntimeError("Sampler ended inside a microbatch cycle")
                yield batch
        if not self.drop_last and next(indices, None) is not None:
            raise RuntimeError("Microbatch cycles did not consume the sampler")

    def __len__(self) -> int:
        return self.cycles * len(self.pattern)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--evaluate", default=None, help="Evaluate this checkpoint")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also evaluate the held-out test split (only after locking the design)",
    )
    parser.add_argument("--resume", default=None, help="Resume a last.pt checkpoint")
    parser.add_argument(
        "--init",
        default=None,
        help="Initialize compatible weights from a v2/v3 checkpoint without optimizer state",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=0,
        help="Exit cleanly after this epoch without evaluating the held-out test split",
    )
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
    temporary.replace(path)


def atomic_checkpoint(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


class ModelEMA:
    """Low-overhead parameter EMA kept on each training rank."""

    def __init__(self, model: nn.Module, decay: float, update_every: int = 1) -> None:
        self.decay = float(decay)
        self.update_every = int(update_every)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in unwrap_model(model).state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        if self.updates % self.update_every:
            return
        current = unwrap_model(model).state_dict()
        for name, value in current.items():
            if self.shadow[name].is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[name].copy_(value.detach())

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        target = unwrap_model(model)
        current = {
            name: value.detach().clone()
            for name, value in target.state_dict().items()
        }
        target.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            target.load_state_dict(current, strict=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "update_every": self.update_every,
            "updates": self.updates,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any], device: torch.device) -> None:
        self.decay = float(state["decay"])
        self.update_every = int(state["update_every"])
        self.updates = int(state["updates"])
        self.shadow = {name: value.to(device) for name, value in state["shadow"].items()}


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def release_cached_cuda_memory(device: torch.device) -> None:
    """Release inactive training segments before full-tile evaluation.

    Training uses 32x32 crops while validation uses 128x128 tiles.  Keeping the
    crop-shaped caching-allocator segments can leave enough total free memory
    but no segment large enough for a validation allocation.  Emptying only the
    inactive cache does not move live tensors or change model numerics.
    """

    if device.type == "cuda":
        torch.cuda.empty_cache()


def make_dataset(cfg: dict[str, Any], stage: str) -> PASTISHDCrops | MTLCCS2Tiles:
    data = cfg["data"]
    if data.get("dataset", "pastis_hd") == "mtlcc":
        return MTLCCS2Tiles(
            root=data["root"],
            stage=stage,
            split_file=data["split_files"][stage],
            stats_file=data["s2_norm_file"],
            temporal_keep_range=tuple(data.get("temporal_keep_range", (0.6, 0.8))),
            ignore_index=data.get("ignore_index", 255),
            max_tiles=data.get("stage_limits", {}).get(stage),
            subset_seed=data.get("subset_seed", cfg["seed"]),
            sample_rng_mode=data.get("sample_rng_mode", "worker"),
            storage_format=data.get("storage_format", "geotiff_legacy"),
        )
    crop_size = (
        data["crop_size"]
        if stage == "train"
        else data.get("eval_crop_size", data["crop_size"])
    )
    return PASTISHDCrops(
        root=data["root"],
        split_dir=data.get("split_dir"),
        stage=stage,
        folds=data.get("folds", {}).get(stage),
        modalities=data.get("modalities", ("spot", "s2", "s1a", "s1d")),
        crop_size=crop_size,
        train_crops_per_tile=data["train_crops_per_tile"],
        temporal_dropout=data["temporal_dropout"],
        s2_clip=tuple(data["s2_clip"]),
        s2_normalization=data.get("s2_normalization", "clip_scale"),
        s2_norm_file=data.get("s2_norm_file", "NORM_S2_patch.json"),
        s2_norm_folds=data.get("s2_norm_folds"),
        s1_clip=tuple(data["s1_clip"]),
        load_parcels=cfg["loss"].get("parcel_boundary_weight", 0.0) > 0,
        load_regions=(
            cfg["loss"].get("region_weight", 0.0) > 0
            or cfg["loss"].get("region_compactness_weight", 0.0) > 0
        ),
        region_source=data.get("region_source", "semantic_components"),
        s2_cache_root=(data.get("s2_cache_root") if stage == "train" else None),
        class_aware_sampling=data.get("class_aware_sampling", False),
        sampling_candidates=data.get("sampling_candidates", 8),
        sampling_probability=data.get("sampling_probability", 0.75),
        sampling_probability_formula=data.get("sampling_probability_formula"),
        sampling_expected_probability=data.get("sampling_expected_probability"),
        sampling_class_weights=data.get("sampling_class_weights"),
        sampling_weight_formula=data.get("sampling_weight_formula"),
        sampling_mode=data.get("sampling_mode", "candidate_score"),
        sampling_min_crops_per_class=data.get("sampling_min_crops_per_class", 0),
        sampling_expected_quota=data.get("sampling_expected_quota"),
        teacher_logits_root=data.get("teacher_logits_root"),
        temporal_sampling=(
            data.get("temporal_sampling")
            if stage == "train"
            else data.get("eval_temporal_sampling", {"mode": "none"})
        ),
        max_tiles=data.get("stage_limits", {}).get(stage),
        subset_seed=data.get("subset_seed", cfg["seed"]),
        sample_rng_mode=data.get("sample_rng_mode", "worker"),
    )


def make_loader(
    dataset: PASTISHDCrops | MTLCCS2Tiles,
    cfg: dict[str, Any],
    rank: int,
    world_size: int,
) -> tuple[DataLoader, Sampler[int] | None]:
    is_train = dataset.stage == "train"
    if world_size > 1 and is_train:
        contract = cfg["train"].get("pipeline_contract", {})
        if contract.get("require_fixed_global_batch", False):
            target = int(cfg["train"]["target_global_batch"])
            if target % world_size:
                raise ValueError(
                    "target_global_batch must be divisible by world_size "
                    "when a fixed optimizer batch is required",
                )
            local_cycle = target // world_size
            pattern = cfg["train"].get("microbatch_pattern")
            if pattern:
                if sum(int(size) for size in pattern) != local_cycle:
                    raise ValueError(
                        "microbatch_pattern must consume exactly one local "
                        "fixed-batch optimizer cycle",
                    )
            elif local_cycle != (
                int(cfg["train"]["batch_size"])
                * int(cfg["train"]["accumulate_steps"])
            ):
                raise ValueError(
                    "batch_size * accumulate_steps * world_size must equal "
                    "target_global_batch for a fixed optimizer batch",
                )
            sampler = CycleAlignedDistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=cfg["seed"],
                samples_per_cycle=local_cycle,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=cfg["seed"],
                drop_last=False,
            )
    elif world_size > 1:
        sampler = DistributedEvalSampler(len(dataset), rank, world_size)
    else:
        sampler = None
    workers = int(cfg["data"]["num_workers"])
    loader_kwargs: dict[str, Any] = {}
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            cfg["data"].get("persistent_workers", False),
        )
        loader_kwargs["prefetch_factor"] = int(
            cfg["data"].get("prefetch_factor", 2),
        )
    microbatch_pattern = cfg["train"].get("microbatch_pattern") if is_train else None
    common_loader_kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_mtlcc if isinstance(dataset, MTLCCS2Tiles) else collate_pastis_hd,
        **loader_kwargs,
    }
    if microbatch_pattern:
        if sampler is None:
            sampler = RandomSampler(dataset)
        loader = DataLoader(
            dataset,
            batch_sampler=RepeatingBatchSizeSampler(
                sampler,
                microbatch_pattern,
                drop_last=bool(
                    cfg["train"].get("microbatch_pattern_drop_last", False),
                ),
            ),
            **common_loader_kwargs,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=(
                cfg["train"]["batch_size"]
                if is_train
                else cfg["train"].get(
                    "eval_batch_size",
                    cfg["train"]["batch_size"],
                )
            ),
            sampler=sampler,
            shuffle=is_train and sampler is None,
            drop_last=is_train,
            **common_loader_kwargs,
        )
    return loader, sampler


def make_scheduler(optimizer: AdamW, cfg: dict[str, Any]) -> LambdaLR:
    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"]["warmup_epochs"]
    minimum_ratio = cfg["train"]["min_learning_rate"] / cfg["train"]["learning_rate"]
    scheduler_type = str(cfg["train"].get("lr_scheduler", "cosine"))
    poly_power = float(cfg["train"].get("poly_power", 0.9))
    warmup_factor = float(cfg["train"].get("warmup_factor", 0.001))
    refinement_start = int(cfg["train"].get("refinement_start_epoch", epochs))
    refinement_ratio = (
        cfg["train"].get("refinement_learning_rate", cfg["train"]["min_learning_rate"])
        / cfg["train"]["learning_rate"]
    )

    def multiplier(epoch: int) -> float:
        if scheduler_type == "poly":
            progress = min(1.0, max(0.0, epoch / max(1, epochs)))
            decay = (1.0 - progress) ** poly_power
            if warmup > 0 and epoch < warmup:
                alpha = epoch / warmup
                warmup_scale = warmup_factor * (1.0 - alpha) + alpha
            else:
                warmup_scale = 1.0
            return minimum_ratio + (1.0 - minimum_ratio) * warmup_scale * decay
        if scheduler_type != "cosine":
            raise ValueError(f"Unknown lr_scheduler: {scheduler_type}")
        if warmup > 0 and epoch < warmup:
            return max(1e-3, (epoch + 1) / warmup)
        if epoch < refinement_start:
            progress = (epoch - warmup) / max(1, refinement_start - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return refinement_ratio + (1.0 - refinement_ratio) * cosine
        progress = (epoch - refinement_start) / max(1, epochs - refinement_start)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return minimum_ratio + (refinement_ratio - minimum_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def _curriculum_progress(epoch: int, start_epoch: int, end_epoch: int) -> float:
    """Cosine interpolation progress for a one-indexed training epoch."""

    if end_epoch < start_epoch:
        raise ValueError("curriculum end_epoch must be >= start_epoch")
    linear = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
    linear = min(1.0, max(0.0, linear))
    return 0.5 - 0.5 * math.cos(math.pi * linear)


def apply_training_curriculum(
    model: nn.Module,
    criterion: nn.Module,
    dataset: PASTISHDCrops,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    """Apply an opt-in, resume-stable semantic-focus curriculum.

    Every value is derived from the immutable YAML baseline and the absolute
    epoch, so restarting from a checkpoint cannot silently change the schedule.
    Configurations without ``train.curriculum`` retain their exact behaviour.
    """

    curriculum = cfg["train"].get("curriculum")
    if not curriculum:
        return {}
    progress = _curriculum_progress(
        epoch,
        int(curriculum["start_epoch"]),
        int(curriculum["end_epoch"]),
    )
    applied: dict[str, float] = {"progress": progress}

    def interpolate(initial: float, final: float) -> float:
        if progress <= 0.0:
            return initial
        if progress >= 1.0:
            return final
        return initial + progress * (final - initial)

    for name, final_value in curriculum.get("loss_final", {}).items():
        if name not in cfg["loss"] or not hasattr(criterion, name):
            raise ValueError(f"Unknown scheduled loss property: {name}")
        initial_value = float(cfg["loss"][name])
        value = interpolate(initial_value, float(final_value))
        setattr(criterion, name, value)
        applied[f"loss.{name}"] = value

    class_weight_mix_final = curriculum.get("class_weights_uniform_mix_final")
    if class_weight_mix_final is not None:
        final_mix = float(class_weight_mix_final)
        if not 0.0 <= final_mix <= 1.0:
            raise ValueError("class_weights_uniform_mix_final must be in [0,1]")
        if criterion.class_weights is None or "class_weights" not in cfg["loss"]:
            raise ValueError(
                "class weight calibration requires loss.class_weights",
            )
        mix = interpolate(0.0, final_mix)
        baseline = torch.as_tensor(
            cfg["loss"]["class_weights"],
            dtype=criterion.class_weights.dtype,
            device=criterion.class_weights.device,
        )
        criterion.class_weights.copy_(baseline.lerp(torch.ones_like(baseline), mix))
        applied["loss.class_weights_uniform_mix"] = mix

    unwrapped = unwrap_model(model)
    for name, final_value in curriculum.get("model_final", {}).items():
        if name not in cfg["model"] or not hasattr(unwrapped, name):
            raise ValueError(f"Unknown scheduled model property: {name}")
        initial_value = float(cfg["model"][name])
        value = interpolate(initial_value, float(final_value))
        setattr(unwrapped, name, value)
        applied[f"model.{name}"] = value

    data_final = curriculum.get("data_final", {})
    if "temporal_keep_range" in data_final:
        if not hasattr(dataset, "temporal_keep_range"):
            raise ValueError("temporal_keep_range schedule requires a temporal tile dataset")
        initial_range = tuple(float(value) for value in cfg["data"]["temporal_keep_range"])
        final_range = tuple(float(value) for value in data_final["temporal_keep_range"])
        if len(initial_range) != 2 or len(final_range) != 2:
            raise ValueError("temporal_keep_range schedule expects [low, high]")
        value = tuple(interpolate(initial, final) for initial, final in zip(initial_range, final_range))
        if not 0.0 < value[0] <= value[1] <= 1.0:
            raise ValueError("scheduled temporal_keep_range must satisfy 0 < low <= high <= 1")
        dataset.temporal_keep_range = value
        applied["data.temporal_keep_range.low"] = value[0]
        applied["data.temporal_keep_range.high"] = value[1]
    if "sampling_probability" in data_final:
        initial_value = float(cfg["data"]["sampling_probability"])
        value = interpolate(initial_value, float(data_final["sampling_probability"]))
        dataset.sampling_probability = value
        applied["data.sampling_probability"] = value
    temporal_final = data_final.get("temporal_dropout", {})
    baseline_dropout = cfg["data"].get("temporal_dropout", {})
    if isinstance(baseline_dropout, (int, float)):
        baseline_dropout = {
            name: float(baseline_dropout) for name in ("s2", "s1a", "s1d")
        }
    for name, final_value in temporal_final.items():
        if name not in dataset.temporal_dropout:
            raise ValueError(f"Unknown scheduled temporal stream: {name}")
        initial_value = float(
            baseline_dropout.get(name, baseline_dropout.get("default", 0.0)),
        )
        value = interpolate(initial_value, float(final_value))
        dataset.temporal_dropout[name] = value
        applied[f"data.temporal_dropout.{name}"] = value
    if "temporal_random_alternative_probability" in data_final:
        initial_value = float(
            cfg["data"].get("temporal_sampling", {}).get(
                "random_alternative_probability",
                0.0,
            ),
        )
        value = interpolate(
            initial_value,
            float(data_final["temporal_random_alternative_probability"]),
        )
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "temporal_random_alternative_probability must be in [0,1]",
            )
        dataset.temporal_random_alternative_probability = value
        applied["data.temporal_random_alternative_probability"] = value
    return applied


def accumulation_group_sizes(
    microbatches: int,
    micro_batch_size: int,
    world_size: int,
    accumulation_steps: int,
    target_global_batch: int | None = None,
) -> list[int]:
    """Partition an epoch into near-uniform optimizer update groups.

    A global batch such as 16 cannot be formed exactly by two ranks with three
    samples each.  Distributing two- and three-microbatch updates across the
    complete epoch preserves the requested *mean* batch and optimizer-step
    count, while allowing a larger GPU-resident microbatch.
    """

    if microbatches < 1 or micro_batch_size < 1 or world_size < 1:
        raise ValueError("Microbatch count, size and world size must be positive")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    if target_global_batch is None:
        full, remainder = divmod(microbatches, accumulation_steps)
        groups = [accumulation_steps] * full
        if remainder:
            groups.append(remainder)
        return groups

    micro_global_batch = micro_batch_size * world_size
    if target_global_batch < micro_global_batch:
        raise ValueError(
            "target_global_batch cannot be smaller than one distributed microbatch",
        )
    updates = round(microbatches * micro_global_batch / target_global_batch)
    updates = max(1, min(microbatches, updates))
    groups = []
    previous = 0
    for update in range(1, updates + 1):
        # Integer nearest rounding avoids floating-point boundary drift.
        boundary = (
            update * microbatches + updates // 2
        ) // updates
        groups.append(boundary - previous)
        previous = boundary
    if not groups or min(groups) < 1 or sum(groups) != microbatches:
        raise RuntimeError("Invalid variable accumulation schedule")
    return groups


def optimizer_accumulation_plan(
    loader: DataLoader,
    cfg: dict[str, Any],
    world_size: int,
) -> tuple[list[int], list[int]]:
    """Return microbatch counts and local sample counts for every update."""

    pattern = cfg["train"].get("microbatch_pattern")
    if pattern:
        pattern = [int(size) for size in pattern]
        if cfg["train"]["accumulate_steps"] != len(pattern):
            raise ValueError(
                "accumulate_steps must equal the length of microbatch_pattern",
            )
        if len(loader) % len(pattern):
            raise RuntimeError("Loader does not contain complete microbatch cycles")
        updates = len(loader) // len(pattern)
        return [len(pattern)] * updates, [sum(pattern)] * updates

    groups = accumulation_group_sizes(
        len(loader),
        cfg["train"]["batch_size"],
        world_size,
        cfg["train"]["accumulate_steps"],
        cfg["train"].get("target_global_batch"),
    )
    return groups, [cfg["train"]["batch_size"] * group for group in groups]


def validate_pipeline_contract(
    cfg: dict[str, Any],
    train_dataset: PASTISHDCrops,
    train_loader: DataLoader,
    world_size: int,
) -> dict[str, Any]:
    """Fail before training when a canonical experiment silently drifts."""

    groups, local_sample_groups = optimizer_accumulation_plan(
        train_loader,
        cfg,
        world_size,
    )
    global_batches = [samples * world_size for samples in local_sample_groups]
    contract = cfg["train"].get("pipeline_contract", {})

    expected_normalization = contract.get("s2_normalization")
    if (
        expected_normalization is not None
        and train_dataset.s2_normalization != expected_normalization
    ):
        raise RuntimeError(
            "Pipeline contract mismatch: "
            f"S2 normalization is {train_dataset.s2_normalization}, "
            f"expected {expected_normalization}",
        )
    expected_rng = contract.get("sample_rng_mode")
    if expected_rng is not None and train_dataset.sample_rng_mode != expected_rng:
        raise RuntimeError(
            "Pipeline contract mismatch: "
            f"sample_rng_mode is {train_dataset.sample_rng_mode}, expected {expected_rng}",
        )
    if contract.get("require_unweighted_ce", False) and cfg["loss"].get("class_weights"):
        raise RuntimeError("Pipeline contract requires unweighted cross entropy")
    if contract.get("require_fixed_global_batch", False):
        expected_batch = int(cfg["train"]["target_global_batch"])
        if set(global_batches) != {expected_batch}:
            raise RuntimeError(
                "Pipeline contract requires a fixed optimizer batch of "
                f"{expected_batch}, observed {sorted(set(global_batches))}",
            )
    expected_train_ids_sha256 = contract.get("sampling_train_ids_sha256")
    if (
        expected_train_ids_sha256 is not None
        and train_dataset.sampling_train_ids_sha256 != expected_train_ids_sha256
    ):
        raise RuntimeError(
            "Pipeline contract train split mismatch: "
            f"observed={train_dataset.sampling_train_ids_sha256}, "
            f"expected={expected_train_ids_sha256}",
        )

    return {
        "s2_normalization": train_dataset.s2_normalization,
        "sample_rng_mode": train_dataset.sample_rng_mode,
        "class_weighted_ce": bool(cfg["loss"].get("class_weights")),
        "sampling_train_ids_sha256": train_dataset.sampling_train_ids_sha256,
        "sampling_pixel_counts": train_dataset.sampling_pixel_counts.tolist(),
        "sampling_class_weights": (
            None
            if train_dataset.sampling_class_weights is None
            else train_dataset.sampling_class_weights.tolist()
        ),
        "sampling_probability": train_dataset.sampling_probability,
        "sampling_probability_formula": train_dataset.sampling_probability_formula,
        "sampling_probability_statistics": (
            train_dataset.sampling_probability_statistics
        ),
        "sampling_weight_formula": train_dataset.sampling_weight_formula,
        "optimizer_global_batch_histogram": {
            str(size): global_batches.count(size) for size in sorted(set(global_batches))
        },
        "s2_stats_sha256": getattr(train_dataset, "stats_sha256", None),
        "sequence_length_min": getattr(train_dataset, "sequence_length_min", None),
        "sequence_length_max": getattr(train_dataset, "sequence_length_max", None),
        "temporal_keep_range": getattr(train_dataset, "temporal_keep_range", None),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    cfg: dict[str, Any],
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    model.train()
    if cfg["train"].get("freeze_legacy", False):
        unwrapped = unwrap_model(model)
        if not hasattr(unwrapped, "legacy"):
            raise ValueError("freeze_legacy requires a model with a legacy path")
        # requires_grad=False freezes weights, but model.train() would still
        # activate dropout and modality dropout inside the inherited function.
        # Keep the v2 teacher/base deterministic while the field branch trains.
        unwrapped.legacy.eval()
    optimizer.zero_grad(set_to_none=True)
    accumulated = {
        name: 0.0
        for name in (
            "total",
            "ce",
            "dice",
            "lovasz",
            "boundary",
            "boundary_consistency",
            "affinity",
            "deep_supervision",
            "parcel_boundary",
            "auxiliary",
            "region",
            "region_compactness",
            "distillation",
            "inheritance",
            "temporal_reconstruction",
            "temporal_prototype",
            "phase_offset_magnitude",
            "phase_offset_smoothness",
        )
    }
    # Accumulate gate sums and element counts on-device so the reported mean
    # is exact over every crop, pixel and distributed rank, rather than an
    # unweighted mean of per-batch means. V4/V4.1 simply leave the counts at
    # zero because they do not emit the V4.2 dual-path gates.
    strict_time_statistics = torch.zeros(4, dtype=torch.float64, device=device)
    samples = 0
    accumulation = cfg["train"]["accumulate_steps"]
    group_sizes, group_local_samples = optimizer_accumulation_plan(
        loader,
        cfg,
        dist.get_world_size() if dist.is_initialized() else 1,
    )
    group_index = 0
    group_end = group_sizes[0]
    amp_enabled = cfg["train"]["amp"] and device.type == "cuda"

    for step, raw_batch in enumerate(loader):
        batch = move_to_device(raw_batch, device)
        batch_size = batch["target"].shape[0]
        should_step = step + 1 == group_end
        group_sample_count = group_local_samples[group_index]
        sync_context = contextlib.nullcontext()
        if isinstance(model, DistributedDataParallel) and not should_step:
            sync_context = model.no_sync()
        with sync_context:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = model(batch, return_aux=True)
                losses = criterion(
                    output,
                    batch["target"],
                    batch.get("parcels"),
                    batch.get("regions"),
                    batch.get("teacher_logits"),
                )
                # Criterion losses are batch means. Weight each microbatch by
                # its true sample count so a 2+...+2+1 cycle is exactly the
                # gradient of 15 samples per rank, not a mean of eight means.
                loss = losses["total"] * batch_size / group_sample_count
            loss.backward()

        with torch.no_grad():
            pooled_gate = output.get("s2_strict_weight")
            if isinstance(pooled_gate, Tensor):
                strict_time_statistics[0] += pooled_gate.detach().double().sum()
                strict_time_statistics[1] += pooled_gate.numel()
            phase_gate = output.get("s2_phase_strict_weight")
            if isinstance(phase_gate, Tensor):
                strict_time_statistics[2] += phase_gate.detach().double().sum()
                strict_time_statistics[3] += phase_gate.numel()

        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            group_index += 1
            if group_index < len(group_sizes):
                group_end += group_sizes[group_index]

        samples += batch_size
        for name in accumulated:
            accumulated[name] += float(losses[name]) * batch_size

    values = torch.cat(
        (
            torch.tensor(
                [samples, *(accumulated[name] for name in accumulated)],
                dtype=torch.float64,
                device=device,
            ),
            strict_time_statistics,
        ),
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if group_index != len(group_sizes):
        raise RuntimeError("Accumulation schedule did not consume every group")
    total_samples = max(1.0, float(values[0]))
    result = {
        name: float(values[index + 1] / total_samples)
        for index, name in enumerate(accumulated)
    }
    statistic_offset = 1 + len(accumulated)
    pooled_count = float(values[statistic_offset + 1])
    phase_count = float(values[statistic_offset + 3])
    if pooled_count > 0:
        result["s2_strict_weight_global_mean"] = float(
            values[statistic_offset] / pooled_count,
        )
    if phase_count > 0:
        result["s2_phase_strict_weight_global_mean"] = float(
            values[statistic_offset + 2] / phase_count,
        )
    return result


def _d4_transform(tensor: Tensor, rotation: int, flip: bool) -> Tensor:
    transformed = torch.rot90(tensor, rotation, dims=(-2, -1))
    return torch.flip(transformed, dims=(-1,)) if flip else transformed


def _d4_inverse(tensor: Tensor, rotation: int, flip: bool) -> Tensor:
    transformed = torch.flip(tensor, dims=(-1,)) if flip else tensor
    return torch.rot90(transformed, -rotation, dims=(-2, -1))


def tta_forward(model: nn.Module, batch: dict[str, Any]) -> dict[str, Tensor]:
    """Average the eight label-preserving dihedral views."""
    spatial_names = ("spot", "s2", "s1a", "s1d")
    logits, boundaries = [], []
    for rotation in range(4):
        for flip in (False, True):
            view = dict(batch)
            for name in spatial_names:
                if name in batch:
                    view[name] = _d4_transform(batch[name], rotation, flip)
            # TTA consumes only semantic and boundary logits. Requesting the
            # full diagnostic contract would repeat phase reconstruction and
            # interaction-map materialization eight times without affecting
            # either averaged prediction.
            output = model(view, return_aux=False)
            logits.append(_d4_inverse(output["logits"], rotation, flip))
            boundaries.append(_d4_inverse(output["boundary_logits"], rotation, flip))
    return {
        "logits": torch.stack(logits).mean(0),
        "boundary_logits": torch.stack(boundaries).mean(0),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict[str, Any],
    save_dir: Path | None = None,
    tta: bool = False,
) -> dict[str, Any]:
    release_cached_cuda_memory(device)
    model.eval()
    metrics = SegmentationMetrics(
        num_classes=cfg["model"]["num_classes"],
        ignore_index=cfg["data"]["ignore_index"],
        device=str(device),
        class_names=cfg["data"].get("class_names"),
    )
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    sample_count = torch.zeros((), dtype=torch.float64, device=device)
    saved = 0
    start = time.perf_counter()

    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=cfg["train"]["amp"] and device.type == "cuda",
        ):
            need_visuals = save_dir is not None and saved < cfg["output"]["save_visualizations"]
            need_aux = need_visuals or any(
                cfg["loss"].get(name, 0.0) > 0
                for name in (
                    "auxiliary_weight",
                    "anchor_weight",
                    "temporal_reconstruction_weight",
                    "temporal_prototype_weight",
                )
            )
            if tta:
                # Auxiliary/affinity channels are orientation-dependent. Use a
                # normal view for the reported validation loss and TTA only for
                # the semantic/boundary metrics.
                loss_output = model(batch, return_aux=need_aux)
                output = tta_forward(model, batch)
            else:
                output = model(batch, return_aux=need_aux)
                loss_output = output
            losses = criterion(
                loss_output,
                batch["target"],
                batch.get("parcels"),
                batch.get("regions"),
                batch.get("teacher_logits"),
            )
        metrics.update(output["logits"], batch["target"], output["boundary_logits"])
        count = batch["target"].shape[0]
        loss_sum += losses["total"].double() * count
        sample_count += count

        if save_dir is not None and saved < cfg["output"]["save_visualizations"]:
            save_dir.mkdir(parents=True, exist_ok=True)
            predictions = output["logits"].argmax(1).cpu().numpy()
            targets = batch["target"].cpu().numpy()
            visual_output = output
            if "geometry_weights" not in visual_output and "availability" not in visual_output:
                visual_output = model(batch, return_aux=True)
            if "geometry_weights" in visual_output:
                routing_name = "geometry_weights"
                geometry_weights = visual_output["geometry_weights"].float().cpu().numpy()
            elif "availability" in visual_output:
                routing_name = "modality_availability"
                geometry_weights = visual_output["availability"].float().cpu().numpy()
            else:
                # Optical-only models have no modality router.  Visualizations
                # remain valid without inventing an availability vector.
                routing_name = None
                geometry_weights = None
            for index in range(min(count, cfg["output"]["save_visualizations"] - saved)):
                routing_payload = (
                    {}
                    if routing_name is None or geometry_weights is None
                    else {routing_name: geometry_weights[index]}
                )
                np.savez_compressed(
                    save_dir / f"{batch['patch_id'][index]}_{saved:03d}.npz",
                    prediction=predictions[index],
                    target=targets[index],
                    **routing_payload,
                    origin=batch["origin"][index].cpu().numpy(),
                )
                saved += 1

    elapsed = time.perf_counter() - start
    metrics.synchronize()
    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    result = metrics.compute()
    result["loss"] = float(loss_sum / sample_count.clamp_min(1.0))
    result["samples"] = int(sample_count)
    result["seconds"] = elapsed
    result["samples_per_second_per_rank"] = float(sample_count / max(elapsed, 1e-6)) / (
        dist.get_world_size() if dist.is_initialized() else 1
    )
    result["tta"] = bool(tta)
    return result


def checkpoint_payload(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    best_miou: float,
    best_epoch: int,
    epochs_without_improvement: int,
    cfg: dict[str, Any],
    ema: ModelEMA | None = None,
) -> dict[str, Any]:
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_miou": best_miou,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "config": cfg,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    return payload


def best_checkpoint_payload(
    model: nn.Module,
    epoch: int,
    best_miou: float,
    cfg: dict[str, Any],
    ema: ModelEMA | None = None,
) -> dict[str, Any]:
    """Weights-only validation-selected artifact; optimizer state stays in last.pt."""
    return {
        "model": (
            {name: value.detach().cpu() for name, value in ema.shadow.items()}
            if ema is not None
            else unwrap_model(model).state_dict()
        ),
        "epoch": epoch,
        "best_val_miou": best_miou,
        "config": cfg,
        "selection_rule": "maximum validation mIoU using EMA weights" if ema else "maximum validation mIoU",
    }


def top_validation_epochs(history: list[dict[str, Any]], count: int) -> list[int]:
    """Return validation epochs ranked by mIoU, with deterministic tie breaks."""

    if count < 0:
        raise ValueError("top validation checkpoint count cannot be negative")
    ranked = [record for record in history if record.get("val") is not None]
    ranked.sort(
        key=lambda record: (
            float(record["val"]["miou"]),
            -int(record["epoch"]),
        ),
        reverse=True,
    )
    return [int(record["epoch"]) for record in ranked[:count]]


def should_validate_epoch(train_config: dict[str, Any], epoch: int) -> bool:
    """Apply an optional early/late validation cadence to one-indexed epochs."""

    if epoch < 1:
        raise ValueError("validation epoch must be one-indexed and positive")
    interval = int(train_config["validate_every"])
    switch_epoch = int(train_config.get("validate_every_switch_epoch", 0))
    if switch_epoch > 0 and epoch > switch_epoch:
        interval = int(train_config.get("validate_every_late", interval))
    if interval < 1:
        raise ValueError("validation intervals must be positive")
    return epoch % interval == 0 or epoch == int(train_config["epochs"])


def initial_validation_selection(
    initial_validation: dict[str, Any] | None,
    enabled: bool,
) -> tuple[float, int]:
    """Optionally let a stage-initialization checkpoint compete with later epochs."""
    if not enabled:
        return -math.inf, -1
    if initial_validation is None:
        raise ValueError("Initial validation selection requires an --init checkpoint")
    return float(initial_validation["miou"]), 0


def main() -> None:
    args = parse_args()
    if args.evaluate and args.resume:
        raise ValueError("--evaluate and --resume are mutually exclusive")
    if args.init and (args.evaluate or args.resume):
        raise ValueError("--init cannot be combined with --evaluate or --resume")
    cfg = load_config(args.config, args.set)
    if (
        cfg["train"].get("require_init", False)
        and not args.init
        and not args.resume
        and not args.evaluate
    ):
        raise ValueError(
            "This training configuration requires --init; use its intended "
            "parent checkpoint or explicitly choose a from-scratch config",
        )
    rank, _local_rank, world_size, device = distributed_context()
    seed_everything(cfg["seed"], rank)
    is_main = rank == 0

    output_dir = Path(cfg["output"]["dir"])
    checkpoint_dir = output_dir / "checkpoints"
    metric_dir = output_dir / "metrics"
    visualization_dir = output_dir / "visualizations"
    if is_main:
        for path in (checkpoint_dir, metric_dir, visualization_dir, output_dir / "logs"):
            path.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)

    train_dataset = make_dataset(cfg, "train")
    val_dataset = make_dataset(cfg, "val")
    test_dataset = make_dataset(cfg, "test")
    train_loader, train_sampler = make_loader(train_dataset, cfg, rank, world_size)
    val_loader, _ = make_loader(val_dataset, cfg, rank, world_size)
    test_loader, _ = make_loader(test_dataset, cfg, rank, world_size)
    pipeline_metadata = validate_pipeline_contract(
        cfg,
        train_dataset,
        train_loader,
        world_size,
    )

    os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
    # Avoid four ranks racing to download the same pretrained checkpoint.
    stagger_pretrained_init = (
        world_size > 1
        and cfg["model"].get("geometry_pretrained", False)
        and cfg["model"].get("geometry_backbone")
        in {"convnext_tiny", "convnext_tiny_truncated", "convnext_base"}
    )
    if stagger_pretrained_init and rank == 0:
        model = build_model(cfg["model"]).to(device)
        dist.barrier(device_ids=[device.index])
    elif stagger_pretrained_init:
        dist.barrier(device_ids=[device.index])
        model = build_model(cfg["model"]).to(device)
    else:
        model = build_model(cfg["model"]).to(device)
    parameters = count_parameters(model)
    parameter_cap = int(cfg["train"].get("max_parameters", 0))
    if parameter_cap > 0 and parameters > parameter_cap:
        raise RuntimeError(
            f"Model has {parameters:,} parameters, exceeding cap {parameter_cap:,}",
        )
    if args.init:
        initialization = torch.load(args.init, map_location="cpu", weights_only=False)
        state = initialization.get("model", initialization)
        allowed_shape_prefixes = tuple(
            str(prefix) for prefix in cfg["train"].get("init_shape_mismatch_prefixes", ())
        )
        model_state = model.state_dict()
        shape_mismatches = sorted(
            name
            for name, value in state.items()
            if name in model_state and tuple(value.shape) != tuple(model_state[name].shape)
        )
        disallowed_shape_mismatches = [
            name
            for name in shape_mismatches
            if not any(name.startswith(prefix) for prefix in allowed_shape_prefixes)
        ]
        if disallowed_shape_mismatches:
            raise RuntimeError(
                "Incompatible --init tensor shapes outside allowed prefixes: "
                f"{disallowed_shape_mismatches}",
            )
        if shape_mismatches:
            state = {name: value for name, value in state.items() if name not in shape_mismatches}
        incompatibility = model.load_state_dict(state, strict=False)
        allowed_missing = {
            name for name in incompatibility.missing_keys if name.startswith("auxiliary_heads.")
        }
        allowed_missing.update(shape_mismatches)
        if cfg["model"].get("architecture") in {
            "temporal_v3_1",
            "temporal_v3_2",
        }:
            allowed_missing.update(
                name
                for name in incompatibility.missing_keys
                if name.startswith("classwise_phenology.")
                or name == "phenology_residual_scale"
            )
        if cfg["model"].get("architecture") == "temporal_v3_5":
            allowed_missing.update(
                name
                for name in incompatibility.missing_keys
                if name.startswith("phase_refiner.")
                or name.startswith("parcel_refiner.")
            )
        if cfg["model"].get("architecture") == "temporal_v3_03":
            allowed_missing.update(
                name
                for name in incompatibility.missing_keys
                if name.startswith("phase_offset_predictor.")
            )
        if cfg["model"].get("architecture") == "sparxo_v5":
            allowed_missing.update(
                name
                for name in incompatibility.missing_keys
                if name.startswith("exchanger.")
                or name.startswith("exchange_update.")
                or name == "exchange_scale"
            )
        if cfg["model"].get("architecture") == "sparxo_v2_u_s1":
            allowed_missing.update(
                name
                for name in incompatibility.missing_keys
                if name.startswith("temporal_encoder.soft_expert_layers.")
            )
        if cfg["model"].get("architecture") == "sparxo_v2_u2_mtlcc_v2":
            allowed_missing.update(
                name
                for name in incompatibility.missing_keys
                if name.startswith("decoder.boundary_semantic_refiner.")
                or name.startswith("sequence_reconstruction.")
            )
        unexpected = list(incompatibility.unexpected_keys)
        disallowed_missing = sorted(set(incompatibility.missing_keys).difference(allowed_missing))
        if disallowed_missing or unexpected:
            raise RuntimeError(
                f"Incompatible --init checkpoint; missing={disallowed_missing}, unexpected={unexpected}",
            )
        if is_main:
            print(
                f"[INIT] loaded={args.init} newly_initialized={sorted(allowed_missing)}",
                flush=True,
            )

    freeze_legacy = bool(cfg["train"].get("freeze_legacy", False))
    if freeze_legacy:
        if cfg["model"].get("architecture") != "field_v3":
            raise ValueError("train.freeze_legacy is only valid for field_v3")
        if args.resume:
            raise ValueError(
                "Resume optimizer state is incompatible with train.freeze_legacy; "
                "start the frozen stage with --init",
            )
        for parameter in model.legacy.parameters():
            parameter.requires_grad_(False)
    trainable_prefixes = tuple(
        str(prefix) for prefix in cfg["train"].get("trainable_prefixes", ())
    )
    if trainable_prefixes:
        matched_prefixes: set[str] = set()
        for name, parameter in model.named_parameters():
            matches = tuple(prefix for prefix in trainable_prefixes if name.startswith(prefix))
            parameter.requires_grad_(bool(matches))
            matched_prefixes.update(matches)
        missing_prefixes = sorted(set(trainable_prefixes).difference(matched_prefixes))
        if missing_prefixes:
            raise ValueError(f"train.trainable_prefixes matched no parameters: {missing_prefixes}")
    trainable_parameters = count_parameters(model)

    if cfg["train"]["compile"]:
        model = torch.compile(model)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    criterion = build_loss(cfg["loss"], cfg["data"]["ignore_index"]).to(device)
    base_lr = cfg["train"]["learning_rate"]
    backbone_multiplier = (
        cfg["train"]["backbone_lr_multiplier"]
        if cfg["model"].get("geometry_pretrained", False)
        else 1.0
    )
    unwrapped = unwrap_model(model)
    if cfg["model"].get("architecture") == "field_v3":
        # The complete legacy path is migrated from v2 and should move
        # conservatively while the zero-initialized field residual opens.
        # Exact-cell projection, FPN laterals, and fusion blocks are new and
        # receive the main learning rate.
        backbone_parameters = list(
            unwrapped.spot_encoder.context.features.parameters(),
        )
        backbone_parameters.extend(unwrapped.legacy.parameters())
    elif isinstance(getattr(unwrapped, "spot_encoder", None), nn.Module):
        backbone_parameters = list(unwrapped.spot_encoder.parameters())
    else:
        # Optical-only SPARXO models have no ImageNet geometry backbone.
        backbone_parameters = []
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
    main_decay, main_no_decay = [], []
    backbone_decay, backbone_no_decay = [], []
    new_decay, new_no_decay = [], []
    new_prefixes = tuple(
        str(prefix)
        for prefix in cfg["train"].get("new_module_prefixes", ())
    )
    new_multiplier = float(cfg["train"].get("new_module_lr_multiplier", 1.0))
    if new_multiplier <= 0.0:
        raise ValueError("train.new_module_lr_multiplier must be positive")
    prefix_lr_multipliers = tuple(
        (str(item["prefix"]), float(item["multiplier"]))
        for item in cfg["train"].get("lr_multiplier_by_prefix", ())
    )
    if any(multiplier <= 0.0 for _, multiplier in prefix_lr_multipliers):
        raise ValueError("train.lr_multiplier_by_prefix multipliers must be positive")
    special_groups: dict[tuple[str, float, bool], list[nn.Parameter]] = {}
    for name, parameter in unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        special = next(
            ((prefix, multiplier) for prefix, multiplier in prefix_lr_multipliers if name.startswith(prefix)),
            None,
        )
        is_backbone = id(parameter) in backbone_parameter_ids
        is_new = bool(new_prefixes) and name.startswith(new_prefixes)
        # Do not decay biases, normalization/scale vectors, SSM rates or
        # calendar frequencies.
        no_decay = parameter.ndim <= 1 or name.endswith(".bias")
        if special is not None:
            prefix, multiplier = special
            special_groups.setdefault((prefix, multiplier, no_decay), []).append(parameter)
        elif is_new and no_decay:
            new_no_decay.append(parameter)
        elif is_new:
            new_decay.append(parameter)
        elif is_backbone and no_decay:
            backbone_no_decay.append(parameter)
        elif is_backbone:
            backbone_decay.append(parameter)
        elif no_decay:
            main_no_decay.append(parameter)
        else:
            main_decay.append(parameter)
    weight_decay = cfg["train"]["weight_decay"]
    optimizer_groups = [
        {"params": main_decay, "lr": base_lr, "weight_decay": weight_decay},
        {"params": main_no_decay, "lr": base_lr, "weight_decay": 0.0},
    ]
    if new_decay:
        optimizer_groups.append(
            {
                "params": new_decay,
                "lr": base_lr * new_multiplier,
                "weight_decay": weight_decay,
            },
        )
    if new_no_decay:
        optimizer_groups.append(
            {
                "params": new_no_decay,
                "lr": base_lr * new_multiplier,
                "weight_decay": 0.0,
            },
        )
    if backbone_decay:
        optimizer_groups.append(
            {
                "params": backbone_decay,
                "lr": base_lr * backbone_multiplier,
                "weight_decay": weight_decay,
            },
        )
    if backbone_no_decay:
        optimizer_groups.append(
            {
                "params": backbone_no_decay,
                "lr": base_lr * backbone_multiplier,
                "weight_decay": 0.0,
            },
        )
    for (prefix, multiplier, no_decay), group_parameters in special_groups.items():
        optimizer_groups.append(
            {
                "params": group_parameters,
                "lr": base_lr * multiplier,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "group_name": prefix,
            },
        )
    optimizer_betas = tuple(float(value) for value in cfg["train"].get("optimizer_betas", (0.9, 0.95)))
    if len(optimizer_betas) != 2 or not all(0.0 <= value < 1.0 for value in optimizer_betas):
        raise ValueError("train.optimizer_betas must contain two values in [0,1)")
    optimizer = AdamW(
        optimizer_groups,
        lr=base_lr,
        weight_decay=weight_decay,
        betas=optimizer_betas,
    )
    scheduler = make_scheduler(optimizer, cfg)

    if is_main:
        accumulation_groups, local_sample_groups = optimizer_accumulation_plan(
            train_loader,
            cfg,
            world_size,
        )
        global_sample_groups = [
            samples * world_size for samples in local_sample_groups
        ]
        mean_global_batch = sum(global_sample_groups) / len(global_sample_groups)
        atomic_json(
            output_dir / "run_metadata.json",
            {
                "parameters": parameters,
                "trainable_parameters": trainable_parameters,
                "trainable_prefixes": list(trainable_prefixes),
                "optimizer_parameter_groups": [
                    {
                        "name": group.get("group_name", "default"),
                        "learning_rate": group["lr"],
                        "weight_decay": group["weight_decay"],
                        "parameters": sum(parameter.numel() for parameter in group["params"]),
                    }
                    for group in optimizer.param_groups
                ],
                "legacy_frozen": freeze_legacy,
                "legacy_runtime_eval": freeze_legacy,
                "parameter_cap": parameter_cap or None,
                "world_size": world_size,
                "train_crops": len(train_dataset),
                "val_crops": len(val_dataset),
                "test_crops": len(test_dataset),
                "effective_global_batch": (
                    cfg["train"].get("target_global_batch")
                    or (
                        cfg["train"]["batch_size"]
                        * world_size
                        * cfg["train"]["accumulate_steps"]
                    )
                ),
                "mean_effective_global_batch": mean_global_batch,
                "optimizer_steps_per_epoch": len(accumulation_groups),
                "microbatch_pattern": cfg["train"].get("microbatch_pattern"),
                "accumulation_group_histogram": {
                    str(size): accumulation_groups.count(size)
                    for size in sorted(set(accumulation_groups))
                },
                "pipeline": pipeline_metadata,
                "train_lowres_pixels_per_epoch": (
                    len(train_dataset) * cfg["data"]["crop_size"] ** 2
                ),
                "class_centered_quota": train_dataset.sampling_quota.tolist(),
                "selection_rule": "best validation mIoU; test evaluated once",
            },
        )
        print(
            f"[START] params={parameters:,} trainable={trainable_parameters:,} "
            f"legacy_frozen={freeze_legacy} legacy_runtime_eval={freeze_legacy} "
            f"train={len(train_dataset)} "
            f"val={len(val_dataset)} test={len(test_dataset)} world={world_size}",
            flush=True,
        )

    initial_validation: dict[str, Any] | None = None
    if args.init:
        # A clean initial evaluation usually fits, but explicitly releasing
        # allocator cache keeps this path consistent with post-train validation.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        initial_validation = evaluate(model, val_loader, criterion, device, cfg)
        if is_main:
            atomic_json(
                metric_dir / "initial_validation.json",
                initial_validation,
            )
            print(
                f"[INITIAL VAL] mIoU={initial_validation['miou']:.4f} "
                f"F1={initial_validation['macro_f1']:.4f}",
                flush=True,
            )
        # Full-resolution epoch-zero validation leaves large cached blocks in
        # the CUDA allocator.  They are reusable for another evaluation but
        # can be badly shaped for the first backward/cuBLAS workspace.  Release
        # only unused cache here; model, optimizer and RNG state are unchanged.
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.evaluate:
        checkpoint = torch.load(args.evaluate, map_location="cpu", weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
        val_metrics = evaluate(model, val_loader, criterion, device, cfg)
        test_metrics = None
        test_tta_metrics = None
        if args.evaluate_test:
            test_metrics = evaluate(
                model,
                test_loader,
                criterion,
                device,
                cfg,
                visualization_dir if is_main else None,
            )
            if cfg["train"].get("test_time_augmentation", False):
                test_tta_metrics = evaluate(model, test_loader, criterion, device, cfg, tta=True)
        if is_main:
            atomic_json(
                metric_dir / "evaluation.json",
                {"val": val_metrics, "test": test_metrics, "test_tta": test_tta_metrics},
            )
            text = f"[EVAL] val_mIoU={val_metrics['miou']:.4f}"
            if test_metrics is not None:
                text += (
                    f" test_mIoU={test_metrics['miou']:.4f}"
                    f" test_OA={test_metrics['overall_accuracy']:.4f}"
                )
            print(text, flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_miou, best_epoch = initial_validation_selection(
        initial_validation,
        bool(cfg["train"].get("select_initial_validation", False))
        and not bool(args.resume),
    )
    epochs_without_improvement = 0
    ema: ModelEMA | None = None
    if best_epoch == 0 and is_main:
        atomic_checkpoint(
            checkpoint_dir / "best_val.pt",
            best_checkpoint_payload(model, 0, best_miou, cfg),
        )
        atomic_json(
            metric_dir / "best_val.json",
            {"epoch": 0, "miou": best_miou, "metrics": initial_validation},
        )
        print(
            f"[INITIAL SELECTED] epoch=0 best_mIoU={best_miou:.4f}",
            flush=True,
        )
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrap_model(model).load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        if "ema" in resume:
            ema = ModelEMA(
                model,
                decay=resume["ema"]["decay"],
                update_every=resume["ema"]["update_every"],
            )
            ema.load_state_dict(resume["ema"], device)
        start_epoch = int(resume["epoch"])
        best_miou = float(resume["best_val_miou"])
        best_epoch = int(resume.get("best_epoch", start_epoch))
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))
        history_path = metric_dir / "history.json"
        if is_main and history_path.exists():
            with history_path.open("r", encoding="utf-8") as stream:
                history = json.load(stream)
        if is_main:
            print(
                f"[RESUME] epoch={start_epoch} best_epoch={best_epoch} "
                f"best_mIoU={best_miou:.4f}",
                flush=True,
            )
    run_start = time.perf_counter()

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        active_curriculum = apply_training_curriculum(
            model,
            criterion,
            train_dataset,
            cfg,
            epoch + 1,
        )
        train_dataset.set_epoch(epoch)
        ema_start = int(cfg["train"].get("ema_start_epoch", 0))
        if ema is None and epoch + 1 >= ema_start and cfg["train"].get("ema_decay", 0.0) > 0:
            ema = ModelEMA(
                model,
                decay=cfg["train"]["ema_decay"],
                update_every=cfg["train"].get("ema_update_every", 1),
            )
        refinement_start = int(
            cfg["train"].get("refinement_start_epoch", cfg["train"]["epochs"] + 1),
        )
        if epoch + 1 == refinement_start:
            criterion.boundary_weight = cfg["loss"].get(
                "refinement_boundary_weight",
                criterion.boundary_weight,
            )
            criterion.label_smoothing = cfg["loss"].get(
                "refinement_label_smoothing",
                criterion.label_smoothing,
            )
            unwrap_model(model).modality_dropout = cfg["model"].get(
                "refinement_modality_dropout",
                unwrap_model(model).modality_dropout,
            )
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg, ema)

        do_validation = should_validate_epoch(cfg["train"], epoch + 1)
        # Full 128x128 evaluation follows a memory-heavy augmented training
        # epoch.  Live tensors are gone when train_one_epoch returns, but the
        # caching allocator can retain fragmented multi-GiB blocks.  Releasing
        # only unused cache here changes no model/optimizer state and prevents
        # validation-only OOMs without reducing crop or batch fidelity.
        if do_validation and device.type == "cuda":
            torch.cuda.empty_cache()
        if do_validation and ema is not None:
            with ema.average_parameters(model):
                val_metrics = evaluate(model, val_loader, criterion, device, cfg)
        else:
            val_metrics = evaluate(model, val_loader, criterion, device, cfg) if do_validation else None
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        peak_memory = torch.zeros(2, device=device)
        if device.type == "cuda":
            peak_memory[0] = torch.cuda.max_memory_allocated(device)
            peak_memory[1] = torch.cuda.max_memory_reserved(device)
        if dist.is_initialized():
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        peak_allocated_gib = float(peak_memory[0].item() / 1024**3)
        peak_reserved_gib = float(peak_memory[1].item() / 1024**3)

        improved = val_metrics is not None and val_metrics["miou"] > best_miou
        if improved:
            best_miou = val_metrics["miou"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        elif do_validation:
            epochs_without_improvement += 1

        if is_main:
            record = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
                "seconds": epoch_seconds,
                "peak_allocated_gib": peak_allocated_gib,
                "peak_reserved_gib": peak_reserved_gib,
                "curriculum": active_curriculum,
            }
            history.append(record)
            atomic_json(metric_dir / "history.json", history)
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_miou,
                best_epoch,
                epochs_without_improvement,
                cfg,
                ema,
            )
            if improved:
                atomic_checkpoint(
                    checkpoint_dir / "best_val.pt",
                    best_checkpoint_payload(model, epoch + 1, best_miou, cfg, ema),
                )
                atomic_json(
                    metric_dir / "best_val.json",
                    {"epoch": best_epoch, "miou": best_miou, "metrics": val_metrics},
                )
            save_top_k = int(cfg["output"].get("save_top_k", 0))
            if save_top_k > 0 and val_metrics is not None:
                retained_epochs = set(top_validation_epochs(history, save_top_k))
                if epoch + 1 in retained_epochs:
                    atomic_checkpoint(
                        checkpoint_dir / f"top_val_epoch_{epoch + 1:03d}.pt",
                        best_checkpoint_payload(
                            model,
                            epoch + 1,
                            float(val_metrics["miou"]),
                            cfg,
                            ema,
                        ),
                    )
                for candidate in checkpoint_dir.glob("top_val_epoch_*.pt"):
                    try:
                        candidate_epoch = int(candidate.stem.rsplit("_", 1)[-1])
                    except ValueError:
                        continue
                    if candidate_epoch not in retained_epochs:
                        candidate.unlink()
            if cfg["output"]["save_last"]:
                atomic_checkpoint(checkpoint_dir / "last.pt", payload)

            completed_this_run = epoch - start_epoch + 1
            eta = (time.perf_counter() - run_start) / completed_this_run * (
                cfg["train"]["epochs"] - epoch - 1
            )
            val_text = (
                f"val_loss={val_metrics['loss']:.4f} val_mIoU={val_metrics['miou']:.4f} "
                f"val_F1={val_metrics['macro_f1']:.4f}"
                if val_metrics is not None
                else "val=skipped"
            )
            strict_weight_text = ""
            if "s2_strict_weight_global_mean" in train_metrics:
                strict_weight_text = (
                    f" s2_strict_w_global="
                    f"{train_metrics['s2_strict_weight_global_mean']:.6f}"
                )
            if "s2_phase_strict_weight_global_mean" in train_metrics:
                strict_weight_text += (
                    f" s2_phase_strict_w_global="
                    f"{train_metrics['s2_phase_strict_weight_global_mean']:.6f}"
                )
            print(
                f"[EPOCH {epoch + 1:03d}/{cfg['train']['epochs']:03d}] "
                f"train_loss={train_metrics['total']:.4f} {val_text} "
                f"{strict_weight_text} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"time={epoch_seconds:.1f}s eta={eta / 3600:.2f}h "
                f"peak_alloc={peak_allocated_gib:.2f}GiB "
                f"peak_reserved={peak_reserved_gib:.2f}GiB",
                flush=True,
            )

        if args.stop_after_epoch > 0 and epoch + 1 >= args.stop_after_epoch:
            if dist.is_initialized():
                dist.barrier()
            if is_main:
                print(
                    f"[CONTROLLED STOP] completed epoch {epoch + 1}; "
                    "held-out test split was not evaluated",
                    flush=True,
                )
            if dist.is_initialized():
                dist.destroy_process_group()
            return

        stop = torch.tensor(
            int(
                do_validation
                and epochs_without_improvement >= cfg["train"]["early_stopping_patience"]
                and epoch + 1 >= cfg["train"].get("early_stopping_min_epoch", 0)
            ),
            device=device,
        )
        if dist.is_initialized():
            dist.broadcast(stop, src=0)
        if bool(stop):
            if is_main:
                print(f"[EARLY STOP] best_epoch={best_epoch} best_mIoU={best_miou:.4f}", flush=True)
            break

    if dist.is_initialized():
        dist.barrier()
    best_checkpoint = torch.load(checkpoint_dir / "best_val.pt", map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(best_checkpoint["model"], strict=True)
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        cfg,
        visualization_dir if is_main else None,
    )
    test_tta_metrics = None
    if cfg["train"].get("test_time_augmentation", False):
        test_tta_metrics = evaluate(model, test_loader, criterion, device, cfg, tta=True)
    if is_main:
        final = {
            "best_epoch": best_epoch,
            "best_val_miou": best_miou,
            "test": test_metrics,
            "test_tta": test_tta_metrics,
            "parameters": parameters,
        }
        atomic_json(metric_dir / "final.json", final)
        print(
            f"[TEST] selected_epoch={best_epoch} mIoU={test_metrics['miou']:.4f} "
            f"F1={test_metrics['macro_f1']:.4f} OA={test_metrics['overall_accuracy']:.4f} "
            f"boundary_F1={test_metrics['boundary_f1']:.4f}",
            flush=True,
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
