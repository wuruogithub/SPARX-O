"""Streaming, physically aligned PASTIS-HD data pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch import Tensor
from torch.utils.data import Dataset

REFERENCE_DATE = datetime(2018, 1, 1)


def _read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [str(row["image"]) for row in csv.DictReader(stream)]


def _parse_dates(values: dict[str, int] | str) -> np.ndarray:
    if isinstance(values, str):
        values = json.loads(values)
    ordered = [values[key] for key in sorted(values, key=lambda value: int(value))]
    days = []
    for value in ordered:
        date = datetime.strptime(str(value), "%Y%m%d")
        days.append((date - REFERENCE_DATE).days)
    return np.asarray(days, dtype=np.float32)


def allocate_class_sampling_quota(
    class_weights: np.ndarray,
    total: int,
    minimum_per_class: int,
) -> np.ndarray:
    """Allocate an exact per-epoch crop budget over positive-weight classes."""
    weights = np.asarray(class_weights, dtype=np.float64)
    quota = np.zeros_like(weights, dtype=np.int64)
    active = np.flatnonzero(weights > 0)
    if total <= 0 or active.size == 0:
        return quota
    minimum = min(max(0, int(minimum_per_class)), total // active.size)
    quota[active] = minimum
    remaining = total - minimum * active.size
    if remaining == 0:
        return quota
    probabilities = weights[active] / weights[active].sum()
    fractional = probabilities * remaining
    addition = np.floor(fractional).astype(np.int64)
    quota[active] += addition
    remainder = remaining - int(addition.sum())
    if remainder:
        order = np.argsort(-(fractional - addition), kind="stable")
        quota[active[order[:remainder]]] += 1
    if int(quota.sum()) != total:
        raise RuntimeError("Class sampling quota does not match the requested total")
    return quota


def derive_class_sampling_weights(
    pixel_counts: np.ndarray,
    specification: dict[str, Any],
) -> np.ndarray:
    """Derive sampler weights from train-only labels with a fixed formula.

    ``enet_inverse_log`` is the class-balancing rule used by the legacy v2.U.1
    constants.  Frequencies are computed over every class except
    ``ignore_classes``; ``zero_weight_classes`` are removed only after the
    formula and normalization so that background can define the same scale
    without being selected as a crop target.
    """
    counts = np.asarray(pixel_counts, dtype=np.float64)
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("pixel_counts must be a non-negative one-dimensional array")
    name = str(specification.get("name", "enet_inverse_log"))
    if name != "enet_inverse_log":
        raise ValueError(f"Unsupported sampling weight formula: {name}")
    ignore = {int(value) for value in specification.get("ignore_classes", [19])}
    zero_weight = {
        int(value) for value in specification.get("zero_weight_classes", [0, 19])
    }
    invalid = sorted((ignore | zero_weight).difference(range(len(counts))))
    if invalid:
        raise ValueError(f"Sampling weight class indices are out of range: {invalid}")
    normalization_classes = np.asarray(
        [index for index in range(len(counts)) if index not in ignore],
        dtype=np.int64,
    )
    supervised_total = float(counts[normalization_classes].sum())
    if supervised_total <= 0:
        raise RuntimeError("The train split contains no supervised pixels")
    offset = float(specification.get("offset", 1.02))
    if offset <= 1.0:
        raise ValueError("enet_inverse_log offset must be greater than 1")
    frequencies = counts[normalization_classes] / supervised_total
    weights = np.zeros_like(counts, dtype=np.float64)
    present = counts[normalization_classes] > 0
    present_classes = normalization_classes[present]
    weights[present_classes] = 1.0 / np.log(offset + frequencies[present])
    if bool(specification.get("normalize_mean", True)):
        weights[present_classes] /= weights[present_classes].mean()
    for class_index in zero_weight:
        weights[class_index] = 0.0
    round_decimals = specification.get("round_decimals")
    if round_decimals is not None:
        weights = np.round(weights, decimals=int(round_decimals))
    return weights.astype(np.float32)


def derive_class_sampling_probability(
    pixel_counts: np.ndarray,
    specification: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Turn train-only class imbalance into a bounded crop-mixture ratio."""
    counts = np.asarray(pixel_counts, dtype=np.float64)
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("pixel_counts must be a non-negative one-dimensional array")
    name = str(specification.get("name", "normalized_entropy_deficit"))
    if name != "normalized_entropy_deficit":
        raise ValueError(f"Unsupported sampling probability formula: {name}")
    ignore = {int(value) for value in specification.get("ignore_classes", [19])}
    invalid = sorted(ignore.difference(range(len(counts))))
    if invalid:
        raise ValueError(f"Sampling probability class indices are out of range: {invalid}")
    classes = np.asarray(
        [index for index in range(len(counts)) if index not in ignore],
        dtype=np.int64,
    )
    total = float(counts[classes].sum())
    if total <= 0 or len(classes) < 2:
        raise RuntimeError("Entropy sampling requires at least two supervised classes")
    probabilities = counts[classes] / total
    positive = probabilities > 0
    entropy = -float(
        np.sum(probabilities[positive] * np.log(probabilities[positive])),
    )
    normalized_entropy = entropy / math.log(len(classes))
    raw_probability = 1.0 - normalized_entropy
    step = float(specification.get("step", 0.05))
    if not 0.0 < step <= 1.0:
        raise ValueError("Entropy sampling quantization step must be in (0, 1]")
    quantization = str(specification.get("quantization", "floor"))
    if quantization != "floor":
        raise ValueError("Entropy sampling currently supports floor quantization only")
    # A tiny tolerance prevents an exact decimal boundary represented just below
    # itself in binary floating point from falling into the preceding bucket.
    quantized = math.floor((raw_probability + 1.0e-12) / step) * step
    minimum = float(specification.get("minimum", 0.0))
    maximum = float(specification.get("maximum", 0.5))
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("Entropy sampling bounds must satisfy 0 <= min <= max <= 1")
    result = round(float(np.clip(quantized, minimum, maximum)), 12)
    return result, {
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "entropy_deficit": raw_probability,
        "quantized_probability": result,
    }


@lru_cache(maxsize=2)
def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Read only date properties from the PASTIS GeoJSON."""
    with path.open("r", encoding="utf-8") as stream:
        geojson = json.load(stream)
    result = {}
    for feature in geojson["features"]:
        props = feature["properties"]
        patch_id = str(props.get("id", props["ID_PATCH"]))
        result[patch_id] = {
            "s2": _parse_dates(props["dates-S2"]),
            "s1a": _parse_dates(props["dates-S1A"]),
            "s1d": _parse_dates(props["dates-S1D"]),
            "fold": int(props["Fold"]),
        }
    return result


class PASTISHDCrops(Dataset[dict[str, Any]]):
    """Online aligned crops with random train and exhaustive evaluation modes."""

    tile_size = 128
    meters_per_lowres_pixel = 10

    def __init__(
        self,
        root: str | Path,
        split_dir: str | Path | None,
        stage: Literal["train", "val", "test"],
        folds: list[int] | tuple[int, ...] | None = None,
        modalities: list[str] | tuple[str, ...] = ("spot", "s2", "s1a", "s1d"),
        crop_size: int = 32,
        train_crops_per_tile: int = 2,
        temporal_dropout: float | dict[str, float] = 0.1,
        augment: bool | None = None,
        s2_clip: tuple[float, float] = (-2000.0, 12000.0),
        s2_normalization: str = "clip_scale",
        s2_norm_file: str = "NORM_S2_patch.json",
        s2_norm_folds: list[int] | tuple[int, ...] | None = None,
        s1_clip: tuple[float, float] = (-40.0, 20.0),
        load_parcels: bool = False,
        load_regions: bool = False,
        region_source: str = "semantic_components",
        s2_cache_root: str | Path | None = None,
        class_aware_sampling: bool = False,
        sampling_candidates: int = 8,
        sampling_probability: float = 0.75,
        sampling_probability_formula: dict[str, Any] | None = None,
        sampling_expected_probability: float | None = None,
        sampling_class_weights: list[float] | None = None,
        sampling_weight_formula: dict[str, Any] | None = None,
        sampling_mode: str = "candidate_score",
        sampling_min_crops_per_class: int = 0,
        sampling_expected_quota: list[int] | None = None,
        teacher_logits_root: str | Path | None = None,
        temporal_sampling: dict[str, Any] | None = None,
        max_tiles: int | None = None,
        subset_seed: int = 42,
        sample_rng_mode: str = "worker",
    ) -> None:
        self.root = Path(root)
        self.stage = stage
        self.modalities = tuple(str(name) for name in modalities)
        unknown_modalities = set(self.modalities).difference({"spot", "s2", "s1a", "s1d"})
        if unknown_modalities or not self.modalities:
            raise ValueError(
                "modalities must contain at least one of spot, s2, s1a and s1d",
            )
        self.crop_size = int(crop_size)
        self.train_crops_per_tile = int(train_crops_per_tile)
        if isinstance(temporal_dropout, dict):
            unknown = set(temporal_dropout).difference({"default", "s2", "s1a", "s1d"})
            if unknown:
                raise ValueError(f"Unknown temporal_dropout streams: {sorted(unknown)}")
            default = float(temporal_dropout.get("default", 0.0))
            self.temporal_dropout = {
                name: float(temporal_dropout.get(name, default)) if stage == "train" else 0.0
                for name in ("s2", "s1a", "s1d")
            }
        else:
            probability = float(temporal_dropout) if stage == "train" else 0.0
            self.temporal_dropout = {name: probability for name in ("s2", "s1a", "s1d")}
        self.augment = stage == "train" if augment is None else augment
        self.s2_clip = tuple(float(v) for v in s2_clip)
        self.s2_normalization = str(s2_normalization)
        if self.s2_normalization not in {"clip_scale", "fold_standardize"}:
            raise ValueError(
                "s2_normalization must be clip_scale or fold_standardize",
            )
        self.s2_mean: np.ndarray | None = None
        self.s2_std: np.ndarray | None = None
        if self.s2_normalization == "fold_standardize":
            normalization_folds = s2_norm_folds if s2_norm_folds is not None else folds
            if not normalization_folds:
                raise ValueError(
                    "fold_standardize requires s2_norm_folds or explicit dataset folds",
                )
            norm_path = self.root / str(s2_norm_file)
            with norm_path.open("r", encoding="utf-8") as stream:
                norm_values = json.load(stream)
            selected = [
                norm_values[f"Fold_{int(fold)}"] for fold in normalization_folds
            ]
            self.s2_mean = np.mean(
                np.asarray([value["mean"] for value in selected], dtype=np.float32),
                axis=0,
            ).reshape(1, -1, 1, 1)
            self.s2_std = np.mean(
                np.asarray([value["std"] for value in selected], dtype=np.float32),
                axis=0,
            ).reshape(1, -1, 1, 1)
            if self.s2_mean.shape[1] != 10 or self.s2_std.shape[1] != 10:
                raise ValueError("PASTIS S2 normalization must contain ten bands")
            if np.any(self.s2_std <= 0):
                raise ValueError("PASTIS S2 standard deviations must be positive")
        self.s1_clip = tuple(float(v) for v in s1_clip)
        self.load_parcels = load_parcels
        self.load_regions = load_regions
        self.region_source = str(region_source)
        self.s2_cache_root = None if s2_cache_root is None else Path(s2_cache_root)
        if self.s2_cache_root is not None and self.s2_normalization != "clip_scale":
            raise ValueError(
                "The existing S2 cache is defined in clip_scale space; "
                "disable it for fold_standardize",
            )
        self.class_aware_sampling = bool(class_aware_sampling) and stage == "train"
        self.sampling_candidates = int(sampling_candidates)
        self.sampling_probability = float(sampling_probability)
        self.sampling_probability_formula = (
            None
            if sampling_probability_formula is None
            else dict(sampling_probability_formula)
        )
        self.sampling_expected_probability = (
            None
            if sampling_expected_probability is None
            else float(sampling_expected_probability)
        )
        self.sampling_probability_statistics: dict[str, float] = {}
        self.sampling_mode = str(sampling_mode)
        self.sampling_min_crops_per_class = int(sampling_min_crops_per_class)
        self.subset_seed = int(subset_seed)
        self.sample_rng_mode = str(sample_rng_mode)
        if self.sample_rng_mode not in {"worker", "epoch_index"}:
            raise ValueError("sample_rng_mode must be worker or epoch_index")
        self.teacher_logits_root = (
            Path(teacher_logits_root)
            if teacher_logits_root is not None and stage == "train"
            else None
        )
        self.sampling_class_weights = (
            None
            if sampling_class_weights is None
            else np.asarray(sampling_class_weights, dtype=np.float32)
        )
        self.sampling_weight_formula = (
            None if sampling_weight_formula is None else dict(sampling_weight_formula)
        )
        self.sampling_expected_quota = (
            None
            if sampling_expected_quota is None
            else np.asarray(sampling_expected_quota, dtype=np.int64)
        )
        self.sampling_pixel_counts = np.zeros(20, dtype=np.int64)
        self.sampling_train_ids_sha256 = ""
        temporal_sampling = {} if temporal_sampling is None else dict(temporal_sampling)
        self.temporal_sampling_mode = str(temporal_sampling.get("mode", "none"))
        self.max_observations = {
            name: int(temporal_sampling.get(name, 0))
            for name in ("s2", "s1a", "s1d")
        }
        self.temporal_random_alternative_probability = float(
            temporal_sampling.get("random_alternative_probability", 0.0),
        )

        if not 0 < self.crop_size <= self.tile_size:
            raise ValueError("crop_size must be in [1, 128]")
        if self.region_source not in {"semantic_components", "parcels"}:
            raise ValueError("region_source must be semantic_components or parcels")
        if self.load_regions and self.region_source == "parcels" and not self.load_parcels:
            raise ValueError("parcel regions require load_parcels=True")
        if stage != "train" and self.tile_size % self.crop_size:
            raise ValueError("Evaluation crop_size must divide the 128-pixel label tile")
        if any(not 0.0 <= value < 1.0 for value in self.temporal_dropout.values()):
            raise ValueError("Each temporal_dropout probability must be in [0, 1)")
        if self.sampling_candidates < 1:
            raise ValueError("sampling_candidates must be positive")
        if not 0.0 <= self.sampling_probability <= 1.0:
            raise ValueError("sampling_probability must be in [0, 1]")
        if self.sampling_mode not in {
            "candidate_score",
            "class_centered",
            "global_class_centered",
        }:
            raise ValueError(
                "sampling_mode must be candidate_score, class_centered, "
                "or global_class_centered",
            )
        if self.sampling_min_crops_per_class < 0:
            raise ValueError("sampling_min_crops_per_class cannot be negative")
        if self.temporal_sampling_mode not in {"none", "seasonal_medoid"}:
            raise ValueError("temporal sampling mode must be none or seasonal_medoid")
        if any(value < 0 for value in self.max_observations.values()):
            raise ValueError("temporal observation limits cannot be negative")
        if not 0.0 <= self.temporal_random_alternative_probability <= 1.0:
            raise ValueError("random_alternative_probability must be in [0,1]")
        if self.class_aware_sampling:
            if (
                self.sampling_class_weights is None
                and self.sampling_weight_formula is None
            ):
                raise ValueError(
                    "class-aware sampling requires sampling_class_weights or "
                    "sampling_weight_formula",
                )
            if (
                self.sampling_class_weights is not None
                and self.sampling_weight_formula is not None
            ):
                raise ValueError(
                    "Use either explicit sampling_class_weights or an automatic "
                    "sampling_weight_formula, not both",
                )
            if (
                self.sampling_class_weights is not None
                and len(self.sampling_class_weights) < 20
            ):
                # Class 19 is ignore; append a zero if only the 19 train classes were supplied.
                if len(self.sampling_class_weights) == 19:
                    self.sampling_class_weights = np.concatenate(
                        (self.sampling_class_weights, np.zeros(1, dtype=np.float32)),
                    )
                else:
                    raise ValueError("sampling_class_weights must contain 19 or 20 values")
            if self.sampling_expected_quota is not None and len(
                self.sampling_expected_quota
            ) != 20:
                raise ValueError("sampling_expected_quota must contain 20 values")

        all_metadata = load_metadata(self.root / "metadata.geojson")
        if split_dir is not None:
            self.ids = _read_ids(Path(split_dir) / f"{stage}.csv")
        else:
            if not folds:
                raise ValueError("folds are required when split_dir is null")
            selected_folds = {int(value) for value in folds}
            if not selected_folds.issubset({1, 2, 3, 4, 5}):
                raise ValueError("PASTIS folds must be selected from 1..5")
            self.ids = [
                patch_id
                for patch_id, metadata in all_metadata.items()
                if int(metadata["fold"]) in selected_folds
            ]
        if max_tiles is not None and int(max_tiles) < len(self.ids):
            if int(max_tiles) <= 0:
                raise ValueError("max_tiles must be positive when supplied")
            subset_rng = np.random.default_rng(int(subset_seed))
            selected = subset_rng.choice(len(self.ids), size=int(max_tiles), replace=False)
            self.ids = [self.ids[index] for index in sorted(selected.tolist())]
        missing = sorted(set(self.ids).difference(all_metadata))
        if missing:
            raise RuntimeError(f"{len(missing)} split IDs are absent from metadata")
        self.dates = {patch_id: all_metadata[patch_id] for patch_id in self.ids}
        self.sampling_train_ids_sha256 = hashlib.sha256(
            "\n".join(self.ids).encode("utf-8"),
        ).hexdigest()
        self.eval_tiles_per_axis = self.tile_size // self.crop_size
        self.class_patch_ids: dict[int, list[str]] = {}
        self.sampling_quota = np.zeros(20, dtype=np.int64)
        self._sampling_target_by_index = np.full(len(self), -1, dtype=np.int16)
        self._sampling_epoch = 0
        if self.class_aware_sampling and self.sampling_mode == "global_class_centered":
            self._build_global_class_catalog()
            self.set_epoch(0)

    def __len__(self) -> int:
        if self.stage == "train":
            return len(self.ids) * self.train_crops_per_tile
        return len(self.ids) * self.eval_tiles_per_axis**2

    def _build_global_class_catalog(self) -> None:
        """Index train-only tiles containing every supervised crop class."""
        catalog = {class_index: [] for class_index in range(20)}
        pixel_counts = np.zeros(20, dtype=np.int64)
        for patch_id in self.ids:
            target = np.load(
                self.root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy",
                mmap_mode="r",
            )[0]
            pixel_counts += np.bincount(np.asarray(target).ravel(), minlength=20)
            for class_index in np.unique(target):
                class_index = int(class_index)
                catalog[class_index].append(patch_id)
        self.sampling_pixel_counts = pixel_counts
        if self.sampling_probability_formula is not None:
            (
                self.sampling_probability,
                self.sampling_probability_statistics,
            ) = derive_class_sampling_probability(
                pixel_counts,
                self.sampling_probability_formula,
            )
        if (
            self.sampling_expected_probability is not None
            and not math.isclose(
                self.sampling_probability,
                self.sampling_expected_probability,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise RuntimeError(
                "Automatic class-sampling probability drifted from the declared "
                f"reference: observed={self.sampling_probability}, "
                f"expected={self.sampling_expected_probability}",
            )
        if self.sampling_weight_formula is not None:
            self.sampling_class_weights = derive_class_sampling_weights(
                pixel_counts,
                self.sampling_weight_formula,
            )
        assert self.sampling_class_weights is not None
        active = set(np.flatnonzero(self.sampling_class_weights > 0).tolist())
        catalog = {class_index: catalog[class_index] for class_index in active}
        absent = sorted(class_index for class_index, patches in catalog.items() if not patches)
        if absent:
            raise RuntimeError(f"Sampling classes absent from the train split: {absent}")
        self.class_patch_ids = catalog

    def set_epoch(self, epoch: int) -> None:
        """Build an exact, deterministic global class-centered crop schedule."""
        self._sampling_epoch = int(epoch)
        if not (
            self.class_aware_sampling
            and self.sampling_mode == "global_class_centered"
        ):
            return
        assert self.sampling_class_weights is not None
        target_count = min(
            len(self),
            int(round(len(self) * self.sampling_probability)),
        )
        self.sampling_quota = allocate_class_sampling_quota(
            self.sampling_class_weights,
            target_count,
            self.sampling_min_crops_per_class,
        )
        if (
            self.sampling_expected_quota is not None
            and not np.array_equal(self.sampling_quota, self.sampling_expected_quota)
        ):
            raise RuntimeError(
                "Automatic class-sampling quota drifted from the declared reference: "
                f"observed={self.sampling_quota.tolist()}, "
                f"expected={self.sampling_expected_quota.tolist()}",
            )
        schedule = np.repeat(
            np.arange(len(self.sampling_quota), dtype=np.int16),
            self.sampling_quota,
        )
        rng = np.random.default_rng(self.subset_seed + 1_000_003 * self._sampling_epoch)
        rng.shuffle(schedule)
        target_indices = rng.choice(len(self), size=target_count, replace=False)
        self._sampling_target_by_index.fill(-1)
        self._sampling_target_by_index[target_indices] = schedule

    def _origin(self, index: int, rng: np.random.Generator) -> tuple[str, int, int]:
        if self.stage == "train":
            patch_id = self.ids[index % len(self.ids)]
            high = self.tile_size - self.crop_size + 1
            global_target_class = (
                int(self._sampling_target_by_index[index])
                if self.sampling_mode == "global_class_centered"
                else -1
            )
            if (
                self.class_aware_sampling
                and (
                    global_target_class >= 0
                    or (
                        self.sampling_mode == "class_centered"
                        and rng.random() < self.sampling_probability
                    )
                )
            ):
                if global_target_class >= 0:
                    target_class = global_target_class
                    patch_choices = self.class_patch_ids[target_class]
                    patch_id = patch_choices[int(rng.integers(0, len(patch_choices)))]
                source = np.load(
                    self.root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy",
                    mmap_mode="r",
                )[0]
                if global_target_class < 0:
                    counts = np.bincount(np.asarray(source).ravel(), minlength=20)
                    present = np.flatnonzero(
                        (counts > 0) & (self.sampling_class_weights > 0),
                    )
                    if present.size:
                        probabilities = self.sampling_class_weights[present].astype(np.float64)
                        probabilities /= probabilities.sum()
                        target_class = int(rng.choice(present, p=probabilities))
                    else:
                        target_class = -1
                if target_class >= 0:
                    target_pixels = np.argwhere(source == target_class)
                    target_y, target_x = target_pixels[rng.integers(0, len(target_pixels))]
                    jitter = max(1, self.crop_size // 4)
                    y = int(target_y) - self.crop_size // 2 + int(
                        rng.integers(-jitter, jitter + 1),
                    )
                    x = int(target_x) - self.crop_size // 2 + int(
                        rng.integers(-jitter, jitter + 1),
                    )
                    return patch_id, int(np.clip(y, 0, high - 1)), int(
                        np.clip(x, 0, high - 1),
                    )
            candidates = rng.integers(0, high, size=(self.sampling_candidates, 2))
            selected = 0
            if (
                self.class_aware_sampling
                and self.sampling_mode == "candidate_score"
                and rng.random() < self.sampling_probability
            ):
                source = np.load(
                    self.root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy",
                    mmap_mode="r",
                )[0]
                scores = []
                for candidate_y, candidate_x in candidates:
                    labels = np.asarray(
                        source[
                            candidate_y : candidate_y + self.crop_size,
                            candidate_x : candidate_x + self.crop_size,
                        ],
                    )
                    scores.append(float(self.sampling_class_weights[labels].mean()))
                selected = int(np.argmax(scores))
            y, x = candidates[selected].tolist()
            return patch_id, int(y), int(x)

        tiles_per_patch = self.eval_tiles_per_axis**2
        patch_id = self.ids[index // tiles_per_patch]
        tile_index = index % tiles_per_patch
        y = (tile_index // self.eval_tiles_per_axis) * self.crop_size
        x = (tile_index % self.eval_tiles_per_axis) * self.crop_size
        return patch_id, y, x

    def _seasonal_medoid_sample(
        self,
        data: np.ndarray,
        dates: np.ndarray,
        name: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select robust observations spread across the acquisition season."""
        limit = self.max_observations[name]
        if (
            self.temporal_sampling_mode == "none"
            or limit <= 0
            or len(dates) <= limit
        ):
            return data, dates
        boundaries = np.linspace(
            float(dates.min()) - 1.0e-3,
            float(dates.max()) + 1.0e-3,
            limit + 1,
        )
        selected: list[int] = []
        summaries = data.mean(axis=(-2, -1), dtype=np.float64)
        for bin_index in range(limit):
            candidates = np.flatnonzero(
                (dates >= boundaries[bin_index])
                & (dates < boundaries[bin_index + 1]),
            )
            if not candidates.size:
                continue
            local = summaries[candidates]
            median = np.median(local, axis=0, keepdims=True)
            distance = np.mean(np.square(local - median), axis=1)
            ranking = candidates[np.argsort(distance)]
            choice = 0
            if (
                self.stage == "train"
                and len(ranking) > 1
                and rng.random() < self.temporal_random_alternative_probability
            ):
                choice = int(rng.integers(0, min(2, len(ranking))))
            selected.append(int(ranking[choice]))
        selected_array = np.asarray(sorted(set(selected)), dtype=np.int64)
        return data[selected_array], dates[selected_array]

    @staticmethod
    def _drop_dates(
        data: np.ndarray,
        dates: np.ndarray,
        probability: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if probability <= 0.0 or len(dates) <= 1:
            return data, dates
        keep = rng.random(len(dates)) >= probability
        if not keep.any():
            keep[rng.integers(0, len(dates))] = True
        return data[keep], dates[keep]

    def _read_series(
        self,
        patch_id: str,
        name: Literal["s2", "s1a", "s1d"],
        y: int,
        x: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        prefixes = {"s2": ("DATA_S2", "S2"), "s1a": ("DATA_S1A", "S1A"), "s1d": ("DATA_S1D", "S1D")}
        folder, prefix = prefixes[name]
        source = np.load(self.root / folder / f"{prefix}_{patch_id}.npy", mmap_mode="r")
        data = np.asarray(
            source[:, :, y : y + self.crop_size, x : x + self.crop_size],
            dtype=np.float32,
        )
        dates = self.dates[patch_id][name].copy()
        if data.shape[0] != len(dates):
            raise RuntimeError(
                f"{patch_id}/{name}: {data.shape[0]} images but {len(dates)} dates",
            )

        if name == "s2":
            if self.s2_normalization == "fold_standardize":
                assert self.s2_mean is not None and self.s2_std is not None
                data = (data - self.s2_mean) / self.s2_std
            else:
                data = np.clip(data, *self.s2_clip) / 10000.0
        else:
            data = np.clip(data, *self.s1_clip) / 20.0
        data, dates = self._seasonal_medoid_sample(data, dates, name, rng)
        return self._drop_dates(data, dates, self.temporal_dropout[name], rng)

    def _read_cached_s2(
        self,
        patch_id: str,
        y: int,
        x: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read offline-selected raw bands plus six cached physical indices."""

        if self.s2_cache_root is None:
            raise RuntimeError("S2 cache was not configured")
        selection_path = self.s2_cache_root / f"S2_SELECTION_{patch_id}.npy"
        physics_path = self.s2_cache_root / f"S2_PHYSICS_{patch_id}.npy"
        dates_path = self.s2_cache_root / f"S2_DATES_{patch_id}.npy"
        if not selection_path.is_file() or not physics_path.is_file() or not dates_path.is_file():
            raise FileNotFoundError(
                f"Incomplete offline S2 cache for tile {patch_id}: "
                f"run scripts/precompute_v33_s2_cache.py first",
            )
        selection = np.load(selection_path)
        physics_source = np.load(physics_path, mmap_mode="r")
        dates = np.load(dates_path)
        expected_tail = (6, self.tile_size, self.tile_size)
        if physics_source.ndim != 4 or physics_source.shape[1:] != expected_tail:
            raise RuntimeError(f"{physics_path} has invalid shape {physics_source.shape}")
        if len(dates) != len(selection) or len(dates) != physics_source.shape[0]:
            raise RuntimeError(f"Incomplete or inconsistent S2 cache for tile {patch_id}")
        raw_source = np.load(
            self.root / "DATA_S2" / f"S2_{patch_id}.npy",
            mmap_mode="r",
        )
        raw = np.asarray(
            raw_source[
                selection,
                :,
                y : y + self.crop_size,
                x : x + self.crop_size,
            ],
            dtype=np.float32,
        )
        raw = np.clip(raw, *self.s2_clip) / 10000.0
        physics = np.asarray(
            physics_source[
                :,
                :,
                y : y + self.crop_size,
                x : x + self.crop_size,
            ],
            dtype=np.float32,
        )
        data = np.concatenate((raw, physics), axis=1)
        return self._drop_dates(data, dates, self.temporal_dropout["s2"], rng)

    def _read_spot(self, patch_id: str, y: int, x: int) -> np.ndarray:
        scale = self.meters_per_lowres_pixel
        spot_path = (
            self.root
            / "DATA_SPOT"
            / "PASTIS_SPOT6_RVB_1M00_2019"
            / f"SPOT6_RVB_1M00_2019_{patch_id}.tif"
        )
        with rasterio.open(spot_path) as source:
            data = source.read(
                window=Window(
                    col_off=x * scale,
                    row_off=y * scale,
                    width=self.crop_size * scale,
                    height=self.crop_size * scale,
                ),
            )
        expected = (3, self.crop_size * scale, self.crop_size * scale)
        if data.shape != expected:
            raise RuntimeError(f"{spot_path} returned {data.shape}, expected {expected}")
        return data.astype(np.float32) / 255.0

    def _read_target(self, patch_id: str, y: int, x: int) -> np.ndarray:
        source = np.load(self.root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy", mmap_mode="r")
        return np.asarray(
            source[0, y : y + self.crop_size, x : x + self.crop_size],
            dtype=np.int64,
        )

    def _read_teacher_logits(self, patch_id: str, y: int, x: int) -> np.ndarray:
        if self.teacher_logits_root is None:
            raise RuntimeError("Teacher logits were not configured")
        path = self.teacher_logits_root / f"LOGITS_{patch_id}.npy"
        source = np.load(path, mmap_mode="r")
        expected = (19, self.tile_size, self.tile_size)
        if source.shape != expected:
            raise RuntimeError(f"{path} has shape {source.shape}, expected {expected}")
        return np.asarray(
            source[:, y : y + self.crop_size, x : x + self.crop_size],
            dtype=np.float32,
        )

    def _read_parcels(self, patch_id: str, y: int, x: int) -> np.ndarray:
        source = np.load(
            self.root / "ANNOTATIONS" / f"ParcelIDs_{patch_id}.npy",
            mmap_mode="r",
        )
        return np.asarray(
            source[y : y + self.crop_size, x : x + self.crop_size],
            dtype=np.int64,
        )

    @staticmethod
    def _semantic_components(target: np.ndarray) -> np.ndarray:
        """Four-connected non-background components derived only from labels."""
        height, width = target.shape
        regions = np.zeros((height, width), dtype=np.int64)
        next_region = 1
        for y in range(height):
            for x in range(width):
                semantic_class = int(target[y, x])
                if semantic_class in (0, 19) or regions[y, x] != 0:
                    continue
                regions[y, x] = next_region
                stack = [(y, x)]
                while stack:
                    current_y, current_x = stack.pop()
                    for neighbor_y, neighbor_x in (
                        (current_y - 1, current_x),
                        (current_y + 1, current_x),
                        (current_y, current_x - 1),
                        (current_y, current_x + 1),
                    ):
                        if (
                            0 <= neighbor_y < height
                            and 0 <= neighbor_x < width
                            and regions[neighbor_y, neighbor_x] == 0
                            and int(target[neighbor_y, neighbor_x]) == semantic_class
                        ):
                            regions[neighbor_y, neighbor_x] = next_region
                            stack.append((neighbor_y, neighbor_x))
                next_region += 1
        return regions

    @staticmethod
    def _spatial_transform(
        arrays: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        k = int(rng.integers(0, 4))
        flip_y = bool(rng.integers(0, 2))
        flip_x = bool(rng.integers(0, 2))
        transformed = {}
        for name, value in arrays.items():
            out = np.rot90(value, k=k, axes=(-2, -1))
            if flip_y:
                out = np.flip(out, axis=-2)
            if flip_x:
                out = np.flip(out, axis=-1)
            transformed[name] = np.ascontiguousarray(out)
        return transformed

    def _sample_seed(self, index: int) -> int:
        if self.sample_rng_mode == "epoch_index":
            # Make crops, augmentation and temporal dropout invariant to worker
            # assignment, micro-batch size and DDP packing. Epoch remains part
            # of the seed so training does not replay the same crop forever.
            return (
                self.subset_seed
                + 1_000_003 * self._sampling_epoch
                + 9_176 * int(index)
            ) % (2**32)
        # Legacy behaviour retained for historical checkpoints/configs.
        return (torch.initial_seed() + index) % (2**32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(self._sample_seed(index))
        patch_id, y, x = self._origin(index, rng)

        arrays = {
            "target": self._read_target(patch_id, y, x),
        }
        dates = {}
        if "spot" in self.modalities:
            arrays["spot"] = self._read_spot(patch_id, y, x)
        for name in ("s2", "s1a", "s1d"):
            if name in self.modalities:
                if name == "s2" and self.s2_cache_root is not None:
                    series, series_dates = self._read_cached_s2(patch_id, y, x, rng)
                else:
                    series, series_dates = self._read_series(patch_id, name, y, x, rng)
                arrays[name] = series
                dates[name] = series_dates
        if self.teacher_logits_root is not None:
            arrays["teacher_logits"] = self._read_teacher_logits(patch_id, y, x)
        if self.load_parcels:
            arrays["parcels"] = self._read_parcels(patch_id, y, x)
        if self.augment:
            arrays = self._spatial_transform(arrays, rng)
        if self.load_regions:
            arrays["regions"] = (
                np.asarray(arrays["parcels"], dtype=np.int64)
                if self.region_source == "parcels"
                else self._semantic_components(arrays["target"])
            )

        item: dict[str, Any] = {
            **{key: torch.from_numpy(value) for key, value in arrays.items()},
            **{
                f"{name}_dates": torch.from_numpy(series_dates)
                for name, series_dates in dates.items()
            },
            "patch_id": patch_id,
            "origin": torch.tensor([y, x], dtype=torch.int64),
        }
        return item


def _pad_series(batch: list[dict[str, Any]], name: str) -> tuple[Tensor, Tensor, Tensor]:
    lengths = torch.tensor([item[name].shape[0] for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    first = batch[0][name]
    data = first.new_zeros((len(batch), max_length, *first.shape[1:]))
    dates = torch.zeros((len(batch), max_length), dtype=torch.float32)
    valid = torch.zeros((len(batch), max_length), dtype=torch.bool)
    for index, item in enumerate(batch):
        length = item[name].shape[0]
        data[index, :length] = item[name]
        dates[index, :length] = item[f"{name}_dates"]
        valid[index, :length] = True
    return data, dates, valid


def collate_pastis_hd(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad each satellite sequence independently."""
    output: dict[str, Any] = {
        "target": torch.stack([item["target"] for item in batch]),
        "patch_id": [item["patch_id"] for item in batch],
        "origin": torch.stack([item["origin"] for item in batch]),
    }
    if "spot" in batch[0]:
        output["spot"] = torch.stack([item["spot"] for item in batch])
    if "parcels" in batch[0]:
        output["parcels"] = torch.stack([item["parcels"] for item in batch])
    if "regions" in batch[0]:
        output["regions"] = torch.stack([item["regions"] for item in batch])
    if "teacher_logits" in batch[0]:
        output["teacher_logits"] = torch.stack(
            [item["teacher_logits"] for item in batch],
        )
    for name in ("s2", "s1a", "s1d"):
        if name not in batch[0]:
            continue
        data, dates, valid = _pad_series(batch, name)
        output[name] = data
        output[f"{name}_dates"] = dates
        output[f"{name}_valid"] = valid
    return output


def estimate_steps_per_epoch(dataset: PASTISHDCrops, batch_size: int, world_size: int) -> int:
    """Number of optimizer-independent loader steps per epoch."""
    return math.ceil(len(dataset) / max(1, batch_size * world_size))
