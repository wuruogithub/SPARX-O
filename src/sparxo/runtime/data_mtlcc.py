"""MTLCC adapter for the complete official 240 px-grid benchmark.

The canonical path uses an immutable, memory-mapped cache produced directly
from the official TFRecords. A legacy GeoTIFF reader remains available only so
old experiments are inspectable; new experiments select ``memmap_v1``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


MTLCC_RAW_CLASS_IDS = (1, 2, 3, 5, 8, 9, 12, 13, 15, 16, 17, 19, 22, 23, 24, 25, 26)
MTLCC_CLASS_NAMES = (
    "Sugar beet", "Summer oat", "Meadow", "Rape", "Hop", "Winter spelt",
    "Winter triticale", "Beans", "Peas", "Potato", "Soybeans", "Asparagus",
    "Winter wheat", "Winter barley", "Winter rye", "Summer barley", "Maize",
)


class MTLCCS2Tiles(Dataset[dict[str, Any]]):
    """Complete 2016 MTLCC fold-0 tiles with a variable retained time axis."""

    input_channels = 10

    def __init__(
        self,
        root: str | Path,
        stage: Literal["train", "val", "test"],
        split_file: str,
        stats_file: str | Path,
        temporal_keep_range: tuple[float, float] = (0.85, 1.0),
        ignore_index: int = 255,
        augment: bool | None = None,
        max_tiles: int | None = None,
        subset_seed: int = 42,
        sample_rng_mode: str = "worker",
        storage_format: str = "geotiff_legacy",
    ) -> None:
        self.root = Path(root)
        self.stage = stage
        self.storage_format = str(storage_format)
        self.temporal_keep_range = tuple(float(value) for value in temporal_keep_range)
        self.ignore_index = int(ignore_index)
        self.augment = stage == "train" if augment is None else bool(augment)
        self.subset_seed = int(subset_seed)
        self.sample_rng_mode = str(sample_rng_mode)
        self.s2_normalization = "mtlcc_train_zscore_10band"
        self._epoch = 0
        low, high = self.temporal_keep_range
        if not 0.0 < low <= high <= 1.0:
            raise ValueError("temporal_keep_range must satisfy 0 < low <= high <= 1")

        lookup = np.full(256, self.ignore_index, dtype=np.int64)
        for class_index, raw_id in enumerate(MTLCC_RAW_CLASS_IDS):
            lookup[raw_id] = class_index
        self.label_lookup = lookup

        stats_path = Path(stats_file)
        if not stats_path.is_absolute():
            stats_path = self.root / stats_path
        statistics = json.loads(stats_path.read_text())
        mean = torch.tensor(statistics["mean"], dtype=torch.float32)
        std = torch.tensor(statistics["std"], dtype=torch.float32)
        if mean.shape != (self.input_channels,) or std.shape != (self.input_channels,):
            raise ValueError("MTLCC statistics must contain ten channel means and stds")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("MTLCC statistics must be finite with strictly positive std")
        self.channel_mean = mean.view(1, -1, 1, 1)
        self.channel_std = std.view(1, -1, 1, 1)
        self.stats_sha256 = hashlib.sha256(stats_path.read_bytes()).hexdigest()

        if self.storage_format == "memmap_v1":
            self._init_memmap(max_tiles)
        elif self.storage_format == "geotiff_legacy":
            self._init_geotiff(split_file, max_tiles)
        else:
            raise ValueError(f"Unsupported MTLCC storage format: {self.storage_format}")

        digest = hashlib.sha256()
        for sample_id in self.sample_ids:
            digest.update(sample_id.encode("utf-8"))
            digest.update(b"\n")
        self.sampling_train_ids_sha256 = digest.hexdigest()
        self.sampling_pixel_counts = (
            self._label_histogram() if stage == "train" else np.zeros(17, dtype=np.int64)
        )
        self.sampling_class_weights = None
        self.sampling_probability = 0.0
        self.sampling_probability_formula = None
        self.sampling_probability_statistics: dict[str, float] = {}
        self.sampling_weight_formula = None
        self.sampling_quota = np.zeros(17, dtype=np.int64)

    def _init_memmap(self, max_tiles: int | None) -> None:
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "mtlcc_memmap_v1" or not manifest.get("complete", False):
            raise RuntimeError(f"Incomplete or incompatible MTLCC cache: {manifest_path}")
        stage_dir = self.root / self.stage
        stage_meta = json.loads((stage_dir / "metadata.json").read_text())
        ids = [line.strip() for line in (stage_dir / "ids.txt").read_text().splitlines() if line.strip()]
        limit = len(ids) if max_tiles is None else min(len(ids), int(max_tiles))
        self.sample_ids = ids[:limit]
        self.x10 = np.load(stage_dir / "x10.npy", mmap_mode="r")
        self.x20 = np.load(stage_dir / "x20.npy", mmap_mode="r")
        self.doy = np.load(stage_dir / "doy.npy", mmap_mode="r")
        self.labels = np.load(stage_dir / "labels.npy", mmap_mode="r")
        self.lengths = np.load(stage_dir / "lengths.npy", mmap_mode="r")
        expected = int(stage_meta["samples"])
        if not (
            expected == len(ids)
            == self.x10.shape[0]
            == self.x20.shape[0]
            == self.doy.shape[0]
            == self.labels.shape[0]
            == self.lengths.shape[0]
        ):
            raise RuntimeError(f"MTLCC cache count mismatch in {stage_dir}")
        if self.x10.shape[2:] != (24, 24, 4) or self.x20.shape[2:] != (12, 12, 6):
            raise RuntimeError("MTLCC cache does not use the official 240-grid band layout")
        if self.x10.shape[1] != self.x20.shape[1] or self.x10.shape[1] != self.doy.shape[1]:
            raise RuntimeError("MTLCC cache time dimensions disagree")
        if self.labels.shape[1:] != (24, 24):
            raise RuntimeError("MTLCC cache labels must be 24x24")
        self.tile_size = 24
        self.sequence_length_min = int(stage_meta["sequence_length_min"])
        self.sequence_length_max = int(stage_meta["sequence_length_max"])

    def _init_geotiff(self, split_file: str, max_tiles: int | None) -> None:
        self.data_dir = self.root / "data"
        split_path = self.root / "tileids" / split_file
        listed = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        samples: list[tuple[str, tuple[str, ...]]] = []
        for sample_id in listed:
            path = self.data_dir / sample_id
            if not path.is_dir() or not (path / "y.tif").is_file():
                continue
            dates = tuple(sorted(file.name.split("_")[0] for file in path.glob("*_10m.tif")))
            dates = tuple(date for date in dates if (path / f"{date}_20m.tif").is_file())
            if dates:
                samples.append((sample_id, dates))
        if max_tiles is not None:
            samples = samples[: int(max_tiles)]
        if not samples:
            raise RuntimeError(f"No valid MTLCC samples for {self.stage} from {split_path}")
        self.samples = samples
        self.sample_ids = [sample_id for sample_id, _ in samples]
        self.tile_size = 48
        self.sequence_length_min = min(len(dates) for _, dates in samples)
        self.sequence_length_max = max(len(dates) for _, dates in samples)

    def _label_histogram(self) -> np.ndarray:
        histogram = np.zeros(17, dtype=np.int64)
        if self.storage_format == "memmap_v1":
            for start in range(0, len(self), 1024):
                raw = np.asarray(self.labels[start : min(len(self), start + 1024)])
                for class_index, raw_id in enumerate(MTLCC_RAW_CLASS_IDS):
                    histogram[class_index] += int(np.count_nonzero(raw == raw_id))
            return histogram
        raw_to_index = {raw_id: index for index, raw_id in enumerate(MTLCC_RAW_CLASS_IDS)}
        for sample_id, _ in self.samples:
            with rasterio.open(self.data_dir / sample_id / "y.tif") as source:
                label = source.read(1)
            values, counts = np.unique(label, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                index = raw_to_index.get(int(value))
                if index is not None:
                    histogram[index] += int(count)
        return histogram

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.sample_ids)

    @staticmethod
    def _read(path: Path) -> Tensor:
        with rasterio.open(path) as source:
            return torch.from_numpy(source.read().astype(np.float32, copy=False))

    def _read_geotiff_series(self, path: Path, dates: tuple[str, ...]) -> Tensor:
        frames: list[Tensor] = []
        for date in dates:
            x10 = self._read(path / f"{date}_10m.tif")
            x20 = F.interpolate(
                self._read(path / f"{date}_20m.tif").unsqueeze(0),
                size=(self.tile_size, self.tile_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            frames.append(torch.cat((x10, x20), dim=0))
        return torch.stack(frames)

    def _read_memmap_series(self, index: int, indices: Tensor) -> Tensor:
        selected = indices.numpy()
        x10 = torch.from_numpy(np.array(self.x10[index, selected], copy=True)).permute(0, 3, 1, 2)
        x20 = torch.from_numpy(np.array(self.x20[index, selected], copy=True)).permute(0, 3, 1, 2)
        x20 = F.interpolate(x20.float(), size=(24, 24), mode="bilinear", align_corners=False)
        return torch.cat((x10.float(), x20), dim=1)

    def _read_geotiff_target(self, path: Path) -> Tensor:
        with rasterio.open(path / "y.tif") as source:
            raw = source.read(1)
        return self._remap_target(raw, str(path))

    def _remap_target(self, raw: np.ndarray, source: str) -> Tensor:
        if raw.min() < 0 or raw.max() >= len(self.label_lookup):
            raise RuntimeError(f"Unexpected MTLCC label range in {source}")
        return torch.from_numpy(self.label_lookup[raw].copy()).long()

    @staticmethod
    def _physical_dates(dates: tuple[str, ...]) -> Tensor:
        parsed = [datetime.strptime(value, "%Y%m%d") for value in dates]
        reference = datetime(parsed[0].year, 1, 1)
        return torch.tensor([(value - reference).days for value in parsed], dtype=torch.float32)

    def _sample_indices(self, length: int) -> Tensor:
        if self.stage != "train":
            return torch.arange(length)
        low, high = self.temporal_keep_range
        keep_fraction = float(torch.empty(()).uniform_(low, high).item())
        keep_count = max(1, int(round(length * keep_fraction)))
        return torch.randperm(length)[:keep_count].sort().values

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.storage_format == "memmap_v1":
            sample_id = self.sample_ids[index]
            length = int(self.lengths[index])
            if not self.sequence_length_min <= length <= self.sequence_length_max:
                raise RuntimeError(f"Invalid cached sequence length for {sample_id}: {length}")
            indices = self._sample_indices(length)
            series = self._read_memmap_series(index, indices)
            dates = torch.from_numpy(np.array(self.doy[index, indices.numpy()], copy=True)).float()
            target = self._remap_target(np.asarray(self.labels[index]), sample_id)
        else:
            sample_id, available_dates = self.samples[index]
            indices = self._sample_indices(len(available_dates))
            selected_dates = tuple(available_dates[position] for position in indices.tolist())
            path = self.data_dir / sample_id
            series = self._read_geotiff_series(path, selected_dates)
            dates = self._physical_dates(selected_dates)
            target = self._read_geotiff_target(path)
        series = (series - self.channel_mean) / self.channel_std
        if self.augment:
            if bool(torch.randint(0, 2, ()).item()):
                series = torch.flip(series, dims=(-2,))
                target = torch.flip(target, dims=(-2,))
            if bool(torch.randint(0, 2, ()).item()):
                series = torch.flip(series, dims=(-1,))
                target = torch.flip(target, dims=(-1,))
        return {
            "s2": series,
            "s2_dates": dates,
            "s2_valid": torch.ones(len(dates), dtype=torch.bool),
            "target": target,
            "patch_id": sample_id,
            "origin": torch.zeros(2, dtype=torch.int64),
        }


def collate_mtlcc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(item["s2"].shape[0] for item in batch)
    batch_size = len(batch)
    channels, height, width = batch[0]["s2"].shape[1:]
    series = torch.zeros(batch_size, max_length, channels, height, width, dtype=torch.float32)
    dates = torch.zeros(batch_size, max_length, dtype=torch.float32)
    valid = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for index, item in enumerate(batch):
        length = item["s2"].shape[0]
        series[index, :length] = item["s2"]
        dates[index, :length] = item["s2_dates"]
        valid[index, :length] = item["s2_valid"]
    return {
        "s2": series,
        "s2_dates": dates,
        "s2_valid": valid,
        "target": torch.stack([item["target"] for item in batch]),
        "patch_id": [item["patch_id"] for item in batch],
        "origin": torch.stack([item["origin"] for item in batch]),
    }
