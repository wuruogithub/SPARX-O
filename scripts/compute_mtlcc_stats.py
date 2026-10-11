#!/usr/bin/env python3
"""Compute Exchanger-style 10-band statistics from MTLCC train_fold0 only."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio


def sample_moments(path_text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    path = Path(path_text)
    sums = np.zeros(10, dtype=np.float64)
    squares = np.zeros(10, dtype=np.float64)
    counts = np.zeros(10, dtype=np.int64)
    observations = 0
    for x10_path in path.glob("*_10m.tif"):
        date = x10_path.name.split("_")[0]
        x20_path = path / f"{date}_20m.tif"
        if not x20_path.is_file():
            continue
        with rasterio.open(x10_path) as source:
            x10 = source.read().astype(np.float64, copy=False)
        with rasterio.open(x20_path) as source:
            x20 = source.read().astype(np.float64, copy=False)
        for offset, array in ((0, x10), (4, x20)):
            sums[offset : offset + array.shape[0]] += array.sum(axis=(1, 2))
            squares[offset : offset + array.shape[0]] += np.square(array).sum(axis=(1, 2))
            counts[offset : offset + array.shape[0]] += array.shape[1] * array.shape[2]
        observations += 1
    return sums, squares, counts, observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="train_fold0.tileids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    ids = [line.strip() for line in (args.root / "tileids" / args.split).read_text().splitlines() if line.strip()]
    paths = [args.root / "data" / sample_id for sample_id in ids]
    paths = [path for path in paths if path.is_dir()]
    total_sum = np.zeros(10, dtype=np.float64)
    total_squares = np.zeros(10, dtype=np.float64)
    total_count = np.zeros(10, dtype=np.int64)
    observations = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for sums, squares, counts, sample_observations in executor.map(
            sample_moments,
            map(str, paths),
            chunksize=8,
        ):
            total_sum += sums
            total_squares += squares
            total_count += counts
            observations += sample_observations
    mean = total_sum / total_count
    variance = total_squares / total_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    result = {
        "protocol": "Exchanger MTLCC train-fold z-normalization; x10 then x20",
        "split": args.split,
        "tiles": len(paths),
        "observations": observations,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "pixel_count_per_channel": total_count.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
